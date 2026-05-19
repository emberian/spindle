//! Active-inference + w-maxing controller substrate for the rigger
//! decision layer (the chamber-alive rebuild).
//!
//! WHY THIS EXISTS (the diagnosis): the old decision layer was
//! *maximally over-specific* in Bennett's w-maxing sense — the throw
//! solver fired only on a near-exact closed-form gate solution (strongest
//! possible precondition ⇒ never throws), and the planner picked the
//! argmin anchor (maximally specific ⇒ local twitch). Nobody played the
//! chamber as a volume; nobody anticipated; riggers almost never threw.
//!
//! THE REBUILD. This module supplies, deterministically (seeded only, no
//! wall-clock, no thread_rng, no HashMap-iteration folds):
//!
//!  1. A **generative model**: short-horizon prediction of the bell and of
//!     agents via the canon Coriolis predictor `crate::trajectory::rk4_step`
//!     + the exact sim skin-bounce. We roll a *band* of bell futures (a
//!     spread of horizons) so the controller can reason about a RANGE of
//!     futures, not a single point.
//!
//!  2. **Expected Free Energy = pragmatic + epistemic.** PRAGMATIC: reach
//!     preferred states (get/keep possession, advance toward the attack
//!     gate, deny the opponent). EPISTEMIC: keep the rigger able to respond
//!     to a *range* of bell futures, and keep the TEAM spread to cover the
//!     chamber volume so there is usually someone kinematically favorable.
//!
//!  3. **W-maxing policy selection (Bennett's razor):** among actions whose
//!     predicted EFE is acceptable ("fit"), choose the WEAKEST = the one
//!     whose success-set is largest / least committal / robust to error.
//!     For a throw that is the *widest tolerance band* (largest set of
//!     release errors that still yield a good outcome), NOT the perfect
//!     argmin shot. For navigation it is the broad region that keeps the
//!     most future options / volume coverage open, NOT the argmin anchor.
//!
//! Everything here is pure geometry + the canon integrator. No RNG is
//! drawn in this module; callers inject difficulty noise exactly where the
//! frozen spec already did (so the determinism guards stay green).

use super::orientation::{attack_ring_x, attack_sign, forward_progress};
use super::types::{PlayerSim, SimState, TeamSide};
use crate::math::Vec3;
use crate::trajectory::{rk4_step, PointState};

const PRED_SKIN_RESTITUTION: f64 = 0.55;

/// Skin bounce — EXACT mirror of the sim Collision.skinBounce the predictor
/// in rigger_ai already used (RESTITUTION 0.55, no rng). Duplicated here so
/// the generative model is self-contained.
pub fn predict_skin_bounce(st: &mut PointState, skin_r: f64) {
    let rho = st.p.y.hypot(st.p.z);
    if rho < skin_r {
        return;
    }
    let ny = st.p.y / rho;
    let nz = st.p.z / rho;
    let vn = st.v.y * ny + st.v.z * nz;
    if vn > 0.0 {
        st.v.y -= (1.0 + PRED_SKIN_RESTITUTION) * vn * ny;
        st.v.z -= (1.0 + PRED_SKIN_RESTITUTION) * vn * nz;
    }
    let s = (skin_r - 1e-3) / rho;
    st.p.y *= s;
    st.p.z *= s;
}

/// Roll a point (bell or agent) forward by `t` seconds with the canon
/// integrator + skin bounce, at a fixed 60 Hz substep. Deterministic.
pub fn roll_forward(p: Vec3, v: Vec3, omega: f64, t: f64, skin_r: f64) -> PointState {
    let mut st = PointState { p, v };
    if t <= 0.0 {
        return st;
    }
    let h: f64 = 1.0 / 60.0;
    let mut acc = 0.0_f64;
    while acc < t {
        let step = h.min(t - acc);
        st = rk4_step(st, omega, step);
        predict_skin_bounce(&mut st, skin_r);
        acc += step;
    }
    st
}

/// The generative model's predicted bell *band*: where the bell is over a
/// spread of short horizons. The spread (not a single point) is what lets
/// the controller reason about — and stay responsive to — a RANGE of
/// futures. Fixed-length, deterministic, no allocation churn beyond this
/// small Vec.
#[derive(Clone, Debug)]
pub struct BellBand {
    /// (t_seconds, predicted_state) samples, near→far.
    pub samples: Vec<(f64, PointState)>,
}

impl BellBand {
    /// Predict the loose/in-flight bell over a fan of horizons. Horizons
    /// are fixed (deterministic): a near, a mid and a far look so the
    /// controller sees the bell's whole near-term envelope.
    pub fn predict(bell_p: Vec3, bell_v: Vec3, omega: f64, skin_r: f64) -> Self {
        // Fixed horizon fan — short enough that the Coriolis prediction is
        // trustworthy, wide enough to span "where could this be soon".
        const HORIZONS: [f64; 5] = [0.25, 0.6, 1.1, 1.8, 2.6];
        let mut samples = Vec::with_capacity(HORIZONS.len());
        for &t in HORIZONS.iter() {
            samples.push((t, roll_forward(bell_p, bell_v, omega, t, skin_r)));
        }
        BellBand { samples }
    }

    /// Mid sample — a stable "representative" future bell point.
    pub fn mid(&self) -> PointState {
        self.samples[self.samples.len() / 2].1
    }

    /// Expected distance for `from` to be able to RESPOND to the bell over
    /// the predicted band, assuming the agent can close at `close_v` m/s.
    /// This is the core epistemic quantity: "how far am I, in time-to-
    /// intercept, from being able to handle whatever the bell does?".
    /// Low = responsive across many futures (preferred). It averages the
    /// shortfall over the whole band so a position good for ONE future but
    /// useless for the rest scores poorly — that is the volume-play drive.
    pub fn expected_response_gap(&self, from: Vec3, close_v: f64) -> f64 {
        let mut acc = 0.0_f64;
        for (t, st) in self.samples.iter() {
            let reach = (from.x - st.p.x).hypot(from.y - st.p.y).hypot(from.z - st.p.z);
            // Time we'd need vs. time available: shortfall, never negative.
            let need_t = reach / close_v.max(1e-3);
            let short = (need_t - t).max(0.0);
            acc += short;
        }
        acc / self.samples.len() as f64
    }

    /// The single band sample this agent is *most* favorable to intercept
    /// (smallest response shortfall). Used to aim a pragmatic possession
    /// action at the part of the bell future this agent can actually win.
    pub fn best_intercept_for(&self, from: Vec3, close_v: f64) -> Vec3 {
        let mut best = self.samples[0].1.p;
        let mut best_short = f64::INFINITY;
        for (t, st) in self.samples.iter() {
            let reach = (from.x - st.p.x).hypot(from.y - st.p.y).hypot(from.z - st.p.z);
            let short = (reach / close_v.max(1e-3) - t).max(0.0);
            if short < best_short {
                best_short = short;
                best = st.p;
            }
        }
        best
    }

    /// Predicted TIME-TO-INTERCEPT (seconds) for an agent at `from` moving
    /// at `from_v`, closing on the loose bell with a confident powered-hook
    /// model `REEL_PULL_SPEED + min(inbound,14)` (the existing decisive-dive
    /// close-speed precedent). We walk the band near→far and return the
    /// earliest horizon `t` at which the agent could already be at that
    /// predicted bell point (reach/close_v ≤ t); if it can never quite catch
    /// up within the band we return the *last* sample's pure travel time
    /// (still a monotone, comparable scalar). Pure geometry, deterministic,
    /// no rng — this is the role-assignment race metric.
    pub fn time_to_intercept(&self, from: Vec3, from_v: Vec3) -> f64 {
        // Inbound-velocity credit toward the bell's CURRENT position (the
        // near sample) so a rigger already driving at the ball is correctly
        // judged the faster racer — mirrors `dive_intercept_gap`.
        let near = self.samples[0].1.p;
        let to_b = Vec3::new(near.x - from.x, near.y - from.y, near.z - from.z);
        let bl = (to_b.x * to_b.x + to_b.y * to_b.y + to_b.z * to_b.z).sqrt();
        let inbound = if bl > 1e-6 {
            ((from_v.x * to_b.x + from_v.y * to_b.y + from_v.z * to_b.z) / bl)
                .max(0.0)
        } else {
            0.0
        };
        let close_v = (crate::tuning::REEL_PULL_SPEED + inbound.min(14.0)).max(1e-3);
        let mut fallback = f64::INFINITY;
        for (t, st) in self.samples.iter() {
            let reach = (from.x - st.p.x)
                .hypot(from.y - st.p.y)
                .hypot(from.z - st.p.z);
            let need_t = reach / close_v;
            fallback = need_t; // last assignment = far-sample travel time
            if need_t <= *t {
                return *t;
            }
        }
        fallback
    }
}

/// Predicted REBOUND LOCUS — where the loose bell ends up if the PRIMARY
/// diver BOBBLES/DEFLECTS it instead of cleanly catching. Deterministic
/// model, faithful to `collision::apply_bobble`:
///   * roll the live bell forward to the primary's predicted touch (the
///     bell state at the moment the dive arrives);
///   * apply the EXACT bobble linear-velocity transform: with a roughly
///     stationary diver at the contact (`player_vel ≈ 0`) the sim sets
///     `bell.v = player_vel + (bell.v − player_vel) * 0.25` ⇒ a clean
///     0.25 damp of the pre-touch linear velocity. (The sim's bobble
///     `clatter((0,4,3))` is an ANGULAR-only impulse — it perturbs the
///     bell's spin `w`, NOT its linear `v`, so it does not enter the
///     point-mass roll-forward; we deliberately do not add a phantom
///     linear kick. The spin perturbation's effect on the path is
///     second-order via Coriolis and is intentionally not modelled here
///     — a stationed shadow only needs the dominant spill direction.)
///   * roll THAT damped bell forward a short spill horizon so the shadow
///     has a distinct, reachable place to wait that is NOT the primary's
///     intercept point.
/// No rng, deterministic.
pub fn rebound_locus(
    bell_p: Vec3,
    bell_v: Vec3,
    omega: f64,
    skin_r: f64,
    t_primary: f64,
) -> Vec3 {
    // 1. Bell state at the primary's predicted touch.
    let touch = roll_forward(bell_p, bell_v, omega, t_primary.max(0.0), skin_r);
    // 2. apply_bobble linear transform with a ~stationary diver: a clean
    //    0.25 damp of the pre-touch linear velocity (the off-axis clatter
    //    is angular-only in the sim and does not move the point mass).
    let bob_v = Vec3::new(touch.v.x * 0.25, touch.v.y * 0.25, touch.v.z * 0.25);
    // 3. Roll the deflected bell forward a short spill horizon — far enough
    //    that the shadow has somewhere distinct to wait, near enough that
    //    the Coriolis prediction is still trustworthy.
    const SPILL_T: f64 = 0.9;
    let spill = roll_forward(touch.p, bob_v, omega, SPILL_T, skin_r).p;
    // Guard against a barely-moving deflection collapsing the shadow onto
    // the primary's exact touch (they MUST hold distinct stations): if the
    // spill is essentially the touch point, nudge along the post-bobble
    // heading by a fixed stand-off so the shadow waits OFF the dive line.
    let dx = spill.x - touch.p.x;
    let dy = spill.y - touch.p.y;
    let dz = spill.z - touch.p.z;
    let sep = (dx * dx + dy * dy + dz * dz).sqrt();
    const MIN_SHADOW_STANDOFF: f64 = 6.0;
    if sep >= MIN_SHADOW_STANDOFF {
        return spill;
    }
    // Deterministic fallback heading: the damped velocity if non-trivial,
    // else straight "up" the tube cross-section (fixed, no rng).
    let bvl = (bob_v.x * bob_v.x + bob_v.y * bob_v.y + bob_v.z * bob_v.z).sqrt();
    let (hx, hy, hz) = if bvl > 1e-6 {
        (bob_v.x / bvl, bob_v.y / bvl, bob_v.z / bvl)
    } else {
        (0.0, 1.0, 0.0)
    };
    Vec3::new(
        touch.p.x + hx * MIN_SHADOW_STANDOFF,
        touch.p.y + hy * MIN_SHADOW_STANDOFF,
        touch.p.z + hz * MIN_SHADOW_STANDOFF,
    )
}

/// TEAM volume coverage — operationalized. The chamber is a tube of radius
/// `skin_r` and (effective) length 2·gate_x along x. "Coverage" reduces to:
/// the team's positions, projected to (forward_progress, cross-angle),
/// should be SPREAD, not collapsed on the ball. We measure it as the mean
/// pairwise closeness penalty among teammates (high when everyone clumps)
/// plus a term for under-occupying the forward axis. An off-ball agent's
/// epistemic action is the one that most REDUCES this — i.e. it moves to
/// fill an empty part of the volume.
pub struct VolumeModel {
    pub team: TeamSide,
    pub gate_x: f64,
    pub skin_r: f64,
}

impl VolumeModel {
    pub fn new(team: TeamSide, gate_x: f64, skin_r: f64) -> Self {
        VolumeModel { team, gate_x, skin_r }
    }

    /// A scalar "crowding" cost of placing `cand` given the rest of the
    /// team `others` (teammate positions, excluding the deciding agent).
    /// Lower = better spread. Pure geometry, deterministic, order-free
    /// (a sum — commutative — so no HashMap-iteration-order hazard even
    /// though callers pass a slice).
    pub fn crowding_at(&self, cand: Vec3, others: &[Vec3]) -> f64 {
        let mut acc = 0.0_f64;
        for o in others {
            let d = (cand.x - o.x)
                .hypot(cand.y - o.y)
                .hypot(cand.z - o.z);
            // A soft repulsion: strong when within ~one body-spacing,
            // negligible far away. Spacing scale ~ chamber radius.
            let s = (self.skin_r * 0.9).max(8.0);
            acc += (-(d * d) / (s * s)).exp();
        }
        acc
    }

    /// How well a forward-progress depth `fp` is *not* already occupied —
    /// rewards spanning the volume along the attack axis. `others_fp` are
    /// teammates' forward-progress values. Lower = this depth band is empty
    /// (good to go fill it).
    pub fn axis_redundancy(&self, fp: f64, others_fp: &[f64]) -> f64 {
        let mut acc = 0.0_f64;
        let band = self.gate_x.abs().max(1.0) * 0.18;
        for &o in others_fp {
            let d = (fp - o).abs();
            acc += (-(d * d) / (band * band)).exp();
        }
        acc
    }
}

// ─── Pass-success generative-model primitives ───────────────────────────────
//
// These score **passing lane quality** rather than the old "response gap"
// measure. They answer: "would a pass TO this position succeed?" and "is
// this throw a good idea?" — enabling off-ball players to seek positions
// with high reception quality (wide basins of successful throws) and the
// throw controller to pick passes that maximize expected value.

/// Defender closing speed assumption (grapple-aided dash), m/s.
const DEF_CLOSE_SPEED: f64 = 22.0;
/// Default ball flight speed when estimating from position only, m/s.
const DEFAULT_BALL_SPEED: f64 = 22.0;
/// Catch radius — generous "can reach the ball" envelope, m.
const CATCH_RADIUS: f64 = 12.0;
/// Pass flight horizon cap, seconds.
const PASS_HORIZON: f64 = 1.5;

/// Score in [0, 1]: "If the carrier threw to a receiver at `candidate_p`
/// moving at `candidate_v`, how likely is successful reception?"
///
/// Models four factors:
/// 1. Lane occlusion — defenders blocking the throw line.
/// 2. Defender closing time — can a defender reach the catch point before
///    the ball arrives?
/// 3. Coriolis-lane quality — does the predicted throw endpoint actually
///    land near the candidate (Coriolis curves it)?
/// 4. Separation — raw distance from nearest defender at the candidate.
pub fn reception_quality(
    carrier_p: Vec3,
    candidate_p: Vec3,
    candidate_v: Vec3,
    defenders: &[Vec3],
    omega: f64,
    skin_r: f64,
) -> f64 {
    let lane = candidate_p.sub(carrier_p);
    let lane_len = lane.len();
    if lane_len < 1e-6 {
        return 0.0;
    }
    let lane_dir = lane.norm();

    // Ball flight time (straight-line estimate).
    let flight_time = lane_len / DEFAULT_BALL_SPEED;

    // ─── 1. Lane occlusion ───
    // For each defender: project onto the throw line, measure perpendicular
    // distance. A defender within a "blocking corridor" along the lane
    // degrades the score.
    let mut occlusion = 0.0_f64;
    let corridor_half_width = 5.0; // meters — how close to the lane center counts
    for &dp in defenders {
        let to_def = dp.sub(carrier_p);
        let proj_t = to_def.dot(lane_dir) / lane_len; // normalized [0,1] along lane
        if proj_t < 0.05 || proj_t > 0.95 {
            continue; // behind carrier or past receiver — not blocking
        }
        // Perpendicular distance from the lane line
        let proj_point = carrier_p.add(lane_dir.scale(proj_t * lane_len));
        let perp_dist = dp.sub(proj_point).len();
        if perp_dist < corridor_half_width {
            // Stronger occlusion when closer to the line center
            occlusion += 1.0 - (perp_dist / corridor_half_width);
        }
    }
    let occlusion_score = 1.0 / (1.0 + occlusion); // 1 = clear lane, → 0 = blocked

    // ─── 2. Defender closing time ───
    // Can any defender reach the catch point before the ball arrives?
    // Predicted catch point: candidate moves during flight time.
    let catch_point = candidate_p.add(candidate_v.scale(flight_time.min(PASS_HORIZON)));
    let mut worst_closing_ratio = 0.0_f64; // 0 = no threat, 1+ = defender arrives first
    for &dp in defenders {
        let def_dist = dp.sub(catch_point).len();
        let def_close_time = def_dist / DEF_CLOSE_SPEED;
        if def_close_time < flight_time {
            // Defender arrives before the ball — threat!
            let ratio = 1.0 - (def_close_time / flight_time.max(1e-6));
            worst_closing_ratio = worst_closing_ratio.max(ratio);
        }
    }
    let closing_score = 1.0 - worst_closing_ratio; // 1 = safe, 0 = defender beats ball

    // ─── 3. Coriolis-lane quality ───
    // Throw the ball at the naive straight-line velocity and see where it
    // actually lands after Coriolis drift.
    let throw_v = lane_dir.scale(DEFAULT_BALL_SPEED);
    let arrived = roll_forward(carrier_p, throw_v, omega, flight_time.min(PASS_HORIZON), skin_r);
    let endpoint_error = arrived.p.sub(catch_point).len();
    // Score: 1.0 when error is 0, falls off as the curve drifts the ball away.
    let coriolis_score = 1.0 / (1.0 + endpoint_error / CATCH_RADIUS);

    // ─── 4. Separation from nearest defender at candidate pos ───
    let mut min_def_dist = f64::INFINITY;
    for &dp in defenders {
        let d = dp.sub(candidate_p).len();
        if d < min_def_dist {
            min_def_dist = d;
        }
    }
    // Saturating sigmoid: full credit at ~20m separation, half at ~8m.
    let sep_score = if defenders.is_empty() {
        1.0
    } else {
        (min_def_dist / 16.0).min(1.0)
    };

    // ─── Combine (geometric mean keeps all factors load-bearing) ───
    let raw = occlusion_score * closing_score * coriolis_score * sep_score;
    raw.clamp(0.0, 1.0)
}

/// Score a specific world-velocity throw. Returns expected value in [0, 1+].
///
/// Rolls the ball with `roll_forward` for ~1.5s, finds which teammate (if any)
/// can reach the predicted endpoint, multiplies by `reception_quality` at that
/// point, adds forward-progress value, penalizes wall/backward throws.
///
/// `teammates` is `&[(position, velocity)]` so we can predict where each
/// teammate will drift during ball flight.
pub fn pass_outcome_ev(
    from_p: Vec3,
    v0: Vec3,
    teammates: &[(Vec3, Vec3)],
    defenders: &[Vec3],
    omega: f64,
    skin_r: f64,
) -> f64 {
    let speed = v0.len();
    if speed < 1e-3 {
        return 0.0;
    }

    // Roll the ball forward for the pass horizon.
    let arrived = roll_forward(from_p, v0, omega, PASS_HORIZON, skin_r);

    // ─── Skin-wall penalty ───
    // If the ball ended up plastered against the skin, it's a bad throw.
    let rho = arrived.p.y.hypot(arrived.p.z);
    let skin_penalty = if rho > skin_r * 0.92 { 0.3 } else { 1.0 };

    // ─── Find best teammate who can reach the endpoint ───
    // Teammates are rolled forward briefly to their predicted position at
    // catch time.
    let mut best_rq = 0.0_f64;
    let mut any_reachable = false;
    for &(tp, tv) in teammates {
        // Predict where the teammate will be at ball-arrival time.
        let tm_future = tp.add(tv.scale(PASS_HORIZON));
        // Can the teammate reach the arrived point within PASS_HORIZON?
        let dist = tm_future.sub(arrived.p).len();
        // Teammate has the flight time to close from their predicted pos.
        let reachable = dist < CATCH_RADIUS + DEF_CLOSE_SPEED * PASS_HORIZON * 0.5;
        if !reachable {
            continue;
        }
        any_reachable = true;
        // Score this teammate's reception quality at the arrival point.
        let rq = reception_quality(
            from_p,
            arrived.p,
            tv,
            defenders,
            omega,
            skin_r,
        );
        if rq > best_rq {
            best_rq = rq;
        }
    }

    if !any_reachable {
        return 0.0;
    }

    // ─── Forward-progress bonus ───
    // Advancing toward the attack end (positive x direction as a convention
    // here — the caller normalizes by team). Measured as fraction of tube
    // length advanced. Mild bonus, not dominant.
    let dx = arrived.p.x - from_p.x;
    let tube_half = crate::tuning::GATE_X;
    let progress_bonus = (dx / tube_half).clamp(-0.2, 0.3);

    // ─── Backward penalty ───
    let backward_penalty = if dx < -20.0 { 0.5 } else { 1.0 };

    let raw = (best_rq + progress_bonus) * skin_penalty * backward_penalty;
    raw.max(0.0)
}

/// Basin width: how MANY throw angles from carrier would result in a
/// successful pass to `candidate_p`?
///
/// Samples a small fan of throw directions from carrier → candidate
/// (varying angle ±), rolls each with the predictor, counts how many
/// arrive within catch range AND aren't intercepted. Returns the fraction
/// [0, 1]. A wide basin means the receiver is in a forgiving position —
/// many throws work — which is what off-ball players should seek.
pub fn reception_basin_width(
    carrier_p: Vec3,
    candidate_p: Vec3,
    candidate_v: Vec3,
    defenders: &[Vec3],
    omega: f64,
    skin_r: f64,
) -> f64 {
    let lane = candidate_p.sub(carrier_p);
    let lane_len = lane.len();
    if lane_len < 1e-6 {
        return 0.0;
    }
    let lane_dir = lane.norm();

    // Flight time estimate.
    let flight_time = (lane_len / DEFAULT_BALL_SPEED).min(PASS_HORIZON);

    // Predicted catch point (receiver drifts during flight).
    let catch_point = candidate_p.add(candidate_v.scale(flight_time));

    // Build a perpendicular basis for fanning throw angles. Pick two
    // vectors orthogonal to lane_dir.
    let perp1 = {
        // Cross with a non-parallel axis to get a perpendicular.
        let seed = if lane_dir.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        lane_dir.cross(seed).norm()
    };
    let perp2 = lane_dir.cross(perp1).norm();

    // Fan: sample 9 throw directions (center + 8 ring at ± angular offset).
    // Angular spread: ±6 degrees ≈ ±0.105 rad — reasonable aiming error band.
    const N_RING: usize = 8;
    const SPREAD_RAD: f64 = 0.105;
    let mut hits = 0_u32;
    let total = 1 + N_RING;

    // Helper: test a single throw direction.
    let test_throw = |dir: Vec3| -> bool {
        let throw_v = dir.scale(DEFAULT_BALL_SPEED);
        let arrived = roll_forward(carrier_p, throw_v, omega, flight_time, skin_r);

        // Does the ball arrive within catch radius of where the receiver will be?
        let miss_dist = arrived.p.sub(catch_point).len();
        if miss_dist > CATCH_RADIUS {
            return false;
        }

        // Is a defender able to intercept? Check if any defender can reach the
        // arrival point before the ball.
        for &dp in defenders {
            let def_dist = dp.sub(arrived.p).len();
            let def_time = def_dist / DEF_CLOSE_SPEED;
            if def_time < flight_time * 0.85 {
                // Defender arrives comfortably before the ball — intercepted.
                return false;
            }
        }
        true
    };

    // Center shot.
    if test_throw(lane_dir) {
        hits += 1;
    }

    // Ring samples.
    for i in 0..N_RING {
        let angle = std::f64::consts::TAU * (i as f64) / (N_RING as f64);
        let offset_y = SPREAD_RAD * angle.cos();
        let offset_z = SPREAD_RAD * angle.sin();
        // Perturbed direction: rotate lane_dir by small angular offset in
        // the perp1/perp2 plane.
        let perturbed = lane_dir
            .add(perp1.scale(offset_y))
            .add(perp2.scale(offset_z))
            .norm();
        if test_throw(perturbed) {
            hits += 1;
        }
    }

    hits as f64 / total as f64
}

/// Pragmatic + epistemic decomposition of Expected Free Energy for a
/// candidate world target the agent could move toward (or hold). LOWER EFE
/// is better (free energy is minimized). Each term is bounded so no single
/// term can dominate pathologically.
#[derive(Clone, Copy, Debug)]
pub struct Efe {
    /// Reach-preferred-state cost (possession / advance / deny).
    pub pragmatic: f64,
    /// Stay-responsive + team-spread cost.
    pub epistemic: f64,
}

impl Efe {
    pub fn total(&self) -> f64 {
        self.pragmatic + self.epistemic
    }
}

/// Forward-progress of a world point for `team` (re-exported convenience).
pub fn fp(team: TeamSide, x: f64) -> f64 {
    forward_progress(team, x)
}

/// Attack ring x / sign convenience re-exports so the controller in
/// rigger_ai keeps a single import surface.
pub fn ring_x(team: TeamSide) -> f64 {
    attack_ring_x(team)
}
pub fn sgn(team: TeamSide) -> f64 {
    attack_sign(team)
}

/// Collect teammate world positions (excluding `me`) on `me`'s team —
/// a deterministic slice (Vec order = state.players order, which is the
/// stable construction order, NOT a HashMap iteration).
pub fn teammate_positions(me: &PlayerSim, state: &SimState) -> Vec<Vec3> {
    state
        .players
        .iter()
        .filter(|p| p.team == me.team && p.id != me.id)
        .map(|p| p.p)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Quat;
    use crate::ai::types::{BellState, RiggerRole};

    const W: f64 = crate::tuning::OMEGA;
    const SKIN: f64 = crate::tuning::R;

    #[test]
    fn roll_forward_is_deterministic_and_axial_inertial() {
        let p = Vec3::new(-100.0, 3.0, -2.0);
        let v = Vec3::new(20.0, 1.0, -1.5);
        let a = roll_forward(p, v, W, 1.3, SKIN);
        let b = roll_forward(p, v, W, 1.3, SKIN);
        assert_eq!(a.p.x.to_bits(), b.p.x.to_bits());
        assert_eq!(a.p.y.to_bits(), b.p.y.to_bits());
        assert_eq!(a.p.z.to_bits(), b.p.z.to_bits());
        // x is inertial in the Coriolis frame: x ≈ x0 + vx·t (no skin in x).
        assert!((a.p.x - (p.x + v.x * 1.3)).abs() < 1e-6);
    }

    #[test]
    fn skin_bounce_keeps_bell_inside_tube() {
        // Fling the bell hard at the skin; the predictor must keep it in.
        let st = roll_forward(
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(5.0, 60.0, 0.0),
            W,
            2.0,
            SKIN,
        );
        let rho = st.p.y.hypot(st.p.z);
        assert!(rho <= SKIN + 1e-6, "rho {rho} escaped skin");
    }

    #[test]
    fn band_response_gap_lower_when_closer() {
        let band = BellBand::predict(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(15.0, 1.0, 0.0),
            W,
            SKIN,
        );
        let near = band.expected_response_gap(band.mid().p, 22.0);
        let far = band.expected_response_gap(Vec3::new(-300.0, 0.0, 0.0), 22.0);
        assert!(near < far, "near {near} should beat far {far}");
        assert!(near.is_finite() && far.is_finite());
    }

    #[test]
    fn crowding_higher_when_clumped() {
        let vm = VolumeModel::new(TeamSide::Home, crate::tuning::GATE_X, SKIN);
        let here = Vec3::new(0.0, 0.0, 0.0);
        let clump = vm.crowding_at(here, &[Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)]);
        let spread = vm.crowding_at(here, &[Vec3::new(200.0, 0.0, 0.0), Vec3::new(-200.0, 30.0, 0.0)]);
        assert!(clump > spread, "clump {clump} should exceed spread {spread}");
    }

    #[test]
    fn best_intercept_is_a_band_sample() {
        let band = BellBand::predict(
            Vec3::new(10.0, 3.0, 1.0),
            Vec3::new(18.0, 2.0, -1.0),
            W,
            SKIN,
        );
        let ip = band.best_intercept_for(Vec3::new(0.0, 0.0, 0.0), 22.0);
        assert!(band.samples.iter().any(|(_, s)| s.p.x.to_bits() == ip.x.to_bits()));
    }

    fn mk_player(id: &str, team: TeamSide, p: Vec3) -> PlayerSim {
        PlayerSim {
            id: id.into(),
            team,
            role: RiggerRole::Spinner,
            p,
            v: Vec3::new(0.0, 0.0, 0.0),
            q: Quat::ident(),
            line: None,
            dv_budget: 0.0,
            contact_ref: None,
            grounded: false,
        }
    }

    // ─── Pass-primitive tests ─────────────────────────────────────────────

    #[test]
    fn reception_quality_clear_lane_beats_blocked() {
        // Use a short pass (~30m) so Coriolis drift is small and the score
        // stays high for a clean lane.
        let carrier = Vec3::new(0.0, 5.0, 0.0);
        let receiver = Vec3::new(28.0, 5.0, 0.0);
        let rv = Vec3::new(0.0, 0.0, 0.0);
        // No defenders — perfect lane.
        let clear = reception_quality(carrier, receiver, rv, &[], W, SKIN);
        // Defender sitting in the middle of the lane.
        let blocked = reception_quality(
            carrier,
            receiver,
            rv,
            &[Vec3::new(14.0, 5.0, 0.0)],
            W,
            SKIN,
        );
        assert!(clear > blocked, "clear {clear} should beat blocked {blocked}");
        assert!(clear > 0.4, "a short open lane should score well: {clear}");
    }

    #[test]
    fn reception_quality_far_defender_is_fine() {
        // Short pass, defender well off to the side.
        let carrier = Vec3::new(0.0, 5.0, 0.0);
        let receiver = Vec3::new(25.0, 5.0, 0.0);
        let rv = Vec3::new(0.0, 0.0, 0.0);
        let score = reception_quality(
            carrier,
            receiver,
            rv,
            &[Vec3::new(12.0, 40.0, 0.0)],
            W,
            SKIN,
        );
        assert!(score > 0.3, "far defender shouldn't kill the score: {score}");
    }

    #[test]
    fn pass_outcome_ev_good_pass_scores_positive() {
        let from = Vec3::new(0.0, 5.0, 0.0);
        // Throw forward toward a teammate.
        let v0 = Vec3::new(22.0, 0.0, 0.0);
        let teammates = &[(Vec3::new(30.0, 5.0, 0.0), Vec3::new(0.0, 0.0, 0.0))];
        let defenders = &[Vec3::new(-50.0, 0.0, 0.0)]; // far away
        let ev = pass_outcome_ev(from, v0, teammates, defenders, W, SKIN);
        assert!(ev > 0.0, "a good forward pass should score positive: {ev}");
    }

    #[test]
    fn pass_outcome_ev_no_teammate_scores_zero() {
        let from = Vec3::new(0.0, 5.0, 0.0);
        let v0 = Vec3::new(22.0, 0.0, 0.0);
        // No teammates anywhere near the landing zone.
        let teammates = &[(Vec3::new(-200.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0))];
        let ev = pass_outcome_ev(from, v0, teammates, &[], W, SKIN);
        assert_eq!(ev, 0.0, "unreachable teammate should yield 0");
    }

    #[test]
    fn basin_width_open_field_is_wide() {
        let carrier = Vec3::new(0.0, 5.0, 0.0);
        let receiver = Vec3::new(25.0, 5.0, 0.0);
        let rv = Vec3::new(0.0, 0.0, 0.0);
        // No defenders — basin should be wide (most angles work).
        let basin = reception_basin_width(carrier, receiver, rv, &[], W, SKIN);
        assert!(basin > 0.5, "open-field basin should be wide: {basin}");
    }

    #[test]
    fn basin_width_shrinks_with_close_defender() {
        let carrier = Vec3::new(0.0, 5.0, 0.0);
        let receiver = Vec3::new(25.0, 5.0, 0.0);
        let rv = Vec3::new(0.0, 0.0, 0.0);
        let open = reception_basin_width(carrier, receiver, rv, &[], W, SKIN);
        // Defender right at the catch point.
        let tight = reception_basin_width(
            carrier,
            receiver,
            rv,
            &[Vec3::new(24.0, 5.0, 0.0)],
            W,
            SKIN,
        );
        assert!(
            open >= tight,
            "defender at catch point should shrink basin: open={open} tight={tight}"
        );
    }

    #[test]
    fn reception_quality_is_deterministic() {
        let c = Vec3::new(-10.0, 3.0, 2.0);
        let r = Vec3::new(40.0, -5.0, 1.0);
        let rv = Vec3::new(2.0, 0.5, 0.0);
        let defs = &[Vec3::new(15.0, 0.0, 0.0), Vec3::new(20.0, -3.0, 1.0)];
        let a = reception_quality(c, r, rv, defs, W, SKIN);
        let b = reception_quality(c, r, rv, defs, W, SKIN);
        assert_eq!(a.to_bits(), b.to_bits(), "must be deterministic");
    }

    #[test]
    fn teammate_positions_excludes_self_and_opponents() {
        let me = mk_player("H1", TeamSide::Home, Vec3::new(0.0, 0.0, 0.0));
        let state = SimState {
            tick: 0.0,
            omega: W,
            bell: BellState {
                p: Vec3::new(0.0, 0.0, 0.0),
                v: Vec3::new(0.0, 0.0, 0.0),
                q: Quat::ident(),
                w: Vec3::new(0.0, 0.0, 0.0),
                chime: 1.0,
                held_by: None,
                thrown_by: None,
                touched_since_throw: false,
                release_pos: Vec3::new(0.0, 0.0, 0.0),
                release_tick: 0.0,
                pass_chain: vec![],
            },
            players: vec![
                me.clone(),
                mk_player("H2", TeamSide::Home, Vec3::new(5.0, 0.0, 0.0)),
                mk_player("A1", TeamSide::Away, Vec3::new(9.0, 0.0, 0.0)),
            ],
        };
        let tp = teammate_positions(&me, &state);
        assert_eq!(tp.len(), 1);
        assert_eq!(tp[0].x.to_bits(), 5.0_f64.to_bits());
    }
}

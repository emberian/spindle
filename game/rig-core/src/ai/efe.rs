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

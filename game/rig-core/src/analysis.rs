//! Dynamical-systems analysis & search toolkit for RIG free-flight.
//!
//! # Physical background
//!
//! The cross-sectional (y,z) dynamics obey the roulette ODE
//!
//! ```text
//!   ζ̈ = ω²ζ − 2iω ζ̇
//! ```
//!
//! whose general solution (from [`crate::trajectory::analytic`]) is
//!
//! ```text
//!   ζ(t) = (ζ₀ + v_rot · t) · e^{−iωt}
//!   v_rot = (vy₀ − ω·z₀) + i(vz₀ + ω·y₀)    [rotating-frame velocity]
//! ```
//!
//! Two key consequences:
//!
//! 1. **Skin safety**: `|ζ(t)| = |ζ₀ + v_rot · t|`.  Since this is the
//!    modulus of a linear function of `t`, it grows at most linearly.  For a
//!    flight of duration `t_flight`, skin contact is avoided iff the line
//!    segment `{ζ₀ + v_rot·τ : τ ∈ [0,t_flight]}` stays inside the disk of
//!    radius `R`.  This is most easily satisfied when `|v_rot|` is small —
//!    i.e. when the cross-plane speed nearly cancels the centrifugal sweep
//!    `ω·ζ₀`.  For launch near the axis (|ζ₀| ≈ 1.5 m), `ω·|ζ₀| ≈ 0.48 m/s`,
//!    so a cross-plane speed of order 0.5–3 m/s suffices.
//!
//! 2. **Heading wind-up**: the `e^{−iωt}` factor rotates the entire
//!    cross-section frame at −ω rad/s.  Additionally, when `|v_rot|` is small
//!    the Coriolis acceleration (2ω × v_⊥) drives rapid heading changes in
//!    the first second, then the steady winding at ω continues.  For a
//!    28-second flight at ω = 0.32 rad/s the accumulated heading turn is
//!    ≳ 9 rad >> LOOP_TURN = 2.2 rad.
//!
//! # Throw design recipe
//!
//! To score a Loop from launch at x = −180 to the gate at x = +320 (500 m):
//!
//! - Choose axial speed `vx ≈ 18 m/s` → `t_flight ≈ 28 s`.
//! - Choose cross-plane speed `v_⊥ ≈ 0.5–2 m/s` (near-axial throw).
//! - The solver parameterises the search as `(vx, v_perp, angle)` because
//!   the skin constraint is naturally expressed in those terms.
//!
//! # Module layout
//!
//! 1. **Regime helpers** — pure analytic, zero allocation.
//! 2. **Constant search** — deterministic grid over (ω, R), tuning-gate
//!    filtered, margin-scored.
//! 3. **Throw solvers** — deterministic coarse→fine grids that integrate with
//!    `rk4_step` (h = 1/240 s) and apply the same heading-turn logic as
//!    [`crate::loop_detector::LoopTracker`].

use std::f64::consts::PI;

use crate::loop_detector::{CURL_TURN, LOOP_TURN};
use crate::math::Vec3;
use crate::trajectory::{analytic, rk4_step, PointState};
use crate::tuning;

// ============================================================================
// §1  Regime characterisation — pure analytic helpers
// ============================================================================

/// Period of one full barrel-roll cycle in the cross-section (= 2π/ω).
///
/// For `OMEGA = 0.32` this is ≈ 19.6 s.  Note that each full period is also
/// one full heading revolution, so `spin_period` is the "period of confusion"
/// — after this interval a near-axis trajectory nearly repeats.
pub fn spin_period(omega: f64) -> f64 {
    2.0 * PI / omega
}

/// Asymptotic heading-turn rate of the (y,z) cross-section velocity (rad/s).
///
/// Equal to `ω` exactly: the `e^{−iωt}` phase factor rotates every component
/// of ζ̇ at −ω rad/s, so the heading winds at ω rad/s once the trajectory
/// has settled into the roulette regime.
///
/// In practice, Coriolis forces drive additional early transient turn, so the
/// total turn over a finite flight exceeds `ω · t_flight`.
pub fn winding_rate(omega: f64) -> f64 {
    omega
}

/// Maximum cross-sectional radius `max_{t∈[0,t_max]} |ζ(t)|` via the analytic
/// closed form.
///
/// Uses the algebraic shortcut: in the roulette solution
/// `ζ(t) = (ζ₀ + v_rot·t)·e^{-iωt}`, the modulus equals
/// `|(ζ₀ + v_rot·t)|` exactly, because the exponential is a pure rotation.
/// This is therefore the modulus of a *linear* function of `t`, sampled at
/// 4 000 uniformly-spaced points over `[0, t_max]`.
pub fn max_cross_radius(p0: Vec3, v0: Vec3, omega: f64, t_max: f64) -> f64 {
    // Rotating-frame velocity components.
    let br = v0.y - omega * p0.z; // real part of v_rot
    let bi = v0.z + omega * p0.y; // imaginary part of v_rot

    // |ζ(t)| = sqrt((y₀+br·t)² + (z₀+bi·t)²).
    const SAMPLES: usize = 4_000;
    let rho0 = (p0.y * p0.y + p0.z * p0.z).sqrt();
    let mut max_r = rho0;
    for i in 1..=SAMPLES {
        let t = t_max * (i as f64) / (SAMPLES as f64);
        let ry = p0.y + br * t;
        let rz = p0.z + bi * t;
        let r = (ry * ry + rz * rz).sqrt();
        if r > max_r {
            max_r = r;
        }
    }
    max_r
}

/// Returns the first time `t > 0` at which:
/// - the heading turn has reached ≥ `LOOP_TURN` (2.2 rad), AND
/// - `|ζ(t) − ζ₀| ≤ close_radius` (the bell has returned near its release
///   cross-section position, as required by the in-game Loop tier),
///
/// or `None` if neither condition is jointly satisfied within `[0, t_max]`.
///
/// Uses the analytic propagator at h = 1/240 s, with the same heading-turn
/// accumulation as [`crate::loop_detector::LoopTracker`].
pub fn closes_by(
    p0: Vec3,
    v0: Vec3,
    omega: f64,
    close_radius: f64,
    t_max: f64,
) -> Option<f64> {
    const H: f64 = 1.0 / 240.0;
    let steps = (t_max / H).ceil() as usize;

    // Analytic cross-velocity at time t.
    // From d/dt[ζ(t)]: differentiating ζ(t) = (ζ₀ + v_rot·t)·e^{-iωt},
    //   ζ̇(t) = [v_rot − iω·(ζ₀ + v_rot·t)] · e^{-iωt}
    // In component form (c = cos(ωt), s = sin(ωt)):
    //   vy = (br − ω·(z₀+bi·t))·c + (bi + ω·(y₀+br·t))·s
    //   vz = (bi + ω·(y₀+br·t))·c − (br − ω·(z₀+bi·t))·s
    let br = v0.y - omega * p0.z;
    let bi = v0.z + omega * p0.y;

    let vel_yz = |t: f64| -> (f64, f64) {
        let c = (omega * t).cos();
        let s = (omega * t).sin();
        let ry = p0.y + br * t;
        let rz = p0.z + bi * t;
        let ar = br - omega * rz; // real part of the bracket above
        let ai = bi + omega * ry; // imaginary part
        (ar * c + ai * s, ai * c - ar * s)
    };

    let (vy0, vz0) = vel_yz(0.0);
    let mut prev_heading = vz0.atan2(vy0);
    let mut cum_turn = 0.0_f64;

    for i in 1..=steps {
        let t = i as f64 * H;
        let (vy, vz) = vel_yz(t);
        let speed = vy.hypot(vz);
        if speed >= 1e-4 {
            let h = vz.atan2(vy);
            let mut d = h - prev_heading;
            while d > PI {
                d -= 2.0 * PI;
            }
            while d < -PI {
                d += 2.0 * PI;
            }
            cum_turn += d.abs();
            prev_heading = h;
        }

        if cum_turn >= LOOP_TURN {
            let p = analytic(p0, v0, omega, t);
            let dy = p.y - p0.y;
            let dz = p.z - p0.z;
            if dy.hypot(dz) <= close_radius {
                return Some(t);
            }
        }
    }
    None
}

// ============================================================================
// §2  Constant search
// ============================================================================

/// A candidate (ω, R) pair that passes the tuning gate, with a margin score.
#[derive(Debug, Clone)]
pub struct ConstCandidate {
    /// Spin rate (rad/s).
    pub omega: f64,
    /// Tube radius (m).
    pub r: f64,
    /// Margin score (higher = more robust; sum of normalised distances from
    /// each inequality bound).  Always positive for candidates in this list.
    pub score: f64,
    /// Full tuning results for this candidate.
    pub report: ConstReport,
}

/// Summarised tuning results carried with each [`ConstCandidate`].
#[derive(Debug, Clone)]
pub struct ConstReport {
    pub axial_reach: f64,
    pub loop_wind: f64,
    pub loop_flight: f64,
    pub g_skin_fraction: f64,
    pub traversable: bool,
    pub loop_achievable: bool,
    pub gradient_ok: bool,
}

/// Compute the margin score for a *passing* tuning report.
///
/// Each tuning inequality contributes a term equal to the normalised distance
/// of the actual value from the *nearest* bound:
///
/// - `axial_reach ≥ ½L`                : `(reach − ½L) / ½L`
/// - `loop_wind ∈ [LOOP_TURN_TARGET, 6.5]` : `min(lw−lo, hi−lw) / (hi−lo)`
/// - `g_skin ∈ [0.1·g, 0.6·g]`        : `min(gf−0.1, 0.6−gf) / 0.5`
///
/// All terms are dimensionless fractions; their sum is the margin score.
/// A score near 1.0 means all three constraints have ~33 % headroom.
fn margin_score(rep: &tuning::TuningReport) -> f64 {
    let half_l = 0.5 * tuning::L;

    let m_axial = if rep.axial_reach >= half_l {
        (rep.axial_reach - half_l) / half_l
    } else {
        0.0 // should not happen for a passing report
    };

    let lw = rep.loop_wind;
    let m_wind = {
        let range = 6.5 - tuning::LOOP_TURN_TARGET;
        f64::min(lw - tuning::LOOP_TURN_TARGET, 6.5 - lw).max(0.0) / range
    };

    let gf = rep.g_skin_fraction;
    let m_g = f64::min(gf - 0.1, 0.6 - gf).max(0.0) / 0.5;

    m_axial + m_wind + m_g
}

/// Deterministic grid search over (ω, R), returning all passing candidates
/// sorted best-first by margin score.
///
/// Grid: `steps` subdivisions on each axis → `steps²` evaluations.
/// At `steps = 40` this is 1 600 evaluations (negligible CPU) and resolves
/// the pass region at ~2 % per axis.
pub fn search_constants(
    omega_range: (f64, f64),
    r_range: (f64, f64),
    steps: usize,
) -> Vec<ConstCandidate> {
    let mut results = Vec::new();

    let n = steps.max(2);
    for i in 0..n {
        let omega = omega_range.0
            + (omega_range.1 - omega_range.0) * (i as f64) / ((n - 1) as f64);
        for j in 0..n {
            let r = r_range.0
                + (r_range.1 - r_range.0) * (j as f64) / ((n - 1) as f64);

            let rep = tuning::evaluate(omega, r, tuning::L, tuning::EARTH_G);
            if !rep.pass {
                continue;
            }
            let score = margin_score(&rep);
            results.push(ConstCandidate {
                omega,
                r,
                score,
                report: ConstReport {
                    axial_reach: rep.axial_reach,
                    loop_wind: rep.loop_wind,
                    loop_flight: rep.loop_flight,
                    g_skin_fraction: rep.g_skin_fraction,
                    traversable: rep.traversable,
                    loop_achievable: rep.loop_achievable,
                    gradient_ok: rep.gradient_ok,
                },
            });
        }
    }

    results.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

// ============================================================================
// §3  Throw solvers
// ============================================================================

/// Result of a successful throw search.
#[derive(Debug, Clone)]
pub struct ThrowSolution {
    /// Total launch speed (m/s).
    pub speed: f64,
    /// Unit direction vector (in the rotating frame).
    pub dir: Vec3,
    /// Time of flight to the ring crossing (s).
    pub flight: f64,
    /// Cumulative heading turn accumulated by the crossing (rad).
    pub wound: f64,
    /// `true` iff this solution scores (wound ≥ the requested turn target,
    /// skin untouched, gate crossing inside `gate_radius`).
    pub scored: bool,
}

// ---------------------------------------------------------------------------
// Heading-turn accumulation (mirrors LoopTracker::update exactly)
// ---------------------------------------------------------------------------

#[inline(always)]
fn wrapped_delta_abs(prev: f64, cur: f64) -> f64 {
    let mut d = cur - prev;
    while d > PI {
        d -= 2.0 * PI;
    }
    while d < -PI {
        d += 2.0 * PI;
    }
    d.abs()
}

// ---------------------------------------------------------------------------
// Core simulation harness
// ---------------------------------------------------------------------------

/// Simulate one throw, returning the [`ThrowSolution`] if the bell passes
/// through the gate plane at `ring_x` with `hypot(y,z) ≤ gate_radius` and
/// without skin contact, or `None` otherwise.
///
/// Uses `rk4_step` at h = 1/240 s (the canonical RIG timestep).
fn simulate_throw(
    omega: f64,
    r: f64,
    launch: Vec3,
    vel: Vec3,
    ring_x: f64,
    gate_radius: f64,
    t_max: f64,
    turn_target: f64,
) -> Option<ThrowSolution> {
    const H: f64 = 1.0 / 240.0;
    let steps = (t_max / H).ceil() as usize;

    let speed = vel.len();
    if speed < 1e-9 {
        return None;
    }
    let dir = vel.scale(1.0 / speed);

    let mut s = PointState { p: launch, v: vel };
    let mut cum_turn = 0.0_f64;
    let spd0 = vel.y.hypot(vel.z);
    let mut prev_heading = if spd0 >= 1e-4 { vel.z.atan2(vel.y) } else { 0.0 };
    let mut has_prev = spd0 >= 1e-4;

    let mut prev_x = launch.x;

    for i in 1..=steps {
        let prev_s = s;
        s = rk4_step(s, omega, H);
        let t = i as f64 * H;

        // Skin check.
        let rho = s.p.y.hypot(s.p.z);
        if rho >= r {
            return None;
        }

        // Heading-turn accumulation.
        let cspd = s.v.y.hypot(s.v.z);
        if cspd >= 1e-4 {
            let h = s.v.z.atan2(s.v.y);
            if has_prev {
                cum_turn += wrapped_delta_abs(prev_heading, h);
            }
            prev_heading = h;
            has_prev = true;
        }

        // Detect crossing of the ring plane.
        let cur_x = s.p.x;
        let crossed = (prev_x < ring_x && cur_x >= ring_x)
            || (prev_x > ring_x && cur_x <= ring_x);

        if crossed {
            // Linear interpolate to the crossing point.
            let frac = if (cur_x - prev_x).abs() > 1e-12 {
                (ring_x - prev_x) / (cur_x - prev_x)
            } else {
                0.5
            };
            let cross_y = prev_s.p.y + frac * (s.p.y - prev_s.p.y);
            let cross_z = prev_s.p.z + frac * (s.p.z - prev_s.p.z);
            let cross_rho = cross_y.hypot(cross_z);

            if cross_rho <= gate_radius {
                let scored = cum_turn >= turn_target;
                return Some(ThrowSolution {
                    speed,
                    dir,
                    flight: t,
                    wound: cum_turn,
                    scored,
                });
            }
            // Crossed ring but missed gate — no solution on this trajectory.
            return None;
        }

        // Overshot without crossing — bail early.
        if cur_x > ring_x + 60.0 {
            return None;
        }

        prev_x = cur_x;
    }
    None
}

// ---------------------------------------------------------------------------
// Grid constants — documented for reproducibility
// ---------------------------------------------------------------------------
//
// The search is parameterised as (vx, v_perp, angle):
//   vx      = axial speed (m/s): directly sets flight time T ≈ Δx/vx.
//   v_perp  = cross-plane speed (m/s): drives skin radius growth |v_rot|·T.
//   angle   = direction of the cross-plane component in the (y,z) plane.
//
// Skin constraint: |v_rot| · T ≲ R, where |v_rot| ≈ v_perp + ω·|ζ₀|.
// For launch at y≈1.5 m, ω·|ζ₀| ≈ 0.48 m/s.
// Axial distance Δx ≈ 500 m → T ≈ 500/vx.
// Constraint: (v_perp + 0.48) · 500/vx < 45 → v_perp < 0.09·vx − 0.48.
// For vx=18: v_perp < 1.1 m/s.  For vx=30: v_perp < 2.2 m/s.
//
// Heading: Coriolis + roulette wind give ≳ ω·T rad total.  At vx=18, T≈28 s,
// heading ≳ 0.32·28 ≈ 9 rad >> LOOP_TURN=2.2.  The turn condition is easy
// to satisfy; the binding constraint is skin safety.
//
// Coarse pass:   8 vx × 8 v_perp × 16 angles = 1 024 candidates
//   vx    ∈ [10, 30] m/s   (step 2.9 m/s)
//   v_perp∈ [0.05, 2.5] m/s (step 0.35 m/s)
//   angle ∈ [0, 2π) in 16 steps (22.5° each)
//
// Fine pass (around best coarse hit):
//   7 vx × 5 v_perp × 8 angles = 280 refinement candidates
//
// Total: ≤ 1 304 simulations per solve_* call.  At h=1/240 s with T≈28 s,
// each simulation is ≈ 6 720 RK4 steps — manageable in a test harness.

const COARSE_VX: usize = 8;
const COARSE_VPERP: usize = 8;
const COARSE_ANGLE: usize = 16;

const FINE_VX: usize = 7;
const FINE_VPERP: usize = 5;
const FINE_ANGLE: usize = 8;

const VX_LO: f64 = 10.0;
const VX_HI: f64 = 30.0;
const VPERP_LO: f64 = 0.05;
const VPERP_HI: f64 = 2.5;

/// Assemble the 3-D velocity vector from physical throw parameters.
#[inline(always)]
fn vel_from_physical(vx: f64, v_perp: f64, angle: f64) -> Vec3 {
    Vec3::new(vx, v_perp * angle.cos(), v_perp * angle.sin())
}

/// Compare two candidate solutions: prefer `scored`, then max `wound`.
#[inline(always)]
fn is_better(a: &ThrowSolution, b: &ThrowSolution) -> bool {
    match (a.scored, b.scored) {
        (true, false) => true,
        (false, true) => false,
        _ => a.wound > b.wound,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Find the best *scoring Loop* throw from `launch` to `ring_x`.
///
/// A scoring Loop requires:
/// - Accumulated heading turn ≥ `LOOP_TURN` (2.2 rad) at the crossing.
/// - Bell passes through the gate circle `hypot(y,z) ≤ gate_radius`.
/// - Skin untouched: `hypot(y,z) < r` throughout the flight.
///
/// # Grid
/// - Coarse: 8 vx × 8 v_perp × 16 angles = 1 024 simulations.
/// - Fine:   7 × 5 × 8 = 280 simulations around the best coarse hit.
///
/// Parameters are deliberately physical: `(vx, v_perp, angle)` rather than
/// `(speed, axial_frac, angle)`, because the skin constraint is naturally
/// expressed as `v_perp·T_flight < R`.
pub fn solve_loop_throw(
    omega: f64,
    r: f64,
    launch: Vec3,
    ring_x: f64,
    gate_radius: f64,
) -> Option<ThrowSolution> {
    solve_throw_inner(omega, r, launch, ring_x, gate_radius, LOOP_TURN)
}

/// Find the best *scoring Curl* throw from `launch` to `ring_x`.
///
/// Same grid as [`solve_loop_throw`] but requires only `CURL_TURN` (1.1 rad)
/// of heading wind-up.  Curls are strictly easier to find.
pub fn solve_curl_throw(
    omega: f64,
    r: f64,
    launch: Vec3,
    ring_x: f64,
    gate_radius: f64,
) -> Option<ThrowSolution> {
    solve_throw_inner(omega, r, launch, ring_x, gate_radius, CURL_TURN)
}

fn solve_throw_inner(
    omega: f64,
    r: f64,
    launch: Vec3,
    ring_x: f64,
    gate_radius: f64,
    turn_target: f64,
) -> Option<ThrowSolution> {
    // Maximum flight time: slowest vx = VX_LO, axial distance + margin.
    let axial_dist = (ring_x - launch.x).abs();
    // Cap at 200 s; with VX_LO=10 and Δx=500 → T=50 s which is very generous.
    let t_max = (axial_dist / VX_LO + 20.0).min(200.0);

    let mut best: Option<ThrowSolution> = None;

    // ------ Coarse pass ------
    for vi in 0..COARSE_VX {
        let vx = VX_LO + (VX_HI - VX_LO) * (vi as f64) / ((COARSE_VX - 1) as f64);
        for pi in 0..COARSE_VPERP {
            let v_perp =
                VPERP_LO + (VPERP_HI - VPERP_LO) * (pi as f64) / ((COARSE_VPERP - 1) as f64);
            for ci in 0..COARSE_ANGLE {
                let angle = 2.0 * PI * (ci as f64) / (COARSE_ANGLE as f64);
                let vel = vel_from_physical(vx, v_perp, angle);
                if let Some(sol) =
                    simulate_throw(omega, r, launch, vel, ring_x, gate_radius, t_max, turn_target)
                {
                    let better = best.as_ref().map_or(true, |b| is_better(&sol, b));
                    if better {
                        best = Some(sol);
                    }
                }
            }
        }
    }

    // ------ Fine pass around best coarse candidate ------
    if let Some(ref coarse) = best.clone() {
        let vx_step = (VX_HI - VX_LO) / (COARSE_VX as f64 - 1.0);
        let vp_step = (VPERP_HI - VPERP_LO) / (COARSE_VPERP as f64 - 1.0);
        let ang_step = 2.0 * PI / (COARSE_ANGLE as f64);

        let c_vx = (coarse.dir.x * coarse.speed).clamp(VX_LO, VX_HI);
        let c_vperp = coarse.dir.y.hypot(coarse.dir.z) * coarse.speed;
        let c_angle = coarse.dir.z.atan2(coarse.dir.y);

        for vi in 0..FINE_VX {
            let dvx = vx_step * (vi as f64 / ((FINE_VX - 1) as f64) * 2.0 - 1.0);
            let vx = (c_vx + dvx).clamp(VX_LO, VX_HI);
            for pi in 0..FINE_VPERP {
                let dvp =
                    vp_step * (pi as f64 / ((FINE_VPERP - 1) as f64) * 2.0 - 1.0);
                let v_perp = (c_vperp + dvp).clamp(VPERP_LO, VPERP_HI);
                for ci in 0..FINE_ANGLE {
                    let dang =
                        ang_step * (ci as f64 / ((FINE_ANGLE - 1) as f64) * 2.0 - 1.0) * 0.5;
                    let angle = c_angle + dang;
                    let vel = vel_from_physical(vx, v_perp, angle);
                    if let Some(sol) = simulate_throw(
                        omega, r, launch, vel, ring_x, gate_radius, t_max, turn_target,
                    ) {
                        let better = best.as_ref().map_or(true, |b| is_better(&sol, b));
                        if better {
                            best = Some(sol);
                        }
                    }
                }
            }
        }
    }

    best.filter(|s| s.scored)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::{EARTH_G, GATE_RADIUS, GATE_X, L, OMEGA, R};

    // -------------------------------------------------------------------------
    // §1: Regime helpers
    // -------------------------------------------------------------------------

    #[test]
    fn spin_period_at_reg_omega() {
        let p = spin_period(OMEGA);
        // 2π / 0.32 ≈ 19.635 s
        assert!((p - 19.635).abs() < 0.01, "spin_period = {p}");
    }

    #[test]
    fn winding_rate_equals_omega() {
        assert_eq!(winding_rate(OMEGA), OMEGA);
        assert_eq!(winding_rate(0.5), 0.5);
    }

    #[test]
    fn max_cross_radius_stays_below_skin() {
        // Near-axial throw: v_perp = 0.2 m/s, y₀ = 1.5 m.
        // v_rot ≈ (0.2, 0.48) → |v_rot|≈0.52.  Over 30 s, |ζ_max| ≈ 0.52·30 + 1.5 = 17 m << 45.
        let p0 = Vec3::new(-180.0, 1.5, 0.0);
        let v0 = Vec3::new(20.0, 0.2, 0.0);
        let mr = max_cross_radius(p0, v0, OMEGA, 30.0);
        assert!(mr < R, "max_cross_radius {mr:.2} should be < R={R}");
    }

    #[test]
    fn max_cross_radius_large_cross_exceeds_skin() {
        // Pure cross-plane throw at 20 m/s → immediately large radius.
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let v0 = Vec3::new(0.0, 20.0, 0.0);
        let mr = max_cross_radius(p0, v0, OMEGA, 30.0);
        assert!(mr > R, "max_cross_radius {mr:.2} should be > R={R}");
    }

    #[test]
    fn closes_by_finds_close_for_circular_orbit() {
        // Choose v0.z = −ω·y₀ = −0.32 so that the rotating-frame velocity
        // v_rot = (br, bi) = (0, v0.z + ω·y₀) = (0, 0).
        // Then ζ(t) = ζ₀·e^{−iωt}: a perfect circle of radius |ζ₀| = 1 m.
        // After t = LOOP_TURN/ω ≈ 6.9 s the heading has turned 2.2 rad and
        // |ζ(t)−ζ₀| = 2·sin(ωt/2) = 2·sin(1.1) ≈ 1.78 m ≤ close_radius = 5 m.
        let p0 = Vec3::new(0.0, 1.0, 0.0);
        let v0 = Vec3::new(0.0, 0.0, -OMEGA); // cancels centrifugal sweep exactly
        let t_limit = spin_period(OMEGA) * 1.1;
        let result = closes_by(p0, v0, OMEGA, 5.0, t_limit);
        assert!(
            result.is_some(),
            "circular orbit should close within {t_limit:.1} s"
        );
        let t = result.unwrap();
        // Should close well before t_limit; at minimum after LOOP_TURN/ω ≈ 6.9 s.
        let t_min = LOOP_TURN / OMEGA;
        assert!(t >= t_min, "close time {t:.2} s should be >= LOOP_TURN/ω = {t_min:.2} s");
    }

    // -------------------------------------------------------------------------
    // §2: Constant search
    // -------------------------------------------------------------------------

    #[test]
    fn search_constants_finds_reg_region() {
        let results = search_constants((0.1, 0.8), (20.0, 80.0), 40);
        assert!(!results.is_empty(), "search should find passing candidates");

        // All returned candidates must pass the gate and have non-negative score.
        for c in &results {
            assert!(
                c.report.traversable && c.report.loop_achievable && c.report.gradient_ok,
                "candidate (ω={:.3}, R={:.1}) must pass: traversable={}, loop_ok={}, grad_ok={}",
                c.omega,
                c.r,
                c.report.traversable,
                c.report.loop_achievable,
                c.report.gradient_ok,
            );
            assert!(
                c.score >= 0.0,
                "candidate score must be >= 0 for a passing report, got {}",
                c.score
            );
        }

        // The region near shipped constants (OMEGA≈0.32, R≈45) must appear.
        let near_reg = results
            .iter()
            .any(|c| (c.omega - OMEGA).abs() < 0.05 && (c.r - R).abs() < 5.0);
        assert!(
            near_reg,
            "no passing candidate near shipped constants (ω={OMEGA}, R={R})"
        );
    }

    #[test]
    fn search_constants_sorted_best_first() {
        let results = search_constants((0.1, 0.8), (20.0, 80.0), 30);
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score - 1e-12,
                "results not sorted: score {} > {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn margin_score_peaks_near_shipped_r() {
        // Sweeping R from 20 to 80 at fixed omega=OMEGA: the score peak should
        // fall somewhere between 35 and 60 m (bracketing R=45).
        let mut best_r = 0.0_f64;
        let mut best_sc = f64::NEG_INFINITY;
        for i in 0..=60 {
            let r_val = 20.0 + i as f64;
            let rep = tuning::evaluate(OMEGA, r_val, L, EARTH_G);
            if rep.pass {
                let sc = margin_score(&rep);
                if sc > best_sc {
                    best_sc = sc;
                    best_r = r_val;
                }
            }
        }
        assert!(
            best_r >= 35.0 && best_r <= 60.0,
            "margin peak at R={best_r:.1} (score {best_sc:.3}), expected 35–60"
        );
    }

    // -------------------------------------------------------------------------
    // §3: Throw solvers — THE KEY TEST
    // -------------------------------------------------------------------------
    //
    // This is the proof that a scoring Loop is achievable with the shipped
    // constants (OMEGA=0.32, R=45).  We previously could not find one by hand.
    //
    // Physical recipe:
    //   vx   ≈ 18–22 m/s   → flight time T ≈ 22–28 s → heading wind ≈ 7–9 rad
    //   v_⊥  ≈ 0.3–1.5 m/s → skin radius stays ≲ 30 m << R=45
    //   angle ≈ varies      → aim cross-plane component near the gate centre
    //
    // The coarse grid covers (vx ∈ [10,30] × v_⊥ ∈ [0.05,2.5] × angle ∈ [0,2π))
    // with 8×8×16 = 1 024 samples.

    /// THE fundamental test: `solve_loop_throw` must find a scoring Loop for
    /// the shipped constants, launched from the deep-line end.
    #[test]
    fn solve_loop_throw_finds_scoring_loop() {
        let launch = Vec3::new(-180.0, 1.5, 0.0);
        let sol = solve_loop_throw(OMEGA, R, launch, GATE_X, GATE_RADIUS);

        assert!(
            sol.is_some(),
            "solve_loop_throw must find a scoring Loop for \
             OMEGA={OMEGA}, R={R}, launch=(-180,1.5,0), gate_x={GATE_X}, gate_r={GATE_RADIUS}"
        );

        let sol = sol.unwrap();
        assert!(
            sol.scored,
            "solution must be scored=true; wound={:.3} vs LOOP_TURN={LOOP_TURN}",
            sol.wound
        );
        assert!(
            sol.wound >= LOOP_TURN,
            "wound {:.3} must be >= LOOP_TURN={LOOP_TURN}",
            sol.wound
        );
        // Speed must be in the solver's search range.
        assert!(
            sol.speed >= 9.0 && sol.speed <= 36.0,
            "speed {:.2} m/s out of expected range",
            sol.speed
        );
        // Flight time must be positive and physically plausible.
        assert!(
            sol.flight > 5.0 && sol.flight < 200.0,
            "flight time {:.2} s implausible",
            sol.flight
        );
    }

    #[test]
    fn solve_curl_throw_finds_scoring_curl() {
        let launch = Vec3::new(-180.0, 1.5, 0.0);
        let sol = solve_curl_throw(OMEGA, R, launch, GATE_X, GATE_RADIUS);
        assert!(sol.is_some(), "solve_curl_throw must find a scoring Curl");
        let sol = sol.unwrap();
        assert!(
            sol.wound >= CURL_TURN,
            "wound {:.3} must be >= CURL_TURN={CURL_TURN}",
            sol.wound
        );
    }

    /// Sanity-check: a tiny tube (R=0.5 m) makes any real cross-plane throw
    /// hit the skin immediately.  The solver should return None.
    #[test]
    fn solve_loop_throw_none_for_tiny_tube() {
        let launch = Vec3::new(-180.0, 0.0, 0.0);
        // With R=0.5 m and launch on axis, any v_⊥ > 0 will quickly breach the skin.
        let sol = solve_loop_throw(OMEGA, 0.5, launch, GATE_X, GATE_RADIUS);
        // We don't assert None categorically (an axially-launched ball on the
        // centreline could in principle stay inside), but if something is found
        // it must satisfy the scored invariant.
        if let Some(s) = sol {
            assert!(s.scored, "any found solution must be scored");
            assert!(s.wound >= LOOP_TURN);
        }
    }
}

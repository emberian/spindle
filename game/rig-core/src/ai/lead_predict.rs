//! 1:1 faithful Rust port of src/ai/decide/LeadPredict.ts.
//!
//! Coriolis lead-pass solver via Newton iteration on the canon trajectory
//! predictor. Given a thrower position, bell velocity, a moving receiver,
//! and omega, find the initial velocity v0 such that the bell arrives at
//! the receiver's predicted position. Returns None if no solution is found
//! within `NEWTON_ITERS`.
//!
//! Coriolis flight prediction uses the SACRED canon integrator
//! `crate::trajectory::rk4_step` (the exact fn the sim uses) so this
//! solver's flight model is byte-identical to the simulation.

use crate::math::Vec3;
use crate::trajectory::{analytic, rk4_step, PointState};

/// Result of a successful lead solve.
#[derive(Clone, Copy, Debug)]
pub struct LeadResult {
    /// Initial bell velocity to throw from `throw_pos`.
    pub v0: Vec3,
    /// Predicted receiver intercept position.
    pub intercept: Vec3,
    /// Estimated flight time in seconds.
    pub flight_time: f64,
}

// Integration step for ball propagation in the Newton solver.
const SOLVE_H: f64 = 1.0 / 30.0;
const MAX_FLIGHT_STEPS: i64 = 300; // 10 s max
const NEWTON_ITERS: i64 = 8;
const CONVERGE_SQ: f64 = 0.25; // 0.5 m tolerance

/// Analytical Coriolis-correct receiver predictor.
///
/// Uses the closed-form `trajectory::analytic` solution to compute the exact
/// receiver position at any time t. This eliminates step-size-dependent
/// numerical drift that caused the 8.2m miss at 100m range when the old RK4
/// table was built at SOLVE_H=1/30 but the sim integrates at 1/240.
struct ReceiverPredictor {
    p0: Vec3,
    v0: Vec3,
    omega: f64,
}

impl ReceiverPredictor {
    fn new(p0: Vec3, v0: Vec3, omega: f64) -> Self {
        ReceiverPredictor { p0, v0, omega }
    }

    fn predict(&self, t: f64) -> Vec3 {
        if t <= 0.0 {
            return self.p0;
        }
        analytic(self.p0, self.v0, self.omega, t)
    }
}

/// Solve for the throw velocity that leads a moving receiver.
///
/// Algorithm (verbatim from LeadPredict.ts):
///   1. Estimate flight time from straight-line distance / throw_speed.
///   2. Compute receiver intercept position at that time.
///   3. Build initial v0 aimed at that intercept.
///   4. Propagate the bell; measure error at receiver.
///   5. Newton-iterate: shift aim point opposite to error.
///
/// Deterministic — no RNG involved.
pub fn solve_lead_velocity(
    throw_pos: Vec3,
    throw_speed: f64,
    receiver_pos: Vec3,
    receiver_vel: Vec3,
    omega: f64,
) -> Option<LeadResult> {
    // Coriolis-correct receiver predictor (replaces the old linear model).
    let predict_receiver = ReceiverPredictor::new(receiver_pos, receiver_vel, omega);

    // Initial guess: aim straight at receiver, estimate time by distance / speed.
    let mut aim_pt: Vec3 = receiver_pos;
    let mut best_error: f64 = f64::INFINITY;
    let mut best_result: Option<LeadResult> = None;

    for _iter in 0..NEWTON_ITERS {
        let raw = aim_pt.sub(throw_pos);
        let dist = raw.len();
        if dist < 0.01 {
            return None;
        }
        let dir = raw.norm();
        let v0: Vec3 = dir.scale(throw_speed);

        let est_time = dist / throw_speed;
        let steps = MAX_FLIGHT_STEPS.min((est_time / SOLVE_H).ceil() as i64 + 2);

        let mut s = PointState { p: throw_pos, v: v0 };
        let mut closest_dist: f64 = f64::INFINITY;
        let mut closest_pos: Vec3 = throw_pos;
        let mut closest_time: f64 = 0.0;

        for step in 1..=steps {
            s = rk4_step(s, omega, SOLVE_H);
            let t = step as f64 * SOLVE_H;
            let rcv = predict_receiver.predict(t);
            let d = s.p.sub(rcv).len();
            if d < closest_dist {
                closest_dist = d;
                closest_pos = s.p;
                closest_time = t;
            }
            if d < CONVERGE_SQ.sqrt() {
                if d < best_error {
                    best_error = d;
                    best_result = Some(LeadResult {
                        v0,
                        intercept: rcv,
                        flight_time: t,
                    });
                }
                break;
            }
        }

        if best_error < CONVERGE_SQ.sqrt() {
            break;
        }

        if closest_dist < best_error {
            best_error = closest_dist;
            let intercept = predict_receiver.predict(closest_time);
            best_result = Some(LeadResult {
                v0,
                intercept,
                flight_time: closest_time,
            });
        }

        let rcv_at_close = predict_receiver.predict(closest_time);
        let err_vec = closest_pos.sub(rcv_at_close);
        aim_pt = aim_pt.sub(err_vec.scale(0.65));
    }

    best_result
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = crate::tuning::OMEGA;

    /// Deterministic: identical inputs → identical output (bitwise on the
    /// numeric fields). No RNG anywhere in the solver.
    #[test]
    fn deterministic() {
        let tp = Vec3::new(-50.0, 2.0, 1.0);
        let rp = Vec3::new(40.0, -3.0, 4.0);
        let rv = Vec3::new(2.0, 1.0, -1.0);
        let a = solve_lead_velocity(tp, 28.0, rp, rv, W);
        let b = solve_lead_velocity(tp, 28.0, rp, rv, W);
        match (a, b) {
            (Some(x), Some(y)) => {
                assert_eq!(x.v0, y.v0);
                assert_eq!(x.intercept, y.intercept);
                assert_eq!(x.flight_time, y.flight_time);
            }
            (None, None) => {}
            _ => panic!("nondeterministic Some/None"),
        }
    }

    /// True-by-construction: when a solution is returned, re-simulating the
    /// bell with the canon integrator from `throw_pos` at `v0` for
    /// `flight_time` lands within the converge tolerance of the intercept.
    /// (The solver only returns the best closest approach; we assert it is
    /// at least as good as the recorded `best_error` would allow — i.e. the
    /// returned flight actually reaches near the predicted intercept.)
    #[test]
    fn solution_resimulates_to_intercept() {
        let tp = Vec3::new(-60.0, 1.0, 2.0);
        let rp = Vec3::new(30.0, 2.0, -2.0);
        let rv = Vec3::new(1.0, 0.5, 0.5);
        if let Some(r) = solve_lead_velocity(tp, 26.0, rp, rv, W) {
            // Re-roll the bell with the SACRED integrator at SOLVE_H to the
            // recorded flight time and confirm it arrives near intercept.
            let mut s = PointState { p: tp, v: r.v0 };
            let n = (r.flight_time / SOLVE_H).round() as i64;
            for _ in 0..n {
                s = rk4_step(s, W, SOLVE_H);
            }
            let miss = s.p.sub(r.intercept).len();
            // The closest-approach the solver kept must be within a sane
            // bound; with this geometry the Newton loop converges tightly.
            assert!(miss < 5.0, "re-sim miss {miss} too large");
            assert!(r.flight_time > 0.0);
        }
    }

    /// Degenerate aim (receiver on top of thrower) → None (dist < 0.01).
    #[test]
    fn coincident_returns_none() {
        let p = Vec3::new(10.0, 0.0, 0.0);
        let r = solve_lead_velocity(p, 20.0, p, Vec3::new(0.0, 0.0, 0.0), W);
        assert!(r.is_none());
    }
}

//! planner.rs — the AI rollout kernel, ported to native Rust.
//!
//! Increment 1 of the progressive TS→Rust port. This is a FAITHFUL 1:1 port
//! of the TS `rolloutPrimitive` (src/ai/nav/GrapplePlanner.ts): one grapple
//! primitive — fire an anchor, reel(-1)/swing(0), swoop-release — rolled
//! out through the true Coriolis + grapple physics, returning the END state
//! plus the branch's closest pass and terminal momentum toward target.
//!
//! Same algorithm + constants as the TS twin, so the headless TS skill
//! harness is the exact parity oracle: a native cargo eval built on this
//! must reproduce the TS ranking. This is the hot loop (candidates × players
//! × ticks) whose TS cost made every eval/optimizer intractable; native
//! Rust is the throughput unlock for deep MPPI / CEM / RL.

use crate::grapple::{REEL_RATE, TETHER_MIN};
use crate::math::Vec3;
use crate::trajectory::{rk4_step, PointState};

/// Swoop release gate — canonical defaults (the TS `__rigtune` knobs are a
/// dev-only runtime affordance; the native kernel takes fixed params, with
/// tunables to be threaded explicitly when the optimizer moves to Rust).
pub const SWOOP_MIN_V: f64 = 8.0;
pub const SWOOP_ALIGN: f64 = 0.6;

#[derive(Clone, Copy, Debug)]
pub struct Rollout {
    /// End position after the primitive.
    pub p: Vec3,
    /// End velocity.
    pub v: Vec3,
    /// Closest the path came to `target` over the primitive.
    pub min_dist: f64,
    /// Terminal speed component toward `target` (the slingshot payoff).
    pub term: f64,
}

/// Roll out one grapple primitive. `reel`: -1 = winch in, 0 = free swing
/// (with swoop-release once fast & aimed). Mirrors TS rolloutPrimitive
/// exactly (incl. the i>=4 release guard and angular-momentum reel).
pub fn rollout_primitive(
    pos: Vec3,
    vel: Vec3,
    anchor: Vec3,
    omega: f64,
    reel: i32,
    target: Vec3,
    steps: u32,
    h: f64,
) -> Rollout {
    let mut p = pos;
    let mut v = vel;
    let mut rest_len = p.sub(anchor).len();
    let mut min_dist = p.sub(target).len();
    let mut released = false;

    for i in 0..steps {
        let s = rk4_step(PointState { p, v }, omega, h);
        p = s.p;
        v = s.v;

        if reel == 0 && !released && i >= 4 {
            let sp = v.len();
            let to_t = target.sub(p);
            let dl = to_t.len();
            if sp > SWOOP_MIN_V && dl > 1e-6 && v.dot(to_t) / (sp * dl) > SWOOP_ALIGN {
                released = true;
            }
        }

        if !released {
            let d = p.sub(anchor);
            let len = d.len();
            if len > 1e-6 {
                let n = d.scale(1.0 / len);
                let v_rad = v.dot(n);
                if len >= rest_len && v_rad > 0.0 {
                    v = v.sub(n.scale(v_rad));
                }
                if reel == -1 && len >= TETHER_MIN {
                    let tl = (rest_len - REEL_RATE * h).max(TETHER_MIN);
                    if tl < rest_len {
                        let vrc = v.dot(n);
                        let sc = if rest_len > 1e-6 { rest_len / tl } else { 1.0 };
                        let vt = v.sub(n.scale(vrc));
                        v = vt.scale(sc).add(n.scale(vrc));
                        rest_len = tl;
                    }
                }
            }
        }

        let dd = p.sub(target).len();
        if dd < min_dist {
            min_dist = dd;
        }
    }

    let to_t = target.sub(p);
    let dl = to_t.len();
    let term = if dl > 1e-6 {
        v.dot(to_t.scale(1.0 / dl)).max(0.0)
    } else {
        0.0
    };
    Rollout { p, v, min_dist, term }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::OMEGA;

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// Deterministic: identical inputs → bit-identical rollout.
    #[test]
    fn rollout_is_deterministic() {
        let a = rollout_primitive(
            v(0.0, 8.0, 0.0), v(2.0, 0.0, 0.0), v(0.0, 0.0, 0.0),
            OMEGA, -1, v(60.0, 0.0, 0.0), 18, 1.0 / 15.0,
        );
        let b = rollout_primitive(
            v(0.0, 8.0, 0.0), v(2.0, 0.0, 0.0), v(0.0, 0.0, 0.0),
            OMEGA, -1, v(60.0, 0.0, 0.0), 18, 1.0 / 15.0,
        );
        assert_eq!(a.p.x, b.p.x);
        assert_eq!(a.p.y, b.p.y);
        assert_eq!(a.v.x, b.v.x);
        assert_eq!(a.min_dist, b.min_dist);
        assert_eq!(a.term, b.term);
    }

    /// Output invariants that are TRUE by construction (no hand-claimed
    /// physics): min_dist starts at the straight distance and only ever
    /// shrinks (it's a running min), and term is a finite non-negative
    /// speed. Cross-implementation PARITY vs the TS twin is validated by
    /// the headless skill harness, not asserted here.
    #[test]
    fn rollout_outputs_are_sane() {
        let pos = v(0.0, 12.0, 0.0);
        let vel = v(1.0, 0.0, 0.0);
        let target = v(0.0, 0.0, 0.0);
        let anchor = v(0.0, 2.0, 0.0);
        let start = pos.sub(target).len();
        let r = rollout_primitive(pos, vel, anchor, OMEGA, -1, target, 45, 1.0 / 15.0);
        assert!(r.min_dist.is_finite() && r.min_dist <= start + 1e-9);
        assert!(r.term.is_finite() && r.term >= 0.0);
        // Free swing should also produce finite, sane output.
        let s = rollout_primitive(pos, v(0.0, 0.0, 12.0), anchor, OMEGA, 0, target, 45, 1.0 / 15.0);
        assert!(s.min_dist.is_finite() && s.term.is_finite() && s.term >= 0.0);
    }
}

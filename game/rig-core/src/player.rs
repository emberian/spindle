//! Player kinematics — port of TS sim/Player.ts.
//! A rigger is a point mass that moves ONLY by: free ballistic drift (rotating
//! frame), pushing off structure, the rig line, or small capped thrumbler
//! nudges. No flight, ever (canon).

use crate::grapple::{Body, Line, resolve_line};
use crate::math::Vec3;
use crate::trajectory::{rk4_step, PointState};
use crate::tuning::{OMEGA, R};

pub const PLAYER_MASS: f64 = 78.0; // kg
pub const THRUMBLER_CAP: f64 = 2.0; // m/s delta-v per possession

/// Full kinematic state for a rigger in the rotating frame.
pub struct PlayerBody {
    pub p: Vec3,
    pub v: Vec3,
    /// 1/PLAYER_MASS, or 0.0 while clipped (immovable anchor).
    pub inv_mass: f64,
    pub line: Option<Line>,
    /// Clipped to a spar/ring/teammate.
    pub contact: bool,
    /// Touched the skin — out of the calm.
    pub grounded: bool,
    pub dv_budget: f64,
}

pub fn make_player(p: Vec3) -> PlayerBody {
    PlayerBody {
        p,
        v: Vec3::new(0.0, 0.0, 0.0),
        inv_mass: 1.0 / PLAYER_MASS,
        line: None,
        contact: false,
        grounded: false,
        dv_budget: THRUMBLER_CAP,
    }
}

/// Push the player off a contact surface in `dir` at `speed`. Only fires
/// when the player is in contact; clears the contact flag afterward.
pub fn push_off(pl: &mut PlayerBody, dir: Vec3, speed: f64) {
    if !pl.contact {
        return;
    }
    let l = dir.len().max(1e-12);
    pl.v = pl.v.add(dir.scale(speed / l));
    pl.contact = false;
}

/// Apply a small thrumbler nudge, capped by the remaining `dv_budget`.
pub fn thrumbler(pl: &mut PlayerBody, dv: Vec3) {
    let mag = dv.len();
    if mag < 1e-6 || pl.dv_budget <= 0.0 {
        return;
    }
    let (dv, used) = if mag > pl.dv_budget {
        (dv.scale(pl.dv_budget / mag), pl.dv_budget)
    } else {
        (dv, mag)
    };
    pl.v = pl.v.add(dv);
    pl.dv_budget = (pl.dv_budget - used).max(0.0);
}

/// Refill the thrumbler budget at the start of a new possession.
pub fn new_possession(pl: &mut PlayerBody) {
    pl.dv_budget = THRUMBLER_CAP;
}

/// Integrate one fixed sub-step:
///   1. Free-flight RK4 (unless contact/grounded).
///   2. Line constraint (if a line is attached).
///   3. Skin clamp — grounding if the player reaches the cylinder wall.
pub fn step_player(pl: &mut PlayerBody, h: f64, reel: i32) {
    // 1. Free-flight
    if !pl.contact && !pl.grounded {
        let s = PointState { p: pl.p, v: pl.v };
        let next = rk4_step(s, OMEGA, h);
        pl.p = next.p;
        pl.v = next.v;
    }

    // 2. Line constraint (static anchor only — None passed for anchor body)
    if let Some(ref mut line) = pl.line {
        let inv_mass = if pl.contact { 0.0 } else { pl.inv_mass };
        let mut body = Body { p: pl.p, v: pl.v, inv_mass };
        resolve_line(&mut body, line, None, reel, h);
        pl.p = body.p;
        pl.v = body.v;
    }

    // 3. Skin clamp
    let rho = pl.p.y.hypot(pl.p.z);
    if rho >= R {
        pl.grounded = true;
        let s = R / rho.max(1e-12);
        pl.p = Vec3::new(pl.p.x, pl.p.y * s, pl.p.z * s);
        pl.v = Vec3::new(0.0, 0.0, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_player_defaults() {
        let pl = make_player(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(pl.p, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(pl.v, Vec3::new(0.0, 0.0, 0.0));
        assert!((pl.inv_mass - 1.0 / PLAYER_MASS).abs() < 1e-12);
        assert!(!pl.contact);
        assert!(!pl.grounded);
        assert!((pl.dv_budget - THRUMBLER_CAP).abs() < 1e-12);
    }

    #[test]
    fn push_off_requires_contact() {
        let mut pl = make_player(Vec3::new(0.0, 0.0, 0.0));
        pl.contact = false;
        push_off(&mut pl, Vec3::new(1.0, 0.0, 0.0), 5.0);
        assert_eq!(pl.v, Vec3::new(0.0, 0.0, 0.0)); // no-op
    }

    #[test]
    fn push_off_clears_contact() {
        let mut pl = make_player(Vec3::new(0.0, 0.0, 0.0));
        pl.contact = true;
        push_off(&mut pl, Vec3::new(1.0, 0.0, 0.0), 5.0);
        assert!(!pl.contact);
        assert!((pl.v.x - 5.0).abs() < 1e-12);
    }

    #[test]
    fn thrumbler_capped_by_budget() {
        let mut pl = make_player(Vec3::new(0.0, 0.0, 0.0));
        pl.dv_budget = 1.0;
        // Ask for 3 m/s; only 1 m/s budget remains — should be capped.
        thrumbler(&mut pl, Vec3::new(3.0, 0.0, 0.0));
        assert!((pl.v.x - 1.0).abs() < 1e-10, "v.x = {}", pl.v.x);
        assert!((pl.dv_budget).abs() < 1e-12);
    }

    #[test]
    fn new_possession_restores_budget() {
        let mut pl = make_player(Vec3::new(0.0, 0.0, 0.0));
        thrumbler(&mut pl, Vec3::new(2.0, 0.0, 0.0));
        assert!(pl.dv_budget < 1e-12);
        new_possession(&mut pl);
        assert!((pl.dv_budget - THRUMBLER_CAP).abs() < 1e-12);
    }

    #[test]
    fn step_grounds_at_skin() {
        let mut pl = make_player(Vec3::new(0.0, R + 1.0, 0.0));
        step_player(&mut pl, 1.0 / 240.0, 0);
        assert!(pl.grounded);
        assert!(pl.p.y.hypot(pl.p.z) <= R + 1e-9);
        assert_eq!(pl.v, Vec3::new(0.0, 0.0, 0.0));
    }
}

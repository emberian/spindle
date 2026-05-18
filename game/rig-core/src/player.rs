//! Player kinematics — port of TS sim/Player.ts.
//! A rigger is a point mass that moves ONLY by: free ballistic drift (rotating
//! frame), pushing off structure, the rig line, or small capped thrumbler
//! nudges. No flight, ever (canon).
//!
//! Fluidity Spike: every instantaneous response is now CONTINUOUS.
//!   • grounding  = penetration push-out spring + exponential tangential
//!                  friction bleed via the PRECOMPUTED `FRICTION_DECAY`
//!                  constant (no per-step exp()); `grounded` is latched so
//!                  `PlayerSkinned` still fires exactly once.
//!   • contact    = a stiff stick-spring toward the latched contact anchor
//!                  (replaces the inv_mass = 0 freeze toggle).
//!   • pushoff /
//!     thrumbler  = the SAME total Δv, but ramped over a fixed number of
//!                  ticks; the dv-budget is consumed immediately so cap /
//!                  determinism semantics are unchanged.
//! Free-flight (rk4 canon ζ̈ = ω²ζ − 2iωζ̇) is untouched.

use crate::grapple::{Body, Line, resolve_line};
use crate::math::Vec3;
use crate::trajectory::{rk4_step, PointState};
use crate::tuning::{
    CONTACT_C, CONTACT_K, FRICTION_DECAY, GROUND_C, GROUND_K, OMEGA, PUSHOFF_RAMP_TICKS, R,
    THRUMBLER_RAMP_TICKS,
};

pub const PLAYER_MASS: f64 = 78.0; // kg
pub const THRUMBLER_CAP: f64 = 2.0; // m/s delta-v per possession

/// A ramped Δv: per-tick velocity increment applied for `ticks` more steps.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ramp {
    per_tick: Vec3,
    ticks: u32,
}

impl Ramp {
    const NONE: Ramp = Ramp {
        per_tick: Vec3::new(0.0, 0.0, 0.0),
        ticks: 0,
    };
}

/// Full kinematic state for a rigger in the rotating frame.
#[derive(Clone)]
pub struct PlayerBody {
    pub p: Vec3,
    pub v: Vec3,
    /// 1/PLAYER_MASS. (Kept for the rig-line solve; the contact freeze is now
    /// a stick-spring, so this no longer goes to 0.)
    pub inv_mass: f64,
    pub line: Option<Line>,
    /// GRAPPLE LATENCY: the sim tick at/after which this player may fire a
    /// new line. Set when a line is released, or when an in-flight claw is
    /// cancelled by an explicit release. A fire input is IGNORED while
    /// `sim.tick < refire_ready_tick` (re-fire cooldown) or while a line is
    /// already present (claw in flight OR attached — committed to target).
    /// Tick-counted (integer), no wall clock ⇒ deterministic.
    pub refire_ready_tick: u64,
    /// Clipped to a spar/ring/teammate.
    pub contact: bool,
    /// Touched the skin — out of the calm. LATCHED (never cleared here) so the
    /// one-shot `PlayerSkinned` event in SimWorld still fires exactly once.
    pub grounded: bool,
    pub dv_budget: f64,
    // ── internal continuous-feel state (NOT in the snapshot) ──────────────
    /// Latched world anchor the stick-spring pulls toward while `contact`.
    contact_anchor: Vec3,
    /// True once `contact_anchor` has been latched for the current contact.
    contact_latched: bool,
    pushoff_ramp: Ramp,
    thrumbler_ramp: Ramp,
}

pub fn make_player(p: Vec3) -> PlayerBody {
    PlayerBody {
        p,
        v: Vec3::new(0.0, 0.0, 0.0),
        inv_mass: 1.0 / PLAYER_MASS,
        line: None,
        refire_ready_tick: 0,
        contact: false,
        grounded: false,
        dv_budget: THRUMBLER_CAP,
        contact_anchor: Vec3::new(0.0, 0.0, 0.0),
        contact_latched: false,
        pushoff_ramp: Ramp::NONE,
        thrumbler_ramp: Ramp::NONE,
    }
}

/// Push the player off a contact surface in `dir` at `speed`. Only fires
/// when the player is in contact; clears the contact flag afterward. The Δv
/// is the SAME total magnitude, but eased over `PUSHOFF_RAMP_TICKS` ticks.
pub fn push_off(pl: &mut PlayerBody, dir: Vec3, speed: f64) {
    if !pl.contact {
        return;
    }
    let l = dir.len().max(1e-12);
    let total = dir.scale(speed / l); // identical total Δv to the old impulse
    pl.pushoff_ramp = Ramp {
        per_tick: total.scale(1.0 / PUSHOFF_RAMP_TICKS as f64),
        ticks: PUSHOFF_RAMP_TICKS,
    };
    pl.contact = false;
    pl.contact_latched = false;
}

/// Apply a small thrumbler nudge, capped by the remaining `dv_budget`.
/// Budget is consumed IMMEDIATELY (cap/determinism unchanged); the velocity
/// is delivered eased over `THRUMBLER_RAMP_TICKS` ticks.
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
    // Accumulate onto any in-flight ramp so rapid taps still sum to the
    // budget-capped total (total Δv preserved exactly).
    let remaining_prev = pl
        .thrumbler_ramp
        .per_tick
        .scale(pl.thrumbler_ramp.ticks as f64);
    let total = remaining_prev.add(dv);
    pl.thrumbler_ramp = Ramp {
        per_tick: total.scale(1.0 / THRUMBLER_RAMP_TICKS as f64),
        ticks: THRUMBLER_RAMP_TICKS,
    };
    pl.dv_budget = (pl.dv_budget - used).max(0.0);
}

/// Refill the thrumbler budget at the start of a new possession.
pub fn new_possession(pl: &mut PlayerBody) {
    pl.dv_budget = THRUMBLER_CAP;
}

fn apply_ramp(ramp: &mut Ramp) -> Vec3 {
    if ramp.ticks == 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    ramp.ticks -= 1;
    ramp.per_tick
}

/// Integrate one fixed sub-step:
///   1. Eased actuators (pushoff / thrumbler ramps).
///   2. Free-flight RK4 (unless contact/grounded).
///   3. Line constraint (if a line is attached).
///   4. Contact stick-spring (if clipped).
///   5. Soft grounding — penetration push-out + tangential friction bleed.
pub fn step_player(pl: &mut PlayerBody, h: f64, reel: i32) {
    step_player_anchored(pl, h, reel, None)
}

/// Same as [`step_player`] but with an explicit moving anchor body for a
/// player↔player line. `anchor` is the CURRENT body of the player this
/// line was fired at (teammate OR opponent); the rig constraint then
/// applies an equal-and-opposite impulse to it (momentum-conserving
/// slingshot / tether). When `anchor` is `None` the line uses its static
/// world `anchor_pos` exactly as before (byte-identical). The caller
/// (`sim_world::step`) is responsible for the deterministic disjoint
/// split-borrow of the stepped player vs. the anchor player.
pub fn step_player_anchored(
    pl: &mut PlayerBody,
    h: f64,
    reel: i32,
    mut anchor: Option<&mut Body>,
) {
    // 1. Eased actuators — deliver the ramped Δv this tick.
    let dv = apply_ramp(&mut pl.pushoff_ramp).add(apply_ramp(&mut pl.thrumbler_ramp));
    pl.v = pl.v.add(dv);

    // 2. Free-flight (canon ζ̈ = ω²ζ − 2iωζ̇, RK4 — IDENTICAL to before and
    //    untouched). It now ALSO runs while grounded so position keeps
    //    integrating; the soft-grounding spring (step 5) is a corrective
    //    force layered on top that holds the rigger on the skin. While in
    //    contact the stick-spring (step 4) does the holding instead of a
    //    hard freeze, so free-flight still runs and the spring resists it.
    {
        let s = PointState { p: pl.p, v: pl.v };
        let next = rk4_step(s, OMEGA, h);
        pl.p = next.p;
        pl.v = next.v;
    }

    // 3. Line constraint. A static-anchor line passes `None` (byte-identical
    //    to before). A player-bound line passes the moving anchor body so the
    //    spring is momentum-conserving (equal-and-opposite recoil on the
    //    anchor). The anchor body is mutated in place; the caller writes the
    //    recoil back onto the anchor player after the step.
    if let Some(ref mut line) = pl.line {
        let mut body = Body {
            p: pl.p,
            v: pl.v,
            inv_mass: pl.inv_mass,
        };
        match anchor.as_mut() {
            Some(a) => resolve_line(&mut body, line, Some(*a), reel, h),
            None => resolve_line(&mut body, line, None, reel, h),
        }
        pl.p = body.p;
        pl.v = body.v;
    }

    // 4. Contact stick-spring: pull toward the latched anchor instead of the
    //    old inv_mass = 0 freeze. Semi-implicit (ω_n·h ≈ 0.037 ≪ 2).
    if pl.contact {
        if !pl.contact_latched {
            pl.contact_anchor = pl.p;
            pl.contact_latched = true;
        }
        let d = pl.p.sub(pl.contact_anchor);
        // F = −K·d − C·v  applied as a Δv via semi-implicit Euler.
        let fx = -CONTACT_K * d.x - CONTACT_C * pl.v.x;
        let fy = -CONTACT_K * d.y - CONTACT_C * pl.v.y;
        let fz = -CONTACT_K * d.z - CONTACT_C * pl.v.z;
        pl.v = Vec3::new(
            pl.v.x + fx * pl.inv_mass * h,
            pl.v.y + fy * pl.inv_mass * h,
            pl.v.z + fz * pl.inv_mass * h,
        );
    } else {
        pl.contact_latched = false;
    }

    // 5. Soft grounding. Penetration push-out spring + exponential tangential
    //    friction bleed using the PRECOMPUTED `FRICTION_DECAY` (no exp()).
    let rho = pl.p.y.hypot(pl.p.z);
    if rho >= R {
        // Latch grounded so the one-shot PlayerSkinned event still fires once.
        pl.grounded = true;
        let inv_rho = 1.0 / rho.max(1e-12);
        let ny = pl.p.y * inv_rho;
        let nz = pl.p.z * inv_rho;
        let pen = rho - R; // ≥ 0 penetration depth
        let v_rad = pl.v.y * ny + pl.v.z * nz;
        // Push-out spring−damper along the inward normal (−n): F·(−n).
        let f = -GROUND_K * pen - GROUND_C * v_rad;
        pl.v = Vec3::new(
            pl.v.x,
            pl.v.y + f * ny * pl.inv_mass * h,
            pl.v.z + f * nz * pl.inv_mass * h,
        );
        // Tangential velocity = total minus the (updated) radial part; bleed
        // it with the precomputed per-step decay (identical f64 both sides).
        let v_rad2 = pl.v.y * ny + pl.v.z * nz;
        let vt_y = pl.v.y - v_rad2 * ny;
        let vt_z = pl.v.z - v_rad2 * nz;
        pl.v = Vec3::new(
            pl.v.x * FRICTION_DECAY,
            v_rad2 * ny + vt_y * FRICTION_DECAY,
            v_rad2 * nz + vt_z * FRICTION_DECAY,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: f64 = 1.0 / 240.0;

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
        // No ramp armed, stepping changes nothing from the actuator.
        step_player(&mut pl, H, 0);
        // Only free-flight from rest at origin ⇒ still ~zero.
        assert!(pl.v.len() < 1e-9, "no-op expected, got {:?}", pl.v);
    }

    /// Pushoff delivers the SAME total Δv as the old impulse, just eased: after
    /// PUSHOFF_RAMP_TICKS steps the integrated speed ≈ requested speed.
    #[test]
    fn push_off_total_dv_preserved_and_eased() {
        let mut pl = make_player(Vec3::new(0.0, 0.0, 0.0));
        pl.contact = true;
        push_off(&mut pl, Vec3::new(1.0, 0.0, 0.0), 5.0);
        assert!(!pl.contact, "contact cleared on pushoff");
        // After one step only a fraction of the 5 m/s has been applied.
        step_player(&mut pl, H, 0);
        assert!(pl.v.x > 0.0 && pl.v.x < 5.0, "eased, got {}", pl.v.x);
        // After the full ramp the cumulative Δv is the full 5 m/s (free-flight
        // from the origin contributes negligibly over 18·h).
        for _ in 0..(PUSHOFF_RAMP_TICKS as usize) {
            step_player(&mut pl, H, 0);
        }
        assert!((pl.v.x - 5.0).abs() < 1e-3, "total Δv preserved, got {}", pl.v.x);
    }

    #[test]
    fn thrumbler_capped_by_budget_total_preserved() {
        let mut pl = make_player(Vec3::new(0.0, 0.0, 0.0));
        pl.dv_budget = 1.0;
        // Ask for 3 m/s; only 1 m/s budget remains — should be capped.
        thrumbler(&mut pl, Vec3::new(3.0, 0.0, 0.0));
        assert!(pl.dv_budget.abs() < 1e-12, "budget consumed immediately");
        for _ in 0..(THRUMBLER_RAMP_TICKS as usize) {
            step_player(&mut pl, H, 0);
        }
        assert!((pl.v.x - 1.0).abs() < 1e-3, "capped total Δv = 1, got {}", pl.v.x);
    }

    #[test]
    fn new_possession_restores_budget() {
        let mut pl = make_player(Vec3::new(0.0, 0.0, 0.0));
        thrumbler(&mut pl, Vec3::new(2.0, 0.0, 0.0));
        assert!(pl.dv_budget < 1e-12);
        new_possession(&mut pl);
        assert!((pl.dv_budget - THRUMBLER_CAP).abs() < 1e-12);
    }

    /// Soft grounding: `grounded` latches (one-shot event), the player is
    /// pushed back inside the skin over a few ticks, and tangential motion
    /// bleeds away (it no longer hard-snaps to v = 0 in a single step).
    #[test]
    fn step_grounds_softly_at_skin() {
        let mut pl = make_player(Vec3::new(0.0, R + 1.0, 0.0));
        // Give it tangential (z) speed so we can watch friction bleed it.
        pl.v = Vec3::new(0.0, 0.0, 6.0);
        step_player(&mut pl, H, 0);
        assert!(pl.grounded, "grounded must latch");
        // Single step does NOT teleport-clamp to exactly R or zero velocity.
        assert!(pl.v.len() > 0.0, "velocity not hard-zeroed");
        // Drive it for ~1.5 s: penetration spring settles it at/inside R and
        // friction bleeds the tangential speed well down.
        for _ in 0..360 {
            step_player(&mut pl, H, 0);
        }
        let rho = pl.p.y.hypot(pl.p.z);
        // Steady state is a small physical penetration where the push-out
        // spring balances spin-gravity: pen ≈ ω²·R·m / GROUND_K
        // = 0.32²·45·78 / 5000 ≈ 0.072 m. Soft, bounded, no hard clamp.
        assert!(
            rho <= R + 0.15,
            "settled just inside the skin (soft), rho = {}",
            rho
        );
        assert!(pl.v.z.abs() < 1.5, "tangential bled by friction, vz = {}", pl.v.z);
        assert!(pl.grounded, "grounded stays latched");
    }

    /// The slingshot is EMERGENT, not scripted. With a static rig line and
    /// real free-flight RK4, reeling the rest_len in pulls the player onto a
    /// tighter orbit *continuously* — there is NO position teleport (the old
    /// `p = anchor + n·rest_len`) and NO explicit v_tan rescale anywhere.
    /// (The exact final speed depends on LINE_K/LINE_C, which the orchestrator
    /// re-tunes for gameplay viability — this test pins the *mechanism*.)
    #[test]
    fn reel_in_slingshot_is_emergent() {
        // Orbit a static anchor at the origin: 20 m out on +x with tangential
        // (y) speed. rho ≪ R so grounding never engages here.
        let mut pl = make_player(Vec3::new(20.0, 0.0, 0.0));
        pl.v = Vec3::new(0.0, 5.0, 0.0);
        pl.line = Some(Line {
            anchor_pos: Vec3::new(0.0, 0.0, 0.0),
            rest_len: 20.0,
            taut: false,
            attached: true,
            attach_tick: 0,
            anchor_player: None,
        });

        let mut max_step_jump = 0.0_f64;
        let mut prev_p = pl.p;
        for _ in 0..288 {
            step_player(&mut pl, H, -1);
            let jump = pl.p.sub(prev_p).len();
            if jump > max_step_jump {
                max_step_jump = jump;
            }
            prev_p = pl.p;
        }

        // Reeled inward (the rig is pulling the player onto a tighter orbit).
        assert!(pl.p.len() < 20.0, "reeled inward, |p| = {}", pl.p.len());
        // The line is taut and actively doing centripetal work.
        assert!(
            pl.line.as_ref().unwrap().taut,
            "the reeled-in line must be taut and pulling"
        );
        // CONTINUITY: no per-step teleport. A scripted `p = a + n·rest_len`
        // jump would move the player metres in one 1/240 s step; the spring
        // model keeps every step well under the free-flight bound.
        assert!(
            max_step_jump < 0.2,
            "motion must be continuous (no teleport); max step = {}",
            max_step_jump
        );
    }

    /// Contact uses a stick-spring (not an inv_mass=0 freeze): a contacting
    /// player stays pinned near its latch point against free-flight drift.
    #[test]
    fn contact_stick_spring_pins_player() {
        let mut pl = make_player(Vec3::new(10.0, 3.0, 0.0));
        pl.contact = true;
        let p0 = pl.p;
        for _ in 0..240 {
            step_player(&mut pl, H, 0);
        }
        let drift = pl.p.sub(p0).len();
        assert!(drift < 0.5, "stick-spring should pin the player, drift = {}", drift);
    }
}

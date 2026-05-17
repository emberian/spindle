//! Skin bounce, relative-velocity catch, and contest clatter — port of TS
//! sim/Collision.ts. Pure functions on bodies; callers emit events.

use crate::bell::{clatter, trueness, BellBody};
use crate::math::Vec3;
use crate::tuning::R;

pub const RESTITUTION: f64 = 0.55;
// LOCKSTEP with src/sim/Collision.ts. Widened from 0.7 m / 9 m/s — that
// envelope was unreachable for grapple locomotion vs a Coriolis bell (zero
// catches/match, no possession). A tether-claw snare: real but skill-gated.
pub const ARM_REACH: f64 = 1.8; // m — claw/tether snare radius
pub const CATCH_SPEED_BASE: f64 = 16.0; // m/s rel-speed a baseline rigger can absorb

/// Bounce a free body off the cylinder skin (cross-axis radius = R).
/// Reflects the outward-pointing radial velocity component by RESTITUTION;
/// clamps position inside the skin. Axial (x) component is untouched.
/// Returns `true` if a bounce occurred this step.
pub fn skin_bounce(p: &mut Vec3, v: &mut Vec3) -> bool {
    let rho = p.y.hypot(p.z);
    if rho < R {
        return false;
    }
    let ny = p.y / rho;
    let nz = p.z / rho;
    let vn = v.y * ny + v.z * nz; // radial component of velocity
    if vn > 0.0 {
        v.y -= (1.0 + RESTITUTION) * vn * ny;
        v.z -= (1.0 + RESTITUTION) * vn * nz;
    }
    let s = (R - 1e-3) / rho;
    p.y *= s;
    p.z *= s;
    true
}

/// Outcome of a catch attempt.
pub enum CatchResult {
    /// Clean catch: bell absorbed.
    Caught,
    /// Partial: bell partially damped and knocked off true.
    Bobble,
    /// Too far, moving away, or moving too fast.
    Miss,
}

/// A catch is physical: in arm's reach, closing, and slow enough relative to
/// the catcher (skill widens the absorbable speed). Otherwise a bobble
/// (partial damp + the bell loses trueness — an audible clatter).
pub fn try_catch(
    bell_pos: Vec3,
    bell_vel: Vec3,
    player_pos: Vec3,
    player_vel: Vec3,
    skill: f64,
) -> CatchResult {
    let d = bell_pos.sub(player_pos);
    let gap = d.len();
    if gap > ARM_REACH {
        return CatchResult::Miss;
    }
    let rel = bell_vel.sub(player_vel);
    let rel_speed = rel.len();
    // Closing speed: positive means the bell is approaching the player.
    let closing = if gap < 1e-6 { 1.0 } else { -rel.dot(d.norm()) };
    if closing < -0.5 {
        return CatchResult::Miss;
    }
    if rel_speed <= CATCH_SPEED_BASE * skill {
        CatchResult::Caught
    } else {
        CatchResult::Bobble
    }
}

/// Apply a bobble: kill most of the relative velocity (damp to 25%), then
/// knock the bell off its true spin axis with a fixed off-axis impulse.
pub fn apply_bobble(bell: &mut BellBody, player_vel: Vec3) {
    let rel = bell.v.sub(player_vel);
    bell.v = Vec3::new(
        player_vel.x + rel.x * 0.25,
        player_vel.y + rel.y * 0.25,
        player_vel.z + rel.z * 0.25,
    );
    clatter(bell, Vec3::new(0.0, 4.0, 3.0));
}

/// A defender contesting the live bell knocks it off true. Strength scales
/// with the contester's closing speed (deep/violent contests hit harder).
pub fn contest_clatter(bell: &mut BellBody, contester_vel: Vec3, strength: f64) {
    let rel = bell.v.sub(contester_vel).len();
    let mag = (4.0 + rel * 0.4).min(18.0) * strength;
    // Off the ring axis (body y/z) so trueness drops.
    clatter(bell, Vec3::new(0.0, 0.8 * mag, 0.6 * mag));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bell::BellBody;
    use crate::math::{Quat, Vec3};

    fn make_bell(wx: f64) -> BellBody {
        BellBody {
            p: Vec3::new(0.0, 0.0, 0.0),
            v: Vec3::new(20.0, 0.0, 0.0),
            q: Quat::ident(),
            w: Vec3::new(wx, 0.0, 0.0),
        }
    }

    /// Skin bounce reflects radial velocity and keeps the body inside R.
    #[test]
    fn skin_bounce_reflects_radial_and_keeps_axial() {
        let mut p = Vec3::new(0.0, R + 2.0, 0.0);
        let mut v = Vec3::new(5.0, 8.0, 0.0);
        let hit = skin_bounce(&mut p, &mut v);
        assert!(hit);
        assert!(p.y.hypot(p.z) <= R);
        assert!(v.y < 0.0, "radial velocity should be reflected inward");
        assert!((v.x - 5.0).abs() < 1e-12, "axial component must be untouched");
    }

    /// No bounce when already inside the skin.
    #[test]
    fn skin_bounce_no_hit_when_inside() {
        let mut p = Vec3::new(0.0, R - 1.0, 0.0);
        let mut v = Vec3::new(1.0, 2.0, 0.0);
        let hit = skin_bounce(&mut p, &mut v);
        assert!(!hit);
    }

    /// A miss when the bell is farther than ARM_REACH (not 5 m).
    #[test]
    fn catch_requires_arm_reach_not_5m() {
        let result = try_catch(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            1.0,
        );
        assert!(matches!(result, CatchResult::Miss));
    }

    /// Slow, closing bell within arm's reach → Caught; fast one → Bobble.
    #[test]
    fn slow_in_reach_is_caught_fast_is_bobble() {
        let pos = Vec3::new(0.0, 0.0, 0.0);
        let pv = Vec3::new(0.0, 0.0, 0.0);

        let slow = try_catch(
            Vec3::new(0.3, 0.0, 0.0),
            Vec3::new(-3.0, 0.0, 0.0),
            pos,
            pv,
            1.0,
        );
        assert!(matches!(slow, CatchResult::Caught));

        let fast = try_catch(
            Vec3::new(0.3, 0.0, 0.0),
            Vec3::new(-30.0, 0.0, 0.0),
            pos,
            pv,
            1.0,
        );
        assert!(matches!(fast, CatchResult::Bobble));
    }

    /// contest_clatter must drop the trueness of an otherwise true bell.
    #[test]
    fn contest_clatter_drops_trueness() {
        let mut bell = make_bell(26.0);
        assert!(trueness(bell.w) > 0.999, "trueness = {}", trueness(bell.w));
        contest_clatter(&mut bell, Vec3::new(0.0, 0.0, 0.0), 1.0);
        assert!(trueness(bell.w) < 0.9, "trueness after clatter = {}", trueness(bell.w));
    }
}

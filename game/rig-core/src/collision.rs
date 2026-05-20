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

// ── OFFENSE REBUILD: committed-catch envelope ────────────────────────────────
// The default snare (ARM_REACH/CATCH_SPEED_BASE) is reflex-only — it cannot
// take a real played pass off a Coriolis bell, which is exactly why
// possession never formed (heldFrac≈0.01, gateClears=0). A player who has
// COMMITTED to the catch (carrier's intended receiver, recover pack, or a
// defender stepping into a pick) reaches and absorbs much more: a real but
// still skill-gated "tether-claw snare". Minimal + deterministic.
pub const COMMIT_ARM_REACH: f64 = 7.0; // m — committed snare radius
pub const COMMIT_CATCH_SPEED: f64 = 38.0; // m/s rel-speed absorbable when committed

/// A catch is physical: in (effective) arm's reach, closing, and slow
/// enough relative to the catcher. `committed` widens both the reach and
/// the absorbable relative speed (a real, deliberate two-hand snare) so a
/// played pass / pick actually completes. Otherwise a bobble (partial damp
/// + the bell loses trueness — an audible clatter).
pub fn try_catch(
    bell_pos: Vec3,
    bell_vel: Vec3,
    player_pos: Vec3,
    player_vel: Vec3,
    skill: f64,
) -> CatchResult {
    try_catch_ex(bell_pos, bell_vel, player_pos, player_vel, skill, false)
}

pub fn try_catch_ex(
    bell_pos: Vec3,
    bell_vel: Vec3,
    player_pos: Vec3,
    player_vel: Vec3,
    skill: f64,
    committed: bool,
) -> CatchResult {
    let reach = if committed { COMMIT_ARM_REACH } else { ARM_REACH };
    let absorb = if committed {
        COMMIT_CATCH_SPEED
    } else {
        CATCH_SPEED_BASE
    } * skill;
    let d = bell_pos.sub(player_pos);
    let gap = d.len();
    if gap > reach {
        return CatchResult::Miss;
    }
    let rel = bell_vel.sub(player_vel);
    let rel_speed = rel.len();
    // Closing speed: positive means the bell is approaching the player.
    let closing = if gap < 1e-6 { 1.0 } else { -rel.dot(d.norm()) };
    // A committed catcher (tether-claw dive) is deliberately putting
    // themselves in the bell's path — closing direction is irrelevant; the
    // rel_speed check is the real skill gate. Reflex catch needs the ball
    // approaching.
    if !committed && closing < -0.5 {
        return CatchResult::Miss;
    }
    if rel_speed <= absorb {
        CatchResult::Caught
    } else if committed && rel_speed <= absorb * 1.6 {
        // Committed but the ball is hot — controlled bobble, not a clean
        // take, but it still TOUCHES (spends the throw → gate can advance).
        CatchResult::Bobble
    } else {
        CatchResult::Bobble
    }
}

/// OFFENSE REBUILD: a defender within strip range of the bell-CARRIER rips
/// the bell loose. Returns the loose-bell velocity (carrier's velocity plus
/// a shove along the defender→away direction). Caller emits the turnover /
/// loose-ball events; this is the only place a held bell is contested, so
/// possession is genuinely fought for (denial > 0).
pub const STRIP_RANGE: f64 = 3.4; // m — must crowd the carrier to strip

pub fn strip_velocity(carrier_vel: Vec3, defender_vel: Vec3) -> Vec3 {
    // Knock it off the carrier's line, biased by the defender's motion.
    Vec3::new(
        carrier_vel.x * 0.4 + defender_vel.x * 0.5,
        carrier_vel.y * 0.4 + defender_vel.y * 0.5 + 1.5,
        carrier_vel.z * 0.4 + defender_vel.z * 0.5,
    )
}

/// CONTEST CONSEQUENCE: a sustained, committed strip should genuinely flip
/// spatial control — not just nudge the bell where both players re-grab it.
/// On top of the base knock we shove the loose bell into the CONTESTER's
/// space (carrier→defender direction) at a magnitude scaling with how long
/// the press was sustained PAST threshold. A marginal strip (just at the
/// hysteresis edge) stays soft and contestable; a hard-fought one decisively
/// cedes the volume to the contester — so crowding a carrier is a real risk
/// (you must commit and track) and a real reward (a won contest is a true
/// turnover, not a coin-flip scramble).
///
/// `away_dir` = unit carrier→defender direction (the side the contester owns).
/// `over_press` = press ticks held PAST the strip threshold (≥ 0).
/// Pure & deterministic: a fixed function of its inputs only (tick count, no
/// wall-clock; no RNG).
pub fn strip_velocity_dir(
    carrier_vel: Vec3,
    defender_vel: Vec3,
    away_dir: Vec3,
    over_press: u32,
) -> Vec3 {
    let base = strip_velocity(carrier_vel, defender_vel);
    // Commitment ramp: 0 extra at threshold, saturating to a firm shove by
    // ~0.3 s of extra sustained press (tick count only — determinism-safe).
    let commit = (over_press as f64 / 70.0).min(1.0);
    let shove = 3.0 + 4.0 * commit; // 3 → 7 m/s into the contester's space
    Vec3::new(
        base.x + away_dir.x * shove,
        base.y + away_dir.y * shove,
        base.z + away_dir.z * shove,
    )
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

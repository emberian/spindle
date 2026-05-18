//! 1:1 faithful port of src/ai/roles/Freewing.ts (FROZEN SPEC).
//!
//! The Freewing — the artist, antispinward side; attacks the Free end
//! (the hardest, most valuable direction) and plays HIGH (near the axis)
//! to exploit Coriolis curves. Decision tick: 10 Hz.
//!
//! Public surface ported (Freewing.ts:1-139):
//!   - `FreewingIntent`           ← interface FreewingIntent  (Freewing.ts:22-26)
//!   - `FreewingIntentKind`       ← the `intent` string union  (Freewing.ts:24)
//!   - `freewing_policy`          ← freewingPolicy            (Freewing.ts:38-127)
//!   - `freewing_navigate`        ← freewingNavigate          (Freewing.ts:129-138)
//!
//! `RoleStyle` (Freewing.ts:33-36) is the frozen substrate
//! `crate::ai::types::RoleStyle`; not redefined here.

use std::f64::consts::PI;

use super::super::orientation::{attack_ring_x, defend_ring_x};
use super::super::plan_bridge::{self, GrapplePlan};
use super::super::profile::TeamProfile;
use super::super::rng::AiRng;
use super::super::types::{MatchState, PlayerSim, RoleStyle, SimState};
use crate::math::Vec3;

// Freewings operate HIGH — near the axis to exploit Coriolis curves.
// Freewing.ts:19-20. REG.R == crate::tuning::R.
const FREEWING_ATTACK_RADIUS: f64 = crate::tuning::R * 0.22;
const FREEWING_HOLD_RADIUS: f64 = crate::tuning::R * 0.38;

/// The `intent` string-literal union of `FreewingIntent` (Freewing.ts:24):
/// `'free-attack' | 'loop-receive' | 'high-hold' | 'cross-contest'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreewingIntentKind {
    FreeAttack,
    LoopReceive,
    HighHold,
    CrossContest,
}

/// 1:1 with `interface FreewingIntent` (Freewing.ts:22-26).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreewingIntent {
    pub target_pos: Vec3,
    pub intent: FreewingIntentKind,
    pub is_attacking: bool,
}

/// 1:1 port of `freewingPolicy` (Freewing.ts:38-127).
///
/// `rng: () => number` → `rng: &mut AiRng` (`rng()` → `rng.next()`).
/// `style?: RoleStyle` → `style: Option<RoleStyle>`. The `sAngle()` helper
/// (Freewing.ts:46) is reproduced inline as a closure with the identical
/// `style ? style.angle : rng()` semantics, preserving the exact rng draw
/// order/count: the `freeEndBias` coin flip (Freewing.ts:58) stays on
/// inline `rng.next()` in both modes, exactly as the TS comment mandates.
pub fn freewing_policy(
    player: &PlayerSim,
    state: &SimState,
    match_state: &MatchState,
    profile: &TeamProfile,
    rng: &mut AiRng,
    style: Option<RoleStyle>,
) -> FreewingIntent {
    // const sAngle = () => (style ? style.angle : rng());  (Freewing.ts:46)
    // Closure form keeps the draw-on-call semantics: a fresh rng draw is
    // taken iff `style` is None, at the exact point `sAngle()` is invoked.
    let mut s_angle = |rng: &mut AiRng| -> f64 {
        match style {
            Some(s) => s.angle,
            None => rng.next(),
        }
    };

    let bell = &state.bell; // Freewing.ts:47
    let possession = match_state.possession == player.team; // Freewing.ts:48
    // ORIENTATION-CORRECT: still plays HIGH/near-axis for the Free (5-pt)
    // curve game, but DIRECTION is the ring THIS team attacks.
    let free_x = attack_ring_x(player.team); // Freewing.ts:52
    let def_x = defend_ring_x(player.team); // Freewing.ts:53

    if possession {
        // Freewing.ts:54
        let bell_dist_to_free = (bell.p.x - free_x).abs(); // Freewing.ts:55

        // Strong free-end bias: aggressively go for the Free ring.
        // Freewing.ts:58 — the `||` short-circuits exactly as in JS, so the
        // `rng.next()` coin draw happens iff freeEndBias <= 0.55.
        if profile.free_end_bias > 0.55 || rng.next() < profile.free_end_bias {
            if bell_dist_to_free < crate::tuning::L * 0.35 {
                // Free-end attack: near axis (cross-spin curve territory).
                // Freewing.ts:61
                let attack_x = free_x + if free_x > 0.0 { -20.0 } else { 20.0 };
                let angle = PI * (0.05 + s_angle(rng) * 0.12); // Freewing.ts:62
                return FreewingIntent {
                    target_pos: Vec3 {
                        x: attack_x,
                        y: FREEWING_ATTACK_RADIUS * angle.cos(),
                        z: FREEWING_ATTACK_RADIUS * angle.sin(),
                    },
                    intent: FreewingIntentKind::FreeAttack,
                    is_attacking: true,
                }; // Freewing.ts:63-71
            }

            // Loop receive: near axis, ready for a loop-setter feed.
            if profile.loop_propensity > 0.4 {
                // Freewing.ts:75
                let loop_x = bell.p.x
                    + (if free_x > 0.0 { 80.0 } else { -80.0 }) * profile.loop_propensity; // Freewing.ts:76
                let angle = s_angle(rng) * PI * 0.1; // Freewing.ts:77
                return FreewingIntent {
                    target_pos: Vec3 {
                        x: loop_x,
                        y: FREEWING_ATTACK_RADIUS * angle.cos(),
                        z: FREEWING_ATTACK_RADIUS * angle.sin(),
                    },
                    intent: FreewingIntentKind::LoopReceive,
                    is_attacking: true,
                }; // Freewing.ts:78-86
            }
        }

        // Midfield high hold: wait for the pass.  (Freewing.ts:90-101)
        let mid_x = bell.p.x + (if free_x > 0.0 { 60.0 } else { -60.0 }) * 0.6; // Freewing.ts:91
        let angle = PI * (0.15 + s_angle(rng) * 0.15); // Freewing.ts:92
        FreewingIntent {
            target_pos: Vec3 {
                x: mid_x,
                y: FREEWING_HOLD_RADIUS * angle.cos(),
                z: FREEWING_HOLD_RADIUS * angle.sin(),
            },
            intent: FreewingIntentKind::HighHold,
            is_attacking: false,
        } // Freewing.ts:93-101
    } else {
        // Defense: cross-spin contest position — force the thrower
        // Faith-side.  (Freewing.ts:102-126)
        // const oppHolder = state.players.find(p => p.id === bell.heldBy)
        // Freewing.ts:104. bell.heldBy is `string | null`; when null the TS
        // `===` can never match, so `find` yields undefined → None here.
        let opp_holder = match &bell.held_by {
            Some(held_by) => state.players.iter().find(|p| &p.id == held_by),
            None => None,
        };
        if let Some(opp_holder) = opp_holder {
            // Freewing.ts:105
            // Position between the holder and OUR defended ring, near axis.
            let free_offset = if def_x > 0.0 { 15.0 } else { -15.0 }; // Freewing.ts:107
            return FreewingIntent {
                target_pos: Vec3 {
                    x: opp_holder.p.x + free_offset,
                    y: opp_holder.p.y * 0.5,
                    z: opp_holder.p.z * 0.5,
                },
                intent: FreewingIntentKind::CrossContest,
                is_attacking: false,
            }; // Freewing.ts:108-116
        }

        // Guard OUR OWN ring.  (Freewing.ts:119-125)
        let guard_x = def_x + if def_x > 0.0 { -30.0 } else { 30.0 }; // Freewing.ts:120
        FreewingIntent {
            target_pos: Vec3 {
                x: guard_x,
                y: FREEWING_HOLD_RADIUS,
                z: 0.0,
            },
            intent: FreewingIntentKind::CrossContest,
            is_attacking: false,
        } // Freewing.ts:121-125
    }
}

/// 1:1 port of `freewingNavigate` (Freewing.ts:129-138).
///
/// TS calls `freewingPolicy(...)` with NO `style` arg → `style = None`
/// here (Freewing.ts:136), then `planGrapple(player, targetPos, state)`
/// with `avoidDefenders` defaulted true and no sticky (Freewing.ts:137).
/// Returns `GrapplePlan | null` → `Option<GrapplePlan>`.
pub fn freewing_navigate(
    player: &PlayerSim,
    state: &SimState,
    match_state: &MatchState,
    profile: &TeamProfile,
    rng: &mut AiRng,
) -> Option<GrapplePlan> {
    let intent = freewing_policy(player, state, match_state, profile, rng, None);
    plan_bridge::plan_grapple(player, intent.target_pos, state, true, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{BellState, MatchPhase, RiggerRole, TeamSide};
    use crate::math::Quat;

    fn vz() -> Vec3 {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    fn mk_player(id: &str, team: TeamSide, p: Vec3) -> PlayerSim {
        PlayerSim {
            id: id.to_string(),
            team,
            role: RiggerRole::Freewing,
            p,
            v: vz(),
            q: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
            line: None,
            dv_budget: 0.0,
            contact_ref: None,
            grounded: false,
        }
    }

    fn mk_state(bell_p: Vec3, held_by: Option<String>, players: Vec<PlayerSim>) -> SimState {
        SimState {
            tick: 0.0,
            omega: crate::tuning::OMEGA,
            bell: BellState {
                p: bell_p,
                v: vz(),
                q: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                w: vz(),
                chime: 1.0,
                held_by,
                thrown_by: None,
                touched_since_throw: false,
                release_pos: vz(),
                release_tick: 0.0,
                pass_chain: vec![],
            },
            players,
        }
    }

    fn mk_match(possession: TeamSide) -> MatchState {
        use crate::ai::types::{Cast, FaithEnd, Gate};
        MatchState {
            inning: 0.0,
            spine: false,
            possession,
            faith_end: FaithEnd::PlusX,
            cast: Cast { throws_left: 3, gate: Gate::First, spot_x: 0.0 },
            contest: None,
            score_home: 0.0,
            score_away: 0.0,
            phase: MatchPhase::Live,
            message: String::new(),
            winner: None,
        }
    }

    // Determinism: identical inputs → bit-identical output (same rng seed).
    #[test]
    fn deterministic_same_seed() {
        let player = mk_player("H1", TeamSide::Home, Vec3 { x: 0.0, y: 5.0, z: 0.0 });
        let st = mk_state(vz(), None, vec![player.clone()]);
        let m = mk_match(TeamSide::Home);
        let prof = TeamProfile::baseline();

        let mut r1 = AiRng::make(42, 100, 0);
        let mut r2 = AiRng::make(42, 100, 0);
        let a = freewing_policy(&player, &st, &m, &prof, &mut r1, None);
        let b = freewing_policy(&player, &st, &m, &prof, &mut r2, None);
        assert_eq!(a.target_pos, b.target_pos);
        assert_eq!(a.intent, b.intent);
        assert_eq!(a.is_attacking, b.is_attacking);
    }

    // Supplied style removes rng dependence for the angle draw, so two
    // different rng seeds must yield the SAME positional output (the
    // freeEndBias coin still draws, but baseline 0.5 > 0.55 is false; with
    // a high free_end_bias the coin is short-circuited away entirely).
    #[test]
    fn style_makes_angle_rng_independent() {
        let player = mk_player("H1", TeamSide::Home, vz());
        // bell near the Free ring so we hit the free-attack branch.
        let free_x = attack_ring_x(TeamSide::Home);
        let st = mk_state(Vec3 { x: free_x, y: 0.0, z: 0.0 }, None, vec![player.clone()]);
        let m = mk_match(TeamSide::Home);
        let mut prof = TeamProfile::baseline();
        prof.free_end_bias = 0.9; // > 0.55 → coin draw short-circuited

        let style = Some(RoleStyle { angle: 0.3, radius: 0.5 });
        let mut r1 = AiRng::make(1, 1, 0);
        let mut r2 = AiRng::make(999, 7, 1);
        let a = freewing_policy(&player, &st, &m, &prof, &mut r1, style);
        let b = freewing_policy(&player, &st, &m, &prof, &mut r2, style);
        assert_eq!(a.intent, FreewingIntentKind::FreeAttack);
        assert_eq!(a.target_pos, b.target_pos);
    }

    // Free-attack branch: by construction is_attacking == true and the
    // target sits on the FREEWING_ATTACK_RADIUS cross-section circle.
    #[test]
    fn free_attack_is_on_attack_radius_circle() {
        let player = mk_player("H1", TeamSide::Home, vz());
        let free_x = attack_ring_x(TeamSide::Home);
        let st = mk_state(Vec3 { x: free_x, y: 0.0, z: 0.0 }, None, vec![player.clone()]);
        let m = mk_match(TeamSide::Home);
        let mut prof = TeamProfile::baseline();
        prof.free_end_bias = 0.9;

        let mut rng = AiRng::make(7, 7, 0);
        let intent = freewing_policy(
            &player,
            &st,
            &m,
            &prof,
            &mut rng,
            Some(RoleStyle { angle: 0.0, radius: 0.0 }),
        );
        assert_eq!(intent.intent, FreewingIntentKind::FreeAttack);
        assert!(intent.is_attacking);
        let r = (intent.target_pos.y * intent.target_pos.y
            + intent.target_pos.z * intent.target_pos.z)
            .sqrt();
        assert!((r - FREEWING_ATTACK_RADIUS).abs() < 1e-9);
        // attackX = freeX - 20 (freeX > 0 for Home).
        assert!((intent.target_pos.x - (free_x - 20.0)).abs() < 1e-9);
    }

    // Defense with an identifiable opp holder → cross-contest, not
    // attacking, x offset toward our defended ring.
    #[test]
    fn defense_cross_contest_on_holder() {
        let me = mk_player("H1", TeamSide::Home, vz());
        let holder = mk_player("A1", TeamSide::Away, Vec3 { x: 10.0, y: 4.0, z: 6.0 });
        let st = mk_state(
            Vec3 { x: 10.0, y: 4.0, z: 6.0 },
            Some("A1".to_string()),
            vec![me.clone(), holder.clone()],
        );
        // possession is Away → `me` (Home) is on defense.
        let m = mk_match(TeamSide::Away);
        let prof = TeamProfile::baseline();
        let mut rng = AiRng::make(3, 3, 0);
        let intent = freewing_policy(&me, &st, &m, &prof, &mut rng, None);
        assert_eq!(intent.intent, FreewingIntentKind::CrossContest);
        assert!(!intent.is_attacking);
        let def_x = defend_ring_x(TeamSide::Home);
        let off = if def_x > 0.0 { 15.0 } else { -15.0 };
        assert!((intent.target_pos.x - (10.0 + off)).abs() < 1e-9);
        assert!((intent.target_pos.y - 4.0 * 0.5).abs() < 1e-9);
        assert!((intent.target_pos.z - 6.0 * 0.5).abs() < 1e-9);
    }

    // Defense, no holder (bell loose) → guard OUR ring at hold radius.
    #[test]
    fn defense_guard_ring_when_no_holder() {
        let me = mk_player("H1", TeamSide::Home, vz());
        let st = mk_state(vz(), None, vec![me.clone()]);
        let m = mk_match(TeamSide::Away); // Home on defense
        let prof = TeamProfile::baseline();
        let mut rng = AiRng::make(5, 5, 0);
        let intent = freewing_policy(&me, &st, &m, &prof, &mut rng, None);
        assert_eq!(intent.intent, FreewingIntentKind::CrossContest);
        assert!(!intent.is_attacking);
        let def_x = defend_ring_x(TeamSide::Home);
        let guard_x = def_x + if def_x > 0.0 { -30.0 } else { 30.0 };
        assert!((intent.target_pos.x - guard_x).abs() < 1e-9);
        assert!((intent.target_pos.y - FREEWING_HOLD_RADIUS).abs() < 1e-9);
        assert!(intent.target_pos.z == 0.0);
    }

    // navigate uses style=None and forwards to the planner; close target
    // (player already at target) → None, exactly like plan_grapple's <3m.
    #[test]
    fn navigate_returns_none_when_already_at_target() {
        // Force the guard-ring branch then place the player AT guard_x.
        let def_x = defend_ring_x(TeamSide::Home);
        let guard_x = def_x + if def_x > 0.0 { -30.0 } else { 30.0 };
        let me = mk_player(
            "H1",
            TeamSide::Home,
            Vec3 { x: guard_x, y: FREEWING_HOLD_RADIUS, z: 0.0 },
        );
        let st = mk_state(vz(), None, vec![me.clone()]);
        let m = mk_match(TeamSide::Away);
        let prof = TeamProfile::baseline();
        let mut rng = AiRng::make(11, 11, 0);
        let plan = freewing_navigate(&me, &st, &m, &prof, &mut rng);
        assert!(plan.is_none());
    }
}

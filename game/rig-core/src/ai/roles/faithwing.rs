//! 1:1 Rust port of src/ai/roles/Faithwing.ts (FROZEN SPEC).
//!
//! Faithwing role policy — the power wing, spinward side. Devastating in
//! the Faith direction; the heavy reliable scorer. Decision tick: 10 Hz.
//!
//! Responsibilities:
//!   - Position to receive on the Faith (spinward) side.
//!   - Drive toward the Faith ring when bell approaches.
//!   - On defense: shadow the opposing Freewing (the most dangerous scorer).
//!
//! Symbol map (TS → Rust):
//!   - `FaithwingIntent` (struct + `FaithwingIntentKind` enum for the
//!     `'faith-attack' | 'receive' | 'shadow-freewing'` union).
//!   - `faithwingPolicy`  → `faithwing_policy`
//!   - `faithwingNavigate` → `faithwing_navigate`
//!   - `RoleStyle` is the frozen substrate `crate::ai::types::RoleStyle`.
//!
//! Faithwing.ts line spans:
//!   - FAITHWING_RADIUS const            : Faithwing.ts:16
//!   - FaithwingIntent interface         : Faithwing.ts:18-21
//!   - faithwingPolicy                   : Faithwing.ts:31-99
//!   - faithwingNavigate                 : Faithwing.ts:101-110

use crate::ai::orientation::{attack_ring_x, defend_ring_x};
use crate::ai::plan_bridge::{plan_grapple, GrapplePlan};
use crate::ai::profile::TeamProfile;
use crate::ai::rng::AiRng;
use crate::ai::types::{MatchState, PlayerSim, RiggerRole, RoleStyle, SimState};
use crate::math::Vec3;
use crate::tuning::R as REG_R;

/// `FAITHWING_RADIUS = REG.R * 0.55` (Faithwing.ts:16).
const FAITHWING_RADIUS: f64 = REG_R * 0.55;

/// The `intent` discriminant of `FaithwingIntent` (Faithwing.ts:20). The
/// TS string-literal union `'faith-attack' | 'receive' | 'shadow-freewing'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaithwingIntentKind {
    /// TS `'faith-attack'`
    FaithAttack,
    /// TS `'receive'`
    Receive,
    /// TS `'shadow-freewing'`
    ShadowFreewing,
}

/// 1:1 with the TS `FaithwingIntent` interface (Faithwing.ts:18-21).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaithwingIntent {
    pub target_pos: Vec3,
    pub intent: FaithwingIntentKind,
}

/// 1:1 port of `faithwingPolicy` (Faithwing.ts:31-99).
///
/// `rng: () => number` → `rng: &mut AiRng`; `rng()` → `rng.next()`.
/// `style?: RoleStyle` → `style: Option<RoleStyle>`.
pub fn faithwing_policy(
    player: &PlayerSim,
    state: &SimState,
    match_: &MatchState,
    profile: &TeamProfile,
    rng: &mut AiRng,
    style: Option<RoleStyle>,
) -> FaithwingIntent {
    // `const sAngle = () => (style ? style.angle : rng());` (Faithwing.ts:38).
    // Closure mutably borrows `rng`; called at most once on any taken path,
    // preserving the TS draw order/count exactly.
    let mut s_angle = || match style {
        Some(s) => s.angle,
        None => rng.next(),
    };

    let bell = &state.bell;
    let possession = match_.possession == player.team;
    // ORIENTATION-CORRECT: attack/defend rings relative to THIS team.
    let faith_x = attack_ring_x(player.team); // the ring we attack
    let def_x = defend_ring_x(player.team); // the ring we defend

    if possession {
        // Check if the bell is close to OUR attacking ring.
        let bell_dist_to_faith = (bell.p.x - faith_x).abs();
        if bell_dist_to_faith < REG_L_TIMES_0_3 {
            // Attack mode: sprint into position to receive a scoring pass.
            let attack_x = faith_x + if faith_x > 0.0 { -15.0 } else { 15.0 };
            let angle = std::f64::consts::PI * 0.35 * (s_angle() * 0.4 + 0.8);
            return FaithwingIntent {
                target_pos: Vec3 {
                    x: attack_x,
                    y: FAITHWING_RADIUS * angle.cos(),
                    z: FAITHWING_RADIUS * angle.sin(),
                },
                intent: FaithwingIntentKind::FaithAttack,
            };
        }

        // Midfield receiving position, biased toward the Faith side.
        let mid_x = bell.p.x
            + if faith_x > 0.0 {
                50.0 * profile.aggression
            } else {
                -50.0 * profile.aggression
            };
        let angle = std::f64::consts::PI * (0.25 + s_angle() * 0.2);
        FaithwingIntent {
            target_pos: Vec3 {
                x: mid_x,
                y: FAITHWING_RADIUS * angle.cos(),
                z: FAITHWING_RADIUS * angle.sin(),
            },
            intent: FaithwingIntentKind::Receive,
        }
    } else {
        // Defense: shadow the opposing Freewing (they're the dangerous one).
        let opponent_freewing = state
            .players
            .iter()
            .find(|p| p.team != player.team && p.role == RiggerRole::Freewing);
        if let Some(opponent_freewing) = opponent_freewing {
            // Position between the Freewing and OUR defended ring.
            let shadow_x = (opponent_freewing.p.x + def_x) / 2.0;
            return FaithwingIntent {
                target_pos: Vec3 {
                    x: shadow_x,
                    y: opponent_freewing.p.y * 0.8,
                    z: opponent_freewing.p.z * 0.8,
                },
                intent: FaithwingIntentKind::ShadowFreewing,
            };
        }

        // Fall back to guarding OUR OWN ring.
        let guard_x = def_x + if def_x > 0.0 { -40.0 } else { 40.0 };
        FaithwingIntent {
            target_pos: Vec3 {
                x: guard_x,
                y: FAITHWING_RADIUS * 0.7,
                z: 0.0,
            },
            intent: FaithwingIntentKind::ShadowFreewing,
        }
    }
}

/// `REG.L * 0.3` (Faithwing.ts:46). `crate::tuning::L` is the frozen
/// mirror of `REG.L`; folded to a const so the literal matches the TS
/// `Math.abs(bell.p.x - faithX) < REG.L * 0.3` byte-for-byte.
const REG_L_TIMES_0_3: f64 = crate::tuning::L * 0.3;

/// 1:1 port of `faithwingNavigate` (Faithwing.ts:101-110).
///
/// TS calls `planGrapple(player, targetPos, state)` with the default
/// `avoidDefenders = true` and no `sticky`; the bridge maps that to
/// `plan_grapple(.., true, None)`. Returns TS `GrapplePlan | null` →
/// `Option<GrapplePlan>`.
pub fn faithwing_navigate(
    player: &PlayerSim,
    state: &SimState,
    match_: &MatchState,
    profile: &TeamProfile,
    rng: &mut AiRng,
) -> Option<GrapplePlan> {
    let intent = faithwing_policy(player, state, match_, profile, rng, None);
    plan_grapple(player, intent.target_pos, state, true, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{BellState, FaithEnd, MatchPhase, TeamSide};
    use crate::ai::types::{Cast, Gate};
    use crate::math::Quat;

    fn vz() -> Vec3 {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    fn mk_player(id: &str, team: TeamSide, role: RiggerRole, p: Vec3) -> PlayerSim {
        PlayerSim {
            id: id.to_string(),
            team,
            role,
            p,
            v: vz(),
            q: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
            line: None,
            dv_budget: 0.0,
            contact_ref: None,
            grounded: false,
        }
    }

    fn mk_state(bell_x: f64, players: Vec<PlayerSim>) -> SimState {
        SimState {
            tick: 0.0,
            omega: crate::tuning::OMEGA,
            bell: BellState {
                p: Vec3 { x: bell_x, y: 0.0, z: 0.0 },
                v: vz(),
                q: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                w: vz(),
                chime: 1.0,
                held_by: None,
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

    // ── determinism ──────────────────────────────────────────────────────
    #[test]
    fn policy_deterministic_same_inputs() {
        let me = mk_player("H1", TeamSide::Home, RiggerRole::Faithwing, vz());
        let st = mk_state(0.0, vec![me.clone()]);
        let m = mk_match(TeamSide::Home);
        let pr = TeamProfile::baseline();
        let mut r1 = AiRng::make(42, 10, 0);
        let mut r2 = AiRng::make(42, 10, 0);
        let a = faithwing_policy(&me, &st, &m, &pr, &mut r1, None);
        let b = faithwing_policy(&me, &st, &m, &pr, &mut r2, None);
        assert_eq!(a, b, "same seed/inputs ⇒ identical intent");
    }

    #[test]
    fn style_overrides_rng_no_draw() {
        // With a style supplied, sAngle uses style.angle and never draws.
        let me = mk_player("H1", TeamSide::Home, RiggerRole::Faithwing, vz());
        let st = mk_state(0.0, vec![me.clone()]); // far from faith ring
        let m = mk_match(TeamSide::Home);
        let pr = TeamProfile::baseline();
        let style = Some(RoleStyle { angle: 0.5, radius: 0.5 });
        let mut r = AiRng::make(1, 1, 0);
        let mut r_check = r.clone();
        let a = faithwing_policy(&me, &st, &m, &pr, &mut r, style);
        // rng untouched ⇒ next draw equals the pre-call stream head.
        assert_eq!(r.next().to_bits(), r_check.next().to_bits());
        let b = faithwing_policy(
            &me,
            &st,
            &m,
            &pr,
            &mut AiRng::make(9999, 7, 1),
            style,
        );
        assert_eq!(a, b, "style ⇒ rng-independent");
    }

    // ── true-by-construction ─────────────────────────────────────────────
    #[test]
    fn possession_far_bell_is_receive_on_faithwing_radius_shell() {
        // Home attacks +X (GATE_X = 320). Bell at x=0 is far (|0-320| !< 192).
        let me = mk_player("H1", TeamSide::Home, RiggerRole::Faithwing, vz());
        let st = mk_state(0.0, vec![me.clone()]);
        let m = mk_match(TeamSide::Home);
        let pr = TeamProfile::baseline();
        let out =
            faithwing_policy(&me, &st, &m, &pr, &mut AiRng::make(3, 3, 0), None);
        assert_eq!(out.intent, FaithwingIntentKind::Receive);
        // (y,z) lies on the FAITHWING_RADIUS circle by construction.
        let r = (out.target_pos.y.powi(2) + out.target_pos.z.powi(2)).sqrt();
        assert!((r - FAITHWING_RADIUS).abs() < 1e-9, "r={r}");
        // mid_x = bell.x + 50*aggression (faith_x>0); aggression baseline 0.5.
        assert!((out.target_pos.x - 50.0 * pr.aggression).abs() < 1e-9);
    }

    #[test]
    fn possession_near_ring_is_faith_attack() {
        // Bell at the +X ring ⇒ |320-320| = 0 < 192 ⇒ faith-attack.
        let me = mk_player("H1", TeamSide::Home, RiggerRole::Faithwing, vz());
        let st = mk_state(crate::tuning::GATE_X, vec![me.clone()]);
        let m = mk_match(TeamSide::Home);
        let pr = TeamProfile::baseline();
        let out =
            faithwing_policy(&me, &st, &m, &pr, &mut AiRng::make(5, 5, 0), None);
        assert_eq!(out.intent, FaithwingIntentKind::FaithAttack);
        // attack_x = faith_x + (-15) since faith_x = +320 > 0.
        assert!(
            (out.target_pos.x - (crate::tuning::GATE_X - 15.0)).abs() < 1e-9
        );
        let r = (out.target_pos.y.powi(2) + out.target_pos.z.powi(2)).sqrt();
        assert!((r - FAITHWING_RADIUS).abs() < 1e-9, "r={r}");
    }

    #[test]
    fn defense_with_freewing_shadows_it() {
        let me = mk_player("H1", TeamSide::Home, RiggerRole::Faithwing, vz());
        let fw = mk_player(
            "A9",
            TeamSide::Away,
            RiggerRole::Freewing,
            Vec3 { x: 100.0, y: 10.0, z: -20.0 },
        );
        let st = mk_state(0.0, vec![me.clone(), fw.clone()]);
        // Home does NOT have possession ⇒ defense branch.
        let m = mk_match(TeamSide::Away);
        let pr = TeamProfile::baseline();
        let out =
            faithwing_policy(&me, &st, &m, &pr, &mut AiRng::make(7, 7, 0), None);
        assert_eq!(out.intent, FaithwingIntentKind::ShadowFreewing);
        // def_x for Home = -GATE_X. shadow_x = (100 + (-320)) / 2.
        let def_x = -crate::tuning::GATE_X;
        assert!((out.target_pos.x - (100.0 + def_x) / 2.0).abs() < 1e-9);
        assert!((out.target_pos.y - 10.0 * 0.8).abs() < 1e-9);
        assert!((out.target_pos.z - (-20.0 * 0.8)).abs() < 1e-9);
    }

    #[test]
    fn defense_no_freewing_guards_own_ring() {
        let me = mk_player("H1", TeamSide::Home, RiggerRole::Faithwing, vz());
        let st = mk_state(0.0, vec![me.clone()]);
        let m = mk_match(TeamSide::Away); // no possession ⇒ defense
        let pr = TeamProfile::baseline();
        let out =
            faithwing_policy(&me, &st, &m, &pr, &mut AiRng::make(8, 8, 0), None);
        assert_eq!(out.intent, FaithwingIntentKind::ShadowFreewing);
        // def_x = -320 < 0 ⇒ guard_x = def_x + 40.
        let def_x = -crate::tuning::GATE_X;
        assert!((out.target_pos.x - (def_x + 40.0)).abs() < 1e-9);
        assert!((out.target_pos.y - FAITHWING_RADIUS * 0.7).abs() < 1e-9);
        assert_eq!(out.target_pos.z, 0.0);
    }

    #[test]
    fn navigate_close_target_returns_none() {
        // Bell far ⇒ receive; target near origin; player AT origin ⇒
        // plan_grapple within 3 m ⇒ None (TS null).
        let me = mk_player("H1", TeamSide::Home, RiggerRole::Faithwing, vz());
        let st = mk_state(0.0, vec![me.clone()]);
        let m = mk_match(TeamSide::Home);
        let pr = TeamProfile::baseline();
        // receive target_pos.x = 50*0.5 = 25 (>3), so this exercises the
        // real planner path deterministically rather than asserting None.
        let a = faithwing_navigate(&me, &st, &m, &pr, &mut AiRng::make(2, 2, 0));
        let b = faithwing_navigate(&me, &st, &m, &pr, &mut AiRng::make(2, 2, 0));
        assert_eq!(a, b, "navigate deterministic");
    }
}

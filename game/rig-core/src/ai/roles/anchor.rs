//! 1:1 port of src/ai/roles/Anchor.ts — the Anchor role policy.
//!
//! The Anchor plays deep/heavy, near the rim end of the gradient. It almost
//! never floats high; when caught axis-side the team is exposed.
//! Decision tick: 10 Hz (every 6 sim-ticks at 60 Hz).
//!
//! Responsibilities:
//!   - On offense: position deep to receive from Spinners; anchor the loop play.
//!   - On defense: guard Faith end, pressure throwers going deep.
//!   - Never voluntarily goes higher than ANCHOR_MAX_RADIUS from axis.
//!
//! Ported symbols (Anchor.ts spans):
//!   - constants ANCHOR_TARGET_RADIUS / ANCHOR_MAX_RADIUS  (Anchor.ts:18-19)
//!   - struct RoleIntent + enum Intent                     (Anchor.ts:21-24)
//!   - anchor_policy   (anchorPolicy)                       (Anchor.ts:38-76)
//!   - anchor_navigate (anchorNavigate)                     (Anchor.ts:81-98)
//!
//! `RoleStyle` is the frozen substrate type (ai::types::RoleStyle, the port
//! of Anchor.ts:29-32). The TS `rng: () => number` becomes `rng: &mut AiRng`
//! and `rng()` becomes `rng.next()`; the `sAngle` closure is reproduced with
//! the exact "only draw when style is None" semantics.

use crate::ai::plan_bridge::{plan_grapple, GrapplePlan};
use crate::ai::profile::TeamProfile;
use crate::ai::rng::AiRng;
use crate::ai::types::{MatchState, PlayerSim, RoleStyle, SimState};
use crate::math::Vec3;
use crate::tuning::R;

/// The Anchor wants to be at ~72% skin radius: deep but not skinned.
/// (Anchor.ts:18 — `REG.R * 0.72`.)
const ANCHOR_TARGET_RADIUS: f64 = R * 0.72;
/// Never voluntarily higher than 85% skin radius. (Anchor.ts:19.)
const ANCHOR_MAX_RADIUS: f64 = R * 0.85;

/// Anchor.ts:23 string-union `intent` → Rust enum (same variants).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    /// 'guard-faith'
    GuardFaith,
    /// 'deep-receive'
    DeepReceive,
    /// 'hold-depth'
    HoldDepth,
    /// 'support-throw'
    SupportThrow,
}

/// Anchor.ts:21-24 `interface RoleIntent`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoleIntent {
    pub target_pos: Vec3,
    pub intent: Intent,
}

/// 1:1 port of `anchorPolicy` (Anchor.ts:38-76).
///
/// Decision is pure — no side effects. Mirrors TS arg order; `rng` is a
/// seeded `AiRng` and `style` is `Option<RoleStyle>` (TS `style?: RoleStyle`).
/// `_profile` is unused exactly as in the TS (`_profile: TeamProfile`).
pub fn anchor_policy(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    _profile: &TeamProfile,
    rng: &mut AiRng,
    style: Option<RoleStyle>,
) -> RoleIntent {
    // Anchor.ts:46 `const sAngle = () => (style ? style.angle : rng());`
    // Draw from rng ONLY when style is absent — exact TS draw order/count.
    let mut s_angle = |rng: &mut AiRng| match style {
        Some(s) => s.angle,
        None => rng.next(),
    };

    // Anchor.ts:47-50
    let bell = &state.bell;
    let on_offense = match &bell.held_by {
        Some(held_by) => state
            .players
            .iter()
            .find(|p| &p.id == held_by)
            .map(|p| p.team == player.team)
            .unwrap_or(false),
        None => false,
    };
    let possession = m.possession == player.team;

    // ORIENTATION-CORRECT: position relative to the ring THIS team attacks /
    // defends (home → +X attack, away → -X attack), never faithEnd.
    // Anchor.ts:54-55
    let atk_x = crate::ai::orientation::attack_ring_x(player.team);
    let def_x = crate::ai::orientation::defend_ring_x(player.team);

    if possession || on_offense {
        // Offense: stay deep near OUR attacking ring to receive / contest
        // loops. Anchor.ts:58-66
        let deep_x = atk_x + if atk_x > 0.0 { -80.0 } else { 80.0 }; // 80 m short of the ring
        let angle = std::f64::consts::PI * (0.4 + s_angle(rng) * 0.2);
        let target_y = ANCHOR_TARGET_RADIUS * angle.cos();
        let target_z = ANCHOR_TARGET_RADIUS * angle.sin();
        RoleIntent {
            target_pos: Vec3 { x: deep_x, y: target_y, z: target_z },
            intent: Intent::DeepReceive,
        }
    } else {
        // Defense: guard OUR OWN ring. Anchor.ts:68-74
        let guard_x = def_x + if def_x > 0.0 { -25.0 } else { 25.0 };
        let guard_y = ANCHOR_TARGET_RADIUS * 0.6;
        RoleIntent {
            target_pos: Vec3 { x: guard_x, y: guard_y, z: 0.0 },
            intent: Intent::GuardFaith,
        }
    }
}

/// 1:1 port of `anchorNavigate` (Anchor.ts:81-98).
///
/// Computes a grapple navigation plan for the Anchor toward its role target.
/// Returns `None` (TS `null`) when the planner declines (already within 3 m).
pub fn anchor_navigate(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    profile: &TeamProfile,
    rng: &mut AiRng,
) -> Option<GrapplePlan> {
    // Anchor.ts:88 — anchorNavigate calls anchorPolicy with no style.
    let RoleIntent { target_pos, .. } = anchor_policy(player, state, m, profile, rng, None);

    // If already too deep (above radius limit), aim inward first.
    // Anchor.ts:91-95
    let r = (player.p.y * player.p.y + player.p.z * player.p.z).sqrt();
    if r > ANCHOR_MAX_RADIUS {
        let inward = Vec3 {
            x: player.p.x,
            y: player.p.y * 0.5,
            z: player.p.z * 0.5,
        };
        // TS planGrapple default avoidDefenders=true; no sticky.
        return plan_grapple(player, inward, state, true, None);
    }

    // Anchor.ts:97
    plan_grapple(player, target_pos, state, true, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{
        BellState, Cast, ContestDirection, FaithEnd, Gate, MatchPhase, MatchState, PlayerSim,
        RiggerRole, SimState, TeamSide,
    };
    use crate::math::Quat;

    fn vz() -> Vec3 {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    fn mk_player(id: &str, team: TeamSide, p: Vec3) -> PlayerSim {
        PlayerSim {
            id: id.to_string(),
            team,
            role: RiggerRole::Anchor,
            p,
            v: vz(),
            q: Quat::ident(),
            line: None,
            dv_budget: 0.0,
            contact_ref: None,
            grounded: false,
        }
    }

    fn mk_state(players: Vec<PlayerSim>, held_by: Option<String>) -> SimState {
        SimState {
            tick: 0.0,
            omega: crate::tuning::OMEGA,
            bell: BellState {
                p: vz(),
                v: vz(),
                q: Quat::ident(),
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

    // Determinism: same inputs + freshly-seeded AiRng → identical intent.
    #[test]
    fn policy_is_deterministic() {
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: 0.0, y: 10.0, z: 0.0 });
        let st = mk_state(vec![p.clone()], None);
        let m = mk_match(TeamSide::Home); // possession → offense branch (draws rng)
        let prof = TeamProfile::baseline();

        let mut r1 = AiRng::make(4242, 60, 0);
        let mut r2 = AiRng::make(4242, 60, 0);
        let a = anchor_policy(&p, &st, &m, &prof, &mut r1, None);
        let b = anchor_policy(&p, &st, &m, &prof, &mut r2, None);
        assert_eq!(a.intent, b.intent);
        assert_eq!(a.target_pos.x.to_bits(), b.target_pos.x.to_bits());
        assert_eq!(a.target_pos.y.to_bits(), b.target_pos.y.to_bits());
        assert_eq!(a.target_pos.z.to_bits(), b.target_pos.z.to_bits());
    }

    // Style supplied → no rng draw, angle is exactly style.angle (true by
    // construction: y=Rcos θ, z=Rsin θ with θ=π(0.4+style.angle·0.2)).
    #[test]
    fn style_overrides_rng_draw() {
        let p = mk_player("H1", TeamSide::Home, vz());
        let st = mk_state(vec![p.clone()], None);
        let m = mk_match(TeamSide::Home);
        let prof = TeamProfile::baseline();
        let style = RoleStyle { angle: 0.5, radius: 0.0 };

        // Different rng states must NOT affect the result when style is set.
        let mut r1 = AiRng::make(1, 1, 0);
        let mut r2 = AiRng::make(999, 777, 1);
        let a = anchor_policy(&p, &st, &m, &prof, &mut r1, Some(style));
        let b = anchor_policy(&p, &st, &m, &prof, &mut r2, Some(style));
        assert_eq!(a, b);

        let angle = std::f64::consts::PI * (0.4 + 0.5 * 0.2);
        assert!((a.target_pos.y - ANCHOR_TARGET_RADIUS * angle.cos()).abs() < 1e-12);
        assert!((a.target_pos.z - ANCHOR_TARGET_RADIUS * angle.sin()).abs() < 1e-12);
    }

    // Offense (possession): deep-receive, finite target, radius in (y,z)
    // plane is exactly ANCHOR_TARGET_RADIUS (true by construction).
    #[test]
    fn offense_branch_deep_receive_bounds() {
        let p = mk_player("H1", TeamSide::Home, vz());
        let st = mk_state(vec![p.clone()], None);
        let m = mk_match(TeamSide::Home);
        let prof = TeamProfile::baseline();
        let mut rng = AiRng::make(7, 7, 0);
        let ri = anchor_policy(&p, &st, &m, &prof, &mut rng, None);
        assert_eq!(ri.intent, Intent::DeepReceive);
        assert!(ri.target_pos.x.is_finite());
        assert!(ri.target_pos.y.is_finite());
        assert!(ri.target_pos.z.is_finite());
        // home attacks +X (=+320); deep_x = 320 - 80 = 240.
        assert!((ri.target_pos.x - (crate::tuning::GATE_X - 80.0)).abs() < 1e-9);
        let rad = (ri.target_pos.y * ri.target_pos.y + ri.target_pos.z * ri.target_pos.z).sqrt();
        assert!((rad - ANCHOR_TARGET_RADIUS).abs() < 1e-9);
        assert!(rad <= ANCHOR_MAX_RADIUS);
    }

    // Defense (no possession, no offense): guard-faith on our own ring.
    #[test]
    fn defense_branch_guard_faith() {
        let p = mk_player("H1", TeamSide::Home, vz());
        // bell held by an Away player → on_offense false; possession Away.
        let opp = mk_player("A1", TeamSide::Away, vz());
        let st = mk_state(vec![p.clone(), opp], Some("A1".to_string()));
        let m = mk_match(TeamSide::Away);
        let prof = TeamProfile::baseline();
        let mut rng = AiRng::make(3, 3, 0);
        let ri = anchor_policy(&p, &st, &m, &prof, &mut rng, None);
        assert_eq!(ri.intent, Intent::GuardFaith);
        // home defends -X (=-320); guard_x = -320 + 25 = -295.
        assert!((ri.target_pos.x - (-crate::tuning::GATE_X + 25.0)).abs() < 1e-9);
        assert!((ri.target_pos.y - ANCHOR_TARGET_RADIUS * 0.6).abs() < 1e-9);
        assert_eq!(ri.target_pos.z, 0.0);
    }

    // anchor_navigate: when too deep it aims inward; plan (if any) is finite.
    #[test]
    fn navigate_inward_when_too_deep() {
        let deep = ANCHOR_MAX_RADIUS + 5.0;
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: 0.0, y: deep, z: 0.0 });
        let st = mk_state(vec![p.clone()], None);
        let m = mk_match(TeamSide::Away); // defense branch, but inward override fires first
        let prof = TeamProfile::baseline();
        let mut rng = AiRng::make(11, 11, 0);
        let plan = anchor_navigate(&p, &st, &m, &prof, &mut rng);
        if let Some(pl) = plan {
            assert!(pl.projected_dist.is_finite());
            assert!(pl.reel == -1 || pl.reel == 0);
        }
    }

    #[test]
    fn navigate_is_deterministic() {
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: -100.0, y: 5.0, z: 0.0 });
        let st = mk_state(vec![p.clone()], None);
        let m = mk_match(TeamSide::Home);
        let prof = TeamProfile::baseline();
        let mut r1 = AiRng::make(55, 60, 0);
        let mut r2 = AiRng::make(55, 60, 0);
        let a = anchor_navigate(&p, &st, &m, &prof, &mut r1);
        let b = anchor_navigate(&p, &st, &m, &prof, &mut r2);
        assert_eq!(a, b);
        let _ = ContestDirection::Fair; // touch import
    }
}

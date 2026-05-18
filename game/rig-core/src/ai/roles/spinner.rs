//! 1:1 faithful port of src/ai/roles/Spinner.ts (FROZEN SPEC).
//!
//! Spinner role policy — midfield engine, trades altitude for tempo.
//! Spinners live in the slope, run the lines, and set loop plays.
//! Decision tick: 10 Hz.
//!
//! Responsibilities (Spinner.ts:1-9):
//!   - Maintain midfield position, bridging Anchor and Wings.
//!   - Set loop-setter position when loop play is called.
//!   - Pressure bell carriers on defense.
//!   - High grapple activity — Spinners swing constantly.
//!
//! Public surface ported:
//!   - `SpinnerIntent` (struct) + `SpinnerIntentKind` (the TS `intent`
//!     string-union: 'midfield' | 'loop-setter' | 'pressure' |
//!     'swing-line') + `is_loop_setter: bool` field.
//!   - `spinnerPolicy` → [`spinner_policy`].
//!   - `spinnerNavigate` → [`spinner_navigate`].
//!
//! NOTE: Spinner.ts carries an extra `isLoopSetter: boolean` param in
//! BOTH functions; the Rust arg order mirrors the TS exactly
//! (`..., profile, is_loop_setter, rng, style`).

use crate::ai::orientation::{attack_sign, defend_ring_x};
use crate::ai::plan_bridge::{plan_grapple, GrapplePlan};
use crate::ai::profile::TeamProfile;
use crate::ai::rng::AiRng;
use crate::ai::types::{MatchState, PlayerSim, RoleStyle, SimState};
use crate::math::Vec3;

// Spinner operates in the mid-radius band (Spinner.ts:18-20).
// REG.R == crate::tuning::R.
const SPINNER_TARGET_RADIUS: f64 = crate::tuning::R * 0.42;
const SPINNER_HIGH_RADIUS: f64 = crate::tuning::R * 0.20;

/// The TS `intent` string-union (Spinner.ts:24) as an enum.
/// `'swing-line'` is part of the FROZEN union surface even though the
/// current Spinner.ts body never emits it; ported verbatim for fidelity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpinnerIntentKind {
    /// `'midfield'`
    Midfield,
    /// `'loop-setter'`
    LoopSetter,
    /// `'pressure'`
    Pressure,
    /// `'swing-line'`
    SwingLine,
}

/// 1:1 with the TS `SpinnerIntent` interface (Spinner.ts:22-26).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinnerIntent {
    pub target_pos: Vec3,
    pub intent: SpinnerIntentKind,
    pub is_loop_setter: bool,
}

/// `() => (style ? style.angle : rng())` (Spinner.ts:55).
///
/// Language-forced divergence: TS builds an arrow closure that captures
/// the `rng` thunk; a Rust closure capturing `&mut AiRng` cannot be
/// reused alongside other inline `rng.next()` calls without aliasing the
/// mutable borrow. Hoisting to a free `fn` taking `&mut AiRng` is
/// behaviorally equivalent: TS `?:` short-circuits so `rng()` is
/// evaluated IFF `style` is absent — this `fn` calls `rng.next()` on
/// exactly the same condition and at exactly the same call site, so the
/// rng draw count/order is byte-identical.
#[inline]
fn s_angle(style: Option<RoleStyle>, rng: &mut AiRng) -> f64 {
    match style {
        Some(s) => s.angle,
        None => rng.next(),
    }
}

/// `() => (style ? style.radius : rng())` (Spinner.ts:56). See
/// [`s_angle`] for the closure→fn divergence rationale.
#[inline]
fn s_radius(style: Option<RoleStyle>, rng: &mut AiRng) -> f64 {
    match style {
        Some(s) => s.radius,
        None => rng.next(),
    }
}

/// Spinner decision policy (Spinner.ts:45-107).
/// Spinners alternate between midfield holding and aggressive
/// line-setting.
///
/// `style` mirrors TS `style?: RoleStyle` (Spinner.ts:52): when supplied
/// by RiggerAI's commitment cache the policy uses deterministic functions
/// of these instead of fresh inline `rng()` each Director window; when
/// absent the policy falls back to inline `rng()` so direct callers stay
/// byte-identical.
#[allow(clippy::too_many_arguments)]
pub fn spinner_policy(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    profile: &TeamProfile,
    is_loop_setter: bool,
    rng: &mut AiRng,
    style: Option<RoleStyle>,
) -> SpinnerIntent {
    // Stable draws when committed (C2); inline rng() fallback otherwise.
    // (sAngle/sRadius are the s_angle/s_radius fns — see their docs.)
    let bell = &state.bell;
    let possession = m.possession == player.team;
    // ORIENTATION-CORRECT: advance toward the ring THIS team attacks.
    let sgn = attack_sign(player.team); // +1 home, -1 away
    let our_def_x = defend_ring_x(player.team);

    if possession {
        if is_loop_setter && profile.loop_propensity > 0.3 {
            // Loop-setter: position near axis, ready to launch the loop.
            // Optimal position: mid-field, high (close to axis), with
            // angle. (Spinner.ts:64-75)
            let loop_x = 0.0 + (rng.next() - 0.5) * 60.0;
            let angle = std::f64::consts::PI * (0.1 + s_angle(style, rng) * 0.15);
            let target_y = SPINNER_HIGH_RADIUS * angle.cos();
            let target_z = SPINNER_HIGH_RADIUS * angle.sin();
            return SpinnerIntent {
                target_pos: Vec3 { x: loop_x, y: target_y, z: target_z },
                intent: SpinnerIntentKind::LoopSetter,
                is_loop_setter: true,
            };
        }

        // Normal offense: bridge AHEAD of self toward OUR attacking ring.
        // (Spinner.ts:78-86)
        let mid_x = player.p.x + sgn * 40.0 * (0.6 + profile.aggression);
        let angle = std::f64::consts::PI * (0.3 + s_angle(style, rng) * 0.25);
        let r = SPINNER_TARGET_RADIUS * (0.8 + s_radius(style, rng) * 0.4);
        SpinnerIntent {
            target_pos: Vec3 { x: mid_x, y: r * angle.cos(), z: r * angle.sin() },
            intent: SpinnerIntentKind::Midfield,
            is_loop_setter: false,
        }
    } else {
        // Defense: pressure the ball carrier from midfield.
        // (Spinner.ts:88-98)
        let bell_holder = match &bell.held_by {
            Some(held_by) => state.players.iter().find(|p| &p.id == held_by),
            None => None,
        };
        if let Some(bell_holder) = bell_holder {
            if bell_holder.team != player.team {
                // Pressure goal-side: get between the carrier and OUR
                // defended ring.
                let pressure_x = bell_holder.p.x + if our_def_x > 0.0 { 20.0 } else { -20.0 };
                return SpinnerIntent {
                    target_pos: Vec3 {
                        x: pressure_x,
                        y: bell_holder.p.y * 0.7,
                        z: bell_holder.p.z * 0.7,
                    },
                    intent: SpinnerIntentKind::Pressure,
                    is_loop_setter: false,
                };
            }
        }

        // Hold midfield position on defense. (Spinner.ts:101-105)
        SpinnerIntent {
            target_pos: Vec3 { x: 0.0, y: SPINNER_TARGET_RADIUS * 0.7, z: 0.0 },
            intent: SpinnerIntentKind::Midfield,
            is_loop_setter: false,
        }
    }
}

/// `spinnerNavigate` (Spinner.ts:109-119). Runs the policy then plans a
/// grapple to the resulting target. TS `planGrapple(player, targetPos,
/// state)` uses the GrapplePlanner defaults (avoidDefenders = true,
/// sticky absent) → `plan_grapple(player, target, state, true, None)`.
/// Returns `None` (TS `null`) when already within 3 m of the target.
///
/// `spinnerNavigate` has NO `style` param in the TS (Spinner.ts:109-116);
/// it forwards `None` so the inline-rng fallback path is taken, exactly
/// as the TS does (it passes no `style` arg).
pub fn spinner_navigate(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    profile: &TeamProfile,
    is_loop_setter: bool,
    rng: &mut AiRng,
) -> Option<GrapplePlan> {
    let SpinnerIntent { target_pos, .. } =
        spinner_policy(player, state, m, profile, is_loop_setter, rng, None);
    plan_grapple(player, target_pos, state, true, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{BellState, PlayerSim, RiggerRole, SimState, TeamSide};
    use crate::math::Quat;

    fn vz() -> Vec3 {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    fn mk_player(id: &str, team: TeamSide, p: Vec3) -> PlayerSim {
        PlayerSim {
            id: id.to_string(),
            team,
            role: RiggerRole::Spinner,
            p,
            v: vz(),
            q: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
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
        use crate::ai::types::{Cast, FaithEnd, Gate, MatchPhase};
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

    // Determinism: same (seed, tick, team) rng + same inputs → identical
    // intent, on the inline-rng fallback path (style = None).
    #[test]
    fn policy_deterministic_inline_rng() {
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: 10.0, y: 5.0, z: 0.0 });
        let st = mk_state(vec![p.clone()], None);
        let mt = mk_match(TeamSide::Home); // possession → offense branch
        let prof = TeamProfile::baseline();

        let mut r1 = AiRng::make(424242, 240, 0);
        let mut r2 = AiRng::make(424242, 240, 0);
        let a = spinner_policy(&p, &st, &mt, &prof, false, &mut r1, None);
        let b = spinner_policy(&p, &st, &mt, &prof, false, &mut r2, None);
        assert_eq!(a, b, "policy must be deterministic for equal rng state");
    }

    // true-by-construction: with possession + loop-setter + sufficient
    // loop_propensity, the intent IS LoopSetter and is_loop_setter=true,
    // and the target lies on the SPINNER_HIGH_RADIUS shell in (y,z).
    #[test]
    fn loop_setter_branch_shape() {
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: 0.0, y: 0.0, z: 0.0 });
        let st = mk_state(vec![p.clone()], None);
        let mt = mk_match(TeamSide::Home);
        let mut prof = TeamProfile::baseline();
        prof.loop_propensity = 0.9; // > 0.3 gate

        let mut rng = AiRng::make(7, 1, 0);
        let it = spinner_policy(&p, &st, &mt, &prof, true, &mut rng, None);
        assert_eq!(it.intent, SpinnerIntentKind::LoopSetter);
        assert!(it.is_loop_setter);
        let yz = (it.target_pos.y * it.target_pos.y + it.target_pos.z * it.target_pos.z).sqrt();
        assert!(
            (yz - SPINNER_HIGH_RADIUS).abs() < 1e-9,
            "loop-setter (y,z) must lie on the SPINNER_HIGH_RADIUS shell: {yz}"
        );
        // |loopX| <= 30 by construction: 0 + (rng-0.5)*60, rng in [0,1).
        assert!(it.target_pos.x.abs() <= 30.0);
    }

    // true-by-construction: style present ⇒ NO inline rng draw in the
    // angle/radius slots, so the offense target is a pure function of
    // (style, profile, player) and rng state is irrelevant there.
    #[test]
    fn styled_offense_independent_of_rng() {
        let p = mk_player("A1", TeamSide::Away, Vec3 { x: -5.0, y: 2.0, z: 1.0 });
        let st = mk_state(vec![p.clone()], None);
        let mt = mk_match(TeamSide::Away); // Away has possession → offense
        let prof = TeamProfile::baseline();
        let style = Some(RoleStyle { angle: 0.25, radius: 0.5 });

        let mut r1 = AiRng::make(1, 1, 0);
        let mut r2 = AiRng::make(999, 12345, 3);
        // is_loop_setter=false → skips the always-inline loopX rng draw.
        let a = spinner_policy(&p, &st, &mt, &prof, false, &mut r1, style);
        let b = spinner_policy(&p, &st, &mt, &prof, false, &mut r2, style);
        assert_eq!(a, b, "styled offense must not depend on rng state");
        assert_eq!(a.intent, SpinnerIntentKind::Midfield);
        assert!(!a.is_loop_setter);
    }

    // true-by-construction: defending team, opponent holds the bell ⇒
    // Pressure intent, target offset 20 from carrier toward our def ring.
    #[test]
    fn defense_pressure_branch() {
        let me = mk_player("H1", TeamSide::Home, Vec3 { x: 0.0, y: 0.0, z: 0.0 });
        let carrier = mk_player("A1", TeamSide::Away, Vec3 { x: 50.0, y: 10.0, z: 4.0 });
        let st = mk_state(vec![me.clone(), carrier.clone()], Some("A1".to_string()));
        let mt = mk_match(TeamSide::Away); // Home does NOT have possession
        let prof = TeamProfile::baseline();

        let mut rng = AiRng::make(3, 3, 0);
        let it = spinner_policy(&me, &st, &mt, &prof, false, &mut rng, None);
        assert_eq!(it.intent, SpinnerIntentKind::Pressure);
        assert!(!it.is_loop_setter);
        // Home defends -X (defend_ring_x < 0) → offset is -20.
        assert!((it.target_pos.x - (50.0 - 20.0)).abs() < 1e-9);
        assert!((it.target_pos.y - 10.0 * 0.7).abs() < 1e-9);
        assert!((it.target_pos.z - 4.0 * 0.7).abs() < 1e-9);
    }

    // true-by-construction: defending, no opposing carrier ⇒ hold
    // midfield at (0, SPINNER_TARGET_RADIUS*0.7, 0).
    #[test]
    fn defense_hold_midfield_branch() {
        let me = mk_player("H1", TeamSide::Home, Vec3 { x: 0.0, y: 0.0, z: 0.0 });
        let st = mk_state(vec![me.clone()], None);
        let mt = mk_match(TeamSide::Away);
        let prof = TeamProfile::baseline();

        let mut rng = AiRng::make(5, 5, 0);
        let it = spinner_policy(&me, &st, &mt, &prof, false, &mut rng, None);
        assert_eq!(it.intent, SpinnerIntentKind::Midfield);
        assert_eq!(
            it.target_pos,
            Vec3 { x: 0.0, y: SPINNER_TARGET_RADIUS * 0.7, z: 0.0 }
        );
    }

    // navigate: a far target yields a deterministic plan; an at-target
    // (within 3 m) yields None — exercises the spinner_navigate wiring.
    #[test]
    fn navigate_far_is_deterministic() {
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: -100.0, y: 5.0, z: 0.0 });
        let st = mk_state(vec![p.clone()], None);
        let mt = mk_match(TeamSide::Away); // defense, no carrier → hold mid
        let prof = TeamProfile::baseline();

        let mut r1 = AiRng::make(11, 11, 0);
        let mut r2 = AiRng::make(11, 11, 0);
        let a = spinner_navigate(&p, &st, &mt, &prof, false, &mut r1);
        let b = spinner_navigate(&p, &st, &mt, &prof, false, &mut r2);
        assert_eq!(a, b, "navigate must be deterministic");
    }
}

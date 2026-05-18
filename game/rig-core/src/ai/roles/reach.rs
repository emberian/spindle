//! 1:1 port of src/ai/roles/Reach.ts — the goal-ring keeper, mobile.
//!
//! The Reach defends the scoring arc and is the one player allowed to clip
//! the goal ring itself and pivot on it. Reaches are the sport's lunatics.
//! Decision tick: 5 Hz (every 12 sim-ticks at 60 Hz) — they hold long.
//!
//! Responsibilities:
//!   - Cover the ring mouth on defense.
//!   - On offense: position beyond the ring to contest incoming passes.
//!   - ONLY role allowed to fire their line at the goal ring.
//!
//! Ported symbols (Reach.ts spans):
//!   - constants REACH_ATTACK_RADIUS / KEEP_STANDOFF / KEEP_HORIZON /
//!     APERTURE / PRED_SKIN_RESTITUTION       (Reach.ts:21, 24-26, 35)
//!   - predict_skin_bounce (predictSkinBounce) (Reach.ts:36-49)
//!   - predict_ring_crossing (predictRingCrossing) + RingCrossing return
//!     type ({y,z,t} | null → Option<RingCrossing>)   (Reach.ts:51-68)
//!   - struct ReachIntent + enum Intent (`clip_ring: bool`) (Reach.ts:70-75)
//!   - reach_policy   (reachPolicy)                          (Reach.ts:85-165)
//!   - reach_navigate (reachNavigate)                        (Reach.ts:167-182)
//!
//! SACRED canon integrator: `predict_ring_crossing` predicts the bell's
//! flight by calling `crate::trajectory::{PointState, rk4_step}` directly —
//! the exact same RK4 the sim uses — so the prediction is byte-identical to
//! the bell's real arc. RK4 is NOT reimplemented here.
//!
//! `RoleStyle` is the frozen substrate type (ai::types::RoleStyle, the port
//! of Reach.ts:80-83). The TS `rng: () => number` becomes `rng: &mut AiRng`
//! and `rng()` becomes `rng.next()`; the `sAngle` closure is reproduced with
//! the exact "only draw when style is None" semantics. Reach.ts:174 calls
//! `reachPolicy` with NO style argument (→ `None`), so `reach_navigate`
//! forwards `None` and the inline rng() path is preserved byte-identically.

use crate::ai::plan_bridge::{plan_grapple, GrapplePlan};
use crate::ai::orientation::{attack_ring_x, defend_ring_x};
use crate::ai::profile::TeamProfile;
use crate::ai::rng::AiRng;
use crate::ai::types::{MatchState, PlayerSim, RoleStyle, SimState};
use crate::math::Vec3;
use crate::trajectory::{rk4_step, PointState};
use crate::tuning::{GATE_RADIUS, OMEGA, R};

/// Reach patrols at moderate radius, close to the goal ring.
/// (Reach.ts:21 — `REG.R * 0.35`.)
const REACH_ATTACK_RADIUS: f64 = R * 0.35;

/// m in front of the ring, field-side. (Reach.ts:24.)
const KEEP_STANDOFF: f64 = 14.0;
/// s — predict an incoming shot this far. (Reach.ts:25.)
const KEEP_HORIZON: f64 = 6.0;
/// 8 m: a score only counts inside this. (Reach.ts:26 — `REG.gateRadius`.)
const APERTURE: f64 = GATE_RADIUS;

/// Exact mirror of sim Collision.skinBounce (RESTITUTION 0.55).
/// (Reach.ts:35.)
const PRED_SKIN_RESTITUTION: f64 = 0.55;

/// Reach.ts:51-53 anonymous `{ y; z; t } | null` predictor return →
/// `Option<RingCrossing>` (the `null` case is the `None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingCrossing {
    pub y: f64,
    pub z: f64,
    pub t: f64,
}

/// Reach.ts:74 string-union `intent` → Rust enum (same four variants).
/// `'clear-zone'` is declared in the TS union but never produced by the
/// policy; it is ported for public-surface fidelity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    /// 'defend-mouth'
    DefendMouth,
    /// 'pivot-ring'
    PivotRing,
    /// 'contest-approach'
    ContestApproach,
    /// 'clear-zone'
    ClearZone,
}

/// Reach.ts:70-75 `interface ReachIntent`. `clip_ring` (TS `clipRing`) is
/// the unique Reach-only flag: when true the line is fired at the goal ring
/// itself — no other role sets this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReachIntent {
    pub target_pos: Vec3,
    /// Whether to fire line at the ring itself (only the Reach may do this).
    pub clip_ring: bool,
    pub intent: Intent,
}

/// 1:1 port of `predictSkinBounce` (Reach.ts:36-49). Exact mirror of sim
/// Collision.skinBounce (RESTITUTION 0.55) — deterministic, mutates in place.
fn predict_skin_bounce(st: &mut PointState) {
    // TS Math.hypot(y, z)
    let rho = (st.p.y * st.p.y + st.p.z * st.p.z).sqrt();
    if rho < R {
        return;
    }
    let ny = st.p.y / rho;
    let nz = st.p.z / rho;
    let vn = st.v.y * ny + st.v.z * nz;
    if vn > 0.0 {
        st.v.y -= (1.0 + PRED_SKIN_RESTITUTION) * vn * ny;
        st.v.z -= (1.0 + PRED_SKIN_RESTITUTION) * vn * nz;
    }
    let s = (R - 1e-3) / rho;
    st.p.y *= s;
    st.p.z *= s;
}

/// 1:1 port of `predictRingCrossing` (Reach.ts:51-68).
///
/// Axial x is inertial (accel.x = 0, canon) so the crossing time `t` is
/// exact; (y, z) are integrated with the SACRED canon `rk4_step` — the same
/// predictor the bell uses — and the skin wall is modelled each sub-step so
/// a skin-bouncing bell's ring crossing is predicted where it really
/// arrives, not through the wall. Returns `None` if the bell isn't inbound
/// in time (TS `null`).
fn predict_ring_crossing(bp: Vec3, bv: Vec3, defend_x: f64) -> Option<RingCrossing> {
    if bv.x.abs() < 1e-4 {
        return None;
    }
    let t = (defend_x - bp.x) / bv.x;
    if t <= 0.0 || t > KEEP_HORIZON {
        return None;
    }
    // TS `{ p: { ...bp }, v: { ...bv } }` — Vec3 is Copy so this clones.
    let mut st = PointState { p: bp, v: bv };
    let h_full: f64 = 1.0 / 60.0;
    let mut acc: f64 = 0.0;
    while acc < t {
        let h = h_full.min(t - acc);
        st = rk4_step(st, OMEGA, h);
        predict_skin_bounce(&mut st);
        acc += h;
    }
    Some(RingCrossing { y: st.p.y, z: st.p.z, t })
}

/// 1:1 port of `reachPolicy` (Reach.ts:85-165).
///
/// Decision is pure — no side effects. Mirrors TS arg order; `rng` is a
/// seeded `AiRng` and `style` is `Option<RoleStyle>` (TS `style?: RoleStyle`).
/// `_profile` is unused exactly as in the TS (`_profile: TeamProfile`).
pub fn reach_policy(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    _profile: &TeamProfile,
    rng: &mut AiRng,
    style: Option<RoleStyle>,
) -> ReachIntent {
    // Reach.ts:93 `const sAngle = () => (style ? style.angle : rng());`
    // Draw from rng ONLY when style is absent — exact TS draw order/count.
    let mut s_angle = |rng: &mut AiRng| match style {
        Some(s) => s.angle,
        None => rng.next(),
    };

    // Reach.ts:94-98
    let bell = &state.bell;
    let possession = m.possession == player.team;
    // ORIENTATION-CORRECT: defend OUR ring, contest near the ring WE attack.
    let our_attack_x = attack_ring_x(player.team);
    let our_defend_x = defend_ring_x(player.team);

    if !possession {
        // Defense: man the scoring lane. Reach.ts:100-150
        let defend_x = our_defend_x;
        // Reach.ts:107 `const fieldSign = defendX > 0 ? -1 : 1;`
        let field_sign = if defend_x > 0.0 { -1.0 } else { 1.0 };
        let cross = predict_ring_crossing(bell.p, bell.v, defend_x);

        if let Some(cross) = cross {
            // Reach.ts:111 Math.hypot(cross.y, cross.z)
            let r = (cross.y * cross.y + cross.z * cross.z).sqrt();
            // Reach.ts:115-118
            let threat = r < APERTURE * 2.2;
            let k = if r > 1e-3 {
                (1.0_f64).min((APERTURE * 0.85) / r)
            } else {
                0.0
            };
            let gy = if threat { cross.y * k } else { 0.0 };
            let gz = if threat { cross.z * k } else { 0.0 };

            if cross.t < 0.9 {
                // Imminent: pivot ON the ring at the crossing angle.
                // Reach.ts:120-131
                let a = cross.z.atan2(cross.y);
                return ReachIntent {
                    target_pos: Vec3 {
                        x: defend_x,
                        y: APERTURE * a.cos(),
                        z: APERTURE * a.sin(),
                    },
                    clip_ring: true,
                    intent: Intent::PivotRing,
                };
            }
            // Reach.ts:133-137
            return ReachIntent {
                target_pos: Vec3 {
                    x: defend_x + field_sign * KEEP_STANDOFF,
                    y: gy,
                    z: gz,
                },
                clip_ring: false,
                intent: Intent::ContestApproach,
            };
        }

        // No inbound shot: hold a centred ready post in the mouth.
        // Reach.ts:142-150
        ReachIntent {
            target_pos: Vec3 {
                x: defend_x + field_sign * KEEP_STANDOFF,
                y: APERTURE * 0.4 * (std::f64::consts::PI * s_angle(rng)).cos(),
                z: APERTURE * 0.4 * (std::f64::consts::PI * s_angle(rng)).sin(),
            },
            clip_ring: false,
            intent: Intent::DefendMouth,
        }
    } else {
        // Offense: position near the ring WE attack. Reach.ts:151-164
        let attack_x = our_attack_x + if our_attack_x > 0.0 { -20.0 } else { 20.0 };
        let angle = std::f64::consts::PI * (0.2 + s_angle(rng) * 0.2);
        ReachIntent {
            target_pos: Vec3 {
                x: attack_x,
                y: REACH_ATTACK_RADIUS * angle.cos(),
                z: REACH_ATTACK_RADIUS * angle.sin(),
            },
            clip_ring: false,
            intent: Intent::ContestApproach,
        }
    }
}

/// 1:1 port of `reachNavigate` (Reach.ts:167-182).
///
/// Computes a grapple navigation plan for the Reach toward its role target.
/// Returns `None` (TS `null`) when the planner declines (already within 3 m).
///
/// Reach.ts:176-181: both the `clipRing` and the non-`clipRing` branches
/// call `planGrapple(player, targetPos, state)` identically — the goal-ring
/// anchor is handled by the planner via the target position. The branch is
/// preserved verbatim (a single call site after the dead split) so behaviour
/// is byte-identical. TS `planGrapple` defaults: `avoidDefenders = true`,
/// no `sticky`.
pub fn reach_navigate(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    profile: &TeamProfile,
    rng: &mut AiRng,
) -> Option<GrapplePlan> {
    // Reach.ts:174 — reachNavigate calls reachPolicy with NO style → None.
    let ReachIntent { target_pos, clip_ring, .. } =
        reach_policy(player, state, m, profile, rng, None);

    if clip_ring {
        // Anchor directly to the goal ring — plan_grapple handles this via
        // the target. (Reach.ts:176-179.)
        return plan_grapple(player, target_pos, state, true, None);
    }

    // Reach.ts:181
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
            role: RiggerRole::Reach,
            p,
            v: vz(),
            q: Quat::ident(),
            line: None,
            dv_budget: 0.0,
            contact_ref: None,
            grounded: false,
        }
    }

    fn mk_state(
        players: Vec<PlayerSim>,
        bell_p: Vec3,
        bell_v: Vec3,
    ) -> SimState {
        SimState {
            tick: 0.0,
            omega: crate::tuning::OMEGA,
            bell: BellState {
                p: bell_p,
                v: bell_v,
                q: Quat::ident(),
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

    // Determinism: same inputs + freshly-seeded AiRng → identical intent.
    // Uses the defense/no-cross branch which draws rng twice via sAngle.
    #[test]
    fn policy_is_deterministic() {
        let p = mk_player("H1", TeamSide::Home, vz());
        // bell not inbound (vx≈0) → predict_ring_crossing None → defend-mouth.
        let st = mk_state(vec![p.clone()], Vec3 { x: 0.0, y: 0.0, z: 0.0 }, vz());
        let m = mk_match(TeamSide::Away); // not possession → defense branch
        let prof = TeamProfile::baseline();

        let mut r1 = AiRng::make(4242, 60, 0);
        let mut r2 = AiRng::make(4242, 60, 0);
        let a = reach_policy(&p, &st, &m, &prof, &mut r1, None);
        let b = reach_policy(&p, &st, &m, &prof, &mut r2, None);
        assert_eq!(a.intent, b.intent);
        assert_eq!(a.clip_ring, b.clip_ring);
        assert_eq!(a.target_pos.x.to_bits(), b.target_pos.x.to_bits());
        assert_eq!(a.target_pos.y.to_bits(), b.target_pos.y.to_bits());
        assert_eq!(a.target_pos.z.to_bits(), b.target_pos.z.to_bits());
        assert_eq!(a.intent, Intent::DefendMouth);
    }

    // Style supplied → no rng draw; result independent of rng state.
    #[test]
    fn style_overrides_rng_draw() {
        let p = mk_player("H1", TeamSide::Home, vz());
        let st = mk_state(vec![p.clone()], vz(), vz());
        let m = mk_match(TeamSide::Home); // possession → offense branch
        let prof = TeamProfile::baseline();
        let style = RoleStyle { angle: 0.5, radius: 0.0 };

        let mut r1 = AiRng::make(1, 1, 0);
        let mut r2 = AiRng::make(999, 777, 1);
        let a = reach_policy(&p, &st, &m, &prof, &mut r1, Some(style));
        let b = reach_policy(&p, &st, &m, &prof, &mut r2, Some(style));
        assert_eq!(a, b);

        // True by construction: y=Rcos θ, z=Rsin θ, θ=π(0.2+style.angle·0.2).
        let angle = std::f64::consts::PI * (0.2 + 0.5 * 0.2);
        assert!((a.target_pos.y - REACH_ATTACK_RADIUS * angle.cos()).abs() < 1e-12);
        assert!((a.target_pos.z - REACH_ATTACK_RADIUS * angle.sin()).abs() < 1e-12);
        assert_eq!(a.intent, Intent::ContestApproach);
        assert!(!a.clip_ring);
    }

    // Offense branch: contest-approach, finite target, never clips the ring.
    #[test]
    fn offense_branch_contest_approach() {
        let p = mk_player("H1", TeamSide::Home, vz());
        let st = mk_state(vec![p.clone()], vz(), vz());
        let m = mk_match(TeamSide::Home);
        let prof = TeamProfile::baseline();
        let mut rng = AiRng::make(7, 7, 0);
        let ri = reach_policy(&p, &st, &m, &prof, &mut rng, None);
        assert_eq!(ri.intent, Intent::ContestApproach);
        assert!(!ri.clip_ring);
        assert!(ri.target_pos.x.is_finite());
        assert!(ri.target_pos.y.is_finite());
        assert!(ri.target_pos.z.is_finite());
        // home attacks +X (=+320); attack_x = 320 - 20 = 300.
        assert!((ri.target_pos.x - (crate::tuning::GATE_X - 20.0)).abs() < 1e-9);
    }

    // predict_ring_crossing IS the SACRED canon RK4: stepping the same
    // PointState through crate::trajectory::rk4_step at the predictor's own
    // sub-step schedule (capped at the crossing time, no skin contact here)
    // reproduces the crossing y/z exactly — true by construction, the
    // predictor calls these very fns.
    #[test]
    fn prediction_uses_crate_trajectory_rk4() {
        // Inbound toward the -X ring: x = -100 moving at vx = -90 reaches
        // x = -320 in t = 220/90 ≈ 2.44 s (< KEEP_HORIZON = 6 s).
        let bp = Vec3 { x: -100.0, y: 3.0, z: -1.0 };
        let bv = Vec3 { x: -90.0, y: 2.0, z: -1.5 };
        let defend_x = -crate::tuning::GATE_X; // home defends -X (= -320)
        let cross = predict_ring_crossing(bp, bv, defend_x)
            .expect("inbound within KEEP_HORIZON");
        assert!(cross.y.is_finite() && cross.z.is_finite() && cross.t.is_finite());

        // Re-run the identical integration via crate::trajectory directly.
        let t = (defend_x - bp.x) / bv.x;
        let mut st = PointState { p: bp, v: bv };
        let h_full: f64 = 1.0 / 60.0;
        let mut acc: f64 = 0.0;
        while acc < t {
            let h = h_full.min(t - acc);
            st = rk4_step(st, crate::tuning::OMEGA, h);
            // rho stays < R for this near-axis arc → skin bounce is a no-op,
            // so the bare canon integrator must match the predictor bitwise.
            assert!((st.p.y * st.p.y + st.p.z * st.p.z).sqrt() < crate::tuning::R);
            acc += h;
        }
        assert_eq!(cross.y.to_bits(), st.p.y.to_bits());
        assert_eq!(cross.z.to_bits(), st.p.z.to_bits());
        assert_eq!(cross.t.to_bits(), t.to_bits());
    }

    // No inbound (|vx| < 1e-4) → predictor returns None (TS null).
    #[test]
    fn predict_none_when_not_inbound() {
        let bp = Vec3 { x: 0.0, y: 1.0, z: 1.0 };
        let bv = Vec3 { x: 0.0, y: 5.0, z: 0.0 };
        assert!(predict_ring_crossing(bp, bv, 320.0).is_none());
        // Inbound but beyond KEEP_HORIZON → also None.
        let slow = Vec3 { x: 1.0, y: 0.0, z: 0.0 }; // t = 320/1 = 320 s > 6
        assert!(predict_ring_crossing(Vec3 { x: 0.0, y: 0.0, z: 0.0 }, slow, 320.0).is_none());
    }

    // clip_ring is set ONLY in the imminent pivot-ring branch (cross.t<0.9).
    // Construct a fast, near-axis inbound shot arriving well under 0.9 s.
    #[test]
    fn clip_ring_only_in_pivot_ring_branch() {
        // Home defends -X (= -320). Place the bell just short of the ring,
        // moving fast toward it so t = (defend_x - bp.x)/bv.x < 0.9 s.
        let defend_x = -crate::tuning::GATE_X;
        let bp = Vec3 { x: defend_x + 40.0, y: 1.5, z: 0.5 };
        let bv = Vec3 { x: -100.0, y: 0.0, z: 0.0 }; // t = 40/100 = 0.4 s
        let p = mk_player("H1", TeamSide::Home, vz());
        let st = mk_state(vec![p.clone()], bp, bv);
        let m = mk_match(TeamSide::Away); // defense branch
        let prof = TeamProfile::baseline();
        let mut rng = AiRng::make(9, 9, 0);
        let ri = reach_policy(&p, &st, &m, &prof, &mut rng, None);
        assert_eq!(ri.intent, Intent::PivotRing);
        assert!(ri.clip_ring, "imminent crossing must set clip_ring");
        // On the ring plane, on the aperture circle (true by construction).
        assert!((ri.target_pos.x - defend_x).abs() < 1e-9);
        let rad = (ri.target_pos.y * ri.target_pos.y
            + ri.target_pos.z * ri.target_pos.z)
            .sqrt();
        assert!((rad - APERTURE).abs() < 1e-9);

        // And the non-imminent contest path does NOT clip the ring: same
        // geometry but far enough that t ≥ 0.9 s.
        let bp2 = Vec3 { x: defend_x + 300.0, y: 1.0, z: 0.0 };
        let bv2 = Vec3 { x: -100.0, y: 0.0, z: 0.0 }; // t = 3.0 s
        let st2 = mk_state(vec![p.clone()], bp2, bv2);
        let ri2 = reach_policy(&p, &st2, &m, &prof, &mut rng, None);
        assert_eq!(ri2.intent, Intent::ContestApproach);
        assert!(!ri2.clip_ring);
    }

    #[test]
    fn navigate_is_deterministic() {
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: -100.0, y: 5.0, z: 0.0 });
        let st = mk_state(vec![p.clone()], vz(), vz());
        let m = mk_match(TeamSide::Home);
        let prof = TeamProfile::baseline();
        let mut r1 = AiRng::make(55, 60, 0);
        let mut r2 = AiRng::make(55, 60, 0);
        let a = reach_navigate(&p, &st, &m, &prof, &mut r1);
        let b = reach_navigate(&p, &st, &m, &prof, &mut r2);
        assert_eq!(a, b);
        if let Some(pl) = a {
            assert!(pl.projected_dist.is_finite());
            assert!(pl.reel == -1 || pl.reel == 0);
        }
        let _ = ContestDirection::Fair; // touch import
        let _ = Intent::ClearZone; // touch the public-surface-only variant
    }
}

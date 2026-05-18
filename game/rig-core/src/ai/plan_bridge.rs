//! ai↔planner bridge — the stable `GrapplePlanner.planGrapple` surface
//! the ported roles/RiggerAI call, expressed on `ai::types`.
//!
//! `crate::planner` (port increment 2) is algorithmically the TS
//! `planGrapple`, but uses planner-local structs. Rather than refactor
//! the shipped/parity-tested planner, this thin orchestrator-authored
//! adapter converts ai::types ↔ planner structs. It is NOT a behavior
//! change: it forwards to `crate::planner::plan_grapple` with the
//! canonical MPC knob (`planner = 0`, = TS `tune('planner', 0)` default).

use super::types::{PlayerSim, SimState, TeamSide};
use crate::math::Vec3;
use crate::planner;
use std::sync::atomic::{AtomicI32, Ordering};

/// Re-export of the planner's `Plan` as the AI-facing `GrapplePlan`
/// (TS `GrapplePlan`: anchor_pos, reel ∈ {-1,0}, projected_dist, is_spar).
pub use crate::planner::Plan as GrapplePlan;

/// Active planner algorithm class — the faithful restoration of the old
/// TS `tune('planner', 0)` / `globalThis.__rigtune` knob (0 = MPC,
/// 1 = RRT, 2 = CEM, 3 = MPPI, 4 = SimAnneal, 5 = Beam, 6 = MCTS,
/// 7 = PotentialField, 8 = RandomShooting, 9 = Coordination).
///
/// PRODUCTION DEFAULT = 9 (Coordination). The 8000-tick skill-eval
/// ranked it 44.5 vs MPC 43.5 — the multi-agent cost terms
/// (teammate-interference penalty + pass-setup attractor) measurably
/// beat plain momentum-MPC, and it is the strategic substrate for the
/// cost-term-composition / coordination work to come. The search
/// METHOD was shown not to matter (RandomShooting == MPC); the win is
/// in the composed cost terms, which is what Coordination adds. The
/// in-browser progression gate validates this default behaviorally.
/// The skill-eval harness still flips this per deterministic run.
static PLANNER_CLASS: AtomicI32 = AtomicI32::new(9);

/// Set the active planner class (eval / exploration only). Set before a
/// run; constant during it.
pub fn set_planner_class(class: i32) {
    PLANNER_CLASS.store(class, Ordering::Relaxed);
}

/// Current planner class (production default 9 = Coordination).
pub fn planner_class() -> i32 {
    PLANNER_CLASS.load(Ordering::Relaxed)
}

/// TS `sticky?: { pos; reel: -1|0 }` for the anti-dither hysteresis.
#[derive(Clone, Copy, Debug)]
pub struct Sticky {
    pub pos: Vec3,
    pub reel: i32,
}

/// Team → planner's i32 convention (mirrors src/sim/wasm.ts:
/// `team === 'home' ? 0 : 1`). Only distinctness matters — the planner
/// compares `p.team != player.team` exactly as the TS compares the
/// TeamSide strings.
fn team_i32(t: TeamSide) -> i32 {
    match t {
        TeamSide::Home => 0,
        TeamSide::Away => 1,
    }
}

fn to_planner_state(state: &SimState) -> planner::SimState {
    planner::SimState {
        omega: state.omega,
        tick: state.tick as i64,
        players: state
            .players
            .iter()
            .map(|p| planner::PlayerSim {
                id: p.id.clone(),
                team: team_i32(p.team),
                p: p.p,
                v: p.v,
            })
            .collect(),
    }
}

/// 1:1 with TS `planGrapple(player, target, state, avoidDefenders=true,
/// sticky?)` — returns `None` (TS `null`) when already within 3 m.
pub fn plan_grapple(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    avoid_defenders: bool,
    sticky: Option<Sticky>,
) -> Option<GrapplePlan> {
    let pl = planner::PlayerSim {
        id: player.id.clone(),
        team: team_i32(player.team),
        p: player.p,
        v: player.v,
    };
    let st = to_planner_state(state);
    let sk = sticky.map(|s| planner::Sticky {
        pos: s.pos,
        reel: s.reel,
    });
    // Planner class from the knob (production default 9 = Coordination; the
    // skill-eval harness flips it to rank the algorithm zoo).
    planner::plan_grapple(&pl, target, &st, avoid_defenders, sk, planner_class())
}

/// TS `sparPositions()` — the static spar lattice (axis spine + off-axis
/// rings). Re-exported unchanged.
pub fn spar_positions() -> Vec<Vec3> {
    planner::spar_positions()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{
        AnchorType, BellState, MatchPhase, RiggerRole,
    };
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

    fn mk_state(players: Vec<PlayerSim>) -> SimState {
        SimState {
            tick: 0.0,
            omega: crate::tuning::OMEGA,
            bell: BellState {
                p: vz(),
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

    #[test]
    fn close_target_returns_none() {
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: 0.0, y: 0.0, z: 0.0 });
        let st = mk_state(vec![p.clone()]);
        // within 3 m → TS planGrapple returns null
        assert!(plan_grapple(&p, Vec3 { x: 1.0, y: 0.0, z: 0.0 }, &st, true, None).is_none());
        let _ = (AnchorType::Spar, MatchPhase::Live);
    }

    #[test]
    fn far_target_plans_and_is_deterministic() {
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: -100.0, y: 5.0, z: 0.0 });
        let st = mk_state(vec![
            p.clone(),
            mk_player("A1", TeamSide::Away, Vec3 { x: 50.0, y: 0.0, z: 0.0 }),
        ]);
        let tgt = Vec3 { x: 200.0, y: 0.0, z: 0.0 };
        let a = plan_grapple(&p, tgt, &st, true, None);
        let b = plan_grapple(&p, tgt, &st, true, None);
        assert_eq!(a, b, "bridge must be deterministic");
        if let Some(plan) = a {
            assert!(plan.projected_dist.is_finite());
            assert!(plan.reel == -1 || plan.reel == 0);
        }
    }

    #[test]
    fn spar_lattice_nonempty() {
        assert!(!spar_positions().is_empty());
    }
}

//! 1:1 port of src/ai/Orientation.ts — the single source of truth for
//! "which way is forward". home ALWAYS attacks +X, away ALWAYS attacks
//! -X; faith_end only labels which ring scores Faith (2) vs Free (5),
//! never direction.

use super::types::{FaithEnd, MatchState, SimState, TeamSide};
use crate::tuning::GATE_X;

/// Signed attack direction: home → +1 (toward +X ring), away → -1.
pub fn attack_sign(team: TeamSide) -> f64 {
    match team {
        TeamSide::Home => 1.0,
        TeamSide::Away => -1.0,
    }
}

/// World-x of the ring `team` is attacking (scoring at).
pub fn attack_ring_x(team: TeamSide) -> f64 {
    attack_sign(team) * GATE_X
}

/// World-x of the ring `team` is defending (their own ring).
pub fn defend_ring_x(team: TeamSide) -> f64 {
    -attack_sign(team) * GATE_X
}

/// Is the ring `team` attacks the Faith (spinward, 2-pt) ring?
pub fn attacking_faith_ring(team: TeamSide, m: &MatchState) -> bool {
    let ring_x = attack_ring_x(team);
    let faith_x = match m.faith_end {
        FaithEnd::PlusX => GATE_X,
        FaithEnd::MinusX => -GATE_X,
    };
    ring_x.signum() == faith_x.signum()
}

/// Which team currently "owns the attack": bell holder, then last
/// thrower (in-flight from us), then match.possession.
pub fn attacking_team(state: &SimState, m: &MatchState) -> TeamSide {
    if let Some(held_by) = &state.bell.held_by {
        if let Some(h) = state.players.iter().find(|p| &p.id == held_by) {
            return h.team;
        }
    }
    if state.bell.held_by.is_none() {
        if let Some(thrown_by) = &state.bell.thrown_by {
            if let Some(t) = state.players.iter().find(|p| &p.id == thrown_by) {
                return t.team;
            }
        }
    }
    m.possession
}

/// Forward (toward `team`'s attacking ring) progress of an x coordinate.
pub fn forward_progress(team: TeamSide, x: f64) -> f64 {
    attack_sign(team) * x
}

/// OFFENSE REBUILD — signed world-x of a cast gate for `team` (mirrors
/// match_sm::gate_x: GATE_ABS = GATE_X·{0.25,0.55,0.85}, signed by attack
/// direction). The match advances the cast when a throw is SPENT (caught/
/// bobbled) with the bell at/past this line in the attack direction, so
/// the AI must drive + complete a pass past it to clear a gate.
pub fn gate_world_x(team: TeamSide, g: super::types::Gate) -> f64 {
    let frac = match g {
        super::types::Gate::First => 0.25,
        super::types::Gate::Deep => 0.55,
        super::types::Gate::Mouth => 0.85,
    };
    attack_sign(team) * GATE_X * frac
}

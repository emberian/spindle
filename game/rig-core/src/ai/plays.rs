//! Coordinated play library — interlocking position targets that create
//! passing opportunities by construction. The Director selects a PlayKind
//! based on game state, assigns PlayRoles to players, and each tick
//! `resolve_targets` produces concrete world-space positions the assigned
//! players navigate toward. Pure/deterministic — no rng.

use super::orientation::{attack_ring_x, attack_sign, defend_ring_x, forward_progress, gate_world_x};
use super::types::{MatchState, PlayerSim, RiggerRole, SimState, TeamSide};
use crate::math::Vec3;

// ─── Play geometry constants ─────────────────────────────────────────────────

/// Throw range: carrier wants to be within this distance of the gate to throw.
const THROW_RANGE: f64 = 60.0;
/// On-axis threshold: a receiver on-axis means rho < this.
const ON_AXIS_RHO: f64 = 8.0;
/// Medium radius for staggered receivers.
const MED_RADIUS: f64 = 15.0;
/// Short outlet offset (ahead of carrier).
const OUTLET_AHEAD: f64 = 20.0;
/// Deep cut fraction (toward attack ring from carrier).
const DEEP_CUT_FRAC: f64 = 0.70;
/// Spread stagger depth intervals.
const SPREAD_DEPTH_1: f64 = 40.0;
const SPREAD_DEPTH_2: f64 = 80.0;
/// Pressure detection radius (defender closer than this = pressure).
const PRESSURE_RADIUS: f64 = 12.0;
/// Short dump cut distance (toward carrier).
const DUMP_CUT_DIST: f64 = 15.0;

// ─── Core types ──────────────────────────────────────────────────────────────

/// A coordinated play pattern — interlocking targets for multiple players
/// that create passing opportunities by construction.
pub struct Play {
    pub kind: PlayKind,
    /// Role assignments within this play. Each entry maps a PlayRole to
    /// a target generation function (parameterized by game state).
    pub roles: &'static [PlayRoleSpec],
    /// Minimum ticks the play stays active before the Director can switch.
    pub min_duration_ticks: f64,
    /// Play is "stale" after this many ticks — Director should re-evaluate.
    pub max_duration_ticks: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayKind {
    /// Carrier advances to throw range, gate receiver cuts to axis past gate.
    GateRun,
    /// Two receivers: one cuts deep, one holds short. Carrier reads pressure.
    OutletChain,
    /// After turnover: defined recovery stations that create a funnel.
    RecoveryFormation,
    /// Ball loose: one dives, one shadows, others pre-position for outlet.
    LooseBallRecovery,
    /// Under heavy pressure: quick dump to nearest open teammate.
    PressureRelease,
    /// Standard possession: spread formation, carrier advances, receivers space.
    SpreadAdvance,
}

pub struct PlayRoleSpec {
    pub role: PlayRole,
    /// How to select which player fills this role.
    pub selection: RoleSelection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayRole {
    /// The ball carrier — drives the play.
    Carrier,
    /// Primary receiver: the intended pass target.
    PrimaryReceiver,
    /// Secondary receiver: the safety valve / outlet.
    SecondaryReceiver,
    /// Blocker/screen: positions to obstruct defenders from the pass lane.
    Screen,
    /// Deep safety: holds position for long option or recovery.
    DeepOption,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoleSelection {
    /// Whoever currently holds the ball.
    BallCarrier,
    /// The player with best reception angle from the carrier.
    BestReception,
    /// The player closest to the target gate.
    NearestGate,
    /// The player furthest forward.
    DeepestForward,
    /// The player nearest the bell (for recovery).
    NearestBell,
    /// A specific RiggerRole.
    ByRole(RiggerRole),
}

// ─── Play library (static definitions) ───────────────────────────────────────

pub static GATE_RUN: Play = Play {
    kind: PlayKind::GateRun,
    roles: &[
        PlayRoleSpec { role: PlayRole::Carrier, selection: RoleSelection::BallCarrier },
        PlayRoleSpec { role: PlayRole::PrimaryReceiver, selection: RoleSelection::NearestGate },
        PlayRoleSpec { role: PlayRole::SecondaryReceiver, selection: RoleSelection::BestReception },
    ],
    min_duration_ticks: 60.0,
    max_duration_ticks: 300.0,
};

pub static OUTLET_CHAIN: Play = Play {
    kind: PlayKind::OutletChain,
    roles: &[
        PlayRoleSpec { role: PlayRole::Carrier, selection: RoleSelection::BallCarrier },
        PlayRoleSpec { role: PlayRole::PrimaryReceiver, selection: RoleSelection::DeepestForward },
        PlayRoleSpec { role: PlayRole::SecondaryReceiver, selection: RoleSelection::BestReception },
    ],
    min_duration_ticks: 45.0,
    max_duration_ticks: 240.0,
};

pub static RECOVERY_FORMATION: Play = Play {
    kind: PlayKind::RecoveryFormation,
    roles: &[
        PlayRoleSpec { role: PlayRole::PrimaryReceiver, selection: RoleSelection::NearestBell },
        PlayRoleSpec { role: PlayRole::SecondaryReceiver, selection: RoleSelection::BestReception },
        PlayRoleSpec { role: PlayRole::DeepOption, selection: RoleSelection::DeepestForward },
    ],
    min_duration_ticks: 30.0,
    max_duration_ticks: 180.0,
};

pub static LOOSE_BALL_RECOVERY: Play = Play {
    kind: PlayKind::LooseBallRecovery,
    roles: &[
        PlayRoleSpec { role: PlayRole::PrimaryReceiver, selection: RoleSelection::NearestBell },
        PlayRoleSpec { role: PlayRole::Screen, selection: RoleSelection::BestReception },
        PlayRoleSpec { role: PlayRole::SecondaryReceiver, selection: RoleSelection::DeepestForward },
        PlayRoleSpec { role: PlayRole::DeepOption, selection: RoleSelection::NearestGate },
    ],
    min_duration_ticks: 20.0,
    max_duration_ticks: 150.0,
};

pub static PRESSURE_RELEASE: Play = Play {
    kind: PlayKind::PressureRelease,
    roles: &[
        PlayRoleSpec { role: PlayRole::Carrier, selection: RoleSelection::BallCarrier },
        PlayRoleSpec { role: PlayRole::PrimaryReceiver, selection: RoleSelection::BestReception },
    ],
    min_duration_ticks: 15.0,
    max_duration_ticks: 90.0,
};

pub static SPREAD_ADVANCE: Play = Play {
    kind: PlayKind::SpreadAdvance,
    roles: &[
        PlayRoleSpec { role: PlayRole::Carrier, selection: RoleSelection::BallCarrier },
        PlayRoleSpec { role: PlayRole::PrimaryReceiver, selection: RoleSelection::DeepestForward },
        PlayRoleSpec { role: PlayRole::SecondaryReceiver, selection: RoleSelection::BestReception },
        PlayRoleSpec { role: PlayRole::DeepOption, selection: RoleSelection::NearestGate },
    ],
    min_duration_ticks: 30.0,
    max_duration_ticks: 360.0,
};

// ─── Resolved targets ────────────────────────────────────────────────────────

/// Resolved play targets for the current tick — what each assigned player
/// should actually navigate toward.
pub struct PlayTargets {
    pub carrier_target: Vec3,
    pub primary_target: Vec3,
    pub secondary_target: Vec3,
    pub screen_target: Option<Vec3>,
    pub deep_target: Option<Vec3>,
}

// ─── Target resolution ───────────────────────────────────────────────────────

/// Resolve concrete world-position targets for the given play kind.
/// `carrier` may be None for plays where we don't have possession.
pub fn resolve_targets(
    kind: PlayKind,
    state: &SimState,
    m: &MatchState,
    team: TeamSide,
    carrier: Option<&PlayerSim>,
) -> PlayTargets {
    match kind {
        PlayKind::GateRun => resolve_gate_run(state, m, team, carrier),
        PlayKind::OutletChain => resolve_outlet_chain(state, m, team, carrier),
        PlayKind::RecoveryFormation => resolve_recovery_formation(state, m, team),
        PlayKind::LooseBallRecovery => resolve_loose_ball_recovery(state, m, team),
        PlayKind::PressureRelease => resolve_pressure_release(state, m, team, carrier),
        PlayKind::SpreadAdvance => resolve_spread_advance(state, m, team, carrier),
    }
}

/// GateRun: carrier advances to throw range of the attack gate; primary
/// receiver cuts to just past the gate plane on-axis; secondary holds as
/// a safety outlet at carrier's depth, offset in radius.
fn resolve_gate_run(
    _state: &SimState,
    m: &MatchState,
    team: TeamSide,
    carrier: Option<&PlayerSim>,
) -> PlayTargets {
    let sign = attack_sign(team);
    let _ring_x = attack_ring_x(team);
    let gate_x = gate_world_x(team, m.cast.gate);

    let carrier_x = carrier.map(|c| c.p.x).unwrap_or(0.0);
    let carrier_yz = carrier.map(|c| (c.p.y, c.p.z)).unwrap_or((0.0, 0.0));

    // Carrier target: advance to throw range from the gate, stay near axis.
    let target_x = gate_x - sign * THROW_RANGE;
    // If carrier is already past the target, don't retreat — keep advancing.
    let carrier_target_x = if forward_progress(team, carrier_x) > forward_progress(team, target_x) {
        carrier_x
    } else {
        target_x
    };
    let carrier_target = Vec3::new(carrier_target_x, carrier_yz.0 * 0.3, carrier_yz.1 * 0.3);

    // Primary receiver: just past the gate plane, on-axis (low rho).
    // Position past the gate in the attack direction so a completed pass clears it.
    let primary_target = Vec3::new(gate_x + sign * 10.0, 0.0, ON_AXIS_RHO * 0.5);

    // Secondary: at carrier's depth, offset cross-radius for a quick dump.
    let secondary_target = Vec3::new(
        carrier_x,
        carrier_yz.0 + MED_RADIUS,
        carrier_yz.1,
    );

    PlayTargets {
        carrier_target,
        primary_target,
        secondary_target,
        screen_target: None,
        deep_target: None,
    }
}

/// OutletChain: carrier advances; primary cuts deep (70% toward ring);
/// secondary holds short (outlet). Creates a high-low read.
fn resolve_outlet_chain(
    _state: &SimState,
    _m: &MatchState,
    team: TeamSide,
    carrier: Option<&PlayerSim>,
) -> PlayTargets {
    let sign = attack_sign(team);
    let ring_x = attack_ring_x(team);

    let carrier_x = carrier.map(|c| c.p.x).unwrap_or(0.0);
    let carrier_yz = carrier.map(|c| (c.p.y, c.p.z)).unwrap_or((0.0, 0.0));

    // Carrier: advance forward at moderate pace (toward gate).
    let carrier_target = Vec3::new(
        carrier_x + sign * 15.0,
        carrier_yz.0 * 0.5,
        carrier_yz.1 * 0.5,
    );

    // Primary: cuts deep — 70% of the distance from carrier to attack ring.
    let deep_x = carrier_x + (ring_x - carrier_x) * DEEP_CUT_FRAC;
    let primary_target = Vec3::new(deep_x, MED_RADIUS * 0.7, 0.0);

    // Secondary: holds short — outlet ahead of carrier, low radius.
    let secondary_target = Vec3::new(
        carrier_x + sign * OUTLET_AHEAD,
        0.0,
        ON_AXIS_RHO * 0.4,
    );

    PlayTargets {
        carrier_target,
        primary_target,
        secondary_target,
        screen_target: None,
        deep_target: None,
    }
}

/// RecoveryFormation: opponent has possession. Three recovery stations
/// relative to the bell and our defensive ring.
fn resolve_recovery_formation(
    state: &SimState,
    _m: &MatchState,
    team: TeamSide,
) -> PlayTargets {
    let sign = attack_sign(team);
    let def_x = defend_ring_x(team);
    let bell_x = state.bell.p.x;
    let bell_y = state.bell.p.y;
    let bell_z = state.bell.p.z;

    // Primary: near the bell — the recovery diver.
    let primary_target = Vec3::new(bell_x, bell_y, bell_z);

    // Secondary: midfield outlet — halfway between bell and our ring,
    // offset in y for a clear lane.
    let mid_x = (bell_x + def_x) * 0.5;
    let secondary_target = Vec3::new(mid_x, MED_RADIUS, 0.0);

    // Deep: last back — stationed near our defensive ring.
    let deep_target = Vec3::new(
        def_x - sign * 30.0,
        0.0,
        ON_AXIS_RHO,
    );

    // Carrier target not meaningful here — place at bell as a fallback.
    let carrier_target = primary_target;

    PlayTargets {
        carrier_target,
        primary_target,
        secondary_target,
        screen_target: None,
        deep_target: Some(deep_target),
    }
}

/// LooseBallRecovery: ball is loose — one dives for the intercept, one
/// shadows the predicted rebound, others outlet.
fn resolve_loose_ball_recovery(
    state: &SimState,
    _m: &MatchState,
    team: TeamSide,
) -> PlayTargets {
    let sign = attack_sign(team);
    let bell_p = state.bell.p;
    let bell_v = state.bell.v;

    // Primary diver: target the bell's predicted position (linear extrapolation,
    // ~0.5 s ahead — 120 ticks at 240 Hz).
    let intercept_t = 0.5;
    let primary_target = Vec3::new(
        bell_p.x + bell_v.x * intercept_t,
        bell_p.y + bell_v.y * intercept_t,
        bell_p.z + bell_v.z * intercept_t,
    );

    // Shadow: predicted rebound locus — behind the bell relative to its
    // velocity, offset outward. If the diver bobbles, it spills here.
    let rebound_offset = 20.0;
    let screen_target = Vec3::new(
        bell_p.x - bell_v.x.signum() * rebound_offset,
        bell_p.y + MED_RADIUS * 0.5,
        bell_p.z,
    );

    // Secondary: pre-positioned for outlet pass after recovery — forward of bell.
    let secondary_target = Vec3::new(
        bell_p.x + sign * 40.0,
        0.0,
        ON_AXIS_RHO * 0.6,
    );

    // Deep: near our gate for safety.
    let deep_target = Vec3::new(
        defend_ring_x(team) - sign * 20.0,
        0.0,
        0.0,
    );

    // Carrier target = primary (no carrier in this play).
    let carrier_target = primary_target;

    PlayTargets {
        carrier_target,
        primary_target,
        secondary_target,
        screen_target: Some(screen_target),
        deep_target: Some(deep_target),
    }
}

/// PressureRelease: carrier is under pressure — nearest teammate cuts
/// TOWARD the carrier to shorten the dump pass.
fn resolve_pressure_release(
    state: &SimState,
    _m: &MatchState,
    team: TeamSide,
    carrier: Option<&PlayerSim>,
) -> PlayTargets {
    let carrier_p = carrier.map(|c| c.p).unwrap_or(state.bell.p);

    // Carrier: hold position (don't advance into more pressure).
    let carrier_target = carrier_p;

    // Primary: cut TOWARD the carrier — shorten the pass distance.
    // Find the nearest teammate direction and place target between them and carrier.
    // Without knowing exact teammate positions at this level, we place the target
    // at DUMP_CUT_DIST behind the carrier (in the defend direction) and offset in y.
    let sign = attack_sign(team);
    let primary_target = Vec3::new(
        carrier_p.x - sign * DUMP_CUT_DIST,
        carrier_p.y + 10.0,
        carrier_p.z,
    );

    // Secondary not used in this quick-release play.
    let secondary_target = Vec3::new(
        carrier_p.x - sign * DUMP_CUT_DIST * 2.0,
        carrier_p.y - 10.0,
        carrier_p.z,
    );

    PlayTargets {
        carrier_target,
        primary_target,
        secondary_target,
        screen_target: None,
        deep_target: None,
    }
}

/// SpreadAdvance: default possession play. Carrier advances on-axis;
/// receivers space at staggered depths and radii.
fn resolve_spread_advance(
    _state: &SimState,
    _m: &MatchState,
    team: TeamSide,
    carrier: Option<&PlayerSim>,
) -> PlayTargets {
    let sign = attack_sign(team);
    let ring_x = attack_ring_x(team);

    let carrier_x = carrier.map(|c| c.p.x).unwrap_or(0.0);
    let carrier_yz = carrier.map(|c| (c.p.y, c.p.z)).unwrap_or((0.0, 0.0));

    // Carrier: advance on-axis toward the attack ring.
    let carrier_target = Vec3::new(
        carrier_x + sign * 20.0,
        carrier_yz.0 * 0.2,
        carrier_yz.1 * 0.2,
    );

    // Primary: stagger depth 1, offset in y.
    let primary_target = Vec3::new(
        carrier_x + sign * SPREAD_DEPTH_1,
        MED_RADIUS,
        0.0,
    );

    // Secondary: stagger depth 2, offset opposite in y.
    let secondary_target = Vec3::new(
        carrier_x + sign * SPREAD_DEPTH_2 * 0.5,
        -MED_RADIUS * 0.7,
        ON_AXIS_RHO * 0.3,
    );

    // Deep: far forward near the ring — the long bomb option.
    let deep_x = carrier_x + (ring_x - carrier_x) * 0.8;
    let deep_target = Vec3::new(deep_x, 0.0, ON_AXIS_RHO * 0.5);

    PlayTargets {
        carrier_target,
        primary_target,
        secondary_target,
        screen_target: None,
        deep_target: Some(deep_target),
    }
}

// ─── Play selection / fitness ────────────────────────────────────────────────

/// Evaluate whether a play's preconditions are met given current state.
/// Returns a fitness score [0, 1] — higher means this play is more appropriate.
pub fn play_fitness(kind: PlayKind, state: &SimState, m: &MatchState, team: TeamSide) -> f64 {
    match kind {
        PlayKind::GateRun => fitness_gate_run(state, m, team),
        PlayKind::OutletChain => fitness_outlet_chain(state, m, team),
        PlayKind::RecoveryFormation => fitness_recovery(state, m, team),
        PlayKind::LooseBallRecovery => fitness_loose_ball(state, m, team),
        PlayKind::PressureRelease => fitness_pressure_release(state, m, team),
        PlayKind::SpreadAdvance => fitness_spread_advance(state, m, team),
    }
}

/// Find the carrier for `team` — the player holding the bell.
fn find_carrier<'a>(state: &'a SimState, team: TeamSide) -> Option<&'a PlayerSim> {
    let held_by = state.bell.held_by.as_ref()?;
    state.players.iter().find(|p| &p.id == held_by && p.team == team)
}

/// Count opponents between a point and the attack ring along the x axis.
fn defenders_between(state: &SimState, team: TeamSide, from_x: f64) -> usize {
    let ring_x = attack_ring_x(team);
    state.players.iter().filter(|p| {
        p.team != team
            && forward_progress(team, p.p.x) > forward_progress(team, from_x)
            && forward_progress(team, p.p.x) < forward_progress(team, ring_x)
    }).count()
}

/// Nearest opponent distance to a point.
fn nearest_opponent_dist(state: &SimState, team: TeamSide, pos: Vec3) -> f64 {
    state.players.iter()
        .filter(|p| p.team != team)
        .map(|p| p.p.sub(pos).len())
        .fold(f64::INFINITY, f64::min)
}

/// GateRun fitness: high when carrier is within 120m of gate, has room to
/// advance, and fewer than 2 defenders between carrier and gate.
fn fitness_gate_run(state: &SimState, m: &MatchState, team: TeamSide) -> f64 {
    let carrier = match find_carrier(state, team) {
        Some(c) => c,
        None => return 0.0,
    };

    let gate_x = gate_world_x(team, m.cast.gate);
    let dist_to_gate = (gate_x - carrier.p.x).abs();

    // Must be within 120m of the gate to consider this play.
    if dist_to_gate > 120.0 {
        return 0.1;
    }

    let defenders = defenders_between(state, team, carrier.p.x);
    let defender_penalty = if defenders >= 2 { 0.4 } else { 0.0 };

    // Score: closer to gate = better, fewer defenders = better.
    let proximity = 1.0 - (dist_to_gate / 120.0).min(1.0);
    (0.5 + proximity * 0.5 - defender_penalty).clamp(0.0, 1.0)
}

/// OutletChain fitness: high when carrier has possession with moderate
/// pressure and teammates have space.
fn fitness_outlet_chain(state: &SimState, _m: &MatchState, team: TeamSide) -> f64 {
    let carrier = match find_carrier(state, team) {
        Some(c) => c,
        None => return 0.0,
    };

    let nearest_def = nearest_opponent_dist(state, team, carrier.p);

    // Moderate pressure: defender between 12-30m. Too close = PressureRelease,
    // too far = can just advance freely.
    if nearest_def < PRESSURE_RADIUS {
        return 0.2; // too much pressure for this play
    }
    if nearest_def > 50.0 {
        return 0.3; // no pressure, spread advance is fine
    }

    // Sweet spot: moderate pressure, multiple options needed.
    0.7
}

/// RecoveryFormation fitness: high when the opponent has possession.
fn fitness_recovery(state: &SimState, _m: &MatchState, team: TeamSide) -> f64 {
    // Opponent holds the bell.
    if let Some(held_by) = &state.bell.held_by {
        if let Some(holder) = state.players.iter().find(|p| &p.id == held_by) {
            if holder.team != team {
                return 0.9;
            }
        }
    }
    // Opponent threw the ball (in flight from them).
    if state.bell.held_by.is_none() {
        if let Some(thrown_by) = &state.bell.thrown_by {
            if let Some(thrower) = state.players.iter().find(|p| &p.id == thrown_by) {
                if thrower.team != team {
                    return 0.7;
                }
            }
        }
    }
    0.0
}

/// LooseBallRecovery fitness: high when ball is loose (no holder, no
/// clear thrower ownership, or bell recently released).
fn fitness_loose_ball(state: &SimState, _m: &MatchState, _team: TeamSide) -> f64 {
    if state.bell.held_by.is_some() {
        return 0.0;
    }
    // Ball is in the air — if thrown recently by us, it's a live pass, not loose.
    // Truly "loose" = no recent thrower, or touched since throw.
    if state.bell.touched_since_throw {
        return 0.9;
    }
    if state.bell.thrown_by.is_none() {
        return 0.85;
    }
    // Ball in flight from a throw — could be intercepted but not "loose" yet.
    0.3
}

/// PressureRelease fitness: high when carrier is under immediate pressure
/// (defender < PRESSURE_RADIUS).
fn fitness_pressure_release(state: &SimState, _m: &MatchState, team: TeamSide) -> f64 {
    let carrier = match find_carrier(state, team) {
        Some(c) => c,
        None => return 0.0,
    };

    let nearest_def = nearest_opponent_dist(state, team, carrier.p);
    if nearest_def < PRESSURE_RADIUS {
        // Closer = more urgent.
        let urgency = 1.0 - (nearest_def / PRESSURE_RADIUS);
        (0.6 + urgency * 0.4).min(1.0)
    } else {
        0.0
    }
}

/// SpreadAdvance fitness: moderate always — the fallback when nothing
/// specific is called for.
fn fitness_spread_advance(state: &SimState, _m: &MatchState, team: TeamSide) -> f64 {
    if find_carrier(state, team).is_some() {
        0.4
    } else {
        0.1
    }
}

// ─── Utility: select the best play ──────────────────────────────────────────

/// All play kinds in evaluation order.
pub const ALL_PLAY_KINDS: &[PlayKind] = &[
    PlayKind::PressureRelease,
    PlayKind::LooseBallRecovery,
    PlayKind::RecoveryFormation,
    PlayKind::GateRun,
    PlayKind::OutletChain,
    PlayKind::SpreadAdvance,
];

/// Select the best play for the current situation. Returns the PlayKind
/// with the highest fitness score.
pub fn select_play(state: &SimState, m: &MatchState, team: TeamSide) -> PlayKind {
    let mut best_kind = PlayKind::SpreadAdvance;
    let mut best_score = 0.0_f64;

    for &kind in ALL_PLAY_KINDS {
        let score = play_fitness(kind, state, m, team);
        if score > best_score {
            best_score = score;
            best_kind = kind;
        }
    }

    best_kind
}

/// Get the static Play definition for a given kind.
pub fn play_def(kind: PlayKind) -> &'static Play {
    match kind {
        PlayKind::GateRun => &GATE_RUN,
        PlayKind::OutletChain => &OUTLET_CHAIN,
        PlayKind::RecoveryFormation => &RECOVERY_FORMATION,
        PlayKind::LooseBallRecovery => &LOOSE_BALL_RECOVERY,
        PlayKind::PressureRelease => &PRESSURE_RELEASE,
        PlayKind::SpreadAdvance => &SPREAD_ADVANCE,
    }
}

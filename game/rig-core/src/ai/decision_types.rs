//! Decision-layer substrate — the cross-file shared types of Director /
//! RiggerAI / AiSystem, defined ONCE here (the serialization point) so
//! Director and RiggerAI can be ported in parallel without colliding.
//! 1:1 with src/ai/Director.ts + src/ai/RiggerAI.ts + src/ai/index.ts.

use super::profile::{Difficulty, TeamProfile};
use super::types::TeamSide;
use crate::math::Vec3;
use crate::planner::Plan as GrapplePlan;
use std::collections::HashMap;

/// Director runs at ~2 Hz: once every 30 sim ticks (Director.ts:99).
pub const DIRECTOR_TICK_INTERVAL: f64 = 30.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastPosture {
    Grind,
    Tempo,
    Chase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Job {
    Carry,
    Recover,
    Receive,
    Mark,
    Support,
    Zone,
}

/// A committed per-player assignment, stable until the next Director tick.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerAssignment {
    pub job: Job,
    pub mark_id: Option<String>,
    pub depth_slot: f64,
    pub radius_slot: f64,
    pub pressure: f64,
}

#[derive(Clone, Debug)]
pub struct DirectorState {
    pub attacking_free: bool,
    /// +1 → drive toward +X ring, -1 → toward -X.
    pub attack_sign: f64,
    pub attack_ring_x: f64,
    pub defend_ring_x: f64,
    pub posture: CastPosture,
    pub loop_setter_id: Option<String>,
    pub contest_commit: bool,
    pub ev_faith: f64,
    pub ev_free: f64,
    pub update_in: f64,
    pub has_possession: bool,
    pub bell_loose: bool,
    pub carrier_id: Option<String>,
    pub recover_id: Option<String>,
    pub assignments: HashMap<String, PlayerAssignment>,
    /// Stable seeded jitter in [-1,1] for the Director window.
    pub style_noise: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DifficultyScaling {
    pub reaction_delay: f64,
    pub throw_variance: f64,
    pub read_quality: f64,
    pub drop_chance: f64,
    pub miss_open_chance: f64,
}

/// DIFFICULTY_PROFILES (RiggerAI.ts:102-124), verbatim.
pub fn difficulty_scaling(d: Difficulty) -> DifficultyScaling {
    match d {
        Difficulty::Rookie => DifficultyScaling {
            reaction_delay: 0.35,
            throw_variance: 0.18,
            read_quality: 0.45,
            drop_chance: 0.08,
            miss_open_chance: 0.2,
        },
        Difficulty::Pro => DifficultyScaling {
            reaction_delay: 0.12,
            throw_variance: 0.06,
            read_quality: 0.8,
            drop_chance: 0.02,
            miss_open_chance: 0.06,
        },
        Difficulty::Legend => DifficultyScaling {
            reaction_delay: 0.04,
            throw_variance: 0.02,
            read_quality: 0.97,
            drop_chance: 0.005,
            miss_open_chance: 0.01,
        },
    }
}

/// Per-player decision commitment (RiggerAI.ts:128-176). Field names are
/// the snake_case ports of the TS; semantics identical.
#[derive(Clone, Debug)]
pub struct PlayerCommit {
    pub throw_target_id: Option<String>,
    pub throw_go: bool,
    pub throw_dir: Option<Vec3>,
    pub throw_spin: f64,
    pub throw_charge: f64,
    pub nav_target: Option<Vec3>,
    pub catch_offset: Vec3,
    pub aim_dither: Vec3,
    pub saw_held_by: Option<String>,
    pub saw_contest: bool,
    pub saw_had_bell: bool,
    pub react_gate_until_tick: f64,
    pub last_anchor_pos: Option<Vec3>,
    /// -1 reel in / 0 swing.
    pub last_anchor_reel: i32,
    pub style_angle: f64,
    pub style_radius: f64,
    pub saw_role: Option<String>,
    pub saw_job: Option<String>,
    pub hold_ticks: f64,
    pub rrt_plan: Option<GrapplePlan>,
    pub rrt_plan_tick: f64,
    pub rrt_plan_target: Option<Vec3>,
}

#[derive(Clone, Debug, Default)]
pub struct PlayerCommitCache {
    pub value: Option<PlayerCommit>,
    pub decided_tick: f64,
}

/// src/ai/index.ts TeamConfig.
#[derive(Clone, Debug)]
pub struct TeamConfig {
    pub side: TeamSide,
    pub profile: TeamProfile,
    pub difficulty: Difficulty,
}

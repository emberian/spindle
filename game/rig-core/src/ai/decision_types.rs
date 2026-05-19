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

/// LOOSE-BELL COORDINATION sentinel. A `Job::Recover` assignment whose
/// `depth_slot` is exactly this value is the SHADOW / SAFETY role: it is
/// NOT the committed primary diver — it stations at the predicted rebound
/// locus (where the bell spills if the primary bobbles) and only pounces
/// (becomes the committer) if a deflection actually wins it the race at
/// the next Director window. Encoding the role through an EXISTING job +
/// a reserved slot value (rather than a new `Job` variant) keeps every
/// exhaustive `match Job` and the wasm `job_str` serialization untouched
/// while still being a fully distinct, deterministic role. The value is
/// out of the normal `depth_slot` range (a fraction/offset, never -3) so
/// it can never collide with a real slot. Pure data; no rng.
pub const SHADOW_DEPTH_FLAG: f64 = -3.0;

/// A committed per-player assignment, stable until the next Director tick.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerAssignment {
    pub job: Job,
    pub mark_id: Option<String>,
    pub depth_slot: f64,
    pub radius_slot: f64,
    pub pressure: f64,
}

impl PlayerAssignment {
    /// The committed PRIMARY diver of the loose bell (`recover_id`).
    pub fn recover_primary() -> Self {
        PlayerAssignment {
            job: Job::Recover,
            mark_id: None,
            depth_slot: 0.0,
            radius_slot: 0.0,
            pressure: 0.0,
        }
    }

    /// The SHADOW / SAFETY role — a `Job::Recover` member flagged (via the
    /// reserved `SHADOW_DEPTH_FLAG`) to hold the rebound locus, never to
    /// dive simultaneously with the primary.
    pub fn recover_shadow() -> Self {
        PlayerAssignment {
            job: Job::Recover,
            mark_id: None,
            depth_slot: SHADOW_DEPTH_FLAG,
            radius_slot: 0.0,
            pressure: 0.0,
        }
    }

    /// The down-field OUTLET — a `Job::Receive` body staged deep on the
    /// advance axis so a clean catch has an immediate pass target.
    pub fn recover_outlet() -> Self {
        PlayerAssignment {
            job: Job::Receive,
            mark_id: None,
            depth_slot: 0.85,
            radius_slot: 0.3,
            pressure: 0.0,
        }
    }

    /// Is this the flagged loose-bell SHADOW (the single semantic test,
    /// shared by the Director, the dive-committer guard and the recover
    /// nav so the encoding is defined in exactly one place)?
    pub fn is_shadow(&self) -> bool {
        self.job == Job::Recover
            && (self.depth_slot - SHADOW_DEPTH_FLAG).abs() < 1e-9
    }
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
    /// OFFENSE REBUILD — signed world-x just PAST the next cast gate. The
    /// designated gate receiver stages here so a completed carry→pass
    /// clears the gate (played progression, not pinball).
    pub gate_stage_x: f64,
    /// The receiver running the gate-clearing route this window.
    pub gate_receiver_id: Option<String>,
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
    /// LOOSE-BELL DECISIVE-DIVE hysteresis. Once a rigger is selected as the
    /// committer to hard-dive the loose bell it LATCHES this tick; it stays
    /// the committer (no per-tick flip-flop with coverage / w-max) until the
    /// bell is taken, it catches, or it is clearly beaten. -1 = not
    /// committing. Deterministic (a tick count, no wall-clock).
    pub dive_commit_tick: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PlayerCommitCache {
    pub value: Option<PlayerCommit>,
    pub decided_tick: f64,
}

/// RENDER-ONLY legibility record — one per AI-controlled rigger, produced
/// as a side channel of `AiSystem::tick` and serialized over a SEPARATE
/// wasm method (`RigAi::ai_debug_json` / `RigPolicy::ai_debug_json`),
/// PARALLEL to `snapshot_meta.playerIds`. It is the human-watchable
/// "what is this agent doing and why" surface for the spectate overlay.
///
/// It is NOT part of `Snapshot`, NEVER enters `sim_world::hash_snapshot`
/// or `sim.step`, and is a pure deterministic function of (state, match,
/// director, commit) — exactly like the existing `lineAnchors` render
/// channel, only it lives in the AI layer instead of the sim layer (so it
/// is structurally impossible for it to perturb determinism). No rng.
#[derive(Clone, Debug)]
pub struct AiDebugRec {
    pub id: String,
    /// The rigger's structural role ("anchor"/"spinner"/…).
    pub role: String,
    /// The Director's committed Job verb ("carry"/"recover"/…), or the
    /// fallback "support" when the rigger has no explicit assignment.
    pub job: String,
    /// The committed world-space point the rigger is acting on this
    /// window: its nav target, lead-intercept when catching, the loose
    /// bell when diving, or the contest interception point. `None` only
    /// when no meaningful target exists (e.g. holding & throwing in place).
    pub intent_target: Option<Vec3>,
    /// The SINGLE committed loose-bell diver (Director recover lead).
    pub is_diver: bool,
    /// The SINGLE committed active-defense contester.
    pub is_contester: bool,
    /// Director-named loose-bell PRIMARY (`recover_id`).
    pub is_primary: bool,
    /// The flagged loose-bell SHADOW (rebound safety, never co-dives).
    pub is_shadow: bool,
    /// A staged down-field outlet (`Job::Receive`) — the pass target.
    pub is_outlet: bool,
    /// Which swarm drives this rigger: "baseline" | "rl" | "human".
    pub controlled_by: &'static str,
}

/// src/ai/index.ts TeamConfig.
#[derive(Clone, Debug)]
pub struct TeamConfig {
    pub side: TeamSide,
    pub profile: TeamProfile,
    pub difficulty: Difficulty,
}

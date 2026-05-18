//! 1:1 Rust mirror of the FROZEN src/sim/types.ts contract — only the
//! shapes the AI reads (SimState/MatchState) or produces (InputFrame).
//! Field names match the TS verbatim (snake_case here; the wasm JSON
//! bridge in increment 5 maps camelCase ↔ snake_case at the boundary).

use crate::math::{Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeamSide {
    Home,
    Away,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiggerRole {
    Anchor,
    Spinner,
    Faithwing,
    Freewing,
    Reach,
}

/// match.faithEnd — which physical ring is the spinward/Faith (2-pt) end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaithEnd {
    PlusX,
    MinusX,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorType {
    Spar,
    Skin,
    Player,
    Ring,
}

#[derive(Clone, Debug)]
pub struct BellState {
    pub p: Vec3,
    pub v: Vec3,
    pub q: Quat,
    pub w: Vec3,
    pub chime: f64,
    pub held_by: Option<String>,
    pub thrown_by: Option<String>,
    pub touched_since_throw: bool,
    pub release_pos: Vec3,
    pub release_tick: f64,
    pub pass_chain: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct GrappleState {
    pub anchor_type: AnchorType,
    pub anchor_ref: Option<String>,
    pub anchor_pos: Vec3,
    pub rest_len: f64,
    pub taut: bool,
}

#[derive(Clone, Debug)]
pub struct PlayerSim {
    pub id: String,
    pub team: TeamSide,
    pub role: RiggerRole,
    pub p: Vec3,
    pub v: Vec3,
    pub q: Quat,
    pub line: Option<GrappleState>,
    pub dv_budget: f64,
    pub contact_ref: Option<String>,
    pub grounded: bool,
}

#[derive(Clone, Debug)]
pub struct SimState {
    pub tick: f64,
    pub omega: f64,
    pub bell: BellState,
    pub players: Vec<PlayerSim>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchPhase {
    Set,
    Live,
    Contest,
    Dead,
    InningBreak,
    Spine,
    Final,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    First,
    Deep,
    Mouth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContestDirection {
    Fair,
    Cross,
}

#[derive(Clone, Debug)]
pub struct Cast {
    /// 0..=3
    pub throws_left: u8,
    pub gate: Gate,
    pub spot_x: f64,
}

#[derive(Clone, Debug)]
pub struct MatchContest {
    pub thrower: String,
    pub contester: String,
    /// 0..=3
    pub count: u8,
    pub radius: f64,
    pub direction: ContestDirection,
}

#[derive(Clone, Debug)]
pub struct MatchState {
    pub inning: f64,
    pub spine: bool,
    pub possession: TeamSide,
    pub faith_end: FaithEnd,
    pub cast: Cast,
    pub contest: Option<MatchContest>,
    pub score_home: f64,
    pub score_away: f64,
    pub phase: MatchPhase,
    pub message: String,
    pub winner: Option<TeamSide>,
}

/// One tick of input for one controlled player. The replay unit.
#[derive(Clone, Debug)]
pub struct PlayerInput {
    pub id: String,
    pub aim: Vec3,
    pub fire_line_at: Option<Vec3>,
    /// -1 in / 0 none / 1 out
    pub reel: i32,
    pub release: bool,
    pub pushoff: bool,
    pub throw_charge: f64,
    pub throw_released: bool,
    pub throw_spin: f64,
    pub thrumbler: Vec3,
}

impl PlayerInput {
    /// The neutral input for a player that decides to do nothing this tick
    /// (mirrors the TS default object scattered across RiggerAI).
    pub fn idle(id: &str) -> Self {
        PlayerInput {
            id: id.to_string(),
            aim: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            fire_line_at: None,
            reel: 0,
            release: false,
            pushoff: false,
            throw_charge: 0.0,
            throw_released: false,
            throw_spin: 0.0,
            thrumbler: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
        }
    }
}

#[derive(Clone, Debug)]
pub struct InputFrame {
    pub tick: f64,
    pub players: Vec<PlayerInput>,
}

/// Stable per-commitment style draw (RiggerAI.ts:484 / Anchor.ts:29).
/// When present, role policies use these instead of fresh rng() draws so
/// a committed intent stays positionally consistent across ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoleStyle {
    pub angle: f64,
    pub radius: f64,
}

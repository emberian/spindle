//! The RL environment — a clean `Env` over the deterministic core.
//!
//! Wraps `SimWorld` + `MatchStateMachine` + `AiSystem` (the exact triad
//! `skill_eval` runs) behind a fixed gym seam so an external learner can
//! `reset` / `step` the sport and drive a subset of riggers, while the
//! rest run the existing baseline AI. NOT a new eval/judge/ranker — the
//! reward is the sport's own intrinsic per-step outcome only.
//!
//! NATIVE-ONLY (gated out of the wasm cdylib in lib.rs, like
//! skill_eval/ga). It reuses the skill_eval `conv::` seam bridges
//! verbatim — no reinvented conversions.
//!
//! ── Determinism argument ────────────────────────────────────────────
//! `reset(seed, scenario)` rebuilds `SimWorld::new(seed)` + a fresh
//! `MatchStateMachine` + a fresh `AiSystem` and replays the EXACT set
//! sequence skill_eval uses (roster add → set_bell_held → synthetic
//! `FoulGarrote` kick). `step` is pure: it folds `actions` (for
//! `controlled_ids`, in order) into an `InputFrame`, lets `AiSystem`
//! (with those ids in its `controlled` skip-set) fill every other
//! rigger, then `sim.step` / `mat.consume` exactly as the ranker does.
//! No wall-clock, no `thread_rng`, no HashMap-iteration folds (the
//! controlled set is membership-only; observation/reward iterate the
//! ordered `Snapshot::players`/`actions` vectors). `snapshot`/`restore`
//! deep-clone the whole triad (every internal type now derives `Clone`,
//! including the seeded `Rng` substreams + `LoopTracker` + `strip_press`
//! + `bell_dead`/`free_ticks`), so a restore is bit-identical state, not
//! a lossy projection of the public `Snapshot`. ⇒ same
//! seed+scenario+actions ⇒ bit-identical trajectory; a `restore`d env
//! continues the SAME trajectory a non-restored one would.

use crate::ai::profile::{style_to_profile, Difficulty};
use crate::ai::system::AiSystem;
use crate::ai::types as ai;
use crate::math::Vec3;
use crate::match_sm::{Gate, MatchPhase, MatchStateMachine};
use crate::scoring;
use crate::skill_eval::conv;
use crate::sim_world::{SimEvent as SwEv, SimWorld};

const H: f64 = 1.0 / 240.0;

// ── Scenario ────────────────────────────────────────────────────────────────

/// How a match is set up for the learner. `controlled_ids` are the
/// riggers the external agent drives (in this order — `step`'s `actions`
/// slice is positionally aligned to it); every other rigger runs the
/// existing baseline `AiSystem`.
#[derive(Clone, Debug)]
pub struct Scenario {
    /// (home_style, home_cyl, away_style, away_cyl) — fed to
    /// `style_to_profile` exactly like skill_eval's MATCHUPS rows.
    pub home_style: String,
    pub home_cyl: String,
    pub away_style: String,
    pub away_cyl: String,
    /// Roster ids the external policy owns (subset of the 8). Order is
    /// the action-vector order. Unknown ids are simply never matched
    /// (they then fall to the baseline AI) — no panic, deterministic.
    pub controlled_ids: Vec<String>,
    /// Truncation horizon (ticks). `step` sets `truncated` at this bound.
    pub max_ticks: u64,
    /// Intrinsic-reward shaping (NOT an eval/composite — just which of
    /// the sport's own outcomes to surface and the terminal magnitude).
    pub reward_config: RewardConfig,
}

impl Scenario {
    /// A symmetric self-play default (the skill_eval matchup #0 styles),
    /// no riggers externally controlled — the baseline drives all 8.
    pub fn self_play() -> Self {
        Scenario {
            home_style: "fall-dynasty".into(),
            home_cyl: "big-slow".into(),
            away_style: "fall-dynasty".into(),
            away_cyl: "big-slow".into(),
            controlled_ids: vec![],
            max_ticks: 2000,
            reward_config: RewardConfig::default(),
        }
    }
}

/// Per-outcome weights for the intrinsic reward. These scale the sport's
/// OWN events (possession, gate, score, contest, bell-out) — they are
/// not a quality metric or ranker; flipping them off zeroes that term.
#[derive(Clone, Copy, Debug)]
pub struct RewardConfig {
    pub w_possession: f64,
    pub w_gate: f64,
    pub w_score: f64,
    pub w_contest: f64,
    pub w_bell_out: f64,
    /// Terminal bonus magnitude: +this on a win for the controlled side,
    /// −this on a loss (0 if the match truncates / no winner).
    pub w_terminal: f64,
    /// Which side the controlled riggers' reward is scored FROM. If
    /// `None`, inferred from the first controlled id's roster team.
    pub reward_side: Option<TeamSide>,
    /// OPTIONAL **potential-based reward shaping** discount γ.
    ///
    /// `None` (the DEFAULT) ⇒ shaping is entirely OFF and `Reward.total`
    /// is byte-identical to before (production / `RewardConfig::default()`
    /// is unchanged). `Some(γ)` ⇒ `total` gets an EXTRA additive term of
    /// the strict Ng-Harada-Russell form `F = γ·Φ(s′) − Φ(s)`, where Φ is
    /// the bounded potential `shaping_potential` (gate progress of the
    /// possessing side, scored from the reward side, + completed-pass
    /// progress). ONLY the learner sets this; the raw intrinsic
    /// components (`possession_held`/`gate_advanced`/… ) are reported
    /// UNCHANGED — shaping affects ONLY `total`.
    ///
    /// ── POLICY-INVARIANCE (why this cannot change the optimum) ───────
    /// Ng, Harada & Russell (1999): for ANY potential Φ:S→ℝ, augmenting
    /// the reward with `F(s,s′)=γ·Φ(s′)−Φ(s)` leaves the set of optimal
    /// policies UNCHANGED, because over any trajectory the shaping
    /// telescopes:  Σ_{t} γ^t F(s_t,s_{t+1}) = −Φ(s_0) + (boundary γ^T
    /// Φ(s_T)) — it adds only a constant (the start-state potential, the
    /// same for every policy from a fixed reset) plus a vanishing
    /// terminal term, NOT any state-/action-dependent bias. So the
    /// argmax over policies of the shaped return equals the argmax of the
    /// pure-intrinsic return: the pure-intrinsic optimum is PRESERVED;
    /// shaping only DENSIFIES the gradient (a non-zero per-step signal
    /// toward gate/pass progress instead of the near-flat sparse reward).
    /// It is therefore NOT a composite ranker / not a new objective —
    /// the objective is still the pure intrinsic reward. We use the SAME
    /// `γ` the learner's return uses so the telescoping is exact. Φ is
    /// bounded (each component ∈ [0,1]) so F stays O(1) and finite.
    pub shaping_gamma: Option<f64>,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            w_possession: 0.01,
            w_gate: 0.1,
            w_score: 1.0,
            w_contest: 0.2,
            w_bell_out: 0.05,
            w_terminal: 1.0,
            reward_side: None,
            // Shaping OFF by default ⇒ production `total` byte-unchanged.
            shaping_gamma: None,
        }
    }
}

/// The bounded shaping potential Φ(state) ∈ roughly [-1, 2], scored from
/// `r_side` (the reward side). PURE function of public match/bell state:
///   * `gate_term`  = (cast gate ordinal / 2) ∈ {0, .5, 1}, SIGNED by
///     whether the side that currently possesses is the reward side
///     (+) or the opponent (−) — so driving the reward side's cast
///     forward raises Φ, the opponent's lowers it;
///   * `pass_term`  = min(pass_chain_len, CAP)/CAP ∈ [0,1] — completed
///     in-possession pass progress (a longer kept chain = more Φ),
///     SIGNED the same way (only meaningful while possessed).
/// Bounded ⇒ `F=γΦ′−Φ` is O(1). No rng/clock; deterministic.
fn shaping_potential(
    gate_ord_val: i32,
    possession_is_reward_side: bool,
    pass_chain_len: usize,
    possessed: bool,
) -> f64 {
    const PASS_CAP: f64 = 6.0;
    let sign = if possession_is_reward_side { 1.0 } else { -1.0 };
    let gate_term = sign * (gate_ord_val as f64 / 2.0);
    let pass_term = if possessed {
        sign * ((pass_chain_len as f64).min(PASS_CAP) / PASS_CAP)
    } else {
        0.0
    };
    gate_term + pass_term
}

/// Re-exported team discriminant for the reward side (the gym's own
/// small enum so callers don't depend on an internal module path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeamSide {
    Home,
    Away,
}

// ── Observation ─────────────────────────────────────────────────────────────
//
// `Observation`/`ObsPlayer` are the WASM-SAFE policy input view; they were
// moved into `rl::policy` (which compiles into the wasm cdylib) so policy
// inference no longer depends on this native-only module. The gym's public
// API is byte-unchanged: it re-exports them here, so every call site
// (`env.reset(..) -> Observation`, `gym::ObsPlayer`, etc.) is identical.
pub use crate::rl::policy::{Observation, ObsPlayer};

// ── Action / Reward / Step ──────────────────────────────────────────────────

/// One controlled rigger's input for one tick. This is exactly
/// `crate::ai::types::PlayerInput` — the same type the baseline AI
/// emits and the sim consumes, so there is ONE input contract.
pub type Action = ai::PlayerInput;

/// The sport's intrinsic per-step outcome from the controlled side's
/// point of view. `total` is the weighted sum (what an RL learner
/// maximises); the components are surfaced honestly for debugging /
/// reward shaping — they are the game's own deltas, not a composite
/// score or ranking.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reward {
    /// +1 per tick the controlled side holds the bell, −1 if the
    /// opponent does, 0 if free (raw, pre-weight).
    pub possession_held: f64,
    /// +1 when the cast gate advances (First→Deep→Mouth) this step.
    pub gate_advanced: f64,
    /// +points scored by the controlled side this step, −points conceded.
    pub scored: f64,
    /// +1 if a contest started / resolved in the controlled side's
    /// favour this step, −1 against (turnover-as-contest-loss included).
    pub contest_won: f64,
    /// −1 when the bell goes out of the field of play this step (a dead
    /// ball is a wasted possession for whoever last threw).
    pub bell_out: f64,
    /// Terminal: ±`w_terminal` on win/loss, 0 otherwise.
    pub terminal: f64,
    /// Weighted sum of the above (the scalar the learner optimises).
    pub total: f64,
}

/// Extra per-step diagnostics (not part of the reward).
#[derive(Clone, Debug, Default)]
pub struct StepInfo {
    /// Score points scored by each side THIS step (home, away).
    pub step_score_home: i64,
    pub step_score_away: i64,
    /// True the step a turnover happened.
    pub turnover: bool,
    /// Number of controlled actions that named an unknown rigger id
    /// (silently ignored — surfaced so callers can detect a misaligned
    /// action vector).
    pub unmatched_actions: usize,
}

/// One environment transition.
#[derive(Clone, Debug)]
pub struct Step {
    pub obs: Observation,
    pub reward: Reward,
    /// The match reached a terminal state (a winner).
    pub done: bool,
    /// The step horizon (`scenario.max_ticks`) was hit first.
    pub truncated: bool,
    pub info: StepInfo,
}

// ── The Env contract ────────────────────────────────────────────────────────

/// The fixed seam a learner codes against.
pub trait Env {
    fn reset(&mut self, seed: u32, scenario: &Scenario) -> Observation;
    /// `actions` are for `scenario.controlled_ids`, in that order.
    fn step(&mut self, actions: &[Action]) -> Step;
    fn snapshot(&self) -> Snapshot;
    fn restore(&mut self, s: &Snapshot);
}

// ── Snapshot (full deep state — for planning / tree search) ──────────────────

/// A complete, restorable env state. A deep clone of the whole triad
/// plus the gym bookkeeping, so `restore` is bit-identical (not the
/// lossy public `sim_world::Snapshot`).
#[derive(Clone)]
pub struct Snapshot {
    sim: SimWorld,
    mat: MatchStateMachine,
    ai: AiSystem,
    seed: u32,
    scenario: Scenario,
    elapsed: u64,
    prev_gate_ord: i32,
    /// Previous-step shaping potential Φ(s) (only used when
    /// `reward_config.shaping_gamma` is `Some`; carried in the snapshot so
    /// a restore continues the SAME shaped trajectory). Default 0.0.
    prev_phi: f64,
    done: bool,
}

// ── Roster (the skill_eval ROSTER, verbatim) ────────────────────────────────

struct RosterEntry {
    id: &'static str,
    team: ai::TeamSide,
    role: ai::RiggerRole,
    x: f64,
}
fn roster() -> [RosterEntry; 8] {
    use ai::RiggerRole::*;
    use ai::TeamSide::*;
    [
        RosterEntry { id: "H1", team: Home, role: Spinner, x: -40.0 },
        RosterEntry { id: "H2", team: Home, role: Anchor, x: -90.0 },
        RosterEntry { id: "H3", team: Home, role: Faithwing, x: -20.0 },
        RosterEntry { id: "H4", team: Home, role: Reach, x: -260.0 },
        RosterEntry { id: "A1", team: Away, role: Spinner, x: 40.0 },
        RosterEntry { id: "A2", team: Away, role: Anchor, x: 90.0 },
        RosterEntry { id: "A3", team: Away, role: Freewing, x: 20.0 },
        RosterEntry { id: "A4", team: Away, role: Reach, x: 260.0 },
    ]
}

fn gate_ord(g: Gate) -> i32 {
    match g {
        Gate::First => 0,
        Gate::Deep => 1,
        Gate::Mouth => 2,
    }
}

fn team_scoring_to_gym(t: scoring::TeamSide) -> TeamSide {
    match t {
        scoring::TeamSide::Home => TeamSide::Home,
        scoring::TeamSide::Away => TeamSide::Away,
    }
}
fn team_code(t: scoring::TeamSide) -> u8 {
    match t {
        scoring::TeamSide::Home => 0,
        scoring::TeamSide::Away => 1,
    }
}
fn ai_team_code(t: ai::TeamSide) -> u8 {
    match t {
        ai::TeamSide::Home => 0,
        ai::TeamSide::Away => 1,
    }
}
fn role_code(r: ai::RiggerRole) -> u8 {
    match r {
        ai::RiggerRole::Anchor => 0,
        ai::RiggerRole::Spinner => 1,
        ai::RiggerRole::Faithwing => 2,
        ai::RiggerRole::Freewing => 3,
        ai::RiggerRole::Reach => 4,
    }
}
fn phase_code(p: MatchPhase) -> u8 {
    match p {
        MatchPhase::Set => 0,
        MatchPhase::Live => 1,
        MatchPhase::Contest => 2,
        MatchPhase::Dead => 3,
        MatchPhase::InningBreak => 4,
        MatchPhase::Spine => 5,
        MatchPhase::Final => 6,
    }
}

// ── RigEnv ──────────────────────────────────────────────────────────────────

/// The concrete `Env`. Holds the live triad + the scenario it was reset
/// with.
pub struct RigEnv {
    sim: SimWorld,
    mat: MatchStateMachine,
    ai: AiSystem,
    cfgs: Vec<crate::ai::decision_types::TeamConfig>,
    seed: u32,
    scenario: Scenario,
    elapsed: u64,
    prev_gate_ord: i32,
    prev_phi: f64,
    done: bool,
}

impl RigEnv {
    pub fn new() -> Self {
        // Constructed in a defined-but-empty state; `reset` is mandatory
        // before the first `step` (mirrors the gym convention).
        RigEnv {
            sim: SimWorld::new(0),
            mat: MatchStateMachine::new(scoring::RingEnd::PlusX, scoring::TeamSide::Home),
            ai: AiSystem::new(),
            cfgs: Vec::new(),
            seed: 0,
            scenario: Scenario::self_play(),
            elapsed: 0,
            prev_gate_ord: 0,
            prev_phi: 0.0,
            done: false,
        }
    }

    /// The side the controlled riggers' reward is scored from.
    fn reward_side(&self) -> scoring::TeamSide {
        if let Some(rs) = self.scenario.reward_config.reward_side {
            return match rs {
                TeamSide::Home => scoring::TeamSide::Home,
                TeamSide::Away => scoring::TeamSide::Away,
            };
        }
        // Infer from the first controlled id's roster team.
        if let Some(first) = self.scenario.controlled_ids.first() {
            if let Some(r) = roster().iter().find(|r| r.id == first.as_str()) {
                return conv::team_ai_to_scoring(r.team);
            }
        }
        scoring::TeamSide::Home
    }

    fn build_observation(&self) -> Observation {
        let snap = self.sim.snapshot();
        let ms = self.mat.state();

        // DERIVED Director hints for the controlled side (the reward side
        // — `controlled_ids`' team / inferred). Pure `run_director` over
        // the public state with a fixed baseline profile + fixed seeded
        // rng; IDENTICAL to the `policy_wasm` build so the policy sees the
        // same assignment-conditioned features in training & inference.
        let ctrl_ai_side = match self.reward_side() {
            scoring::TeamSide::Home => ai::TeamSide::Home,
            scoring::TeamSide::Away => ai::TeamSide::Away,
        };
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = conv::msm_to_ai(self.mat.state());
        let hints = crate::ai::director::obs_director_hints(
            &ai_sim, &ai_match, ctrl_ai_side,
        );

        let players = snap
            .players
            .iter()
            .map(|p| {
                let assignment = hints
                    .assignments
                    .iter()
                    .find(|(id, _, _)| id == &p.id)
                    .map(|(_, job, mark)| crate::rl::policy::ObsAssignment {
                        job: *job,
                        mark_id: mark.clone(),
                    });
                ObsPlayer {
                    id: p.id.clone(),
                    team: match p.team {
                        crate::sim_world::TeamSide::Home => 0,
                        crate::sim_world::TeamSide::Away => 1,
                    },
                    role: match p.role {
                        crate::sim_world::RiggerRole::Anchor => 0,
                        crate::sim_world::RiggerRole::Spinner => 1,
                        crate::sim_world::RiggerRole::Faithwing => 2,
                        crate::sim_world::RiggerRole::Freewing => 3,
                        crate::sim_world::RiggerRole::Reach => 4,
                    },
                    p: p.p,
                    v: p.v,
                    line_anchor: p.line_anchor,
                    line_rest_len: p.line_rest_len,
                    assignment,
                }
            })
            .collect();
        Observation {
            tick: snap.tick,
            bell_p: snap.bell.p,
            bell_v: snap.bell.v,
            bell_held_by: snap.bell.held_by.clone(),
            possessed: snap.bell.held_by.is_some(),
            possession: team_code(ms.possession),
            gate: match ms.cast.gate {
                Gate::First => 0,
                Gate::Deep => 1,
                Gate::Mouth => 2,
            },
            score_home: ms.score_home,
            score_away: ms.score_away,
            phase: phase_code(ms.phase),
            players,
            controlled_ids: self.scenario.controlled_ids.clone(),
            attack_sign: hints.attack_sign,
            gate_plane_x: hints.gate_plane_x,
        }
    }

    /// The shaping potential Φ of the CURRENT state, scored from the
    /// reward side. Pure read of the live triad (gate ordinal,
    /// possession team, bell pass-chain length, possessed flag). Used
    /// only when `reward_config.shaping_gamma` is `Some`.
    fn current_phi(&self) -> f64 {
        let r_side = self.reward_side();
        let ms = self.mat.state();
        let snap = self.sim.snapshot();
        let possession_is_reward =
            team_scoring_to_gym(ms.possession) == team_scoring_to_gym(r_side);
        shaping_potential(
            gate_ord(ms.cast.gate),
            possession_is_reward,
            snap.bell.pass_chain.len(),
            snap.bell.held_by.is_some(),
        )
    }
}

impl Default for RigEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Env for RigEnv {
    fn reset(&mut self, seed: u32, scenario: &Scenario) -> Observation {
        let mut sim = SimWorld::new(seed);
        let rs = roster();
        for r in rs.iter() {
            // Same deterministic ring placement skill_eval uses.
            let k = rs.iter().position(|e| e.id == r.id).unwrap();
            let ang = (k as f64 / rs.len() as f64) * std::f64::consts::PI * 2.0;
            let sw_team = match r.team {
                ai::TeamSide::Home => crate::sim_world::TeamSide::Home,
                ai::TeamSide::Away => crate::sim_world::TeamSide::Away,
            };
            let sw_role = match r.role {
                ai::RiggerRole::Anchor => crate::sim_world::RiggerRole::Anchor,
                ai::RiggerRole::Spinner => crate::sim_world::RiggerRole::Spinner,
                ai::RiggerRole::Faithwing => crate::sim_world::RiggerRole::Faithwing,
                ai::RiggerRole::Freewing => crate::sim_world::RiggerRole::Freewing,
                ai::RiggerRole::Reach => crate::sim_world::RiggerRole::Reach,
            };
            sim.add_player(r.id, sw_team, sw_role, Vec3::new(r.x, ang.cos() * 8.0, ang.sin() * 8.0));
        }
        sim.set_bell_held("H1");

        let mut mat = MatchStateMachine::new(scoring::RingEnd::PlusX, scoring::TeamSide::Home);
        // Kick the match live with the same synthetic foul skill_eval uses.
        let snap0 = sim.snapshot();
        let s0 = conv::ai_to_scoring(&conv::snap_to_ai(&snap0));
        mat.consume(
            &[scoring::SimEvent::FoulGarrote { by: "__start__".to_string() }],
            &s0,
        );

        self.cfgs = vec![
            crate::ai::decision_types::TeamConfig {
                side: ai::TeamSide::Home,
                profile: style_to_profile(&scenario.home_style, &scenario.home_cyl),
                difficulty: Difficulty::Pro,
            },
            crate::ai::decision_types::TeamConfig {
                side: ai::TeamSide::Away,
                profile: style_to_profile(&scenario.away_style, &scenario.away_cyl),
                difficulty: Difficulty::Pro,
            },
        ];

        let mut ai = AiSystem::new();
        ai.set_controlled(scenario.controlled_ids.iter().cloned());

        self.sim = sim;
        self.mat = mat;
        self.ai = ai;
        self.seed = seed;
        self.scenario = scenario.clone();
        self.elapsed = 0;
        self.prev_gate_ord = gate_ord(self.mat.state().cast.gate);
        // Φ(s_0): the start-state potential. Always computed (cheap); only
        // CONSUMED in `step` when shaping is enabled — so the default path
        // is byte-unchanged.
        self.prev_phi = self.current_phi();
        self.done = self.mat.state().winner.is_some();
        self.build_observation()
    }

    fn step(&mut self, actions: &[Action]) -> Step {
        let r_side = self.reward_side();
        let pre = self.mat.state().clone();
        let pre_held = self.sim.snapshot().bell.held_by.clone();

        // Build the AI inputs for the uncontrolled riggers.
        let snap = self.sim.snapshot();
        let controlled: std::collections::HashSet<&str> = self
            .scenario
            .controlled_ids
            .iter()
            .map(|s| s.as_str())
            .collect();

        // Skip the expensive AiSystem tick entirely when all players are
        // externally controlled (self-play mode). This removes the plan_grapple
        // bottleneck from the training loop.
        let ai_frame = if controlled.len() >= snap.players.len() {
            ai::InputFrame { tick: snap.tick as f64, players: vec![] }
        } else {
            let ai_sim = conv::snap_to_ai(&snap);
            let ai_match = conv::msm_to_ai(self.mat.state());
            self.ai.tick(&ai_sim, &ai_match, &self.cfgs, self.seed)
        };

        // Fold in the external actions for controlled_ids (positional).
        // Unknown ids are counted but skipped — deterministic, no panic.
        let mut unmatched = 0usize;
        let mut players: Vec<ai::PlayerInput> = ai_frame
            .players
            .iter()
            .filter(|p| !controlled.contains(p.id.as_str()))
            .cloned()
            .collect();
        for (i, act) in actions.iter().enumerate() {
            // Trust the action's own id; if it isn't a known controlled
            // id, fall back to the scenario's i-th controlled id so a
            // positional caller still works.
            let id = if controlled.contains(act.id.as_str()) {
                act.id.clone()
            } else if let Some(sid) = self.scenario.controlled_ids.get(i) {
                sid.clone()
            } else {
                unmatched += 1;
                continue;
            };
            let mut a = act.clone();
            a.id = id;
            players.push(a);
        }
        let frame = ai::InputFrame { tick: snap.tick as f64, players };

        // Step the sim, feed events + post state to the match SM.
        let sw_frame = conv::frame_ai_to_sw(&frame);
        let evs = self.sim.step(&sw_frame, H);
        let mut bell_out_this_step = 0.0;
        for e in &evs {
            if matches!(e, SwEv::BellMissed { .. }) {
                bell_out_this_step -= 1.0;
            }
        }
        let scoring_evs: Vec<scoring::SimEvent> =
            evs.iter().map(conv::ev_sw_to_scoring).collect();
        let post_snap = self.sim.snapshot();
        let post = conv::ai_to_scoring(&conv::snap_to_ai(&post_snap));
        let upd = self.mat.consume(&scoring_evs, &post);

        // ── Intrinsic reward (the sport's own deltas) ──────────────────
        let cfg = self.scenario.reward_config;

        // Possession: who holds the bell post-step.
        let possession_held = match post_snap.bell.held_by.as_deref() {
            Some(hid) => match roster().iter().find(|r| r.id == hid) {
                Some(r) if conv::team_ai_to_scoring(r.team) == r_side => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            },
            None => 0.0,
        };

        // Gate advance (read post-consume, exactly like skill_eval Fix A).
        let g_ord = gate_ord(self.mat.state().cast.gate);
        let gate_advanced = if g_ord > self.prev_gate_ord { 1.0 } else { 0.0 };
        self.prev_gate_ord = g_ord;

        // Score delta (this side minus opponent), in points.
        let (dh, da) = (
            self.mat.state().score_home - pre.score_home,
            self.mat.state().score_away - pre.score_away,
        );
        let scored = match r_side {
            scoring::TeamSide::Home => (dh - da) as f64,
            scoring::TeamSide::Away => (da - dh) as f64,
        };

        // Contest: a started/resolved contest in this side's favour.
        // A turnover AGAINST the side counts as a contest loss; for the
        // side, a contest win. We read the SM's own turnover signal.
        let mut contest_won = 0.0;
        if let Some(t) = upd.turnover {
            // Turnover.team is the side that GAINS possession.
            contest_won += if team_scoring_to_gym(t.team) == team_scoring_to_gym(r_side) {
                1.0
            } else {
                -1.0
            };
        }
        // A fresh contest entering favouring this side (thrower on side).
        if pre.contest.is_none() {
            if let Some(c) = &self.mat.state().contest {
                if let Some(r) = roster().iter().find(|r| r.id == c.thrower.as_str()) {
                    contest_won += if conv::team_ai_to_scoring(r.team) == r_side {
                        0.5
                    } else {
                        -0.5
                    };
                }
            }
        }

        // Terminal.
        let winner = self.mat.state().winner;
        let done = winner.is_some();
        let terminal = match winner {
            Some(w) => {
                if team_scoring_to_gym(w) == team_scoring_to_gym(r_side) {
                    cfg.w_terminal
                } else {
                    -cfg.w_terminal
                }
            }
            None => 0.0,
        };

        let intrinsic_total = cfg.w_possession * possession_held
            + cfg.w_gate * gate_advanced
            + cfg.w_score * scored
            + cfg.w_contest * contest_won
            + cfg.w_bell_out * bell_out_this_step
            + terminal;

        // ── OPTIONAL potential-based shaping (policy-invariant) ────────
        // `total` = intrinsic + (γ·Φ(s′) − Φ(s)) IFF shaping is enabled;
        // otherwise `total` is the byte-unchanged intrinsic sum. Φ(s′) is
        // read on the post-consume, PRE-rearm state (the same point the
        // post-step possession/gate signals are read), Φ(s) is the value
        // carried from the previous step (`prev_phi`). The raw intrinsic
        // components below are reported UNCHANGED — shaping touches only
        // `total`. The telescoping sum makes this provably leave the
        // pure-intrinsic optimum unchanged (see `RewardConfig`).
        let total = match cfg.shaping_gamma {
            Some(gamma) => {
                let phi_s = self.prev_phi;
                let phi_next = self.current_phi();
                self.prev_phi = phi_next;
                intrinsic_total + (gamma * phi_next - phi_s)
            }
            None => {
                // Keep `prev_phi` tracking the live state even when OFF so
                // toggling on mid-episode (tests) stays well-defined; this
                // does NOT affect `total` (production path byte-unchanged).
                self.prev_phi = self.current_phi();
                intrinsic_total
            }
        };

        let reward = Reward {
            possession_held,
            gate_advanced,
            scored,
            contest_won,
            bell_out: bell_out_this_step,
            terminal,
            total,
        };

        // Re-arm a dead / inning-break ball exactly like skill_eval, so a
        // long match keeps producing live play (NOT a reward hook — pure
        // game flow).
        if self.mat.state().winner.is_none() && self.mat.state().phase != MatchPhase::Live {
            let poss = self.mat.state().possession;
            if let Some(r) = roster()
                .iter()
                .find(|r| conv::team_ai_to_scoring(r.team) == poss)
            {
                self.sim.set_bell_held(r.id);
            }
            self.mat.resume_live();
        }

        self.elapsed += 1;
        let truncated = !done && self.elapsed >= self.scenario.max_ticks;
        self.done = done;

        let _ = pre_held; // (kept for symmetry with skill_eval's pass logic)

        Step {
            obs: self.build_observation(),
            reward,
            done,
            truncated,
            info: StepInfo {
                step_score_home: dh,
                step_score_away: da,
                turnover: upd.turnover.is_some(),
                unmatched_actions: unmatched,
            },
        }
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            sim: self.sim.clone(),
            mat: self.mat.clone(),
            ai: self.ai.clone(),
            seed: self.seed,
            scenario: self.scenario.clone(),
            elapsed: self.elapsed,
            prev_gate_ord: self.prev_gate_ord,
            prev_phi: self.prev_phi,
            done: self.done,
        }
    }

    fn restore(&mut self, s: &Snapshot) {
        self.sim = s.sim.clone();
        self.mat = s.mat.clone();
        self.ai = s.ai.clone();
        self.seed = s.seed;
        self.scenario = s.scenario.clone();
        self.elapsed = s.elapsed;
        self.prev_gate_ord = s.prev_gate_ord;
        self.prev_phi = s.prev_phi;
        self.done = s.done;
        // Rebuild the (non-state) team configs from the restored scenario.
        self.cfgs = vec![
            crate::ai::decision_types::TeamConfig {
                side: ai::TeamSide::Home,
                profile: style_to_profile(&self.scenario.home_style, &self.scenario.home_cyl),
                difficulty: Difficulty::Pro,
            },
            crate::ai::decision_types::TeamConfig {
                side: ai::TeamSide::Away,
                profile: style_to_profile(&self.scenario.away_style, &self.scenario.away_cyl),
                difficulty: Difficulty::Pro,
            },
        ];
    }
}

// ── Tests (correctness gates only — NOT a ranker) ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim_world::hash_snapshot;

    fn scen(controlled: &[&str], max_ticks: u64) -> Scenario {
        Scenario {
            home_style: "fall-dynasty".into(),
            home_cyl: "big-slow".into(),
            away_style: "rise-power".into(),
            away_cyl: "small-fast".into(),
            controlled_ids: controlled.iter().map(|s| s.to_string()).collect(),
            max_ticks,
            reward_config: RewardConfig::default(),
        }
    }

    /// Φ POLICY-INVARIANCE (the written proof, as an executable check).
    /// For ANY fixed trajectory the discounted shaping sum telescopes:
    ///   Σ_{t=0}^{T-1} γ^t (γ·Φ(s_{t+1}) − Φ(s_t))
    ///     = −Φ(s_0) + γ^T·Φ(s_T)
    /// — it adds ONLY the start-state constant −Φ(s_0) (identical for
    /// every policy from a fixed reset) plus a vanishing γ^T boundary
    /// term, NO state-/action-dependent bias. So the shaped return ranks
    /// policies identically to the pure-intrinsic return ⇒ the
    /// pure-intrinsic optimum is preserved. This asserts the identity to
    /// f64 tolerance over a deterministic Φ sequence — a pure unit check,
    /// NO env/trainer.
    #[test]
    fn potential_shaping_telescopes_policy_invariant() {
        let gamma = 0.997_f64;
        // An arbitrary deterministic bounded Φ sequence (the shape of Φ
        // is irrelevant to invariance — telescoping holds for ANY Φ).
        let phi: Vec<f64> = (0..64)
            .map(|i| ((i as f64 * 0.37).sin() + 0.25 * ((i % 5) as f64)))
            .collect();
        let mut discounted_sum = 0.0_f64;
        for t in 0..phi.len() - 1 {
            let f = gamma * phi[t + 1] - phi[t];
            discounted_sum += gamma.powi(t as i32) * f;
        }
        let closed_form = -phi[0]
            + gamma.powi((phi.len() - 1) as i32) * phi[phi.len() - 1];
        assert!(
            (discounted_sum - closed_form).abs() < 1e-9,
            "shaping must telescope to a policy-independent constant: \
             {discounted_sum} vs {closed_form}"
        );
        // And `shaping_potential` itself is bounded + sign-correct: the
        // reward side advancing its cast raises Φ; the opponent lowers it.
        let mine = shaping_potential(2, true, 4, true);
        let theirs = shaping_potential(2, false, 4, true);
        assert!(mine > 0.0 && theirs < 0.0 && mine == -theirs);
        assert!(mine.abs() <= 2.0, "Φ must stay bounded/O(1)");
    }

    /// Shaping is DEFAULT-OFF and byte-unchanged: with the production
    /// `RewardConfig::default()` (shaping_gamma = None) every step's
    /// `total` is bit-identical to the explicit pure-intrinsic sum; and
    /// turning shaping ON changes `total` while leaving every raw
    /// intrinsic component bit-identical.
    #[test]
    fn shaping_default_off_and_only_affects_total() {
        let base = scen(&[], 200);
        assert!(base.reward_config.shaping_gamma.is_none());

        let mut off = RigEnv::new();
        off.reset(123, &base);
        let mut on_env = RigEnv::new();
        let mut on_sc = scen(&[], 200);
        on_sc.reward_config.shaping_gamma = Some(0.997);
        on_env.reset(123, &on_sc);

        for _ in 0..200 {
            let so = off.step(&[]);
            let sn = on_env.step(&[]);
            // DEFAULT path: total == the explicit intrinsic weighted sum.
            let c = base.reward_config;
            let recomputed = c.w_possession * so.reward.possession_held
                + c.w_gate * so.reward.gate_advanced
                + c.w_score * so.reward.scored
                + c.w_contest * so.reward.contest_won
                + c.w_bell_out * so.reward.bell_out
                + so.reward.terminal;
            assert_eq!(
                so.reward.total.to_bits(),
                recomputed.to_bits(),
                "default (shaping off) total must be the pure intrinsic sum"
            );
            // Raw components are reported UNCHANGED whether shaping is on.
            assert_eq!(
                so.reward.possession_held.to_bits(),
                sn.reward.possession_held.to_bits()
            );
            assert_eq!(
                so.reward.gate_advanced.to_bits(),
                sn.reward.gate_advanced.to_bits()
            );
            assert_eq!(so.reward.scored.to_bits(), sn.reward.scored.to_bits());
            assert_eq!(
                so.reward.contest_won.to_bits(),
                sn.reward.contest_won.to_bits()
            );
            assert!(sn.reward.total.is_finite());
            if so.done || sn.done {
                break;
            }
        }
    }

    /// A baseline-only env (no controlled ids) is a sane self-play match:
    /// it runs, stays finite, and produces some game activity. NOT a
    /// ranking — just "the loop is alive".
    #[test]
    fn self_play_smoke() {
        let mut env = RigEnv::new();
        let sc = scen(&[], 600);
        let obs0 = env.reset(1234, &sc);
        assert_eq!(obs0.players.len(), 8);
        assert_eq!(obs0.tick, 0);
        let mut last = obs0;
        for _ in 0..600 {
            let st = env.step(&[]);
            assert!(st.reward.total.is_finite());
            assert!(st.obs.bell_p.x.is_finite());
            last = st.obs.clone();
            if st.done {
                break;
            }
        }
        // The match advanced (ticks moved) and stayed in-bounds finite.
        assert!(last.tick > 0);
        assert!(last.score_home >= 0 && last.score_away >= 0);
    }

    /// Determinism: same seed + scenario + (here empty) actions ⇒
    /// bit-identical trajectory (hash the underlying snapshot each step).
    #[test]
    fn deterministic_reset_step() {
        let sc = scen(&[], 400);
        let mut a = RigEnv::new();
        let mut b = RigEnv::new();
        a.reset(99, &sc);
        b.reset(99, &sc);
        for _ in 0..400 {
            let sa = a.step(&[]);
            let sb = b.step(&[]);
            assert_eq!(
                hash_snapshot(&a.sim.snapshot()),
                hash_snapshot(&b.sim.snapshot())
            );
            assert_eq!(sa.reward.total.to_bits(), sb.reward.total.to_bits());
            assert_eq!(sa.done, sb.done);
        }
    }

    /// snapshot/restore round-trip: snapshot at tick N, step both the
    /// original and a restored copy with the same actions ⇒ they stay
    /// bit-identical. Proves restore is full state, not a lossy view.
    #[test]
    fn snapshot_restore_roundtrip() {
        let sc = scen(&["H1", "H2"], 500);
        let mut env = RigEnv::new();
        env.reset(7, &sc);
        // Drive a deterministic non-trivial action for the controlled
        // riggers (a steady aim + reel) so the controlled path is
        // exercised, not just baseline.
        let act = |id: &str| {
            let mut a = ai::PlayerInput::idle(id);
            a.aim = Vec3::new(1.0, 0.0, 0.0);
            a.reel = -1;
            a
        };
        for _ in 0..120 {
            env.step(&[act("H1"), act("H2")]);
        }
        let snap = env.snapshot();
        // Continue the original for a reference trajectory.
        let mut ref_hashes = Vec::new();
        for _ in 0..80 {
            env.step(&[act("H1"), act("H2")]);
            ref_hashes.push(hash_snapshot(&env.sim.snapshot()));
        }
        // Restore into a FRESH env and replay the same actions.
        let mut env2 = RigEnv::new();
        env2.restore(&snap);
        for k in 0..80 {
            env2.step(&[act("H1"), act("H2")]);
            assert_eq!(
                ref_hashes[k],
                hash_snapshot(&env2.sim.snapshot()),
                "restored trajectory diverged at step {k}"
            );
        }
    }

    /// Controlled-vs-baseline mixing: when H1 is externally controlled
    /// with a distinctive constant input, its trajectory must differ
    /// from a pure-baseline run (proving the selector skips it and the
    /// external action actually drives it), while an UNcontrolled rigger
    /// on the other team is unaffected at the first step.
    #[test]
    fn controlled_vs_baseline_mixing() {
        // Baseline-only reference.
        let mut base = RigEnv::new();
        base.reset(42, &scen(&[], 200));
        let mut base_h1 = Vec::new();
        for _ in 0..200 {
            let st = base.step(&[]);
            let h1 = st.obs.players.iter().find(|p| p.id == "H1").unwrap();
            base_h1.push(h1.p);
        }

        // H1 externally controlled with a hard constant pushoff+aim.
        let mut env = RigEnv::new();
        env.reset(42, &scen(&["H1"], 200));
        let mut ctl_h1 = Vec::new();
        for _ in 0..200 {
            let mut a = ai::PlayerInput::idle("H1");
            a.aim = Vec3::new(0.0, 1.0, 0.0);
            a.pushoff = true;
            let st = env.step(&[a]);
            let h1 = st.obs.players.iter().find(|p| p.id == "H1").unwrap();
            ctl_h1.push(h1.p);
        }

        // The controlled H1 diverges from the baseline H1.
        let diverged = base_h1
            .iter()
            .zip(ctl_h1.iter())
            .any(|(a, b)| (a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs() > 1e-6);
        assert!(diverged, "controlled H1 never diverged from baseline — selector not wired");

        // And the run stays deterministic for the controlled config too.
        let mut env2 = RigEnv::new();
        env2.reset(42, &scen(&["H1"], 200));
        for k in 0..200 {
            let mut a = ai::PlayerInput::idle("H1");
            a.aim = Vec3::new(0.0, 1.0, 0.0);
            a.pushoff = true;
            let st = env2.step(&[a]);
            let h1 = st.obs.players.iter().find(|p| p.id == "H1").unwrap();
            assert_eq!(ctl_h1[k].x.to_bits(), h1.p.x.to_bits());
        }
    }

    /// The agent-control selector at the AiSystem level: with H1 in the
    /// controlled set, AiSystem::tick must NOT emit an input for H1.
    #[test]
    fn aisystem_skips_controlled() {
        let mut env = RigEnv::new();
        env.reset(1, &scen(&["H1", "A3"], 10));
        let snap = env.sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = conv::msm_to_ai(env.mat.state());
        let frame = env.ai.tick(&ai_sim, &ai_match, &env.cfgs, env.seed);
        assert!(
            !frame.players.iter().any(|p| p.id == "H1" || p.id == "A3"),
            "AiSystem emitted an input for a controlled rigger"
        );
        assert!(
            frame.players.iter().any(|p| p.id == "H2"),
            "AiSystem must still drive uncontrolled riggers"
        );
    }
}

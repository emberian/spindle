//! WASM-bindgen facade for a TRAINED RL policy (`crate::rl::policy`).
//!
//! `RigPolicy` is the browser's seam to the learned shared-parameter MLP:
//! a team the user opts into is driven by the policy instead of the
//! baseline `RigAi`. It mirrors `RigAi`'s JSON idiom EXACTLY — same
//! frozen `SimState`/`MatchState` JSON in, same `InputFrame` JSON out
//! (every field incl. `catchIntent`, via the SHARED `ai_wasm` parsers and
//! `emit_input_frame` — no second JSON dialect). The only extra arg is
//! `controlled_ids_json` (`["H1","H2",...]`): those riggers' inputs come
//! from `rl::policy::RlPolicy::act`; every OTHER rigger falls to the
//! internal baseline `AiSystem` (its `controlled` skip-set = the policy
//! ids, exactly the gym's split).
//!
//! ── Determinism ─────────────────────────────────────────────────────────
//! The policy forward is pure f64 (no rng, no wall-clock); the egocentric
//! featurization is the SAME `rl::policy::featurize` training used. Given
//! identical (sim, match, controlled, seed) the output is bit-identical.
//! The baseline `AiSystem` half is the existing deterministic `RigAi`
//! path verbatim. ⇒ same seed ⇒ identical match ⇒ replay-stable (the
//! recorded `InputFrame`s already capture the policy's emitted inputs).

use wasm_bindgen::prelude::*;

use crate::ai::system::AiSystem;
use crate::ai::types::{
    Gate, InputFrame, MatchPhase, MatchState, PlayerInput, RiggerRole, SimState, TeamSide,
};
use crate::ai::decision_types::TeamConfig;
use crate::ai_wasm::{
    emit_input_frame, parse_match_state, parse_sim_state, parse_str_array,
};
use crate::rl::policy::{Observation, ObsPlayer, RlPolicy};
use crate::rl::attention::AttentionPolicy;

// ── SimState/MatchState → wasm-safe Observation ─────────────────────────────
//
// EXACTLY the gym's `build_observation` mapping (team/role/gate/phase
// integer codes), but sourced from the parsed FROZEN-contract JSON rather
// than a native `Snapshot` — so the policy sees the SAME featurization in
// the browser it saw in training. Pure; no rng/clock.

fn team_code(t: TeamSide) -> u8 {
    match t {
        TeamSide::Home => 0,
        TeamSide::Away => 1,
    }
}

fn role_code(r: RiggerRole) -> u8 {
    match r {
        RiggerRole::Anchor => 0,
        RiggerRole::Spinner => 1,
        RiggerRole::Faithwing => 2,
        RiggerRole::Freewing => 3,
        RiggerRole::Reach => 4,
    }
}

fn gate_code(g: Gate) -> u8 {
    match g {
        Gate::First => 0,
        Gate::Deep => 1,
        Gate::Mouth => 2,
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

fn build_observation(
    sim: &SimState,
    mat: &MatchState,
    controlled_ids: &[String],
) -> Observation {
    // Controlled side = the team of the controlled ids (default Home if
    // none / mixed). Same inference the gym's `reward_side` makes.
    let ctrl_side = if !controlled_ids.is_empty()
        && controlled_ids.iter().all(|id| {
            sim.players
                .iter()
                .find(|p| &p.id == id)
                .map(|p| p.team == TeamSide::Away)
                .unwrap_or(false)
        }) {
        TeamSide::Away
    } else {
        TeamSide::Home
    };
    // DERIVED Director hints — the SAME pure `obs_director_hints` the
    // native gym uses (fixed baseline profile + fixed seeded rng), so the
    // browser policy sees identical assignment/gate features to training.
    let hints =
        crate::ai::director::obs_director_hints(sim, mat, ctrl_side);

    let players = sim
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
                team: team_code(p.team),
                role: role_code(p.role),
                p: p.p,
                v: p.v,
                // The frozen contract carries a static `anchorPos` +
                // `restLen` when a line is out.
                line_anchor: p.line.as_ref().map(|l| l.anchor_pos),
                line_rest_len: p.line.as_ref().map(|l| l.rest_len),
                assignment,
            }
        })
        .collect();
    Observation {
        tick: sim.tick as u64,
        bell_p: sim.bell.p,
        bell_v: sim.bell.v,
        bell_held_by: sim.bell.held_by.clone(),
        possessed: sim.bell.held_by.is_some(),
        possession: team_code(mat.possession),
        gate: gate_code(mat.cast.gate),
        score_home: mat.score_home as i64,
        score_away: mat.score_away as i64,
        phase: phase_code(mat.phase),
        players,
        controlled_ids: controlled_ids.to_vec(),
        attack_sign: hints.attack_sign,
        gate_plane_x: hints.gate_plane_x,
    }
}

// ── Unified policy enum (MLP or Attention) ──────────────────────────────────

/// Either the MLP policy or the entity-attention policy. Both implement the
/// same `act(&obs, id) → PlayerInput` interface, so the wasm tick path is
/// unified.
enum PolicyKind {
    Mlp(RlPolicy),
    Attention(AttentionPolicy),
}

impl PolicyKind {
    fn act(&self, obs: &Observation, self_id: &str) -> PlayerInput {
        match self {
            PolicyKind::Mlp(p) => p.act(obs, self_id),
            PolicyKind::Attention(p) => p.act(obs, self_id),
        }
    }
}

// ── RigPolicy (wasm-bindgen public surface) ─────────────────────────────────

#[wasm_bindgen]
pub struct RigPolicy {
    policy: PolicyKind,
    /// The baseline AI fills every rigger NOT driven by the policy
    /// (its `controlled` skip-set is reset to the policy ids each tick).
    ai: AiSystem,
    /// RENDER-ONLY legibility records for the POLICY-driven riggers from
    /// the last `tick` (the baseline half's records live on `ai`). Merged
    /// with the baseline half by `ai_debug_json`. NEVER a sim/determinism
    /// surface — the spectate overlay channel only.
    rl_debug: Vec<crate::ai::decision_types::AiDebugRec>,
}

impl RigPolicy {
    /// Pure, native-testable fallible constructor: parse the flat
    /// `weights` array out of an artifact JSON blob and build the policy.
    /// `Err(String)` (NOT `JsValue`) so it is callable off-wasm — the
    /// wasm `new` constructor just maps the message into a `JsValue`. A
    /// wrong-dim / missing array is rejected (must NOT silently drive the
    /// sim with garbage), matching the rest of the JSON seam's rigor.
    ///
    /// Detects the policy type from the artifact: if `"type":"attention"`
    /// is present, loads as `AttentionPolicy` (ATTN_PARAM_W weights);
    /// otherwise loads as the MLP `RlPolicy` (PARAM_W weights).
    pub fn try_new(weights_json: &str) -> Result<RigPolicy, String> {
        let is_attention = weights_json.contains("\"type\":\"attention\"");
        let policy = if is_attention {
            let w = crate::rl::attention::attn_weights_from_json(weights_json)
                .ok_or_else(|| {
                    "RigPolicy: attention weights artifact missing/!= ATTN_PARAM_W f64"
                        .to_string()
                })?;
            PolicyKind::Attention(AttentionPolicy::from_weights(w))
        } else {
            let w = crate::rl::policy::weights_from_json(weights_json)
                .ok_or_else(|| {
                    "RigPolicy: weights artifact missing/!= PARAM_W f64 in \"weights\":[...]"
                        .to_string()
                })?;
            PolicyKind::Mlp(RlPolicy::from_weights(w))
        };
        Ok(RigPolicy {
            policy,
            ai: AiSystem::new(),
            rl_debug: Vec::new(),
        })
    }
}

#[wasm_bindgen]
impl RigPolicy {
    /// Build from a trained-weights artifact JSON blob (the committed
    /// `policy-v*.json`: `{"seed","config","dims","weights":[...]}`).
    /// Detects the policy type: if `"type":"attention"` is present, loads
    /// as `AttentionPolicy` (ATTN_PARAM_W weights); otherwise loads as
    /// the MLP `RlPolicy` (PARAM_W weights). A wrong-dim asset throws
    /// — matching the rigor of the rest of the JSON seam.
    #[wasm_bindgen(constructor)]
    pub fn new(weights_json: &str) -> Result<RigPolicy, JsValue> {
        Self::try_new(weights_json).map_err(|e| JsValue::from_str(&e))
    }

    /// One policy tick. Same JSON contract as `RigAi::tick` plus
    /// `controlled_ids_json` (a bare `["H1",...]` array). Policy ids get
    /// `RlPolicy::act`; all others get the baseline `AiSystem`. Returns
    /// the merged `InputFrame` JSON (policy players first, then AI).
    pub fn tick(
        &mut self,
        sim_json: &str,
        match_json: &str,
        controlled_ids_json: &str,
        seed: u32,
    ) -> String {
        let sim = parse_sim_state(sim_json);
        let mat = parse_match_state(match_json);
        // The controlled-ids arg is a bare JSON array; wrap it so the
        // shared `"key":`-anchored splitter can find it (same trick
        // `ai_wasm::parse_configs` uses for the bare configs array).
        let wrapped = format!("{{\"ids\":{}}}", controlled_ids_json.trim());
        let controlled = parse_str_array(&wrapped, "ids");

        let obs = build_observation(&sim, &mat, &controlled);

        // Policy half: one egocentric action per controlled id, IN the
        // given order (the same per-agent shared-weight forward training
        // ran). Skip ids not present in the sim (defensive — `act`
        // already no-ops an unknown id, but we also drop it so the frame
        // never carries a phantom rigger).
        let mut players: Vec<PlayerInput> = Vec::with_capacity(sim.players.len());
        self.rl_debug.clear();
        for id in &controlled {
            if let Some(sp) = sim.players.iter().find(|p| &p.id == id) {
                let act = self.policy.act(&obs, id);
                // RENDER-ONLY legibility for a learned rigger. The policy
                // is a single shared MLP — it has NO Director Job — so the
                // honest, watchable label is its ROLE + that it is
                // RL-controlled + the point it is acting on this tick (its
                // committed grapple target, or the bell when it signals a
                // catch). Pure read of the just-decoded action; no rng;
                // never a sim/determinism surface.
                let intent_target = if let Some(f) = act.fire_line_at {
                    Some(f)
                } else if act.catch_intent {
                    Some(sim.bell.p)
                } else {
                    None
                };
                self.rl_debug.push(crate::ai::decision_types::AiDebugRec {
                    id: id.clone(),
                    role: crate::ai::rigger_ai::role_str(sp.role).to_string(),
                    // The learned policy has no explicit Job; "learned" is
                    // the honest verb for the overlay (distinct from every
                    // baseline Job string).
                    job: "learned".to_string(),
                    intent_target,
                    is_diver: false,
                    is_contester: false,
                    is_primary: false,
                    is_shadow: false,
                    is_outlet: false,
                    controlled_by: "rl",
                });
                players.push(act);
            }
        }

        // Baseline half: the internal AiSystem drives every rigger NOT in
        // the policy set (its `controlled` skip-set IS the policy ids —
        // exactly the gym's split). The configs the baseline needs are
        // derived from the match — but `RigAi::tick` takes explicit
        // configs; here the spectate caller does not pass them, so we
        // reuse the baseline default profiles per side (the same the
        // production spectate path feeds, see main.ts cfgs).
        self.ai
            .set_controlled(controlled.iter().cloned().collect::<Vec<_>>());
        let configs = vec![
            TeamConfig {
                side: TeamSide::Home,
                profile: crate::ai::profile::TeamProfile::baseline(),
                difficulty: crate::ai::profile::Difficulty::Pro,
            },
            TeamConfig {
                side: TeamSide::Away,
                profile: crate::ai::profile::TeamProfile::baseline(),
                difficulty: crate::ai::profile::Difficulty::Pro,
            },
        ];
        let ai_frame = self.ai.tick(&sim, &mat, &configs, seed);
        for pl in ai_frame.players {
            players.push(pl);
        }

        emit_input_frame(&InputFrame {
            tick: sim.tick,
            players,
        })
    }

    /// RENDER-ONLY legibility seam (mirrors `RigAi::ai_debug_json`).
    /// Returns the merged per-rigger records IN THE SAME ORDER the last
    /// `tick` emitted players: the POLICY-driven riggers first
    /// (`controlledBy:"rl"`), then the baseline half (`"baseline"`). A
    /// separate method, a pure deterministic read, NEVER a sim or
    /// `hash_snapshot` surface — the spectate-overlay channel only.
    pub fn ai_debug_json(&self) -> String {
        let mut merged = self.rl_debug.clone();
        merged.extend_from_slice(self.ai.last_debug());
        crate::ai::system::ai_debug_to_json(&merged)
    }

    /// Reset the baseline AI's director/commitment caches (new inning).
    /// The policy itself is stateless (a pure forward) — nothing to clear.
    pub fn reset(&mut self) {
        self.ai.reset();
        self.rl_debug.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::policy::{weights_to_json, PARAM_W};

    fn sim_json() -> String {
        r#"{
          "tick": 0,
          "omega": 0.5,
          "bell": {
            "p": {"x":0.0,"y":0.0,"z":0.0},
            "v": {"x":0.0,"y":0.0,"z":0.0},
            "q": {"x":0.0,"y":0.0,"z":0.0,"w":1.0},
            "w": {"x":0.0,"y":0.0,"z":0.0},
            "chime": 1.0,
            "heldBy": "H1",
            "thrownBy": null,
            "touchedSinceThrow": false,
            "releasePos": {"x":0.0,"y":0.0,"z":0.0},
            "releaseTick": 0,
            "passChain": ["H1"]
          },
          "players": [
            {"id":"H1","team":"home","role":"spinner","p":{"x":-40.0,"y":4.0,"z":0.0},"v":{"x":0.0,"y":0.0,"z":0.0},"q":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"line":null,"dvBudget":6.0,"contactRef":null,"grounded":false},
            {"id":"H2","team":"home","role":"anchor","p":{"x":-90.0,"y":4.0,"z":0.0},"v":{"x":0.0,"y":0.0,"z":0.0},"q":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"line":{"anchorType":"spar","anchorRef":null,"anchorPos":{"x":-90.0,"y":0.0,"z":0.0},"restLen":12.0,"taut":true},"dvBudget":6.0,"contactRef":null,"grounded":false},
            {"id":"A1","team":"away","role":"spinner","p":{"x":40.0,"y":4.0,"z":0.0},"v":{"x":0.0,"y":0.0,"z":0.0},"q":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"line":null,"dvBudget":6.0,"contactRef":null,"grounded":false},
            {"id":"A2","team":"away","role":"reach","p":{"x":90.0,"y":4.0,"z":0.0},"v":{"x":0.0,"y":0.0,"z":0.0},"q":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"line":null,"dvBudget":6.0,"contactRef":null,"grounded":false}
          ]
        }"#
        .to_string()
    }

    fn match_json() -> String {
        r#"{
          "inning": 1, "spine": false, "possession": "home", "faithEnd": "+x",
          "cast": {"throwsLeft": 3, "gate": "first", "spotX": 0.0},
          "contest": null, "scoreHome": 0, "scoreAway": 0, "phase": "live",
          "message": "play ball", "winner": null
        }"#
        .to_string()
    }

    fn art() -> String {
        // A deterministic non-trivial weight pattern (no rng), wrapped in
        // the artifact JSON the constructor parses.
        let w: Vec<f64> = (0..PARAM_W)
            .map(|i| ((i % 17) as f64 - 8.0) * 0.013)
            .collect();
        weights_to_json(&w, 7, "unit-test", 0.0, 0.0)
    }

    #[test]
    fn policy_tick_well_shaped_and_deterministic() {
        let a = RigPolicy::try_new(&art())
            .unwrap()
            .tick(&sim_json(), &match_json(), r#"["H1","H2"]"#, 99);
        let b = RigPolicy::try_new(&art())
            .unwrap()
            .tick(&sim_json(), &match_json(), r#"["H1","H2"]"#, 99);
        assert_eq!(a, b, "same seed/weights must be bit-identical");
        assert!(a.starts_with("{\"tick\":"));
        // Policy-controlled + baseline-controlled riggers both present.
        assert!(a.contains("\"id\":\"H1\""));
        assert!(a.contains("\"id\":\"A1\""));
        // The frozen InputFrame field set — incl. catchIntent (the seam
        // lesson: every field must cross intact).
        for k in [
            "\"aim\"", "\"fireLineAt\"", "\"reel\"", "\"release\"",
            "\"pushoff\"", "\"throwCharge\"", "\"throwReleased\"",
            "\"throwSpin\"", "\"thrumbler\"", "\"catchIntent\"",
        ] {
            assert!(a.contains(k), "missing {k}");
        }
        assert!(!a.contains("NaN") && !a.contains("inf"));
    }

    #[test]
    fn rejects_wrong_dim_artifact() {
        assert!(RigPolicy::try_new(r#"{"weights":[1.0,2.0,3.0]}"#).is_err());
        assert!(RigPolicy::try_new(r#"{"weights":[]}"#).is_err());
        assert!(RigPolicy::try_new(r#"{"nope":1}"#).is_err());
    }

    #[test]
    fn empty_controlled_falls_entirely_to_baseline() {
        // No policy ids ⇒ pure baseline AI for all (production-equivalent
        // path); still a well-formed frame.
        let out = RigPolicy::try_new(&art())
            .unwrap()
            .tick(&sim_json(), &match_json(), r#"[]"#, 1);
        assert!(out.contains("\"id\":\"H1\""));
        assert!(out.contains("\"players\""));
    }

    #[test]
    fn attention_policy_loads_and_ticks() {
        use crate::rl::attention::{attn_weights_to_json, ATTN_PARAM_W};
        let w: Vec<f64> = (0..ATTN_PARAM_W)
            .map(|i| ((i % 23) as f64 - 11.0) * 0.007)
            .collect();
        let json = attn_weights_to_json(&w, 42, "attn-test", 0.0, 1.0);
        // Must detect as attention type and load successfully.
        let mut p = RigPolicy::try_new(&json).expect("attention artifact must load");
        let out = p.tick(&sim_json(), &match_json(), r#"["H1","H2"]"#, 99);
        assert!(out.starts_with("{\"tick\":"));
        assert!(out.contains("\"id\":\"H1\""));
        assert!(out.contains("\"id\":\"A1\""));
        assert!(!out.contains("NaN") && !out.contains("inf"));
    }
}

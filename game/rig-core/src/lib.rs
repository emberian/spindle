//! RIG deterministic physics core. Native (`cargo test`) for the test
//! oracle; `wasm-bindgen` cdylib for the browser. Ported module-by-module
//! against the green TS suite (the parity oracle). P2.75 increment 1:
//! foundations (math, rng, trajectory, the Number). Bell/Grapple/Player/
//! Collision/LoopDetector/SimWorld land in subsequent increments.

// Modules are standalone libraries until SimWorld + the wasm-bindgen
// step/snapshot API wire them together (P2.75 increment 3). Allow the
// not-yet-consumed surface; tightened when the facade lands.
#![allow(dead_code, unused_imports)]

pub mod math;
mod rng;
mod trajectory;
mod tuning;
// Swarm increment 2 — files authored in parallel, converged & built by the
// orchestrator. Declared up front so agents never collide on lib.rs.
mod bell;
mod collision;
mod grapple;
mod loop_detector;
mod player;
mod analysis;
// Progressive TS→Rust port, increment 1: the AI rollout kernel (the hot
// loop that made every TS eval/optimizer intractable). Native speed here
// unlocks deep MPPI/CEM/RL; parity oracle = the headless TS skill harness.
mod planner;
// Stage-1 keystone refactor: the composable CostTerm/Profile cost extraction
// (pure behavior-preserving; planner.rs routes its cost arithmetic through
// this). WASM-safe (no rayon/threads) — it is in the production cdylib path.
mod planner_cost;
// Progressive port increment 2 (swarm): match logic — 1:1 ports of
// src/match/{Scoring,Contest,MatchStateMachine}.ts. scoring is the shared
// base; contest and match_sm depend on it (wired via crate:: paths).
mod scoring;
mod contest;
mod match_sm;
// Progressive port increment 3: the AI substrate (coherent dependency
// root for the src/ai/** port — shared frozen types, bit-exact ai_rng,
// Orientation, TeamProfile). decide/* + roles/* + RiggerAI/Director/
// AiSystem land on this in subsequent sub-increments.
pub mod ai;
// Increment 5: the wasm-bindgen RigAi boundary — exposes the ported
// AiSystem to TS so the browser runs the Rust AI (TS AI deleted).
mod ai_wasm;
// The native skill-eval harness — the fast measuring instrument for
// ranking the planner algorithm zoo (replaces the deleted TS headless).
// NATIVE-ONLY: it uses rayon (gated out of the wasm cdylib in
// Cargo.toml) and has no place in the browser bundle anyway, so the
// module itself is excluded from the wasm32 target.
#[cfg(not(target_arch = "wasm32"))]
mod skill_eval;
#[cfg(not(target_arch = "wasm32"))]
pub mod drill;
#[cfg(not(target_arch = "wasm32"))]
pub mod match_analytics;
#[cfg(not(target_arch = "wasm32"))]
pub mod adversarial_eval;
#[cfg(not(target_arch = "wasm32"))]
pub mod curriculum;
#[cfg(not(target_arch = "wasm32"))]
pub mod generalization_eval;
// Stage 3: the offline GA over planner profile-space (deterministic,
// seeded ChaCha8). ADDITIVE & NATIVE-ONLY — it does not change production
// behavior (production still uses the default Coordination profile); the
// GA is an offline search tool over the cost-term WEIGHTS. Gated out of
// the wasm cdylib EXACTLY like skill_eval (it uses rand/rayon and the
// native eval_profile).
#[cfg(not(target_arch = "wasm32"))]
mod ga;
// THE COORDINATION LEARNER: a deterministic seeded self-play / evolutionary
// loop that optimizes the EFE + w-maxing controller (ai::efe +
// ai::rigger_ai) for team coordination. Fitness = the game's OWN intrinsic
// outcome for the controlled team (net score, possession, gate advance,
// contests, bell-in-play) — NOT a meta-instrument. w-maxing is kept as the
// generalization criterion (the controller it tunes is w-maxing; the
// learner also tie-breaks toward the weakest-sufficient genome). NATIVE-
// ONLY, gated out of the wasm cdylib EXACTLY like skill_eval / ga (it uses
// rand/rayon and the native episode driver).
#[cfg(not(target_arch = "wasm32"))]
mod coord_learner;
// Increment 3 (swarm): the deterministic world + the wasm-bindgen facade.
mod sim_world;
mod wasm;
// The RL environment / gym facade: exposes the deterministic
// SimWorld+MatchStateMachine+AiSystem as a clean `Env` (reset/step/
// snapshot/restore) so external learners can drive a subset of riggers.
// NATIVE-ONLY (gated EXACTLY like skill_eval/ga): it reuses the
// skill_eval `conv::` seam bridges (which live behind the same
// not(wasm32) cfg) and has no place in the browser cdylib. Determinism
// is preserved by construction — see gym.rs for the argument.
#[cfg(not(target_arch = "wasm32"))]
pub mod gym;
// THE RL AGENT: a shared-parameter deterministic MLP policy + a seeded
// gradient-free CEM trainer driving the gym (fitness = the gym's PURE
// intrinsic Reward, NO shaping), plus an OFFLINE strategy judge.
//
// SPLIT for the browser: `rl::policy` (featurize + forward + decode +
// the wasm-safe Observation view + flat weight (de)serialization) is
// pure f64 — NO rand/rayon/clock — so the whole `rl` module is now
// declared in BOTH builds and `rl::policy` compiles into the wasm
// cdylib (the browser drives a team via `policy_wasm::RigPolicy`).
// `rl::train` (CEM/rayon/rand_chacha + gym::RigEnv) is gated
// `cfg(not(target_arch = "wasm32"))` INSIDE rl/mod.rs exactly like
// gym/coord_learner/skill_eval/ga, so the wasm build still pulls
// neither rayon nor rand.
pub mod rl;

// The browser seam to the TRAINED policy: `RigPolicy` (wasm-bindgen),
// mirroring `RigAi`'s JSON contract. Ungated — it compiles into the
// cdylib and only uses `rl::policy` (pure f64) + the existing AiSystem.
mod policy_wasm;

use wasm_bindgen::prelude::*;

/// Smoke export so the WASM bundle has a callable surface this increment.
/// Returns 1 if the shipped Reg constants pass the gameplay-inequality gate.
#[wasm_bindgen]
pub fn reg_gate_ok() -> u32 {
    let r = tuning::evaluate(tuning::OMEGA, tuning::R, tuning::L, tuning::EARTH_G);
    if r.pass {
        1
    } else {
        0
    }
}

/// Expose the tuned reference spin (parity check against TS REG.omega).
#[wasm_bindgen]
pub fn reg_omega() -> f64 {
    tuning::OMEGA
}

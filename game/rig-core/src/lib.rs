//! RIG deterministic physics core. Native (`cargo test`) for the test
//! oracle; `wasm-bindgen` cdylib for the browser. Ported module-by-module
//! against the green TS suite (the parity oracle). P2.75 increment 1:
//! foundations (math, rng, trajectory, the Number). Bell/Grapple/Player/
//! Collision/LoopDetector/SimWorld land in subsequent increments.

// Modules are standalone libraries until SimWorld + the wasm-bindgen
// step/snapshot API wire them together (P2.75 increment 3). Allow the
// not-yet-consumed surface; tightened when the facade lands.
#![allow(dead_code, unused_imports)]

mod math;
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
mod ai;
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
// Increment 3 (swarm): the deterministic world + the wasm-bindgen facade.
mod sim_world;
mod wasm;

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

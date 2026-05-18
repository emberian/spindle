//! AI substrate — the coherent dependency root for the progressive
//! src/ai/** → Rust port (port increment 3+). The TS AI is the parity
//! oracle; these are 1:1 ports verified module-by-module against it.
//!
//! `types` mirrors the FROZEN src/sim/types.ts contract (the shapes the
//! AI reads/produces). `rng` is the bit-exact port of index.ts `makeRng`
//! (single-draw sfc32 — NOT crate::rng's 53-bit two-draw Substream).
//! `orientation` / `profile` port src/ai/Orientation.ts + the
//! league/teams.ts TeamProfile surface. Leaf decide/* and roles/* land
//! on top of this frozen substrate.

pub mod orientation;
pub mod profile;
pub mod rng;
pub mod types;
// Port increment 3 leaves (swarm-authored, orchestrator-converged):
// pure deterministic decide/* — no RNG, on the frozen substrate.
pub mod gate_solve;
pub mod lead_predict;
pub mod score_ev;
pub mod throw_score;
// Port increment 4: ai↔planner bridge (orchestrator-authored coherence
// glue) — the stable GrapplePlanner.planGrapple surface on ai::types
// that the swarmed role ports call.
pub mod plan_bridge;
// Port increment 4: per-role tactical policies (swarm-authored).
pub mod roles;
// Port increment 4b decision-layer substrate: the cross-file shared
// types of Director/RiggerAI/AiSystem (serialization point).
pub mod decision_types;
// Chamber-alive rebuild: active-inference + w-maxing controller substrate
// (generative model, EFE pragmatic+epistemic terms, volume coverage).
pub mod efe;
// Port increment 4b: the decision integrator + orchestrator.
pub mod director;
pub mod rigger_ai;
pub mod system;

pub use orientation::*;
pub use profile::{style_to_profile, Difficulty, TeamProfile};
pub use rng::AiRng;
pub use types::*;

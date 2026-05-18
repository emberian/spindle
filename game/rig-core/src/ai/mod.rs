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

pub use orientation::*;
pub use profile::{style_to_profile, Difficulty, TeamProfile};
pub use rng::AiRng;
pub use types::*;

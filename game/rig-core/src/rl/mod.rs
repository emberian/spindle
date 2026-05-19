//! rl/ — a real learned RL agent on the deterministic gym (drop 1).
//!
//! `policy`  — the SHARED-PARAMETER deterministic MLP: egocentric
//!             featurization, pure-f64 forward pass, action decode into
//!             the exact `ai::PlayerInput` contract.
//! `train`   — the seeded gradient-free CEM trainer (mirrors
//!             `coord_learner`'s determinism/parallel template exactly;
//!             fitness = the gym's PURE intrinsic `Reward`), plus the
//!             OFFLINE strategy judge (reporting/validation ONLY — never
//!             fitness/reward).
//!
//! NATIVE-ONLY (gated `cfg(not(target_arch = "wasm32"))` in lib.rs like
//! gym/coord_learner/skill_eval/ga). This drop does NOT wire a policy into
//! the shipped browser game — that is a deliberate phase 2.

#![cfg(not(target_arch = "wasm32"))]

pub mod policy;
pub mod train;

pub use policy::{RlPolicy, FEAT_W, K, OUT_W, PARAM_W};
pub use train::{judge, train_policy, JudgeSignals, TrainConfig, Trained};

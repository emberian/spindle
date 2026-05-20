//! rl/ — a real learned RL agent on the deterministic gym.
//!
//! `policy`  — the SHARED-PARAMETER deterministic MLP: egocentric
//!             featurization, pure-f64 forward pass, action decode into
//!             the exact `ai::PlayerInput` contract, plus the
//!             wasm-safe `Observation`/`ObsPlayer` view it consumes and
//!             flat weight (de)serialization. **WASM-SAFE**: pure f64,
//!             no rand/rayon/clock — compiled into BOTH the native lib
//!             and the wasm cdylib so the browser can drive a team with
//!             a trained policy (see `policy_wasm::RigPolicy`).
//! `attention` — the ENTITY-ATTENTION transformer policy (MAPPO actor):
//!             multi-head self-attention over entities (self, teammates,
//!             opponents, bell). **WASM-SAFE**: pure f64 matmuls + stable
//!             softmax. Same `act(&obs, id) → PlayerInput` interface.
//! `value`   — the CENTRALIZED VALUE FUNCTION (MAPPO critic): sees the
//!             full global state during training for credit assignment.
//!             **NATIVE-ONLY** (training aid, not deployed to wasm).
//! `train`   — the seeded gradient-free CEM trainer (mirrors
//!             `coord_learner`'s determinism/parallel template exactly;
//!             fitness = the gym's PURE intrinsic `Reward`), plus the
//!             OFFLINE strategy judge (reporting/validation ONLY — never
//!             fitness/reward). **NATIVE-ONLY** (rand_chacha/rayon +
//!             gym::RigEnv): gated `cfg(not(target_arch = "wasm32"))`
//!             in lib.rs exactly like gym/coord_learner/skill_eval/ga.
//!
//! The split lets `rl::policy` and `rl::attention` inference compile into
//! the wasm cdylib WITHOUT pulling rayon/rand (those are only `use`d
//! inside `train` and `value`, which the cfg gate keeps out of the wasm
//! build entirely).

pub mod policy;
pub mod attention;

#[cfg(not(target_arch = "wasm32"))]
pub mod value;

#[cfg(not(target_arch = "wasm32"))]
pub mod train;

#[cfg(not(target_arch = "wasm32"))]
pub mod self_play;

#[cfg(not(target_arch = "wasm32"))]
pub mod mappo;

pub use policy::{
    Observation, ObsPlayer, RlPolicy, FEAT_W, K, OUT_W, PARAM_W,
    N_INTENTS, INTENT_GATE_IDX, INTENT_LOGITS_BASE,
};

pub use attention::{AttentionPolicy, ATTN_PARAM_W, N_ENTITIES, D_MODEL};

#[cfg(not(target_arch = "wasm32"))]
pub use train::{judge, train_policy, JudgeSignals, TrainConfig, Trained};

#[cfg(not(target_arch = "wasm32"))]
pub use self_play::{
    run_self_play, Population, SelfPlayConfig, SelfPlayEnv, SelfPlayResult,
};

#[cfg(not(target_arch = "wasm32"))]
pub use value::{CentralizedValue, VALUE_PARAM_W, VALUE_FEAT_W, compute_gae};

#[cfg(not(target_arch = "wasm32"))]
pub use mappo::{train_mappo, MappoConfig, MappoResult, MappoGenReport, mappo_weights_to_json};

//! Per-role tactical policies — 1:1 ports of src/ai/roles/*.ts (swarm-
//! authored, orchestrator-converged) on the frozen substrate + the
//! ai↔planner bridge. Each exposes a `*_policy` (intent) and a
//! `*_navigate` (Option<GrapplePlan>) mirroring the TS surface.

pub mod anchor;
pub mod faithwing;
pub mod freewing;
pub mod reach;
pub mod spinner;

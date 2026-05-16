//! RIG deterministic physics core. Native (`cargo test`) for the test
//! oracle; `wasm-bindgen` cdylib for the browser. Ported module-by-module
//! against the green TS suite (the parity oracle). P2.75 increment 1:
//! foundations (math, rng, trajectory, the Number). Bell/Grapple/Player/
//! Collision/LoopDetector/SimWorld land in subsequent increments.

mod math;
mod rng;
mod trajectory;
mod tuning;

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

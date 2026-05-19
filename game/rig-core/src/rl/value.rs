//! rl/value.rs — Centralized value function (MAPPO critic, training-only).
//!
//! The value function sees the FULL global state: all 8 players' absolute
//! positions/velocities/roles/line-states + bell state + match state +
//! score. NOT egocentric — it is PRIVILEGED during training (this is what
//! gives credit assignment in CTDE). Used by MAPPO's GAE advantage:
//!   A_t = δ_t + γλδ_{t+1} + ...
//! where δ_t = r_t + γV(s_{t+1}) − V(s_t).
//!
//! NATIVE-ONLY: this module is gated out of the wasm build (it is only
//! needed during training, never at inference time). It has its own weight
//! vector separate from the actor.
//!
//! Architecture: a small MLP (2 layers, 128 hidden, tanh) mapping the
//! concatenated global state → scalar V(s).

use crate::math::Vec3;
use crate::tuning;

use super::policy::Observation;

// ── Normalization (same scales as actor) ───────────────────────────────────
const POS_SCALE: f64 = tuning::L;
const VEL_SCALE: f64 = tuning::REEL_PULL_SPEED;
const LEN_SCALE: f64 = tuning::L;
const SCORE_SCALE: f64 = 8.0;
const CLOCK_SCALE: f64 = 8000.0;
const ROLE_W: usize = 5;

// ── Global state feature dimensions ───────────────────────────────────────

/// Per-player global features: abs_pos(3) + abs_vel(3) + role(5)
///   + team(1) + line_attached(1) + line_rest_len(1) = 14
const PLAYER_FEAT: usize = 14;
/// Number of players in a full match.
const N_PLAYERS: usize = 8;
/// Bell features: abs_pos(3) + abs_vel(3) + held(1) + held_team(2) = 9
const BELL_FEAT: usize = 9;
/// Match context: possession(2) + gate(3) + phase(7) + score_margin(1)
///   + clock(1) + attack_sign(1) + gate_plane_x(1) = 16
const MATCH_FEAT: usize = 16;

/// Total value function input dimension.
pub const VALUE_FEAT_W: usize = N_PLAYERS * PLAYER_FEAT + BELL_FEAT + MATCH_FEAT;
// = 8*14 + 9 + 16 = 137

/// Hidden layer width.
pub const VALUE_HID: usize = 128;

/// Total value function parameter count:
/// W1(HID x FEAT) + b1(HID) + W2(HID x HID) + b2(HID) + W3(1 x HID) + b3(1)
pub const VALUE_PARAM_W: usize =
    VALUE_HID * VALUE_FEAT_W + VALUE_HID
    + VALUE_HID * VALUE_HID + VALUE_HID
    + VALUE_HID + 1;

// ── Global state featurization ─────────────────────────────────────────────

/// Build the global state feature vector from the full observation.
/// NOT egocentric — uses absolute positions (the value function is
/// privileged during training). Deterministic: players are ordered by
/// the observation's player vector (which comes from the ordered roster).
pub fn value_featurize(obs: &Observation) -> Vec<f64> {
    let mut f = vec![0.0_f64; VALUE_FEAT_W];
    let mut off = 0usize;

    // Per-player features (in roster order from obs.players).
    for (pi, p) in obs.players.iter().enumerate() {
        if pi >= N_PLAYERS {
            break;
        }
        let base = off + pi * PLAYER_FEAT;
        f[base] = p.p.x / POS_SCALE;
        f[base + 1] = p.p.y / POS_SCALE;
        f[base + 2] = p.p.z / POS_SCALE;
        f[base + 3] = p.v.x / VEL_SCALE;
        f[base + 4] = p.v.y / VEL_SCALE;
        f[base + 5] = p.v.z / VEL_SCALE;
        // role one-hot
        let r = (p.role as usize).min(ROLE_W - 1);
        f[base + 6 + r] = 1.0;
        // team
        f[base + 11] = p.team as f64;
        // line attached
        f[base + 12] = if p.line_anchor.is_some() { 1.0 } else { 0.0 };
        // line rest len
        f[base + 13] = p.line_rest_len.unwrap_or(0.0) / LEN_SCALE;
    }
    off += N_PLAYERS * PLAYER_FEAT;

    // Bell features (absolute).
    f[off] = obs.bell_p.x / POS_SCALE;
    f[off + 1] = obs.bell_p.y / POS_SCALE;
    f[off + 2] = obs.bell_p.z / POS_SCALE;
    f[off + 3] = obs.bell_v.x / VEL_SCALE;
    f[off + 4] = obs.bell_v.y / VEL_SCALE;
    f[off + 5] = obs.bell_v.z / VEL_SCALE;
    f[off + 6] = if obs.bell_held_by.is_some() { 1.0 } else { 0.0 };
    // held-by team: {home=1, away=1, none=0,0}
    if let Some(hid) = &obs.bell_held_by {
        if let Some(hp) = obs.players.iter().find(|p| &p.id == hid) {
            if hp.team == 0 {
                f[off + 7] = 1.0;
            } else {
                f[off + 8] = 1.0;
            }
        }
    }
    off += BELL_FEAT;

    // Match context.
    // possession one-hot
    if obs.possession == 0 {
        f[off] = 1.0;
    } else {
        f[off + 1] = 1.0;
    }
    // gate one-hot (3)
    let g = (obs.gate as usize).min(2);
    f[off + 2 + g] = 1.0;
    // phase one-hot (7)
    let ph = (obs.phase as usize).min(6);
    f[off + 5 + ph] = 1.0;
    // score margin
    let margin = (obs.score_home - obs.score_away) as f64;
    f[off + 12] = (margin / SCORE_SCALE).clamp(-1.0, 1.0);
    // clock
    f[off + 13] = ((obs.tick as f64) / CLOCK_SCALE).clamp(0.0, 1.0);
    // attack_sign
    f[off + 14] = obs.attack_sign;
    // gate_plane_x normalized
    f[off + 15] = obs.gate_plane_x / POS_SCALE;

    f
}

// ── Centralized Value Function ─────────────────────────────────────────────

/// Centralized value function (MAPPO critic). Training-only; has its own
/// weight vector separate from the actor.
#[derive(Clone)]
pub struct CentralizedValue {
    pub weights: Vec<f64>,
}

impl CentralizedValue {
    pub fn zeros() -> Self {
        CentralizedValue {
            weights: vec![0.0; VALUE_PARAM_W],
        }
    }

    pub fn from_weights(w: Vec<f64>) -> Self {
        assert_eq!(
            w.len(),
            VALUE_PARAM_W,
            "CentralizedValue weight length must be VALUE_PARAM_W={}",
            VALUE_PARAM_W
        );
        CentralizedValue { weights: w }
    }

    /// Pure deterministic forward pass: global features → scalar V(s).
    /// 2-layer MLP with tanh activations.
    pub fn forward(&self, features: &[f64]) -> f64 {
        debug_assert_eq!(features.len(), VALUE_FEAT_W);
        let w = &self.weights;
        let mut off = 0usize;

        // Layer 1: VALUE_HID x VALUE_FEAT_W + VALUE_HID bias, tanh.
        let mut h1 = [0.0_f64; VALUE_HID];
        for i in 0..VALUE_HID {
            let mut acc = 0.0;
            let row = off + i * VALUE_FEAT_W;
            for j in 0..VALUE_FEAT_W {
                acc += w[row + j] * features[j];
            }
            h1[i] = acc;
        }
        off += VALUE_HID * VALUE_FEAT_W;
        for i in 0..VALUE_HID {
            h1[i] = (h1[i] + w[off + i]).tanh();
        }
        off += VALUE_HID;

        // Layer 2: VALUE_HID x VALUE_HID + VALUE_HID bias, tanh.
        let mut h2 = [0.0_f64; VALUE_HID];
        for i in 0..VALUE_HID {
            let mut acc = 0.0;
            let row = off + i * VALUE_HID;
            for j in 0..VALUE_HID {
                acc += w[row + j] * h1[j];
            }
            h2[i] = acc;
        }
        off += VALUE_HID * VALUE_HID;
        for i in 0..VALUE_HID {
            h2[i] = (h2[i] + w[off + i]).tanh();
        }
        off += VALUE_HID;

        // Output layer: 1 x VALUE_HID + 1 bias, linear.
        let mut v = 0.0_f64;
        for j in 0..VALUE_HID {
            v += w[off + j] * h2[j];
        }
        off += VALUE_HID;
        v += w[off];

        v
    }

    /// Convenience: featurize + forward for a given observation.
    pub fn evaluate(&self, obs: &Observation) -> f64 {
        let features = value_featurize(obs);
        self.forward(&features)
    }
}

// ── GAE computation ────────────────────────────────────────────────────────

/// Compute Generalized Advantage Estimation (GAE) from a trajectory of
/// rewards and value estimates. Returns (advantages, returns).
///
/// GAE: A_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}
/// where δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
///
/// `values` has length T+1 (includes the bootstrap V(s_T)),
/// `rewards` has length T, `dones` has length T.
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    gamma: f64,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let t = rewards.len();
    assert_eq!(values.len(), t + 1);
    assert_eq!(dones.len(), t);

    let mut advantages = vec![0.0_f64; t];
    let mut gae = 0.0_f64;

    // Backward pass through time.
    for i in (0..t).rev() {
        let mask = if dones[i] { 0.0 } else { 1.0 };
        let delta = rewards[i] + gamma * values[i + 1] * mask - values[i];
        gae = delta + gamma * lambda * mask * gae;
        advantages[i] = gae;
    }

    // Returns = advantages + values[0..T]
    let returns: Vec<f64> = advantages
        .iter()
        .zip(values[..t].iter())
        .map(|(a, v)| a + v)
        .collect();

    (advantages, returns)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    use crate::gym::{Env, RigEnv, Scenario, RewardConfig};

    /// Value function produces finite scalars.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn value_function_produces_finite_scalar() {
        let mut env = RigEnv::new();
        let sc = Scenario {
            home_style: "fall-dynasty".into(),
            home_cyl: "big-slow".into(),
            away_style: "fall-dynasty".into(),
            away_cyl: "big-slow".into(),
            controlled_ids: vec![
                "H1".into(),
                "H2".into(),
                "H3".into(),
                "H4".into(),
            ],
            max_ticks: 50,
            reward_config: RewardConfig::default(),
        };
        let obs = env.reset(7, &sc);

        // Non-trivial weights.
        let w: Vec<f64> = (0..VALUE_PARAM_W)
            .map(|i| ((i % 19) as f64 - 9.0) * 0.005)
            .collect();
        let vf = CentralizedValue::from_weights(w);

        let v = vf.evaluate(&obs);
        assert!(v.is_finite(), "value function output not finite: {v}");
    }

    /// Value function determinism: same obs → bit-identical V(s).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn value_function_is_deterministic() {
        let mut env = RigEnv::new();
        let sc = Scenario {
            home_style: "fall-dynasty".into(),
            home_cyl: "big-slow".into(),
            away_style: "fall-dynasty".into(),
            away_cyl: "big-slow".into(),
            controlled_ids: vec![
                "H1".into(),
                "H2".into(),
                "H3".into(),
                "H4".into(),
            ],
            max_ticks: 50,
            reward_config: RewardConfig::default(),
        };
        let obs = env.reset(42, &sc);

        let w: Vec<f64> = (0..VALUE_PARAM_W)
            .map(|i| ((i % 13) as f64 - 6.0) * 0.003)
            .collect();
        let vf = CentralizedValue::from_weights(w);

        let v1 = vf.evaluate(&obs);
        let v2 = vf.evaluate(&obs);
        assert_eq!(v1.to_bits(), v2.to_bits(), "value not bit-stable");
    }

    /// GAE computation: known trajectory gives expected advantages.
    #[test]
    fn gae_computation_correct() {
        // Simple 3-step trajectory, gamma=0.99, lambda=0.95
        let rewards = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 1.0, 1.5, 2.0]; // T+1 values
        let dones = vec![false, false, false];
        let gamma = 0.99;
        let lambda = 0.95;

        let (advantages, returns) = compute_gae(&rewards, &values, &dones, gamma, lambda);

        assert_eq!(advantages.len(), 3);
        assert_eq!(returns.len(), 3);

        // Verify δ_2 = r_2 + γ*V(3) - V(2) = 3 + 0.99*2.0 - 1.5 = 3.48
        let d2 = 3.0 + gamma * 2.0 - 1.5;
        assert!((advantages[2] - d2).abs() < 1e-12);

        // All should be finite.
        for a in &advantages {
            assert!(a.is_finite());
        }
        for r in &returns {
            assert!(r.is_finite());
        }
    }

    /// GAE with done=true resets the advantage accumulator.
    #[test]
    fn gae_resets_on_done() {
        let rewards = vec![1.0, 2.0];
        let values = vec![0.5, 1.0, 1.5];
        let dones = vec![true, false]; // episode ends after step 0
        let gamma = 0.99;
        let lambda = 0.95;

        let (advantages, _) = compute_gae(&rewards, &values, &dones, gamma, lambda);

        // δ_0 = r_0 + γ*V(1)*0 - V(0) = 1.0 + 0 - 0.5 = 0.5
        // A_0 = δ_0 (no future terms because done=true masks them)
        assert!((advantages[0] - 0.5).abs() < 1e-12);
    }

    /// Feature dimensions match constants.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn value_feature_dim_correct() {
        let mut env = RigEnv::new();
        let sc = Scenario {
            home_style: "fall-dynasty".into(),
            home_cyl: "big-slow".into(),
            away_style: "fall-dynasty".into(),
            away_cyl: "big-slow".into(),
            controlled_ids: vec![
                "H1".into(),
                "H2".into(),
                "H3".into(),
                "H4".into(),
            ],
            max_ticks: 50,
            reward_config: RewardConfig::default(),
        };
        let obs = env.reset(1, &sc);
        let feat = value_featurize(&obs);
        assert_eq!(feat.len(), VALUE_FEAT_W);
        assert!(feat.iter().all(|v| v.is_finite()));
    }
}

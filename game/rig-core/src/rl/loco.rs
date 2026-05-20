//! rl/loco.rs — Learned locomotion policy (WASM-safe inference).
//!
//! A small MLP that replaces `plan_grapple`/`navigate_to`. Given the player's
//! current state and a target position, outputs the low-level grapple controls
//! (aim, fire_line_at, reel, release, pushoff, thrumbler).
//!
//! Trained offline on a reaching task (see `loco_train.rs`). Deployed to both
//! native and WASM. Pure deterministic f64 forward pass — no rand, no rayon.

use crate::math::Vec3;
use crate::tuning;

// ── Normalization scales ────────────────────────────────────────────────────
const POS_SCALE: f64 = tuning::L; // 640
const VEL_SCALE: f64 = tuning::REEL_PULL_SPEED; // 26
const LEN_SCALE: f64 = tuning::L;

/// Number of nearest spars in the observation.
const K_SPAR: usize = 3;

/// Feature width of the locomotion observation.
pub const LOCO_FEAT_W: usize = 40;

/// Output width of the locomotion policy.
pub const LOCO_OUT_W: usize = 13;

/// Hidden layer width.
const HID: usize = 48;

/// Total trainable parameters.
pub const LOCO_PARAM_W: usize =
    HID * LOCO_FEAT_W + HID + HID * HID + HID + LOCO_OUT_W * HID + LOCO_OUT_W;
// = 1920 + 48 + 2304 + 48 + 624 + 13 = 4957

/// Range for the fire direction → anchor point computation (metres).
const FIRE_RANGE: f64 = 60.0;

// ── Observation ─────────────────────────────────────────────────────────────

/// Compact locomotion observation (40 floats).
pub struct LocoObs {
    pub feat: [f64; LOCO_FEAT_W],
}

/// Grapple-relevant player state for featurization.
pub struct LocoPlayerState {
    pub p: Vec3,
    pub v: Vec3,
    pub line_anchor: Option<Vec3>,
    pub line_rest_len: Option<f64>,
    pub line_taut: bool,
    pub line_in_flight: bool,
    pub grounded: bool,
    pub omega: f64,
}

/// Build the locomotion observation from player state + target + spars.
pub fn loco_featurize(
    player: &LocoPlayerState,
    target: Vec3,
    spars: &[Vec3],
) -> LocoObs {
    let mut f = [0.0_f64; LOCO_FEAT_W];

    // [0..3) self pos / POS_SCALE
    f[0] = player.p.x / POS_SCALE;
    f[1] = player.p.y / POS_SCALE;
    f[2] = player.p.z / POS_SCALE;

    // [3..6) self vel / VEL_SCALE
    f[3] = player.v.x / VEL_SCALE;
    f[4] = player.v.y / VEL_SCALE;
    f[5] = player.v.z / VEL_SCALE;

    // [6..9) target rel pos / POS_SCALE
    let rel = Vec3::new(target.x - player.p.x, target.y - player.p.y, target.z - player.p.z);
    f[6] = rel.x / POS_SCALE;
    f[7] = rel.y / POS_SCALE;
    f[8] = rel.z / POS_SCALE;

    // [9] target distance / POS_SCALE
    let dist = rel.len();
    f[9] = dist / POS_SCALE;

    // [10..13) self-to-target unit direction
    if dist > 1e-6 {
        f[10] = rel.x / dist;
        f[11] = rel.y / dist;
        f[12] = rel.z / dist;
    }

    // [13..16) velocity decomposition: toward-target + tangent components
    let speed = player.v.len();
    if dist > 1e-6 && speed > 1e-6 {
        let tgt_dir = Vec3::new(rel.x / dist, rel.y / dist, rel.z / dist);
        let v_toward = player.v.x * tgt_dir.x + player.v.y * tgt_dir.y + player.v.z * tgt_dir.z;
        f[13] = v_toward / VEL_SCALE;
        // Tangent velocity (perpendicular to target direction)
        let v_tang = Vec3::new(
            player.v.x - tgt_dir.x * v_toward,
            player.v.y - tgt_dir.y * v_toward,
            player.v.z - tgt_dir.z * v_toward,
        );
        f[14] = v_tang.y / VEL_SCALE;
        f[15] = v_tang.z / VEL_SCALE;
    }

    // [16..19) line anchor rel pos / POS_SCALE
    if let Some(anchor) = player.line_anchor {
        f[16] = (anchor.x - player.p.x) / POS_SCALE;
        f[17] = (anchor.y - player.p.y) / POS_SCALE;
        f[18] = (anchor.z - player.p.z) / POS_SCALE;
    }

    // [19] line rest_len / LEN_SCALE
    if let Some(rl) = player.line_rest_len {
        f[19] = rl / LEN_SCALE;
    }

    // [20] line attached (taut)
    f[20] = if player.line_taut { 1.0 } else { 0.0 };

    // [21] line in-flight
    f[21] = if player.line_in_flight { 1.0 } else { 0.0 };

    // [22] refire cooldown (TODO: wire from sim state; 0 for now)
    f[22] = 0.0;

    // [23..32) 3 nearest spars, rel pos / POS_SCALE
    let mut spar_dists: Vec<(u64, usize)> = spars
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let dx = s.x - player.p.x;
            let dy = s.y - player.p.y;
            let dz = s.z - player.p.z;
            let d2 = dx * dx + dy * dy + dz * dz;
            (d2.to_bits(), i)
        })
        .collect();
    spar_dists.sort_unstable();
    for k in 0..K_SPAR {
        if k < spar_dists.len() {
            let s = spars[spar_dists[k].1];
            let base = 23 + k * 3;
            f[base] = (s.x - player.p.x) / POS_SCALE;
            f[base + 1] = (s.y - player.p.y) / POS_SCALE;
            f[base + 2] = (s.z - player.p.z) / POS_SCALE;
        }
    }

    // [32] radial distance to skin / R
    let rho = player.p.y.hypot(player.p.z);
    f[32] = rho / tuning::R;

    // [33..36) centrifugal direction (radial outward unit)
    if rho > 1e-6 {
        f[33] = 0.0; // x component is 0 (radial is in y-z plane)
        f[34] = player.p.y / rho;
        f[35] = player.p.z / rho;
    }

    // [36] speed / VEL_SCALE
    f[36] = speed / VEL_SCALE;

    // [37] omega
    f[37] = player.omega;

    // [38] grounded flag
    f[38] = if player.grounded { 1.0 } else { 0.0 };

    // [39] bias
    f[39] = 1.0;

    LocoObs { feat: f }
}

// ── Policy (MLP) ────────────────────────────────────────────────────────────

/// The locomotion policy network (2-layer MLP, tanh activations).
#[derive(Clone)]
pub struct LocoPolicy {
    pub params: Vec<f64>,
}

impl LocoPolicy {
    pub fn new_zeros() -> Self {
        LocoPolicy {
            params: vec![0.0; LOCO_PARAM_W],
        }
    }

    pub fn from_params(params: Vec<f64>) -> Self {
        assert_eq!(params.len(), LOCO_PARAM_W);
        LocoPolicy { params }
    }

    /// Forward pass: obs features → raw output logits/heads.
    pub fn forward(&self, obs: &LocoObs) -> [f64; LOCO_OUT_W] {
        let w = &self.params;
        let mut idx = 0;

        // Layer 1: input → hidden (LOCO_FEAT_W × HID)
        let mut h1 = [0.0_f64; HID];
        for j in 0..HID {
            let mut sum = 0.0;
            for i in 0..LOCO_FEAT_W {
                sum += obs.feat[i] * w[idx];
                idx += 1;
            }
            h1[j] = sum;
        }
        // Bias 1
        for j in 0..HID {
            h1[j] = (h1[j] + w[idx]).tanh();
            idx += 1;
        }

        // Layer 2: hidden → hidden (HID × HID)
        let mut h2 = [0.0_f64; HID];
        for j in 0..HID {
            let mut sum = 0.0;
            for i in 0..HID {
                sum += h1[i] * w[idx];
                idx += 1;
            }
            h2[j] = sum;
        }
        // Bias 2
        for j in 0..HID {
            h2[j] = (h2[j] + w[idx]).tanh();
            idx += 1;
        }

        // Output layer: hidden → output (HID × LOCO_OUT_W)
        let mut out = [0.0_f64; LOCO_OUT_W];
        for j in 0..LOCO_OUT_W {
            let mut sum = 0.0;
            for i in 0..HID {
                sum += h2[i] * w[idx];
                idx += 1;
            }
            out[j] = sum;
        }
        // Output bias
        for j in 0..LOCO_OUT_W {
            out[j] += w[idx];
            idx += 1;
        }

        debug_assert_eq!(idx, LOCO_PARAM_W);
        out
    }
}

// ── Decode ──────────────────────────────────────────────────────────────────

/// Decoded locomotion action.
pub struct LocoAction {
    pub aim: Vec3,
    pub fire_line_at: Option<Vec3>,
    pub reel: i32,
    pub release: bool,
    pub pushoff: bool,
    pub thrumbler: Vec3,
}

/// Decode raw policy output into a locomotion action.
pub fn loco_decode(out: &[f64; LOCO_OUT_W], player_p: Vec3, target: Vec3) -> LocoAction {
    // [0..3) fire direction head → normalize
    let fd_raw = Vec3::new(out[0], out[1], out[2]);
    let fd_len = fd_raw.len();
    let fire_dir = if fd_len > 1e-9 {
        Vec3::new(fd_raw.x / fd_len, fd_raw.y / fd_len, fd_raw.z / fd_len)
    } else {
        let to_t = Vec3::new(target.x - player_p.x, target.y - player_p.y, target.z - player_p.z);
        let tl = to_t.len();
        if tl > 1e-6 { Vec3::new(to_t.x / tl, to_t.y / tl, to_t.z / tl) }
        else { Vec3::new(1.0, 0.0, 0.0) }
    };

    // [3] fire gate
    let fire_gate = out[3] > 0.0;

    // [4..7) reel logits → argmax
    let reel = if out[4] >= out[5] && out[4] >= out[6] {
        -1 // winch in
    } else if out[5] >= out[4] && out[5] >= out[6] {
        0 // neutral / swing
    } else {
        1 // reel out
    };

    // [7] release
    let release = out[7].tanh() > 0.0;

    // [8] pushoff
    let pushoff = out[8].tanh() > 0.0;

    // [9..12) thrumbler (tanh → [-1,1] per axis, scaled by max)
    let thrumbler = Vec3::new(
        out[9].tanh() * 3.5,
        out[10].tanh() * 3.5,
        out[11].tanh() * 3.5,
    );

    // [12] aim override: 1 = aim toward target, 0 = aim along fire_dir
    let aim_override = out[12].tanh() > 0.0;
    let aim = if aim_override {
        let to_t = Vec3::new(target.x - player_p.x, target.y - player_p.y, target.z - player_p.z);
        let tl = to_t.len();
        if tl > 1e-6 { Vec3::new(to_t.x / tl, to_t.y / tl, to_t.z / tl) }
        else { Vec3::new(1.0, 0.0, 0.0) }
    } else {
        fire_dir
    };

    // Fire target: anchor point = player.p + fire_dir * FIRE_RANGE
    let fire_line_at = if fire_gate {
        Some(Vec3::new(
            player_p.x + fire_dir.x * FIRE_RANGE,
            player_p.y + fire_dir.y * FIRE_RANGE,
            player_p.z + fire_dir.z * FIRE_RANGE,
        ))
    } else {
        None
    };

    LocoAction {
        aim,
        fire_line_at,
        reel,
        release,
        pushoff,
        thrumbler,
    }
}

// ── Weight serialization ────────────────────────────────────────────────────

pub fn loco_weights_to_json(policy: &LocoPolicy) -> String {
    let nums: Vec<String> = policy.params.iter().map(|v| format!("{:.8}", v)).collect();
    format!("[{}]", nums.join(","))
}

pub fn loco_weights_from_json(json: &str) -> Option<LocoPolicy> {
    let trimmed = json.trim().trim_start_matches('[').trim_end_matches(']');
    let params: Vec<f64> = trimmed
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    if params.len() == LOCO_PARAM_W {
        Some(LocoPolicy::from_params(params))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_zero_weights_produces_valid_output() {
        let policy = LocoPolicy::new_zeros();
        let player = LocoPlayerState {
            p: Vec3::new(0.0, 10.0, 0.0),
            v: Vec3::new(5.0, 0.0, 0.0),
            line_anchor: None,
            line_rest_len: None,
            line_taut: false,
            line_in_flight: false,
            grounded: false,
            omega: 0.32,
        };
        let target = Vec3::new(100.0, 10.0, 0.0);
        let spars = crate::planner::spar_positions();
        let obs = loco_featurize(&player, target, &spars);
        let out = policy.forward(&obs);
        // Zero weights → zero outputs (bias layer is zero too)
        for &v in out.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn featurize_is_deterministic() {
        let player = LocoPlayerState {
            p: Vec3::new(-50.0, 20.0, -5.0),
            v: Vec3::new(12.0, -3.0, 1.0),
            line_anchor: Some(Vec3::new(-30.0, 25.0, 0.0)),
            line_rest_len: Some(22.0),
            line_taut: true,
            line_in_flight: false,
            grounded: false,
            omega: 0.32,
        };
        let target = Vec3::new(80.0, -10.0, 15.0);
        let spars = crate::planner::spar_positions();
        let a = loco_featurize(&player, target, &spars);
        let b = loco_featurize(&player, target, &spars);
        assert_eq!(a.feat, b.feat);
    }

    #[test]
    fn decode_produces_valid_action() {
        let out = [0.5, 0.1, -0.3, 0.8, -0.2, 0.5, 0.1, -0.5, 0.3, 0.2, -0.1, 0.4, -0.6];
        let p = Vec3::new(0.0, 10.0, 0.0);
        let t = Vec3::new(100.0, 10.0, 0.0);
        let action = loco_decode(&out, p, t);
        assert!(action.aim.len() > 0.9);
        assert!(action.fire_line_at.is_some()); // gate > 0
        assert_eq!(action.reel, 0); // out[5] = 0.5 is max
    }

    #[test]
    fn param_count_matches() {
        assert_eq!(LOCO_PARAM_W, 4957);
    }
}

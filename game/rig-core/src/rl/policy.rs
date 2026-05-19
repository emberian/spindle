//! rl/policy.rs — the SHARED-PARAMETER policy.
//!
//! ONE small deterministic MLP. Every controlled rigger feeds its OWN
//! egocentric observation through the SAME weight vector and decodes the
//! SAME-shaped output into an `ai::PlayerInput`. Coordination is not
//! hand-coded anywhere here: it can only EMERGE from (a) shared weights,
//! (b) per-agent egocentric obs, (c) a shared team reward in the trainer.
//!
//! ── Determinism ─────────────────────────────────────────────────────────
//! `featurize` and `forward` are pure deterministic f64. No `exp` (the
//! squash is `tanh`, which is bit-stable across the std libm we target),
//! no HashMap iteration (the K-nearest lists are built from the ORDERED
//! `Observation::players` vector and sorted by an explicit
//! `(distance_bits, id)` key — never hash order). Same weights + same obs
//! ⇒ bit-identical `PlayerInput`.
//!
//! NATIVE-ONLY: this module is gated `cfg(not(target_arch = "wasm32"))`
//! in lib.rs exactly like gym/coord_learner/skill_eval/ga. It does NOT
//! wire a policy into the shipped browser game (that is a deliberate
//! phase 2).

use crate::ai::types as ai;
use crate::gym::Observation;
use crate::math::Vec3;
use crate::tuning;

// ── Canon normalization scales (from tuning.rs) ─────────────────────────────
// Positions are normalized by the ring length `L`; velocities by the
// winch cruise speed (the rigger's only real propulsion, so the natural
// O(1) speed scale); the line rest length by `L`. These keep every
// feature O(1) so a small fixed MLP sees well-conditioned inputs.
const POS_SCALE: f64 = tuning::L; // 640
const VEL_SCALE: f64 = tuning::REEL_PULL_SPEED; // 26
const LEN_SCALE: f64 = tuning::L; // 640

/// Nearest-K teammates and nearest-K opponents in the feature vector.
pub const K: usize = 3;

/// Per-other-rigger feature block: rel p (3) + rel v (3).
const OTHER_W: usize = 6;

/// Fixed, documented feature width. Layout (all egocentric, normalized):
///   [0..3)   self p / POS_SCALE
///   [3..6)   self v / VEL_SCALE
///   [6..9)   bell rel p / POS_SCALE
///   [9..12)  bell rel v / VEL_SCALE
///   [12..15) bell-held one-hot: {mine-team, opp-team, free}
///   [15..17) possession one-hot: {home, away}
///   [17..20) cast gate one-hot: {First, Deep, Mouth}
///   [20..27) match phase one-hot (7 phases)
///   [27..30) self line anchor rel p / POS_SCALE (0 if no line)
///   [30]     self line rest_len / LEN_SCALE (0 if no line)
///   [31]     self line attached flag (1 if a line is out, else 0)
///   [32..50) K=3 nearest TEAMMATES, each rel p (3) + rel v (3),
///            sorted by (distance, id)
///   [50..68) K=3 nearest OPPONENTS, same block + sort key
///   [68]     constant bias 1.0
pub const FEAT_W: usize = 32 + 2 * K * OTHER_W + 1; // = 69

/// Raw network outputs before decode.
///   [0..3)  aim head (normalized to a unit-ish 3-vec)
///   [3..6)  reel logits (argmax → -1 / 0 / +1)
///   [6]     fire gate logit (>0 ⇒ fire this tick)
///   [7..10) fire direction head (normalized; target = self.p + dir*range)
///   [10]    release logit (tanh>0)
///   [11]    pushoff logit (tanh>0)
///   [12]    catch_intent logit (tanh>0)
///   [13]    throw_charge head (squashed to [0,1])
///   [14]    throw_spin head (tanh → [-1,1])
///   [15..18) thrumbler head (tanh per-axis → [-1,1]^3)
pub const OUT_W: usize = 18;

/// Hidden layer width (two hidden layers).
pub const HID: usize = 64;

/// The fire target is placed this many metres downrange along the fire
/// direction head when the fire gate fires (O(ring) so it is a reachable
/// grapple point; the sim clamps/ignores it under latency/cooldown).
const FIRE_RANGE: f64 = 60.0;

/// Total trainable parameter count: W1(HID×FEAT_W)+b1 + W2(HID×HID)+b2 +
/// W3(OUT_W×HID)+b3.
pub const PARAM_W: usize =
    HID * FEAT_W + HID + HID * HID + HID + OUT_W * HID + OUT_W;

#[inline]
fn d2(a: Vec3, b: Vec3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

#[inline]
fn norm3(v: Vec3) -> Vec3 {
    let m = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    if m > 1e-9 {
        Vec3::new(v.x / m, v.y / m, v.z / m)
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    }
}

/// Build the egocentric, normalized, deterministic feature vector for the
/// controlled rigger `self_id` from `obs`. Pure: no rng, no clock, no
/// HashMap iteration. The K-nearest lists come from the ORDERED
/// `obs.players` and are sorted by `(distance.to_bits(), id)` — a total,
/// reproducible key.
pub fn featurize(obs: &Observation, self_id: &str) -> [f64; FEAT_W] {
    let mut f = [0.0_f64; FEAT_W];

    let me = match obs.players.iter().find(|p| p.id == self_id) {
        Some(m) => m,
        // Defensive: an unknown id ⇒ all-zero features + bias (the decode
        // still yields a valid idle-ish input). Never panics.
        None => {
            f[FEAT_W - 1] = 1.0;
            return f;
        }
    };
    let sp = me.p;

    // self p / v
    f[0] = sp.x / POS_SCALE;
    f[1] = sp.y / POS_SCALE;
    f[2] = sp.z / POS_SCALE;
    f[3] = me.v.x / VEL_SCALE;
    f[4] = me.v.y / VEL_SCALE;
    f[5] = me.v.z / VEL_SCALE;

    // bell relative p / v
    f[6] = (obs.bell_p.x - sp.x) / POS_SCALE;
    f[7] = (obs.bell_p.y - sp.y) / POS_SCALE;
    f[8] = (obs.bell_p.z - sp.z) / POS_SCALE;
    f[9] = obs.bell_v.x / VEL_SCALE;
    f[10] = obs.bell_v.y / VEL_SCALE;
    f[11] = obs.bell_v.z / VEL_SCALE;

    // bell-held one-hot {mine-team, opp-team, free}
    match &obs.bell_held_by {
        Some(hid) => match obs.players.iter().find(|p| &p.id == hid) {
            Some(h) if h.team == me.team => f[12] = 1.0,
            Some(_) => f[13] = 1.0,
            None => f[14] = 1.0,
        },
        None => f[14] = 1.0,
    }

    // possession one-hot {home, away}
    if obs.possession == 0 {
        f[15] = 1.0;
    } else {
        f[16] = 1.0;
    }

    // gate one-hot {First, Deep, Mouth}
    let g = (obs.gate as usize).min(2);
    f[17 + g] = 1.0;

    // phase one-hot (7)
    let ph = (obs.phase as usize).min(6);
    f[20 + ph] = 1.0;

    // self line state
    if let Some(anc) = me.line_anchor {
        f[27] = (anc.x - sp.x) / POS_SCALE;
        f[28] = (anc.y - sp.y) / POS_SCALE;
        f[29] = (anc.z - sp.z) / POS_SCALE;
        f[30] = me.line_rest_len.unwrap_or(0.0) / LEN_SCALE;
        f[31] = 1.0;
    }

    // K nearest teammates / opponents — built from the ORDERED players
    // vec, sorted by an explicit total key (distance bits, then id).
    let mut mates: Vec<(u64, &str, &crate::gym::ObsPlayer)> = Vec::new();
    let mut opps: Vec<(u64, &str, &crate::gym::ObsPlayer)> = Vec::new();
    for p in obs.players.iter() {
        if p.id == self_id {
            continue;
        }
        let key = d2(p.p, sp).to_bits();
        if p.team == me.team {
            mates.push((key, p.id.as_str(), p));
        } else {
            opps.push((key, p.id.as_str(), p));
        }
    }
    let sort_key =
        |a: &(u64, &str, &crate::gym::ObsPlayer),
         b: &(u64, &str, &crate::gym::ObsPlayer)| {
            a.0.cmp(&b.0).then(a.1.cmp(b.1))
        };
    mates.sort_by(sort_key);
    opps.sort_by(sort_key);

    let mut put = |base: usize, list: &[(u64, &str, &crate::gym::ObsPlayer)]| {
        for k in 0..K {
            let o = base + k * OTHER_W;
            if let Some((_, _, p)) = list.get(k) {
                f[o] = (p.p.x - sp.x) / POS_SCALE;
                f[o + 1] = (p.p.y - sp.y) / POS_SCALE;
                f[o + 2] = (p.p.z - sp.z) / POS_SCALE;
                f[o + 3] = p.v.x / VEL_SCALE;
                f[o + 4] = p.v.y / VEL_SCALE;
                f[o + 5] = p.v.z / VEL_SCALE;
            }
        }
    };
    put(32, &mates);
    put(32 + K * OTHER_W, &opps);

    // constant bias feature
    f[FEAT_W - 1] = 1.0;
    f
}

/// The shared MLP. Weights are the flat trainable vector (length
/// `PARAM_W`); the forward pass is a pure deterministic f64 fold (tanh
/// hidden activations — no `exp`, no divergence risk).
#[derive(Clone)]
pub struct RlPolicy {
    pub weights: Vec<f64>,
}

impl RlPolicy {
    /// A zero-weight policy (used only as a constructor base; the trainer
    /// seeds real weights). Forward of all-zero weights is the constant
    /// zero output ⇒ a valid, idle-ish `PlayerInput`.
    pub fn zeros() -> Self {
        RlPolicy { weights: vec![0.0; PARAM_W] }
    }

    pub fn from_weights(w: Vec<f64>) -> Self {
        assert_eq!(w.len(), PARAM_W, "RlPolicy weight length must be PARAM_W");
        RlPolicy { weights: w }
    }

    /// Pure deterministic forward pass: FEAT_W → HID → HID → OUT_W with
    /// tanh hidden activations. Slices the flat weight vector in a fixed
    /// order (no allocation per layer beyond the activation buffers).
    pub fn forward(&self, x: &[f64; FEAT_W]) -> [f64; OUT_W] {
        let w = &self.weights;
        let mut o = 0usize;

        // Layer 1: HID x FEAT_W + HID bias, tanh.
        let mut h1 = [0.0_f64; HID];
        for (i, hv) in h1.iter_mut().enumerate() {
            let mut acc = 0.0;
            let row = o + i * FEAT_W;
            for j in 0..FEAT_W {
                acc += w[row + j] * x[j];
            }
            *hv = acc;
        }
        o += HID * FEAT_W;
        for (i, hv) in h1.iter_mut().enumerate() {
            *hv = (*hv + w[o + i]).tanh();
        }
        o += HID;

        // Layer 2: HID x HID + HID bias, tanh.
        let mut h2 = [0.0_f64; HID];
        for (i, hv) in h2.iter_mut().enumerate() {
            let mut acc = 0.0;
            let row = o + i * HID;
            for j in 0..HID {
                acc += w[row + j] * h1[j];
            }
            *hv = acc;
        }
        o += HID * HID;
        for (i, hv) in h2.iter_mut().enumerate() {
            *hv = (*hv + w[o + i]).tanh();
        }
        o += HID;

        // Layer 3: OUT_W x HID + OUT_W bias, linear (decode applies the
        // per-head squashing).
        let mut out = [0.0_f64; OUT_W];
        for (i, ov) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            let row = o + i * HID;
            for j in 0..HID {
                acc += w[row + j] * h2[j];
            }
            *ov = acc;
        }
        o += OUT_W * HID;
        for (i, ov) in out.iter_mut().enumerate() {
            *ov += w[o + i];
        }
        out
    }

    /// Decode raw outputs into the EXACT `ai::PlayerInput` contract for
    /// `self_id`. Every field is always valid: aim/fire-dir normalized,
    /// reel ∈ {-1,0,1} (argmax of 3 logits), bools thresholded,
    /// throw_charge ∈ [0,1], throw_spin ∈ [-1,1], thrumbler ∈ [-1,1]^3.
    pub fn decode(&self, out: &[f64; OUT_W], self_id: &str, self_p: Vec3) -> ai::PlayerInput {
        let mut a = ai::PlayerInput::idle(self_id);

        // aim: normalized 3-vec head.
        a.aim = norm3(Vec3::new(out[0], out[1], out[2]));

        // reel: argmax of a 3-logit head → {-1, 0, +1}.
        let reel_logits = [out[3], out[4], out[5]];
        let mut bi = 0usize;
        for i in 1..3 {
            if reel_logits[i] > reel_logits[bi] {
                bi = i;
            }
        }
        a.reel = (bi as i32) - 1;

        // fire: gated bool + a target point head.
        if out[6] > 0.0 {
            let dir = norm3(Vec3::new(out[7], out[8], out[9]));
            a.fire_line_at = Some(Vec3::new(
                self_p.x + dir.x * FIRE_RANGE,
                self_p.y + dir.y * FIRE_RANGE,
                self_p.z + dir.z * FIRE_RANGE,
            ));
        }

        // thresholded tanh bool heads.
        a.release = out[10].tanh() > 0.0;
        a.pushoff = out[11].tanh() > 0.0;
        a.catch_intent = out[12].tanh() > 0.0;

        // bounded continuous heads.
        a.throw_charge = 0.5 * (out[13].tanh() + 1.0); // [0,1]
        a.throw_released = a.throw_charge > 0.6 && out[13] > 0.0;
        a.throw_spin = out[14].tanh(); // [-1,1]
        a.thrumbler = Vec3::new(out[15].tanh(), out[16].tanh(), out[17].tanh());

        a
    }

    /// Convenience: featurize + forward + decode for one rigger.
    pub fn act(&self, obs: &Observation, self_id: &str) -> ai::PlayerInput {
        let self_p = obs
            .players
            .iter()
            .find(|p| p.id == self_id)
            .map(|p| p.p)
            .unwrap_or(Vec3::new(0.0, 0.0, 0.0));
        let feat = featurize(obs, self_id);
        let out = self.forward(&feat);
        self.decode(&out, self_id, self_p)
    }
}

//! rl/attention.rs — Entity-attention transformer policy (MAPPO actor).
//!
//! DECENTRALIZED ACTOR: each agent sees ONLY its own observation (the same
//! `Observation` the MLP policy sees). Entities (self, K teammates, K
//! opponents, bell) attend to each other via multi-head self-attention,
//! then the self-entity's output embedding drives the same action heads as
//! the MLP policy. DETERMINISTIC: entity ordering is fixed (self first,
//! teammates sorted by id, opponents sorted by id, bell last) — no
//! permutation variance, no HashMap iteration.
//!
//! WASM-SAFE: the forward pass is pure f64 matmuls + stable softmax (max-
//! subtraction before exp). No rand, no rayon, no clock, no external deps.
//! Compiles into BOTH the native lib AND the wasm cdylib.
//!
//! ── Architecture ────────────────────────────────────────────────────────
//! Entity count:  N_ENTITIES = 1 (self) + K (teammates) + K (opponents)
//!                           + 1 (bell) = 2*K + 2 = 8
//! Per-entity embed dim: D_ENTITY (after the per-entity linear encoder)
//! Attention: ATTN_LAYERS layers, ATTN_HEADS heads, D_MODEL dim
//! Action head: linear from the self-entity's post-attention embedding
//!              to OUT_W (reusing the MLP's action decode logic).
//!
//! Weight layout (flat Vec<f64>, same contract as RlPolicy):
//!   Entity encoder weights (per-entity-type projections)
//!   Attention layer weights (QKV + output projections per layer)
//!   Action head weights (linear → OUT_W)

use crate::ai::types as ai;
use crate::math::Vec3;
use crate::tuning;

use super::policy::{
    featurize, norm3, sorted_others, Observation, ObsPlayer, K, OUT_W,
};

// ── Architecture constants ─────────────────────────────────────────────────

/// Number of entities in the attention set: self + K mates + K opps + bell.
pub const N_ENTITIES: usize = 2 * K + 2; // = 8

/// Per-entity RAW feature widths (before the linear encoder).
/// Self-entity: pos(3) + vel(3) + role(5) + line_state(5) + assignment(9)
///            + gate_geometry(4) + context(3) = 32
pub const SELF_RAW: usize = 32;
/// Other player entity: rel_pos(3) + rel_vel(3) + role(5) + team(1)
///                     + has_ball(1) + line_attached(1) = 14
pub const OTHER_RAW: usize = 14;
/// Bell entity: rel_pos(3) + rel_vel(3) + held_by_one_hot(3) + possession(2)
///            + gate(3) + phase(7) = 21
pub const BELL_RAW: usize = 21;

/// The max raw width (for the encoder weight matrix — smaller entities
/// just have trailing zeros).
pub const MAX_RAW: usize = SELF_RAW; // = 32

/// Model dimension (the entity embedding after encoding).
pub const D_MODEL: usize = 48;

/// Number of attention layers.
pub const ATTN_LAYERS: usize = 2;

/// Number of attention heads per layer.
pub const ATTN_HEADS: usize = 4;

/// Per-head dimension.
pub const D_HEAD: usize = D_MODEL / ATTN_HEADS; // = 12

// ── Weight count ───────────────────────────────────────────────────────────

/// Entity encoder: MAX_RAW → D_MODEL linear (weight + bias).
const ENCODER_W: usize = D_MODEL * MAX_RAW + D_MODEL;

/// Per attention layer: QKV projection (3 * D_MODEL * D_MODEL + 3*D_MODEL)
/// + output projection (D_MODEL * D_MODEL + D_MODEL).
const PER_LAYER_W: usize =
    3 * D_MODEL * D_MODEL + 3 * D_MODEL + D_MODEL * D_MODEL + D_MODEL;

/// Action head: D_MODEL → OUT_W linear (weight + bias).
const ACTION_HEAD_W: usize = OUT_W * D_MODEL + OUT_W;

/// Total trainable actor parameter count.
pub const ATTN_PARAM_W: usize =
    ENCODER_W + ATTN_LAYERS * PER_LAYER_W + ACTION_HEAD_W;

// ── Normalization scales (shared with policy.rs) ───────────────────────────
const POS_SCALE: f64 = tuning::L;
const VEL_SCALE: f64 = tuning::REEL_PULL_SPEED;
const LEN_SCALE: f64 = tuning::L;
const SCORE_SCALE: f64 = 8.0;
const CLOCK_SCALE: f64 = 8000.0;
const ROLE_W: usize = 5;
const JOB_W: usize = 6;
const FIRE_RANGE: f64 = 60.0;

// ── Entity featurization ───────────────────────────────────────────────────

/// Build the raw feature vector for the self-entity.
fn self_entity_features(obs: &Observation, me: &ObsPlayer) -> [f64; MAX_RAW] {
    let mut f = [0.0_f64; MAX_RAW];
    let sp = me.p;
    // pos (normalized)
    f[0] = sp.x / POS_SCALE;
    f[1] = sp.y / POS_SCALE;
    f[2] = sp.z / POS_SCALE;
    // vel
    f[3] = me.v.x / VEL_SCALE;
    f[4] = me.v.y / VEL_SCALE;
    f[5] = me.v.z / VEL_SCALE;
    // role one-hot (5)
    let r = (me.role as usize).min(ROLE_W - 1);
    f[6 + r] = 1.0;
    // line state (5): anchor_rel(3) + rest_len(1) + attached_flag(1)
    if let Some(anc) = me.line_anchor {
        f[11] = (anc.x - sp.x) / POS_SCALE;
        f[12] = (anc.y - sp.y) / POS_SCALE;
        f[13] = (anc.z - sp.z) / POS_SCALE;
        f[14] = me.line_rest_len.unwrap_or(0.0) / LEN_SCALE;
        f[15] = 1.0;
    }
    // assignment (9): job one-hot(6) + mark_rel_pos(3)
    if let Some(asg) = &me.assignment {
        let j = (asg.job as usize).min(JOB_W - 1);
        f[16 + j] = 1.0;
        if let Some(mid) = &asg.mark_id {
            if let Some(mp) = obs.players.iter().find(|p| &p.id == mid) {
                f[22] = (mp.p.x - sp.x) / POS_SCALE;
                f[23] = (mp.p.y - sp.y) / POS_SCALE;
                f[24] = (mp.p.z - sp.z) / POS_SCALE;
            }
        }
    }
    // gate geometry (4): gate_offset, attack_sign, forward_progress, opp_closing
    f[25] = ((obs.gate_plane_x - sp.x) * obs.attack_sign) / POS_SCALE;
    f[26] = obs.attack_sign;
    f[27] = (sp.x * obs.attack_sign) / POS_SCALE;
    // (opp_closing computed below if we had opps — left at 0 for the
    // self-entity; it's on the bell entity or computed externally)
    f[28] = 0.0;
    // context (3): score_margin, clock, bias
    let ctrl_is_away = !obs.controlled_ids.is_empty()
        && obs.controlled_ids.iter().all(|id| {
            obs.players
                .iter()
                .find(|p| &p.id == id)
                .map(|p| p.team == 1)
                .unwrap_or(false)
        });
    let margin = if ctrl_is_away {
        (obs.score_away - obs.score_home) as f64
    } else {
        (obs.score_home - obs.score_away) as f64
    };
    f[29] = (margin / SCORE_SCALE).clamp(-1.0, 1.0);
    f[30] = ((obs.tick as f64) / CLOCK_SCALE).clamp(0.0, 1.0);
    f[31] = 1.0; // bias
    f
}

/// Build the raw feature vector for another player entity (teammate or opp).
fn other_entity_features(
    obs: &Observation,
    me: &ObsPlayer,
    other: &ObsPlayer,
) -> [f64; MAX_RAW] {
    let mut f = [0.0_f64; MAX_RAW];
    let sp = me.p;
    // relative position
    f[0] = (other.p.x - sp.x) / POS_SCALE;
    f[1] = (other.p.y - sp.y) / POS_SCALE;
    f[2] = (other.p.z - sp.z) / POS_SCALE;
    // relative velocity
    f[3] = (other.v.x - me.v.x) / VEL_SCALE;
    f[4] = (other.v.y - me.v.y) / VEL_SCALE;
    f[5] = (other.v.z - me.v.z) / VEL_SCALE;
    // role one-hot (5)
    let r = (other.role as usize).min(ROLE_W - 1);
    f[6 + r] = 1.0;
    // team (0=same, 1=different)
    f[11] = if other.team == me.team { 0.0 } else { 1.0 };
    // has ball
    f[12] = if obs.bell_held_by.as_deref() == Some(&other.id) {
        1.0
    } else {
        0.0
    };
    // line attached
    f[13] = if other.line_anchor.is_some() { 1.0 } else { 0.0 };
    // Remaining slots [14..MAX_RAW) are zero-padded.
    f
}

/// Build the raw feature vector for the bell entity.
fn bell_entity_features(
    obs: &Observation,
    me: &ObsPlayer,
) -> [f64; MAX_RAW] {
    let mut f = [0.0_f64; MAX_RAW];
    let sp = me.p;
    // relative position
    f[0] = (obs.bell_p.x - sp.x) / POS_SCALE;
    f[1] = (obs.bell_p.y - sp.y) / POS_SCALE;
    f[2] = (obs.bell_p.z - sp.z) / POS_SCALE;
    // relative velocity
    f[3] = (obs.bell_v.x - me.v.x) / VEL_SCALE;
    f[4] = (obs.bell_v.y - me.v.y) / VEL_SCALE;
    f[5] = (obs.bell_v.z - me.v.z) / VEL_SCALE;
    // held-by one-hot: {my-team, opp-team, free}
    match &obs.bell_held_by {
        Some(hid) => match obs.players.iter().find(|p| &p.id == hid) {
            Some(h) if h.team == me.team => f[6] = 1.0,
            Some(_) => f[7] = 1.0,
            None => f[8] = 1.0,
        },
        None => f[8] = 1.0,
    }
    // possession one-hot {home, away}
    if obs.possession == 0 {
        f[9] = 1.0;
    } else {
        f[10] = 1.0;
    }
    // gate one-hot {First, Deep, Mouth}
    let g = (obs.gate as usize).min(2);
    f[11 + g] = 1.0;
    // phase one-hot (7)
    let ph = (obs.phase as usize).min(6);
    f[14 + ph] = 1.0;
    // Remaining slots [21..MAX_RAW) are zero-padded.
    f
}

// ── The Attention Policy ───────────────────────────────────────────────────

/// Entity-attention actor (MAPPO decentralized policy). Shared parameters,
/// each agent sees only its own observation. Weight layout is a flat
/// `Vec<f64>` of length `ATTN_PARAM_W`.
#[derive(Clone)]
pub struct AttentionPolicy {
    pub weights: Vec<f64>,
}

impl AttentionPolicy {
    pub fn zeros() -> Self {
        AttentionPolicy {
            weights: vec![0.0; ATTN_PARAM_W],
        }
    }

    pub fn from_weights(w: Vec<f64>) -> Self {
        assert_eq!(
            w.len(),
            ATTN_PARAM_W,
            "AttentionPolicy weight length must be ATTN_PARAM_W={}",
            ATTN_PARAM_W
        );
        AttentionPolicy { weights: w }
    }

    /// Pure deterministic forward pass: entities → attention → action logits.
    /// Returns raw OUT_W logits (same shape as RlPolicy::forward).
    pub fn forward(&self, obs: &Observation, self_id: &str) -> [f64; OUT_W] {
        let me = match obs.players.iter().find(|p| p.id == self_id) {
            Some(m) => m,
            None => return [0.0; OUT_W],
        };

        // ── Build entity raw features (fixed ordering) ───────────────────
        // Order: [self, teammate_0..K-1, opponent_0..K-1, bell]
        let (mates, opps) = sorted_others(obs, me, self_id);

        let mut raw_entities = [[0.0_f64; MAX_RAW]; N_ENTITIES];
        raw_entities[0] = self_entity_features(obs, me);
        for k in 0..K {
            if let Some(tm) = mates.get(k) {
                raw_entities[1 + k] = other_entity_features(obs, me, tm);
            }
            // else: zero-padded (absent entity)
        }
        for k in 0..K {
            if let Some(op) = opps.get(k) {
                raw_entities[1 + K + k] = other_entity_features(obs, me, op);
            }
        }
        raw_entities[1 + 2 * K] = bell_entity_features(obs, me);

        let w = &self.weights;
        let mut off = 0usize;

        // ── Entity encoder: MAX_RAW → D_MODEL with tanh ──────────────────
        let mut embeddings = [[0.0_f64; D_MODEL]; N_ENTITIES];
        // Weight: D_MODEL x MAX_RAW matrix + D_MODEL bias
        for e in 0..N_ENTITIES {
            for i in 0..D_MODEL {
                let mut acc = 0.0;
                let row_off = off + i * MAX_RAW;
                for j in 0..MAX_RAW {
                    acc += w[row_off + j] * raw_entities[e][j];
                }
                // bias
                embeddings[e][i] = (acc + w[off + D_MODEL * MAX_RAW + i]).tanh();
            }
        }
        off += ENCODER_W;

        // ── Multi-head self-attention layers ──────────────────────────────
        for _layer in 0..ATTN_LAYERS {
            // QKV projection: 3 * D_MODEL x D_MODEL + 3*D_MODEL bias
            let mut qkv = [[0.0_f64; 3 * D_MODEL]; N_ENTITIES];
            for e in 0..N_ENTITIES {
                for i in 0..(3 * D_MODEL) {
                    let mut acc = 0.0;
                    let row_off = off + i * D_MODEL;
                    for j in 0..D_MODEL {
                        acc += w[row_off + j] * embeddings[e][j];
                    }
                    qkv[e][i] = acc + w[off + 3 * D_MODEL * D_MODEL + i];
                }
            }
            off += 3 * D_MODEL * D_MODEL + 3 * D_MODEL;

            // Multi-head attention: split Q, K, V into heads, compute
            // scaled dot-product attention per head, concat.
            let scale = 1.0 / (D_HEAD as f64).sqrt();
            let mut attn_out = [[0.0_f64; D_MODEL]; N_ENTITIES];

            for h in 0..ATTN_HEADS {
                let hoff = h * D_HEAD;
                // For each query entity, compute attention over all keys.
                for qi in 0..N_ENTITIES {
                    // Compute attention scores (qi attends to all kj).
                    let mut scores = [0.0_f64; N_ENTITIES];
                    let mut max_score = f64::NEG_INFINITY;
                    for kj in 0..N_ENTITIES {
                        let mut dot = 0.0;
                        for d in 0..D_HEAD {
                            // Q from qkv[qi][hoff+d], K from qkv[kj][D_MODEL+hoff+d]
                            dot += qkv[qi][hoff + d]
                                * qkv[kj][D_MODEL + hoff + d];
                        }
                        scores[kj] = dot * scale;
                        if scores[kj] > max_score {
                            max_score = scores[kj];
                        }
                    }
                    // Stable softmax: subtract max, exp, normalize.
                    let mut attn_weights = [0.0_f64; N_ENTITIES];
                    let mut sum_exp = 0.0_f64;
                    for kj in 0..N_ENTITIES {
                        let e = (scores[kj] - max_score).exp();
                        attn_weights[kj] = e;
                        sum_exp += e;
                    }
                    if sum_exp > 0.0 {
                        for kj in 0..N_ENTITIES {
                            attn_weights[kj] /= sum_exp;
                        }
                    }
                    // Weighted sum of values.
                    for d in 0..D_HEAD {
                        let mut val = 0.0;
                        for kj in 0..N_ENTITIES {
                            // V from qkv[kj][2*D_MODEL+hoff+d]
                            val += attn_weights[kj]
                                * qkv[kj][2 * D_MODEL + hoff + d];
                        }
                        attn_out[qi][hoff + d] = val;
                    }
                }
            }

            // Output projection: D_MODEL x D_MODEL + D_MODEL bias, then
            // residual connection + tanh activation.
            for e in 0..N_ENTITIES {
                let mut proj = [0.0_f64; D_MODEL];
                for i in 0..D_MODEL {
                    let mut acc = 0.0;
                    let row_off = off + i * D_MODEL;
                    for j in 0..D_MODEL {
                        acc += w[row_off + j] * attn_out[e][j];
                    }
                    proj[i] = acc + w[off + D_MODEL * D_MODEL + i];
                }
                // Residual + tanh
                for i in 0..D_MODEL {
                    embeddings[e][i] = (embeddings[e][i] + proj[i]).tanh();
                }
            }
            off += D_MODEL * D_MODEL + D_MODEL;
        }

        // ── Action head: self-entity embedding → OUT_W ───────────────────
        // Entity 0 is always self.
        let self_emb = &embeddings[0];
        let mut out = [0.0_f64; OUT_W];
        for i in 0..OUT_W {
            let mut acc = 0.0;
            let row_off = off + i * D_MODEL;
            for j in 0..D_MODEL {
                acc += w[row_off + j] * self_emb[j];
            }
            out[i] = acc + w[off + OUT_W * D_MODEL + i];
        }

        debug_assert_eq!(off + ACTION_HEAD_W, ATTN_PARAM_W);
        out
    }

    /// Decode raw outputs into `ai::PlayerInput` — reuses the same decode
    /// logic as the MLP policy (aim normalize, reel argmax, fire gate,
    /// relational pass head, etc.).
    ///
    /// HIERARCHICAL INTENTS: when `out[INTENT_GATE_IDX] > 0`, the argmax
    /// over `out[INTENT_LOGITS_BASE..INTENT_LOGITS_BASE+N_INTENTS]` selects
    /// a high-level tactical intent. The intent executor calls proven
    /// baseline controller geometry (lead solver, gate solver, planner) to
    /// produce the full `PlayerInput`. If the executor returns `None`
    /// (solver failure, missing data), the low-level heads are used.
    pub fn decode(
        &self,
        out: &[f64; OUT_W],
        obs: &Observation,
        self_id: &str,
    ) -> ai::PlayerInput {
        use super::policy::{INTENT_GATE_IDX, INTENT_LOGITS_BASE, N_INTENTS};

        // ── Intent gate check ───────────────────────────────────────────
        if out[INTENT_GATE_IDX] > 0.0 {
            let mut best = 0usize;
            for i in 1..N_INTENTS {
                if out[INTENT_LOGITS_BASE + i] > out[INTENT_LOGITS_BASE + best] {
                    best = i;
                }
            }
            if let Some(action) = execute_intent(best, obs, self_id) {
                return action;
            }
        }

        // ── Low-level decode (unchanged) ────────────────────────────────
        let mut a = ai::PlayerInput::idle(self_id);
        let me = obs.players.iter().find(|p| p.id == self_id);
        let self_p = me.map(|p| p.p).unwrap_or(Vec3::new(0.0, 0.0, 0.0));

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
        a.throw_charge = 0.5 * (out[13].tanh() + 1.0);
        a.throw_released = a.throw_charge > 0.6 && out[13] > 0.0;
        a.throw_spin = out[14].tanh();
        a.thrumbler = Vec3::new(out[15].tanh(), out[16].tanh(), out[17].tanh());

        // ── relational pass head ──────────────────────────────────────
        if out[18] > 0.0 {
            if let Some(me) = me {
                let (mates, _) = sorted_others(obs, me, self_id);
                let mut sk = 0usize;
                for k in 1..K {
                    if out[19 + k] > out[19 + sk] {
                        sk = k;
                    }
                }
                if let Some(tm) = mates.get(sk) {
                    if let Some(lr) =
                        crate::ai::lead_predict::solve_lead_velocity(
                            self_p,
                            Self::PASS_SPEED,
                            tm.p,
                            tm.v,
                            crate::tuning::OMEGA,
                        )
                    {
                        a.aim = norm3(Vec3::new(
                            lr.intercept.x - self_p.x,
                            lr.intercept.y - self_p.y,
                            lr.intercept.z - self_p.z,
                        ));
                        a.fire_line_at = Some(lr.intercept);
                        a.throw_charge = a.throw_charge.max(0.8);
                        a.throw_released = true;
                        a.catch_intent = true;
                    }
                }
            }
        }

        a
    }

    const PASS_SPEED: f64 = VEL_SCALE;

    /// Convenience: forward + decode for one rigger.
    pub fn act(&self, obs: &Observation, self_id: &str) -> ai::PlayerInput {
        let out = self.forward(obs, self_id);
        self.decode(&out, obs, self_id)
    }
}

// ── Hierarchical intent execution ─────────────────────────────────────────
//
// Pure, deterministic, wasm-safe intent executors. Each takes the obs and
// self_id, uses existing geometry solvers to produce a complete PlayerInput.
// Returns None on solver failure → the low-level heads are used instead.

const INTENT_PASS_SPEED: f64 = VEL_SCALE; // 26 m/s

fn execute_intent(
    intent_idx: usize,
    obs: &Observation,
    self_id: &str,
) -> Option<ai::PlayerInput> {
    match intent_idx {
        0 => intent_nav_advance(obs, self_id),
        1 => intent_pass_gate_receiver(obs, self_id),
        2 => intent_pass_best(obs, self_id),
        3 => intent_catch_commit(obs, self_id),
        4 => intent_receive_stage(obs, self_id),
        5 => intent_mark_nearest(obs, self_id),
        6 => intent_spread_coverage(obs, self_id),
        _ => None,
    }
}

/// Reconstruct a minimal planner SimState from the observation.
fn obs_to_planner_state(obs: &Observation) -> crate::planner::SimState {
    crate::planner::SimState {
        omega: tuning::OMEGA,
        tick: obs.tick as i64,
        players: obs
            .players
            .iter()
            .map(|p| crate::planner::PlayerSim {
                id: p.id.clone(),
                team: p.team as i32,
                p: p.p,
                v: p.v,
            })
            .collect(),
    }
}

/// Build a PlayerInput by navigating toward a target using the grapple planner.
fn navigate_intent(
    obs: &Observation,
    self_id: &str,
    target: Vec3,
    catch_intent: bool,
) -> Option<ai::PlayerInput> {
    let me = obs.players.iter().find(|p| p.id == self_id)?;
    let state = obs_to_planner_state(obs);
    let player = crate::planner::PlayerSim {
        id: self_id.to_string(),
        team: me.team as i32,
        p: me.p,
        v: me.v,
    };
    let plan = crate::planner::plan_grapple(&player, target, &state, true, None, 0)?;
    let aim = norm3(Vec3::new(
        plan.anchor_pos.x - me.p.x,
        plan.anchor_pos.y - me.p.y,
        plan.anchor_pos.z - me.p.z,
    ));
    Some(ai::PlayerInput {
        id: self_id.to_string(),
        aim,
        fire_line_at: Some(plan.anchor_pos),
        reel: plan.reel,
        release: false,
        pushoff: false,
        throw_charge: 0.0,
        throw_released: false,
        throw_spin: 0.0,
        thrumbler: Vec3::new(0.0, 0.0, 0.0),
        catch_intent,
    })
}

/// Intent 0: Navigate forward toward the attack ring (carrier behavior).
fn intent_nav_advance(obs: &Observation, self_id: &str) -> Option<ai::PlayerInput> {
    let me = obs.players.iter().find(|p| p.id == self_id)?;
    let sgn = obs.attack_sign;
    let step = 110.0_f64.min((obs.gate_plane_x - me.p.x).abs() + 40.0);
    let ahead_x = me.p.x + sgn * step;
    let r = (me.p.y * me.p.y + me.p.z * me.p.z).sqrt();
    let target_r = r.min(6.0);
    let yz_len = if r > 1e-6 { r } else { 1.0 };
    let target = Vec3::new(
        ahead_x,
        (me.p.y / yz_len) * target_r,
        (me.p.z / yz_len) * target_r,
    );
    navigate_intent(obs, self_id, target, false)
}

/// Intent 1: Execute a gate-clearing throw using the closed-form gate solver.
fn intent_pass_gate_receiver(obs: &Observation, self_id: &str) -> Option<ai::PlayerInput> {
    let me = obs.players.iter().find(|p| p.id == self_id)?;
    if obs.bell_held_by.as_deref() != Some(self_id) {
        return None;
    }
    let team = if obs.attack_sign > 0.0 {
        crate::ai::types::TeamSide::Home
    } else {
        crate::ai::types::TeamSide::Away
    };
    let gs = crate::ai::gate_solve::solve_gate_throw(
        me.p,
        team,
        tuning::OMEGA,
        24.0,
        0.5,
        me.v,
    )?;
    let aim = norm3(gs.throw_vec);
    let charge = ((gs.release_speed - crate::ai::gate_solve::THROW_MIN_SPEED)
        / (crate::ai::gate_solve::THROW_MAX_SPEED - crate::ai::gate_solve::THROW_MIN_SPEED))
        .clamp(0.0, 1.0);
    Some(ai::PlayerInput {
        id: self_id.to_string(),
        aim,
        fire_line_at: None,
        reel: 0,
        release: false,
        pushoff: false,
        throw_charge: charge,
        throw_released: true,
        throw_spin: -0.5,
        thrumbler: Vec3::new(0.0, 0.0, 0.0),
        catch_intent: false,
    })
}

/// Intent 2: Execute an advancing pass to the best-positioned teammate.
fn intent_pass_best(obs: &Observation, self_id: &str) -> Option<ai::PlayerInput> {
    let me = obs.players.iter().find(|p| p.id == self_id)?;
    if obs.bell_held_by.as_deref() != Some(self_id) {
        return None;
    }
    let sgn = obs.attack_sign;
    let (mates, _) = sorted_others(obs, me, self_id);
    // Pick the furthest-forward teammate.
    let target = mates
        .iter()
        .max_by(|a, b| {
            let fa = a.p.x * sgn;
            let fb = b.p.x * sgn;
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        })?;
    let lr = crate::ai::lead_predict::solve_lead_velocity(
        me.p,
        INTENT_PASS_SPEED,
        target.p,
        target.v,
        tuning::OMEGA,
    )?;
    let aim = norm3(Vec3::new(
        lr.intercept.x - me.p.x,
        lr.intercept.y - me.p.y,
        lr.intercept.z - me.p.z,
    ));
    let dist = ((lr.intercept.x - me.p.x).powi(2)
        + (lr.intercept.y - me.p.y).powi(2)
        + (lr.intercept.z - me.p.z).powi(2))
    .sqrt();
    let charge = ((dist / lr.flight_time - crate::ai::gate_solve::THROW_MIN_SPEED)
        / (crate::ai::gate_solve::THROW_MAX_SPEED - crate::ai::gate_solve::THROW_MIN_SPEED))
        .clamp(0.3, 1.0);
    Some(ai::PlayerInput {
        id: self_id.to_string(),
        aim,
        fire_line_at: None,
        reel: 0,
        release: false,
        pushoff: false,
        throw_charge: charge,
        throw_released: true,
        throw_spin: 0.0,
        thrumbler: Vec3::new(0.0, 0.0, 0.0),
        catch_intent: false,
    })
}

/// Intent 3: Dive toward the bell's predicted intercept point.
fn intent_catch_commit(obs: &Observation, self_id: &str) -> Option<ai::PlayerInput> {
    let me = obs.players.iter().find(|p| p.id == self_id)?;
    if obs.bell_held_by.is_some() {
        return None;
    }
    // Predict bell intercept using the lead solver in reverse (where will
    // the ball be when I can reach it?). Simple: aim at bell_p + lead based
    // on bell velocity, capped horizon.
    let gap = ((obs.bell_p.x - me.p.x).powi(2)
        + (obs.bell_p.y - me.p.y).powi(2)
        + (obs.bell_p.z - me.p.z).powi(2))
    .sqrt();
    let close_v = 24.0;
    let t_lead = (gap / close_v).max(0.1).min(2.5);
    let intercept = Vec3::new(
        obs.bell_p.x + obs.bell_v.x * t_lead,
        obs.bell_p.y + obs.bell_v.y * t_lead,
        obs.bell_p.z + obs.bell_v.z * t_lead,
    );
    navigate_intent(obs, self_id, intercept, true)
}

/// Intent 4: Hold the gate-receiver staging position.
fn intent_receive_stage(obs: &Observation, self_id: &str) -> Option<ai::PlayerInput> {
    let sgn = obs.attack_sign;
    let gate_lead = 22.0;
    let stage_x = obs.gate_plane_x + sgn * gate_lead;
    let r = 5.0;
    let ang = std::f64::consts::PI * 0.5;
    let target = Vec3::new(stage_x, r * ang.cos(), r * ang.sin());
    navigate_intent(obs, self_id, target, false)
}

/// Intent 5: Track the nearest opponent (defensive positioning).
fn intent_mark_nearest(obs: &Observation, self_id: &str) -> Option<ai::PlayerInput> {
    let me = obs.players.iter().find(|p| p.id == self_id)?;
    let (_, opps) = sorted_others(obs, me, self_id);
    let opp = opps.first()?;
    // Position between the opponent and the bell.
    let to_bell = Vec3::new(
        obs.bell_p.x - opp.p.x,
        obs.bell_p.y - opp.p.y,
        obs.bell_p.z - opp.p.z,
    );
    let d = (to_bell.x * to_bell.x + to_bell.y * to_bell.y + to_bell.z * to_bell.z).sqrt();
    let standoff = 6.0;
    let target = if d > 1e-6 {
        Vec3::new(
            opp.p.x + to_bell.x / d * standoff,
            opp.p.y + to_bell.y / d * standoff,
            opp.p.z + to_bell.z / d * standoff,
        )
    } else {
        opp.p
    };
    navigate_intent(obs, self_id, target, false)
}

/// Intent 6: Spread into an anti-clump coverage position.
fn intent_spread_coverage(obs: &Observation, self_id: &str) -> Option<ai::PlayerInput> {
    let me = obs.players.iter().find(|p| p.id == self_id)?;
    let (mates, _) = sorted_others(obs, me, self_id);
    // Compute teammate centroid.
    let n = mates.len().max(1) as f64;
    let cx = mates.iter().map(|m| m.p.x).sum::<f64>() / n;
    let cy = mates.iter().map(|m| m.p.y).sum::<f64>() / n;
    let cz = mates.iter().map(|m| m.p.z).sum::<f64>() / n;
    // Move away from centroid, biased forward along attack axis.
    let sgn = obs.attack_sign;
    let away = Vec3::new(me.p.x - cx, me.p.y - cy, me.p.z - cz);
    let away_len = (away.x * away.x + away.y * away.y + away.z * away.z).sqrt();
    let spread_dist = 40.0;
    let target = if away_len > 1e-6 {
        Vec3::new(
            me.p.x + away.x / away_len * spread_dist * 0.5 + sgn * 30.0,
            me.p.y + away.y / away_len * spread_dist,
            me.p.z + away.z / away_len * spread_dist,
        )
    } else {
        Vec3::new(me.p.x + sgn * 60.0, me.p.y + 20.0, me.p.z)
    };
    // Clamp inside the cylinder skin.
    let skin_r = tuning::R - 2.0;
    let rho = (target.y * target.y + target.z * target.z).sqrt();
    let target = if rho > skin_r && rho > 1e-6 {
        let s = skin_r / rho;
        Vec3::new(target.x, target.y * s, target.z * s)
    } else {
        target
    };
    navigate_intent(obs, self_id, target, false)
}

// ── Weight (de)serialization ───────────────────────────────────────────────

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{:?}", x)
    } else {
        "0.0".to_string()
    }
}

/// Serialize an attention policy weight vector to artifact JSON.
pub fn attn_weights_to_json(
    weights: &[f64],
    seed: u64,
    config: &str,
    baseline: f64,
    fitness: f64,
) -> String {
    let ws: Vec<String> = weights.iter().map(|w| jnum(*w)).collect();
    format!(
        "{{\"type\":\"attention\",\"seed\":{},\"config\":{:?},\"dims\":{{\"entities\":{},\"d_model\":{},\"layers\":{},\"heads\":{},\"out\":{},\"param\":{}}},\"baseline\":{},\"fitness\":{},\"weights\":[{}]}}",
        seed,
        config,
        N_ENTITIES,
        D_MODEL,
        ATTN_LAYERS,
        ATTN_HEADS,
        OUT_W,
        ATTN_PARAM_W,
        jnum(baseline),
        jnum(fitness),
        ws.join(",")
    )
}

/// Parse an attention policy weight vector from artifact JSON. Returns
/// `None` unless the weights array is exactly `ATTN_PARAM_W` long.
pub fn attn_weights_from_json(src: &str) -> Option<Vec<f64>> {
    let needle = "\"weights\":";
    let start = src.find(needle)? + needle.len();
    let rest = src[start..].trim_start();
    let rest = rest.strip_prefix('[')?;
    let end = rest.find(']')?;
    let inner = &rest[..end];
    let mut out = Vec::with_capacity(ATTN_PARAM_W);
    for tok in inner.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.parse::<f64>().ok()?);
    }
    if out.len() == ATTN_PARAM_W {
        Some(out)
    } else {
        None
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    use crate::gym::{Env, RigEnv, Scenario, RewardConfig};

    /// Deterministic non-trivial weights for testing.
    fn test_weights() -> Vec<f64> {
        (0..ATTN_PARAM_W)
            .map(|i| ((i % 23) as f64 - 11.0) * 0.007)
            .collect()
    }

    /// Attention forward determinism: same weights + same obs → bit-identical
    /// action (the `.to_bits()` contract).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn attention_forward_is_deterministic() {
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

        let pol = AttentionPolicy::from_weights(test_weights());

        let out1 = pol.forward(&obs, "H1");
        let out2 = pol.forward(&obs, "H1");
        for k in 0..OUT_W {
            assert_eq!(
                out1[k].to_bits(),
                out2[k].to_bits(),
                "attention forward not bit-stable at index {k}"
            );
        }

        let a1 = pol.act(&obs, "H1");
        let a2 = pol.act(&obs, "H1");
        assert_eq!(a1.aim.x.to_bits(), a2.aim.x.to_bits());
        assert_eq!(a1.aim.y.to_bits(), a2.aim.y.to_bits());
        assert_eq!(a1.aim.z.to_bits(), a2.aim.z.to_bits());
        assert_eq!(a1.reel, a2.reel);
        assert_eq!(a1.throw_charge.to_bits(), a2.throw_charge.to_bits());
        assert_eq!(a1.throw_spin.to_bits(), a2.throw_spin.to_bits());
    }

    /// Valid action shape: aim unit-ish, reel in {-1,0,1}, ranges valid.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn attention_action_shape_valid() {
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
        let pol = AttentionPolicy::from_weights(test_weights());

        for id in &["H1", "H2", "H3", "H4"] {
            let a = pol.act(&obs, id);
            let m =
                (a.aim.x.powi(2) + a.aim.y.powi(2) + a.aim.z.powi(2)).sqrt();
            assert!(
                m <= 1.0 + 1e-9 && (m == 0.0 || (m - 1.0).abs() < 1e-9),
                "aim not unit for {id}: mag={m}"
            );
            assert!(
                a.reel == -1 || a.reel == 0 || a.reel == 1,
                "reel out of range for {id}"
            );
            assert!((0.0..=1.0).contains(&a.throw_charge));
            assert!((-1.0..=1.0).contains(&a.throw_spin));
            assert!(a.thrumbler.x.is_finite());
            assert!(a.thrumbler.y.is_finite());
            assert!(a.thrumbler.z.is_finite());
            if let Some(f) = a.fire_line_at {
                assert!(f.x.is_finite() && f.y.is_finite() && f.z.is_finite());
            }
        }
    }

    /// The policy works through the gym: drive 100 steps, no panic, reward
    /// finite.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn attention_policy_drives_gym_100_steps() {
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
            max_ticks: 200,
            reward_config: RewardConfig::default(),
        };
        let mut obs = env.reset(123, &sc);
        let pol = AttentionPolicy::from_weights(test_weights());

        for _ in 0..100 {
            let acts: Vec<ai::PlayerInput> = ["H1", "H2", "H3", "H4"]
                .iter()
                .map(|id| pol.act(&obs, id))
                .collect();
            let step = env.step(&acts);
            assert!(
                step.reward.total.is_finite(),
                "reward not finite during gym drive"
            );
            obs = step.obs;
            if step.done {
                break;
            }
        }
    }

    /// Weight serialization round-trip.
    #[test]
    fn attn_weight_serialization_roundtrip() {
        let w = test_weights();
        let json = attn_weights_to_json(&w, 42, "test", 0.0, 1.0);
        let loaded = attn_weights_from_json(&json).expect("parse failed");
        assert_eq!(loaded.len(), ATTN_PARAM_W);
        for (a, b) in w.iter().zip(loaded.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// Wrong-dim artifact is rejected.
    #[test]
    fn attn_rejects_wrong_dim() {
        assert!(attn_weights_from_json(r#"{"weights":[1.0,2.0]}"#).is_none());
    }
}

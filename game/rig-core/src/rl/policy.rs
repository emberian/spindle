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
//! WASM-SAFE: `featurize`/`forward`/`decode`/`act` and the
//! `Observation`/`ObsPlayer` view + flat weight (de)serialization are
//! pure f64 with NO rand/rayon/clock, so this module compiles into BOTH
//! the native lib AND the wasm cdylib. The browser drives a team via
//! `policy_wasm::RigPolicy` (which reuses this verbatim). Only the CEM
//! trainer (`rl::train`, rand_chacha/rayon + gym::RigEnv) stays
//! native-only — it lives behind the lib.rs cfg gate.

use crate::ai::types as ai;
use crate::math::Vec3;
use crate::tuning;

// ── Observation view (wasm-safe; the gym re-exports these) ──────────────────
//
// These were the gym's `Observation`/`ObsPlayer`. They are pure data
// (only `Vec3`, which is wasm-safe — `RigAi` already uses it). They live
// here so `featurize` has its input type WITHOUT depending on the
// native-only `gym` module; `gym.rs` `pub use`s them so its public API
// and every existing call site are byte-unchanged.

/// The Director's committed assignment for one rigger, reduced to the
/// wasm-safe codes the policy needs. This is a DERIVED READ (computed by
/// running the pure `ai::director::run_director` over the same public
/// state at the gym/wasm seam) — it is NOT sim physics state and never
/// enters `hash_snapshot`. Both the native gym and the browser
/// `policy_wasm` build it identically (same fixed baseline profile + the
/// same seeded `AiRng`), so the policy sees the SAME assignment-conditioned
/// features in training and inference.
#[derive(Clone, Debug)]
pub struct ObsAssignment {
    /// `Job` discriminant 0..=5: Carry / Recover / Receive / Mark /
    /// Support / Zone (the `ai::decision_types::Job` order).
    pub job: u8,
    /// The rigger this assignment marks/targets (a mark to cover, or the
    /// receiver/outlet to feed), if any.
    pub mark_id: Option<String>,
}

/// One controlled-or-not rigger's observable state (ordered like the
/// underlying `Snapshot::players`).
#[derive(Clone, Debug)]
pub struct ObsPlayer {
    pub id: String,
    /// Home / Away (0/1) and the role discriminant (0..=4), both as the
    /// stable integer codes so the learner needn't import the enums.
    pub team: u8,
    pub role: u8,
    pub p: Vec3,
    pub v: Vec3,
    /// Line anchor world-pos + rest length if a rig line is out, else
    /// `None` (so a policy can see tether state).
    pub line_anchor: Option<Vec3>,
    pub line_rest_len: Option<f64>,
    /// The Director's committed assignment for this rigger (derived read;
    /// `None` if the seam did not resolve one — featurize then zeroes the
    /// assignment block, never panics).
    pub assignment: Option<ObsAssignment>,
}

/// The documented observation. A flat, derived view of the SimWorld
/// snapshot + match state — everything a policy needs, nothing it
/// shouldn't (no internal ramps/contact-springs).
#[derive(Clone, Debug)]
pub struct Observation {
    pub tick: u64,
    pub bell_p: Vec3,
    pub bell_v: Vec3,
    pub bell_held_by: Option<String>,
    /// True while the bell is in a hand (possession is live).
    pub possessed: bool,
    /// Possession team code (0 Home / 1 Away) per the match SM.
    pub possession: u8,
    /// Gate code: 0 First / 1 Deep / 2 Mouth.
    pub gate: u8,
    pub score_home: i64,
    pub score_away: i64,
    /// Match phase code: 0 Set 1 Live 2 Contest 3 Dead 4 InningBreak
    /// 5 Spine 6 Final.
    pub phase: u8,
    pub players: Vec<ObsPlayer>,
    /// The ids the external agent drives, in action order (echoed from
    /// the scenario so a stateless policy can map actions positionally).
    pub controlled_ids: Vec<String>,
    /// DERIVED gate-geometry of the controlled side (the team whose reward
    /// is scored — `controlled_ids`' team, defaulting Home). Real world
    /// coordinates, not a gate index:
    ///   `attack_sign` ∈ {+1,-1}: the controlled side's downrange dir,
    ///   `gate_plane_x`: signed world-x of the CURRENT cast gate plane the
    ///                   controlled side must drive a completed pass past.
    /// Both are pure functions of (team, match gate) via
    /// `ai::orientation` — a derived read, never sim state. Default
    /// (`attack_sign=1, gate_plane_x=0`) if the seam left it unset.
    pub attack_sign: f64,
    pub gate_plane_x: f64,
}

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

/// Number of `Job` discriminants (Carry/Recover/Receive/Mark/Support/Zone).
const JOB_W: usize = 6;
/// Number of `RiggerRole` discriminants (Anchor/Spinner/Faithwing/
/// Freewing/Reach).
const ROLE_W: usize = 5;

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
///   ── richer (assignment / role / gate-geometry / context) block ──
///   [68..74) Director Job one-hot (JOB_W=6) of self's assignment
///            (all-zero if no assignment resolved)
///   [74..77) assignment mark/target rel p / POS_SCALE (the marked /
///            fed rigger's position relative to self; 0 if no mark)
///   [77..82) self RiggerRole one-hot (ROLE_W=5)
///   [82]     attack-gate plane offset: (gate_plane_x − self.x) ·
///            attack_sign / POS_SCALE — signed *downrange* distance to
///            the cast-gate plane the controlled side must clear (real
///            geometry, NOT a gate index)
///   [83]     controlled-side attack_sign ∈ {+1,−1}
///   [84]     self forward progress: self.x · attack_sign / POS_SCALE
///   [85]     nearest-opponent closing speed onto the bell: that
///            opponent's velocity component toward the bell / VEL_SCALE
///            (0 if no opponent)
///   [86]     score margin from the controlled side:
///            (score_ctrl − score_opp) / SCORE_SCALE, clamped
///   [87]     clock context: tick / CLOCK_SCALE, clamped to [0,1]
///   [88]     constant bias 1.0
pub const FEAT_W: usize =
    32 + 2 * K * OTHER_W + JOB_W + 3 + ROLE_W + 4 + 2 + 1; // = 89

/// Score normalization (a few points decides a match — keep O(1)).
const SCORE_SCALE: f64 = 8.0;
/// Clock normalization (a real episode budget is thousands of ticks; the
/// learner only needs an O(1) "how late" signal, saturated past the cap).
const CLOCK_SCALE: f64 = 8000.0;
/// Index where the richer block begins (right after the K-other blocks).
const RICH_BASE: usize = 32 + 2 * K * OTHER_W; // = 68

/// Raw network outputs before decode.
///   [0..3)   aim head (normalized to a unit-ish 3-vec)
///   [3..6)   reel logits (argmax → -1 / 0 / +1)
///   [6]      fire gate logit (>0 ⇒ fire this tick)
///   [7..10)  fire direction head (normalized; target = self.p + dir*range)
///   [10]     release logit (tanh>0)
///   [11]     pushoff logit (tanh>0)
///   [12]     catch_intent logit (tanh>0)
///   [13]     throw_charge head (squashed to [0,1])
///   [14]     throw_spin head (tanh → [-1,1])
///   [15..18) thrumbler head (tanh per-axis → [-1,1]^3)
///   ── RELATIONAL PASS head (K=3) ──
///   [18]     pass gate logit (>0 ⇒ this tick is a teammate-targeted
///            pass: it OVERRIDES the raw downrange fire point with a
///            Coriolis lead-solved aim at teammate-k's predicted
///            catch point — a relational, coordination-capable action)
///   [19..22) pass-target logits over the K=3 deterministically-sorted
///            teammate slots (argmax → which teammate to feed); if that
///            slot is empty the pass gate is ignored (graceful: falls
///            back to the continuous fire head, every field stays valid)
///   ── HIERARCHICAL INTENT head ──
///   [22]     intent gate logit (>0 ⇒ use high-level intent mode;
///            the baseline controller executes the selected intent)
///   [23..30) intent logits (argmax → which tactical intent to execute)
pub const OUT_W: usize = 18 + 1 + K + 1 + N_INTENTS; // = 30

/// Number of hierarchical intents the attention policy can select.
pub const N_INTENTS: usize = 7;
/// Intent gate logit index in the output vector.
pub const INTENT_GATE_IDX: usize = 22;
/// Base index for the N_INTENTS intent logits.
pub const INTENT_LOGITS_BASE: usize = 23;

/// Hidden layer width (two hidden layers).
pub const HID: usize = 64;

/// The fire target is placed this many metres downrange along the fire
/// direction head when the fire gate fires (O(ring) so it is a reachable
/// grapple point; the sim clamps/ignores it under latency/cooldown).
const FIRE_RANGE: f64 = 60.0;

/// The MLP policy's output width — the low-level heads only (no intents).
/// The MLP does not use the hierarchical intent heads; keeping its param
/// count independent of OUT_W preserves compatibility with committed
/// weight artifacts.
const MLP_OUT_W: usize = 18 + 1 + K; // = 22

/// Total trainable parameter count for the MLP policy.
pub const PARAM_W: usize =
    HID * FEAT_W + HID + HID * HID + HID + MLP_OUT_W * HID + MLP_OUT_W;

#[inline]
fn d2(a: Vec3, b: Vec3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

#[inline]
pub fn norm3(v: Vec3) -> Vec3 {
    let m = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    if m > 1e-9 {
        Vec3::new(v.x / m, v.y / m, v.z / m)
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    }
}

/// Deterministically sort the OTHER riggers into (teammates, opponents)
/// relative to `me`, each ordered by the total key `(distance_bits, id)`
/// — built from the ORDERED `obs.players`, never hash order. Returned by
/// VALUE-INDEX so both `featurize` (feature block) and `decode` (the
/// relational pass head's teammate-k pick) consume the EXACT same slot
/// ordering ⇒ "pass to teammate-k" addresses the same feature the policy
/// saw. Pure: no rng, no clock.
pub fn sorted_others<'a>(
    obs: &'a Observation,
    me: &ObsPlayer,
    self_id: &str,
) -> (Vec<&'a ObsPlayer>, Vec<&'a ObsPlayer>) {
    let sp = me.p;
    let mut mates: Vec<(u64, &str, &ObsPlayer)> = Vec::new();
    let mut opps: Vec<(u64, &str, &ObsPlayer)> = Vec::new();
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
    let sort_key = |a: &(u64, &str, &ObsPlayer), b: &(u64, &str, &ObsPlayer)| {
        a.0.cmp(&b.0).then(a.1.cmp(b.1))
    };
    mates.sort_by(sort_key);
    opps.sort_by(sort_key);
    (
        mates.into_iter().map(|(_, _, p)| p).collect(),
        opps.into_iter().map(|(_, _, p)| p).collect(),
    )
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

    // K nearest teammates / opponents — the SHARED deterministic sort
    // (also consumed by `decode`'s relational pass head).
    let (mates, opps) = sorted_others(obs, me, self_id);

    let mut put = |base: usize, list: &[&ObsPlayer]| {
        for k in 0..K {
            let o = base + k * OTHER_W;
            if let Some(p) = list.get(k) {
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

    // ── richer block (RICH_BASE = 68) ─────────────────────────────────
    let mut o = RICH_BASE;

    // [68..74) Director Job one-hot of self's assignment.
    if let Some(asg) = &me.assignment {
        let j = (asg.job as usize).min(JOB_W - 1);
        f[o + j] = 1.0;
        // [74..77) mark/target rel p (the marked / fed rigger).
        if let Some(mid) = &asg.mark_id {
            if let Some(mp) = obs.players.iter().find(|p| &p.id == mid) {
                f[o + JOB_W] = (mp.p.x - sp.x) / POS_SCALE;
                f[o + JOB_W + 1] = (mp.p.y - sp.y) / POS_SCALE;
                f[o + JOB_W + 2] = (mp.p.z - sp.z) / POS_SCALE;
            }
        }
    }
    o += JOB_W + 3; // → 77

    // [77..82) self RiggerRole one-hot.
    let r = (me.role as usize).min(ROLE_W - 1);
    f[o + r] = 1.0;
    o += ROLE_W; // → 82

    // [82] signed downrange distance to the cast-gate plane the
    // controlled side must clear (real geometry, not a gate index).
    f[o] = ((obs.gate_plane_x - sp.x) * obs.attack_sign) / POS_SCALE;
    // [83] controlled-side attack sign.
    f[o + 1] = obs.attack_sign;
    // [84] self forward progress along the attack axis.
    f[o + 2] = (sp.x * obs.attack_sign) / POS_SCALE;
    // [85] nearest-opponent closing speed onto the bell (its velocity
    // component along the unit vector from that opponent toward the bell).
    if let Some(op) = opps.first() {
        let to_bell = Vec3::new(
            obs.bell_p.x - op.p.x,
            obs.bell_p.y - op.p.y,
            obs.bell_p.z - op.p.z,
        );
        let n = norm3(to_bell);
        f[o + 3] =
            (op.v.x * n.x + op.v.y * n.y + op.v.z * n.z) / VEL_SCALE;
    }
    o += 4; // → 86

    // [86] score margin from the controlled side. The controlled side is
    // the controlled_ids' team (Home unless every controlled id is Away);
    // default Home if none. Clamped to keep O(1).
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
    f[o] = (margin / SCORE_SCALE).clamp(-1.0, 1.0);
    // [87] clock context.
    f[o + 1] = ((obs.tick as f64) / CLOCK_SCALE).clamp(0.0, 1.0);
    o += 2; // → 88

    // [88] constant bias feature.
    debug_assert_eq!(o, FEAT_W - 1);
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

        // Layer 3: MLP_OUT_W x HID + MLP_OUT_W bias, linear (decode applies
        // the per-head squashing). Intent heads [22..30) are zero (the MLP
        // does not produce them; the attention policy does).
        let mut out = [0.0_f64; OUT_W];
        for i in 0..MLP_OUT_W {
            let mut acc = 0.0;
            let row = o + i * HID;
            for j in 0..HID {
                acc += w[row + j] * h2[j];
            }
            out[i] = acc;
        }
        o += MLP_OUT_W * HID;
        for i in 0..MLP_OUT_W {
            out[i] += w[o + i];
        }
        out
    }

    /// A representative throw/grapple speed for the relational lead-solve
    /// (the canon reel/cruise scale — the same O(1) speed `VEL_SCALE`
    /// uses; deterministic, no profile dependence so train==inference).
    const PASS_SPEED: f64 = VEL_SCALE; // 26 m/s

    /// Decode raw outputs into the EXACT `ai::PlayerInput` contract for
    /// `self_id`. Every field is always valid: aim/fire-dir normalized,
    /// reel ∈ {-1,0,1} (argmax of 3 logits), bools thresholded,
    /// throw_charge ∈ [0,1], throw_spin ∈ [-1,1], thrumbler ∈ [-1,1]^3.
    ///
    /// RELATIONAL PASS: when the pass-gate logit `out[18] > 0` AND the
    /// argmax pass-target slot `out[19..22]` indexes an EXISTING teammate
    /// (over the SAME deterministic `sorted_others` mate ordering
    /// `featurize` used), the raw downrange fire point is OVERRIDDEN with
    /// a Coriolis lead-solved aim at that teammate's predicted catch
    /// point (`lead_predict::solve_lead_velocity`, reusing the SACRED
    /// canon integrator — not reinvented), the throw is charged+released
    /// and the aim points at the lead intercept. If the slot is empty or
    /// the solve fails, it gracefully falls back to the continuous fire
    /// head — every `ai::PlayerInput` field stays valid every tick.
    pub fn decode(
        &self,
        out: &[f64; OUT_W],
        obs: &Observation,
        self_id: &str,
    ) -> ai::PlayerInput {
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
        a.throw_charge = 0.5 * (out[13].tanh() + 1.0); // [0,1]
        a.throw_released = a.throw_charge > 0.6 && out[13] > 0.0;
        a.throw_spin = out[14].tanh(); // [-1,1]
        a.thrumbler = Vec3::new(out[15].tanh(), out[16].tanh(), out[17].tanh());

        // ── relational pass head ──────────────────────────────────────
        if out[18] > 0.0 {
            if let Some(me) = me {
                let (mates, _) = sorted_others(obs, me, self_id);
                // argmax over the K pass-target logits → teammate slot.
                let mut sk = 0usize;
                for k in 1..K {
                    if out[19 + k] > out[19 + sk] {
                        sk = k;
                    }
                }
                if let Some(tm) = mates.get(sk) {
                    if let Some(lr) = crate::ai::lead_predict::solve_lead_velocity(
                        self_p,
                        Self::PASS_SPEED,
                        tm.p,
                        tm.v,
                        crate::tuning::OMEGA,
                    ) {
                        // Aim AND fire-target the lead intercept; charge
                        // and release a thrown pass; the teammate is the
                        // committed catcher (widen its catch envelope).
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

    /// Convenience: featurize + forward + decode for one rigger.
    pub fn act(&self, obs: &Observation, self_id: &str) -> ai::PlayerInput {
        let feat = featurize(obs, self_id);
        let out = self.forward(&feat);
        self.decode(&out, obs, self_id)
    }
}

// ── Deterministic flat-weight (de)serialization ─────────────────────────────
//
// The weights artifact is a JSON object so it crosses the same JSON seam
// idiom the rest of the wasm boundary uses (hand-rolled, no serde). Format:
//   {"seed":N,"config":"...","dims":{"feat":F,"out":O,"hid":H,"param":P},
//    "baseline":B,"fitness":G,"weights":[w0,w1,...]}
// Only `weights` is load-bearing for inference; the rest is provenance so
// the asset can be reproduced from its recorded seed/config and cannot
// silently drift (a native test re-runs the recorded config and asserts
// the committed vector matches bit-for-bit).

/// Emit a deterministic f64 — shortest round-tripping repr (`{:?}`),
/// matching `ai_wasm::json_num` exactly so the seam stays one dialect.
fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{:?}", x)
    } else {
        "0.0".to_string()
    }
}

/// Serialize a weight vector to the artifact JSON (provenance fields are
/// caller-supplied; `weights` is the only field inference reads back).
pub fn weights_to_json(
    weights: &[f64],
    seed: u64,
    config: &str,
    baseline: f64,
    fitness: f64,
) -> String {
    let ws: Vec<String> = weights.iter().map(|w| jnum(*w)).collect();
    format!(
        "{{\"seed\":{},\"config\":{:?},\"dims\":{{\"feat\":{},\"out\":{},\"hid\":{},\"param\":{}}},\"baseline\":{},\"fitness\":{},\"weights\":[{}]}}",
        seed,
        config,
        FEAT_W,
        OUT_W,
        HID,
        PARAM_W,
        jnum(baseline),
        jnum(fitness),
        ws.join(",")
    )
}

/// Parse the flat `weights` array out of an artifact JSON blob. Tolerant
/// of field order (anchors on `"weights":[`); returns `None` unless the
/// array length is exactly `PARAM_W` (a wrong-dim asset must not silently
/// drive the sim with garbage). No serde — the same substring-scan idiom
/// as `ai_wasm`/`wasm`.
pub fn weights_from_json(src: &str) -> Option<Vec<f64>> {
    let needle = "\"weights\":";
    let start = src.find(needle)? + needle.len();
    let rest = src[start..].trim_start();
    let rest = rest.strip_prefix('[')?;
    let end = rest.find(']')?;
    let inner = &rest[..end];
    let mut out = Vec::with_capacity(PARAM_W);
    for tok in inner.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.parse::<f64>().ok()?);
    }
    if out.len() == PARAM_W {
        Some(out)
    } else {
        None
    }
}

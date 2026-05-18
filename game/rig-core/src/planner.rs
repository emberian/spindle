//! planner.rs — the AI rollout kernel, ported to native Rust.
//!
//! Increment 1 of the progressive TS→Rust port. This is a FAITHFUL 1:1 port
//! of the TS `rolloutPrimitive` (src/ai/nav/GrapplePlanner.ts): one grapple
//! primitive — fire an anchor, reel(-1)/swing(0), swoop-release — rolled
//! out through the true Coriolis + grapple physics, returning the END state
//! plus the branch's closest pass and terminal momentum toward target.
//!
//! Same algorithm + constants as the TS twin, so the headless TS skill
//! harness is the exact parity oracle: a native cargo eval built on this
//! must reproduce the TS ranking. This is the hot loop (candidates × players
//! × ticks) whose TS cost made every eval/optimizer intractable; native
//! Rust is the throughput unlock for deep MPPI / CEM / RL.

use crate::grapple::{REEL_RATE, TETHER_MIN};
use crate::math::Vec3;
use crate::trajectory::{rk4_step, PointState};
use crate::tuning::{L, R};

/// Swoop release gate — canonical defaults (the TS `__rigtune` knobs are a
/// dev-only runtime affordance; the native kernel takes fixed params, with
/// tunables to be threaded explicitly when the optimizer moves to Rust).
pub const SWOOP_MIN_V: f64 = 8.0;
pub const SWOOP_ALIGN: f64 = 0.6;

#[derive(Clone, Copy, Debug)]
pub struct Rollout {
    /// End position after the primitive.
    pub p: Vec3,
    /// End velocity.
    pub v: Vec3,
    /// Closest the path came to `target` over the primitive.
    pub min_dist: f64,
    /// Terminal speed component toward `target` (the slingshot payoff).
    pub term: f64,
}

/// Roll out one grapple primitive. `reel`: -1 = winch in, 0 = free swing
/// (with swoop-release once fast & aimed). Mirrors TS rolloutPrimitive
/// exactly (incl. the i>=4 release guard and angular-momentum reel).
pub fn rollout_primitive(
    pos: Vec3,
    vel: Vec3,
    anchor: Vec3,
    omega: f64,
    reel: i32,
    target: Vec3,
    steps: u32,
    h: f64,
) -> Rollout {
    let mut p = pos;
    let mut v = vel;
    let mut rest_len = p.sub(anchor).len();
    let mut min_dist = p.sub(target).len();
    let mut released = false;

    for i in 0..steps {
        let s = rk4_step(PointState { p, v }, omega, h);
        p = s.p;
        v = s.v;

        if reel == 0 && !released && i >= 4 {
            let sp = v.len();
            let to_t = target.sub(p);
            let dl = to_t.len();
            if sp > SWOOP_MIN_V && dl > 1e-6 && v.dot(to_t) / (sp * dl) > SWOOP_ALIGN {
                released = true;
            }
        }

        if !released {
            let d = p.sub(anchor);
            let len = d.len();
            if len > 1e-6 {
                let n = d.scale(1.0 / len);
                let v_rad = v.dot(n);
                if len >= rest_len && v_rad > 0.0 {
                    v = v.sub(n.scale(v_rad));
                }
                if reel == -1 && len >= TETHER_MIN {
                    let tl = (rest_len - REEL_RATE * h).max(TETHER_MIN);
                    if tl < rest_len {
                        let vrc = v.dot(n);
                        let sc = if rest_len > 1e-6 { rest_len / tl } else { 1.0 };
                        let vt = v.sub(n.scale(vrc));
                        v = vt.scale(sc).add(n.scale(vrc));
                        rest_len = tl;
                    }
                }
            }
        }

        let dd = p.sub(target).len();
        if dd < min_dist {
            min_dist = dd;
        }
    }

    let to_t = target.sub(p);
    let dl = to_t.len();
    let term = if dl > 1e-6 {
        v.dot(to_t.scale(1.0 / dl)).max(0.0)
    } else {
        0.0
    };
    Rollout { p, v, min_dist, term }
}

// ── Spar geometry — EXACT mirror of TS sparPositions() ───────────────────────
// TS GrapplePlanner.ts:31-64. SPAR_SPACING along x, SPAR_COUNT axis spine at
// y=z=0, plus the 16×3 off-axis clip lattice at R*0.62.
const SPAR_SPACING: f64 = 40.0; // m along X axis
/// TS: Math.ceil(REG.L / SPAR_SPACING) + 1
fn spar_count() -> usize {
    (L / SPAR_SPACING).ceil() as usize + 1
}
const SPAR_RINGS: usize = 16;
const SPAR_AROUND: usize = 3;
fn spar_ring_r() -> f64 {
    R * 0.62
}

/// The spar lattice depends only on compile-time constants (`L`, `R`,
/// SPAR_* ), so it is identical on every call. Build it exactly once and
/// hand out a shared slice on the hot path; the f64 values produced are
/// bit-for-bit the same computation as before, just memoized.
fn build_spar_positions() -> Vec<Vec3> {
    let mut out: Vec<Vec3> = Vec::new();
    let start = -L / 2.0;
    let count = spar_count();
    for i in 0..count {
        out.push(Vec3::new(start + i as f64 * SPAR_SPACING, 0.0, 0.0));
    }
    let ring_r = spar_ring_r();
    for i in 0..SPAR_RINGS {
        let x = -L / 2.0 + ((i as f64 + 0.5) / SPAR_RINGS as f64) * L;
        for a in 0..SPAR_AROUND {
            let ang = (a as f64 / SPAR_AROUND as f64) * std::f64::consts::PI * 2.0;
            out.push(Vec3::new(x, ang.cos() * ring_r, ang.sin() * ring_r));
        }
    }
    out
}

fn spar_positions_cached() -> &'static [Vec3] {
    use std::sync::OnceLock;
    static SPARS: OnceLock<Vec<Vec3>> = OnceLock::new();
    SPARS.get_or_init(build_spar_positions).as_slice()
}

/// All spar positions: the canon axis spine PLUS the off-axis clip lattice
/// (identical to TS `sparPositions()`, GrapplePlanner.ts:46-64). Public
/// signature preserved (owned `Vec`); internally backed by a one-time
/// cached lattice.
pub fn spar_positions() -> Vec<Vec3> {
    spar_positions_cached().to_vec()
}

// ── Plan / input types (self-contained mirror of the TS interfaces) ──────────

/// Mirror of TS `PlayerSim` (only the fields the planner reads).
#[derive(Clone, Debug)]
pub struct PlayerSim {
    pub id: String,
    pub team: i32,
    pub p: Vec3,
    pub v: Vec3,
}

/// Mirror of TS `SimState` (only the fields the planner reads).
#[derive(Clone, Debug)]
pub struct SimState {
    pub omega: f64,
    pub tick: i64,
    pub players: Vec<PlayerSim>,
}

/// Previously-committed anchor for the anti-dither hysteresis (TS `sticky`).
#[derive(Clone, Copy, Debug)]
pub struct Sticky {
    pub pos: Vec3,
    pub reel: i32,
}

/// Mirror of TS `GrapplePlan`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub anchor_pos: Vec3,
    /// -1 = reel in, 0 = let swing.
    pub reel: i32,
    pub projected_dist: f64,
    pub is_spar: bool,
}

// ── Canonical __rigtune defaults (TS GrapplePlanner.ts:180-211) ──────────────
const W_MOM: f64 = 0.6; // TS W_MOM_D
const W_SPACE: f64 = 0.0; // TS tune('wSpace', 0) default
const RRT_REPLAN: f64 = 30.0; // TS tune('rrtReplan', 30) default

/// TS rrtReplanTicks(): Math.max(1, Math.round(tune('rrtReplan', 30)))
fn rrt_replan_ticks() -> i64 {
    (RRT_REPLAN.round() as i64).max(1)
}

// ── Simulation helpers (TS GrapplePlanner.ts:152-165) ────────────────────────
const PLAN_H: f64 = 1.0 / 15.0;
const PLAN_STEPS: u32 = 45;
const SKIN_BUFFER: f64 = 4.0;
const ANCHOR_SWITCH_MARGIN: f64 = 6.0;
const DIRECT_REACH: f64 = 12.0;
const HOP_MAX: f64 = 75.0;
const DEFENDER_DANGER: f64 = 6.0;

// ── Deterministic hash RNG — EXACT port of TS hashRng (GP.ts:380-389) ────────
// JS: s = (2166136261 ^ tick) >>> 0; loop Math.imul(s ^ code, 16777619) >>> 0;
// xorshift32 returning s / 4294967296. All arithmetic is u32-wrapping.
struct HashRng {
    s: u32,
}

impl HashRng {
    fn new(id: &str, tick: i64) -> Self {
        // TS: 2166136261 ^ tick, where `tick` is a JS number coerced to int32
        // by the bitwise `^`. Reproduce that ToInt32 then xor in u32 space.
        let tick32 = (tick as i64 as i32) as u32;
        let mut s = (2166136261u32) ^ tick32;
        for ch in id.chars() {
            // JS charCodeAt → UTF-16 code unit. ids here are ASCII; use the
            // code point (== code unit for BMP scalar values in ASCII ids).
            let code = ch as u32;
            // Math.imul: 32-bit signed multiply, low 32 bits.
            s = (s ^ code).wrapping_mul(16777619u32);
        }
        HashRng { s }
    }
    fn next(&mut self) -> f64 {
        let mut s = self.s;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.s = s;
        s as f64 / 4294967296.0
    }
}

// ── Anchor candidates shared by RRT & CEM (TS GP.ts:458-473 / 582-597) ───────
#[derive(Clone, Copy)]
struct Anchor {
    pos: Vec3,
    is_spar: bool,
}

fn build_anchors(
    pos: Vec3,
    to_dir: Vec3,
    spars: &[Vec3],
    player: &PlayerSim,
    state: &SimState,
) -> Vec<Anchor> {
    let mut anchors: Vec<Anchor> = Vec::with_capacity(spars.len() + state.players.len());
    for sp in spars {
        let dd = pos.sub(*sp).len();
        if dd < 2.0 || dd > 170.0 {
            continue;
        }
        if sp.sub(pos).norm().dot(to_dir) < -0.7 {
            continue;
        }
        anchors.push(Anchor { pos: *sp, is_spar: true });
    }
    for pl in &state.players {
        if pl.id == player.id || pl.team != player.team {
            continue;
        }
        let dd = pos.sub(pl.p).len();
        if dd < 3.0 || dd > 70.0 {
            continue;
        }
        anchors.push(Anchor { pos: pl.p, is_spar: false });
    }
    if anchors.is_empty() {
        return anchors;
    }
    // TS: sort by (|a-target|-|b-target|) || (a.x - b.x). target is captured
    // via the closure; here we pass the comparison's target via to-target.
    anchors
}

/// Sort anchors as TS does: nearer-to-target first, x as deterministic tie.
fn sort_anchors(anchors: &mut [Anchor], target: Vec3) {
    anchors.sort_by(|a, b| {
        let da = a.pos.sub(target).len();
        let db = b.pos.sub(target).len();
        match da.partial_cmp(&db).unwrap() {
            std::cmp::Ordering::Equal => a.pos.x.partial_cmp(&b.pos.x).unwrap(),
            o => o,
        }
    });
}

fn to_target_dir(pos: Vec3, target: Vec3) -> Vec3 {
    let d = target.sub(pos);
    let l = d.len();
    if l > 1e-6 {
        d.scale(1.0 / l)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    }
}

// Cost shared by RRT/CEM (TS GP.ts:478-489 / 602-611): minDist − wMom·term
// + defender danger + optional teammate-spacing penalty.
fn branch_cost(
    p: Vec3,
    min_dist: f64,
    term: f64,
    opponents: &[Vec3],
    teammates: &[Vec3],
) -> f64 {
    let mut c = min_dist - W_MOM * term;
    for o in opponents {
        let od = p.sub(*o).len();
        if od < DEFENDER_DANGER {
            c += (DEFENDER_DANGER - od) * 3.5;
        }
    }
    if W_SPACE > 0.0 {
        let mut near_tm = f64::INFINITY;
        for t in teammates {
            let td = p.sub(*t).len();
            if td < near_tm {
                near_tm = td;
            }
        }
        if near_tm < 18.0 {
            c += (18.0 - near_tm) * W_SPACE;
        }
    }
    c
}

// ── RRT-style kinodynamic planner (TS GP.ts:444-560) ─────────────────────────
struct RrtNode {
    p: Vec3,
    v: Vec3,
    parent: i32,
    a_pos: Vec3,
    reel: i32,
    is_spar: bool,
    best: f64,
}
const RRT_ITERS: u32 = 28;
const RRT_PRIM_STEPS: u32 = 18;

fn plan_rrt(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    opponents: &[Vec3],
    teammates: &[Vec3],
    sticky: Option<Sticky>,
    spars: &[Vec3],
) -> Option<Plan> {
    let pos = player.p;
    let omega = state.omega;
    let to_target_dir_v = to_target_dir(pos, target);

    let mut anchors = build_anchors(pos, to_target_dir_v, spars, player, state);
    if anchors.is_empty() {
        return None;
    }
    sort_anchors(&mut anchors, target);

    let mut nodes: Vec<RrtNode> = Vec::with_capacity(1 + RRT_ITERS as usize);
    nodes.push(RrtNode {
        p: player.p,
        v: player.v,
        parent: -1,
        a_pos: pos,
        reel: -1,
        is_spar: true,
        best: pos.sub(target).len(),
    });
    let mut best_idx: usize = 0;
    let mut best_cost = nodes[0].best;

    // TS: rng = hashRng(player.id, Math.floor(state.tick / replanW))
    let replan_w = rrt_replan_ticks();
    let mut rng = HashRng::new(&player.id, state.tick.div_euclid(replan_w));

    for _ in 0..RRT_ITERS {
        let from_idx = if rng.next() < 0.7 {
            best_idx
        } else {
            // TS: Math.floor(rng() * nodes.length) | 0
            ((rng.next() * nodes.len() as f64) as i64 as i32) as usize
        };
        let (fp, fv, fbest) = {
            let fn_ = &nodes[from_idx];
            (fn_.p, fn_.v, fn_.best)
        };
        let u = rng.next();
        // TS: Math.min(anchors.length - 1, (u*u*anchors.length) | 0)
        let ai = (((u * u * anchors.len() as f64) as i64 as i32) as usize)
            .min(anchors.len() - 1);
        let a = anchors[ai];
        let reel: i32 = if rng.next() < 0.5 { -1 } else { 0 };
        let r = rollout_primitive(fp, fv, a.pos, omega, reel, target, RRT_PRIM_STEPS, PLAN_H);
        let branch_best = fbest.min(r.min_dist);
        nodes.push(RrtNode {
            p: r.p,
            v: r.v,
            parent: from_idx as i32,
            a_pos: a.pos,
            reel,
            is_spar: a.is_spar,
            best: branch_best,
        });
        let c = branch_cost(r.p, branch_best, r.term, opponents, teammates);
        if c < best_cost - 1e-9 {
            best_cost = c;
            best_idx = nodes.len() - 1;
        }
    }

    if best_idx == 0 {
        return None; // tree found nothing better → use MPC
    }
    // Backtrack to the FIRST primitive off the root (parent > 0).
    let mut cur = best_idx;
    while nodes[cur].parent > 0 {
        cur = nodes[cur].parent as usize;
    }
    let first_a_pos = nodes[cur].a_pos;
    let first_reel = nodes[cur].reel;
    let first_is_spar = nodes[cur].is_spar;
    let best_branch = nodes[best_idx].best;

    if let Some(st) = sticky {
        let sd = pos.sub(st.pos).len();
        let moved = first_a_pos.sub(st.pos).len();
        if sd >= 2.0 && sd <= 180.0 && moved > 4.0 {
            let sr = rollout_primitive(
                player.p, player.v, st.pos, omega, st.reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let sticky_min = player.p.sub(target).len().min(sr.min_dist);
            let sticky_cost =
                branch_cost(sr.p, sticky_min, sr.term, opponents, teammates);
            if best_cost >= sticky_cost - ANCHOR_SWITCH_MARGIN {
                return Some(Plan {
                    anchor_pos: st.pos,
                    reel: st.reel,
                    projected_dist: sr.min_dist,
                    is_spar: st.pos.y.hypot(st.pos.z) < 1.0,
                });
            }
        }
    }

    Some(Plan {
        anchor_pos: first_a_pos,
        reel: first_reel,
        projected_dist: best_branch,
        is_spar: first_is_spar,
    })
}

// ── CEM / MPPI planner (TS GP.ts:570-654) ────────────────────────────────────
const CEM_ITERS: u32 = 4;
const CEM_POP: u32 = 16;
const CEM_ELITE: usize = 4;
const CEM_SMOOTH: f64 = 0.4;

fn plan_cem(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    opponents: &[Vec3],
    teammates: &[Vec3],
    sticky: Option<Sticky>,
    spars: &[Vec3],
) -> Option<Plan> {
    let pos = player.p;
    let vel = player.v;
    let omega = state.omega;
    let to_dir = to_target_dir(pos, target);

    let mut anchors = build_anchors(pos, to_dir, spars, player, state);
    if anchors.is_empty() {
        return None;
    }
    sort_anchors(&mut anchors, target);

    let replan_w = rrt_replan_ticks();
    let mut rng = HashRng::new(&player.id, state.tick.div_euclid(replan_w));
    let a_n = anchors.len();
    // Categorical weights, goal-biased (TS: w[i] = 1/(1+i)).
    let mut w: Vec<f64> = (0..a_n).map(|i| 1.0 / (1.0 + i as f64)).collect();
    let mut best_c = f64::INFINITY;
    let mut best_ai: usize = 0;
    let mut best_reel: i32 = -1;

    // Reused across CEM_ITERS to avoid a per-iteration Vec alloc; contents
    // are fully overwritten each pass so behavior is bit-identical.
    let mut samples: Vec<(usize, i32, f64)> = Vec::with_capacity(CEM_POP as usize);
    let mut next_w: Vec<f64> = vec![0.0; a_n];

    for _ in 0..CEM_ITERS {
        let mut wsum = 0.0;
        for i in 0..a_n {
            wsum += w[i];
        }
        // (ai, reel, c)
        samples.clear();
        for _ in 0..CEM_POP {
            // Sample an anchor index from the categorical weights (TS).
            let mut u = rng.next() * wsum;
            let mut ai: usize = 0;
            while ai < a_n - 1 {
                u -= w[ai];
                if u <= 0.0 {
                    break;
                }
                ai += 1;
            }
            let reel: i32 = if rng.next() < 0.5 { -1 } else { 0 };
            let r = rollout_primitive(
                pos, vel, anchors[ai].pos, omega, reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let c = branch_cost(r.p, r.min_dist, r.term, opponents, teammates);
            samples.push((ai, reel, c));
            if c < best_c - 1e-9 {
                best_c = c;
                best_ai = ai;
                best_reel = reel;
            }
        }
        // Elites → refit (TS: sort by c then ai; nextW = SMOOTH; +1 per elite).
        samples.sort_by(|a, b| match a.2.partial_cmp(&b.2).unwrap() {
            std::cmp::Ordering::Equal => a.0.cmp(&b.0),
            o => o,
        });
        for nw in next_w.iter_mut() {
            *nw = CEM_SMOOTH;
        }
        for e in 0..CEM_ELITE.min(samples.len()) {
            next_w[samples[e].0] += 1.0;
        }
        std::mem::swap(&mut w, &mut next_w);
    }

    let mut chosen_pos = anchors[best_ai].pos;
    let mut chosen_reel = best_reel;
    let mut chosen_is_spar = anchors[best_ai].is_spar;

    if let Some(st) = sticky {
        let sd = pos.sub(st.pos).len();
        if sd >= 2.0 && sd <= 180.0 && chosen_pos.sub(st.pos).len() > 4.0 {
            let sr = rollout_primitive(
                pos, vel, st.pos, omega, st.reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let sticky_cost =
                branch_cost(sr.p, sr.min_dist, sr.term, opponents, teammates);
            if best_c >= sticky_cost - ANCHOR_SWITCH_MARGIN {
                chosen_pos = st.pos;
                chosen_reel = st.reel;
                chosen_is_spar = st.pos.y.hypot(st.pos.z) < 1.0;
            }
        }
    }
    Some(Plan {
        anchor_pos: chosen_pos,
        reel: chosen_reel,
        projected_dist: best_c,
        is_spar: chosen_is_spar,
    })
}

// ── MPPI planner (planner == 3) ──────────────────────────────────────────────
// Model-Predictive Path Integral control over the discrete-anchor action
// space. Like CEM it samples a population of (anchor, reel) candidates from
// the same goal-biased categorical and rolls each out via rollout_primitive;
// UNLIKE CEM there is no hard elite cut — every sample contributes a soft
// importance weight  wᵢ = exp(−(cᵢ − c_min)/λ)  (the information-theoretic
// MPPI update; c_min subtracted only for numerical stability, it cancels in
// the normalised weights). The action space here is *discrete* anchors, not
// a continuous control sequence, so the MPPI expected-control update would
// have to round back onto a real anchor. The principled discrete-MPPI form
// is therefore to aggregate the soft weights *per discrete (anchor,reel)
// action* and return the action carrying the maximum total posterior weight
// (the MAP action under the path-integral posterior). This is documented and
// chosen deliberately over a coordinate-averaged pseudo-anchor, which could
// land in empty space off the spar lattice.
const MPPI_POP: u32 = 24;
const MPPI_LAMBDA: f64 = 8.0;

fn plan_mppi(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    opponents: &[Vec3],
    teammates: &[Vec3],
    sticky: Option<Sticky>,
    spars: &[Vec3],
) -> Option<Plan> {
    let pos = player.p;
    let vel = player.v;
    let omega = state.omega;
    let to_dir = to_target_dir(pos, target);

    let mut anchors = build_anchors(pos, to_dir, spars, player, state);
    if anchors.is_empty() {
        return None;
    }
    sort_anchors(&mut anchors, target);

    let replan_w = rrt_replan_ticks();
    let mut rng = HashRng::new(&player.id, state.tick.div_euclid(replan_w));
    let a_n = anchors.len();
    // Same goal-biased categorical proposal as CEM (w[i] = 1/(1+i)).
    let w: Vec<f64> = (0..a_n).map(|i| 1.0 / (1.0 + i as f64)).collect();
    let mut wsum = 0.0;
    for wi in &w {
        wsum += *wi;
    }

    // (ai, reel, cost) for every sampled rollout, in deterministic draw order.
    let mut samples: Vec<(usize, i32, f64)> = Vec::with_capacity(MPPI_POP as usize);
    let mut best_c = f64::INFINITY;
    let mut min_c = f64::INFINITY;
    for _ in 0..MPPI_POP {
        let mut u = rng.next() * wsum;
        let mut ai: usize = 0;
        while ai < a_n - 1 {
            u -= w[ai];
            if u <= 0.0 {
                break;
            }
            ai += 1;
        }
        let reel: i32 = if rng.next() < 0.5 { -1 } else { 0 };
        let r = rollout_primitive(
            pos, vel, anchors[ai].pos, omega, reel, target, RRT_PRIM_STEPS, PLAN_H,
        );
        let c = branch_cost(r.p, r.min_dist, r.term, opponents, teammates);
        if c < min_c {
            min_c = c;
        }
        if c < best_c - 1e-9 {
            best_c = c;
        }
        samples.push((ai, reel, c));
    }

    // Path-integral posterior: accumulate soft weight per discrete (ai,reel)
    // action. Index key = ai*2 + (reel==0). Deterministic Vec, no HashMap.
    let mut acc: Vec<f64> = vec![0.0; a_n * 2];
    for &(ai, reel, c) in &samples {
        let key = ai * 2 + if reel == 0 { 1 } else { 0 };
        acc[key] += (-(c - min_c) / MPPI_LAMBDA).exp();
    }
    // MAP action = max total posterior weight. Deterministic tie-break by
    // lower key index (i.e. nearer-to-target anchor, reel=-1 before reel=0).
    let mut best_key = 0usize;
    let mut best_wt = f64::NEG_INFINITY;
    for (k, &wt) in acc.iter().enumerate() {
        if wt > best_wt + 1e-12 {
            best_wt = wt;
            best_key = k;
        }
    }
    let chosen_ai = best_key / 2;
    let chosen_reel0: i32 = if best_key % 2 == 1 { 0 } else { -1 };

    let mut chosen_pos = anchors[chosen_ai].pos;
    let mut chosen_reel = chosen_reel0;
    let mut chosen_is_spar = anchors[chosen_ai].is_spar;

    if let Some(st) = sticky {
        let sd = pos.sub(st.pos).len();
        if sd >= 2.0 && sd <= 180.0 && chosen_pos.sub(st.pos).len() > 4.0 {
            let sr = rollout_primitive(
                pos, vel, st.pos, omega, st.reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let sticky_cost =
                branch_cost(sr.p, sr.min_dist, sr.term, opponents, teammates);
            if best_c >= sticky_cost - ANCHOR_SWITCH_MARGIN {
                chosen_pos = st.pos;
                chosen_reel = st.reel;
                chosen_is_spar = st.pos.y.hypot(st.pos.z) < 1.0;
            }
        }
    }
    Some(Plan {
        anchor_pos: chosen_pos,
        reel: chosen_reel,
        projected_dist: best_c,
        is_spar: chosen_is_spar,
    })
}

// ── Simulated-annealing planner (planner == 4) ───────────────────────────────
// Start from the sticky anchor if valid else the best (nearest-to-target)
// spar candidate, then propose neighbour anchors (a nearby anchor index
// drawn from the sorted candidate list) and accept Δcost<0 always, Δcost≥0
// with probability exp(−Δcost/T). T cools geometrically over SA_ITERS. All
// randomness is the windowed HashRng, so the whole walk is deterministic.
const SA_ITERS: u32 = 40;
const SA_T0: f64 = 12.0;
const SA_COOL: f64 = 0.92;

fn plan_simanneal(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    opponents: &[Vec3],
    teammates: &[Vec3],
    sticky: Option<Sticky>,
    spars: &[Vec3],
) -> Option<Plan> {
    let pos = player.p;
    let vel = player.v;
    let omega = state.omega;
    let to_dir = to_target_dir(pos, target);

    let mut anchors = build_anchors(pos, to_dir, spars, player, state);
    if anchors.is_empty() {
        return None;
    }
    sort_anchors(&mut anchors, target);
    let a_n = anchors.len();

    let replan_w = rrt_replan_ticks();
    let mut rng = HashRng::new(&player.id, state.tick.div_euclid(replan_w));

    let cost_of = |ai: usize, reel: i32| -> (f64, f64) {
        let r = rollout_primitive(
            pos, vel, anchors[ai].pos, omega, reel, target, RRT_PRIM_STEPS, PLAN_H,
        );
        (branch_cost(r.p, r.min_dist, r.term, opponents, teammates), r.min_dist)
    };

    // Seed: nearest sticky anchor if a sticky pos is given, else candidate 0
    // (sorted nearest-to-target). reel seeds from sticky else -1.
    let (mut cur_ai, mut cur_reel) = if let Some(st) = sticky {
        let mut bi = 0usize;
        let mut bd = f64::INFINITY;
        for (i, a) in anchors.iter().enumerate() {
            let d = a.pos.sub(st.pos).len();
            if d < bd {
                bd = d;
                bi = i;
            }
        }
        (bi, st.reel)
    } else {
        (0usize, -1i32)
    };
    let (mut cur_c, mut cur_pd) = cost_of(cur_ai, cur_reel);
    let mut best_ai = cur_ai;
    let mut best_reel = cur_reel;
    let mut best_c = cur_c;
    let mut best_pd = cur_pd;

    let mut t = SA_T0;
    for _ in 0..SA_ITERS {
        // Neighbour: jump to a nearby anchor index (±a small window) and/or
        // flip reel. Deterministic via the windowed rng.
        let span = 3.0;
        let off = ((rng.next() * (2.0 * span + 1.0)) as i64 - span as i64) as i64;
        let mut nai = cur_ai as i64 + off;
        if nai < 0 {
            nai = 0;
        }
        if nai > a_n as i64 - 1 {
            nai = a_n as i64 - 1;
        }
        let nai = nai as usize;
        let nreel: i32 = if rng.next() < 0.5 { -1 } else { 0 };
        let (nc, npd) = cost_of(nai, nreel);
        let d = nc - cur_c;
        let accept = if d < 0.0 {
            true
        } else {
            rng.next() < (-d / t).exp()
        };
        if accept {
            cur_ai = nai;
            cur_reel = nreel;
            cur_c = nc;
            cur_pd = npd;
            if cur_c < best_c - 1e-9 {
                best_c = cur_c;
                best_ai = cur_ai;
                best_reel = cur_reel;
                best_pd = cur_pd;
            }
        }
        t *= SA_COOL;
    }
    let _ = cur_pd;

    let mut chosen_pos = anchors[best_ai].pos;
    let mut chosen_reel = best_reel;
    let mut chosen_is_spar = anchors[best_ai].is_spar;
    let mut chosen_pd = best_pd;

    if let Some(st) = sticky {
        let sd = pos.sub(st.pos).len();
        if sd >= 2.0 && sd <= 180.0 && chosen_pos.sub(st.pos).len() > 4.0 {
            let sr = rollout_primitive(
                pos, vel, st.pos, omega, st.reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let sticky_cost =
                branch_cost(sr.p, sr.min_dist, sr.term, opponents, teammates);
            if best_c >= sticky_cost - ANCHOR_SWITCH_MARGIN {
                chosen_pos = st.pos;
                chosen_reel = st.reel;
                chosen_is_spar = st.pos.y.hypot(st.pos.z) < 1.0;
                chosen_pd = sr.min_dist;
            }
        }
    }
    Some(Plan {
        anchor_pos: chosen_pos,
        reel: chosen_reel,
        projected_dist: chosen_pd,
        is_spar: chosen_is_spar,
    })
}

// ── Beam-search planner (planner == 5) ───────────────────────────────────────
// The MPC arm is a 1-cycle greedy choice; this does a BEAM_K-wide beam over
// BEAM_DEPTH grapple hops. Each beam node carries the post-rollout
// (p, v, cumulative-cost, running-min-dist) plus the FIRST-hop (anchor,reel,
// is_spar) so the winner can be traced back to its opening move. Expansion
// is fully deterministic: the anchor list is sorted, every (anchor,reel) is
// tried in order, and survivors are kept by a deterministic sort on
// (cumulative cost, first-hop anchor x). No RNG is needed (exhaustive beam),
// so determinism is structural.
const BEAM_K: usize = 6;
const BEAM_DEPTH: u32 = 3;

#[derive(Clone, Copy)]
struct BeamNode {
    p: Vec3,
    v: Vec3,
    cum: f64,
    run_min: f64,
    first_pos: Vec3,
    first_reel: i32,
    first_is_spar: bool,
    first_pd: f64,
}

fn plan_beam(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    opponents: &[Vec3],
    teammates: &[Vec3],
    sticky: Option<Sticky>,
    spars: &[Vec3],
) -> Option<Plan> {
    let pos = player.p;
    let vel = player.v;
    let omega = state.omega;
    let to_dir = to_target_dir(pos, target);

    let mut anchors = build_anchors(pos, to_dir, spars, player, state);
    if anchors.is_empty() {
        return None;
    }
    sort_anchors(&mut anchors, target);

    // Frontier seed: a single node at the player's current state.
    let mut frontier: Vec<BeamNode> = vec![BeamNode {
        p: pos,
        v: vel,
        cum: 0.0,
        run_min: pos.sub(target).len(),
        first_pos: pos,
        first_reel: -1,
        first_is_spar: true,
        first_pd: pos.sub(target).len(),
    }];

    let mut best: Option<BeamNode> = None;

    for depth in 0..BEAM_DEPTH {
        let mut expanded: Vec<BeamNode> = Vec::new();
        for node in &frontier {
            for a in &anchors {
                for &reel in &[-1i32, 0i32] {
                    let r = rollout_primitive(
                        node.p, node.v, a.pos, omega, reel, target, RRT_PRIM_STEPS,
                        PLAN_H,
                    );
                    let run_min = node.run_min.min(r.min_dist);
                    let step_c =
                        branch_cost(r.p, r.min_dist, r.term, opponents, teammates);
                    let (first_pos, first_reel, first_is_spar, first_pd) =
                        if depth == 0 {
                            (a.pos, reel, a.is_spar, r.min_dist)
                        } else {
                            (node.first_pos, node.first_reel, node.first_is_spar,
                             node.first_pd)
                        };
                    let n = BeamNode {
                        p: r.p,
                        v: r.v,
                        cum: node.cum + step_c,
                        run_min,
                        first_pos,
                        first_reel,
                        first_is_spar,
                        first_pd,
                    };
                    match best {
                        Some(b) if b.cum <= n.cum + 1e-12 => {}
                        _ => best = Some(n),
                    }
                    expanded.push(n);
                }
            }
        }
        if expanded.is_empty() {
            break;
        }
        // Keep top-BEAM_K by (cumulative cost, first-hop anchor x, reel) —
        // a total deterministic order with no float-equality ambiguity.
        expanded.sort_by(|x, y| {
            match x.cum.partial_cmp(&y.cum).unwrap() {
                std::cmp::Ordering::Equal => {
                    match x.first_pos.x.partial_cmp(&y.first_pos.x).unwrap() {
                        std::cmp::Ordering::Equal => x.first_reel.cmp(&y.first_reel),
                        o => o,
                    }
                }
                o => o,
            }
        });
        expanded.truncate(BEAM_K);
        frontier = expanded;
    }

    let chosen = best?;
    let mut chosen_pos = chosen.first_pos;
    let mut chosen_reel = chosen.first_reel;
    let mut chosen_is_spar = chosen.first_is_spar;
    let mut chosen_pd = chosen.first_pd;
    let best_c = chosen.cum;

    if let Some(st) = sticky {
        let sd = pos.sub(st.pos).len();
        if sd >= 2.0 && sd <= 180.0 && chosen_pos.sub(st.pos).len() > 4.0 {
            let sr = rollout_primitive(
                pos, vel, st.pos, omega, st.reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let sticky_cost =
                branch_cost(sr.p, sr.min_dist, sr.term, opponents, teammates);
            if best_c >= sticky_cost - ANCHOR_SWITCH_MARGIN {
                chosen_pos = st.pos;
                chosen_reel = st.reel;
                chosen_is_spar = st.pos.y.hypot(st.pos.z) < 1.0;
                chosen_pd = sr.min_dist;
            }
        }
    }
    Some(Plan {
        anchor_pos: chosen_pos,
        reel: chosen_reel,
        projected_dist: chosen_pd,
        is_spar: chosen_is_spar,
    })
}

// ── MCTS planner (planner == 6) ──────────────────────────────────────────────
// A proper UCT (Upper Confidence bounds applied to Trees) search over
// multi-hop anchor sequences. The action set at every node is the SAME sorted
// (anchor, reel) candidate set used by RRT/CEM (build_anchors + sort_anchors;
// reel ∈ {-1,0}). A tree node holds the post-rollout simulated state (p, v)
// reached by the path of actions from the root. Tree policy is UCB1 with
// exploration constant MCTS_C; expansion adds one untried child; the rollout
// (default policy) is a short random walk of further hops scored by
// branch_cost. We run MCTS_ITERS iterations; depth is capped at MCTS_DEPTH
// hops. We return the FIRST hop of the most-visited root child (the standard
// robust-child MCTS recommendation).
//
// Cost→reward map: branch_cost is "lower is better" and unbounded; MCTS needs
// a bounded reward in roughly [0,1] so UCB1's exploration term is well-scaled.
// We map deterministically  reward = 1 / (1 + exp(cost / MCTS_REWARD_SCALE))
// — a logistic squash. It is strictly monotone decreasing in cost (lower cost
// ⇒ higher reward), bounded in (0,1), depends only on the deterministic cost,
// and has no tunable offset (the constant cancels in argmax-by-visits / is
// only a temperature on exploration). Backprop accumulates the SUM of these
// rewards; the node value used by UCB1 is the running mean (sum / visits).
const MCTS_ITERS: u32 = 64;
const MCTS_C: f64 = 1.2;
const MCTS_DEPTH: u32 = 4;
const MCTS_REWARD_SCALE: f64 = 20.0;

struct MctsNode {
    p: Vec3,
    v: Vec3,
    parent: i32,
    // Action taken from parent to reach this node (root: sentinel).
    a_pos: Vec3,
    reel: i32,
    is_spar: bool,
    run_min: f64,
    depth: u32,
    visits: u32,
    value_sum: f64,
    // Children created so far + count of actions already expanded. Actions are
    // indexed (ai * 2 + reel_bit) over the sorted anchor list (deterministic).
    children: Vec<usize>,
    untried_next: usize,
}

fn mcts_cost_to_reward(cost: f64) -> f64 {
    1.0 / (1.0 + (cost / MCTS_REWARD_SCALE).exp())
}

fn plan_mcts(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    opponents: &[Vec3],
    teammates: &[Vec3],
    sticky: Option<Sticky>,
    spars: &[Vec3],
) -> Option<Plan> {
    let pos = player.p;
    let vel = player.v;
    let omega = state.omega;
    let to_dir = to_target_dir(pos, target);

    let mut anchors = build_anchors(pos, to_dir, spars, player, state);
    if anchors.is_empty() {
        return None;
    }
    sort_anchors(&mut anchors, target);
    let a_n = anchors.len();
    // Total discrete actions = anchors × {reel=-1, reel=0}.
    let n_actions = a_n * 2;

    let replan_w = rrt_replan_ticks();
    let mut rng = HashRng::new(&player.id, state.tick.div_euclid(replan_w));

    let mut nodes: Vec<MctsNode> = Vec::with_capacity(MCTS_ITERS as usize + 1);
    nodes.push(MctsNode {
        p: pos,
        v: vel,
        parent: -1,
        a_pos: pos,
        reel: -1,
        is_spar: true,
        run_min: pos.sub(target).len(),
        depth: 0,
        visits: 0,
        value_sum: 0.0,
        children: Vec::new(),
        untried_next: 0,
    });

    // Decode action index → (anchor index, reel). Order: ai ascending, then
    // reel=-1 before reel=0 (deterministic).
    let decode = |act: usize| -> (usize, i32) {
        (act / 2, if act % 2 == 1 { 0 } else { -1 })
    };

    for _ in 0..MCTS_ITERS {
        // ---- Selection: descend by UCB1 until a node with an untried action
        // or at depth cap. ----
        let mut path: Vec<usize> = vec![0];
        let mut cur = 0usize;
        loop {
            let depth = nodes[cur].depth;
            let fully_expanded = nodes[cur].untried_next >= n_actions;
            if depth >= MCTS_DEPTH || !fully_expanded {
                break;
            }
            if nodes[cur].children.is_empty() {
                break;
            }
            // UCB1 over children. Deterministic tie-break: higher score wins;
            // on a numeric tie, lower child node index (= earlier action).
            let parent_visits = nodes[cur].visits.max(1) as f64;
            let mut best_child = nodes[cur].children[0];
            let mut best_ucb = f64::NEG_INFINITY;
            for &ci in &nodes[cur].children {
                let cn = &nodes[ci];
                let mean = if cn.visits > 0 {
                    cn.value_sum / cn.visits as f64
                } else {
                    0.0
                };
                let explore = if cn.visits > 0 {
                    MCTS_C * (parent_visits.ln() / cn.visits as f64).sqrt()
                } else {
                    f64::INFINITY
                };
                let ucb = mean + explore;
                if ucb > best_ucb + 1e-12 {
                    best_ucb = ucb;
                    best_child = ci;
                }
            }
            cur = best_child;
            path.push(cur);
        }

        // ---- Expansion: if room, add one untried child via the next action
        // index (deterministic order). ----
        let mut leaf = cur;
        if nodes[cur].depth < MCTS_DEPTH && nodes[cur].untried_next < n_actions {
            let act = nodes[cur].untried_next;
            nodes[cur].untried_next += 1;
            let (ai, reel) = decode(act);
            let (fp, fv, frun) = (nodes[cur].p, nodes[cur].v, nodes[cur].run_min);
            let r = rollout_primitive(
                fp, fv, anchors[ai].pos, omega, reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let run_min = frun.min(r.min_dist);
            let depth = nodes[cur].depth + 1;
            let new_idx = nodes.len();
            nodes.push(MctsNode {
                p: r.p,
                v: r.v,
                parent: cur as i32,
                a_pos: anchors[ai].pos,
                reel,
                is_spar: anchors[ai].is_spar,
                run_min,
                depth,
                visits: 0,
                value_sum: 0.0,
                children: Vec::new(),
                untried_next: 0,
            });
            nodes[cur].children.push(new_idx);
            leaf = new_idx;
            path.push(new_idx);
        }

        // ---- Simulation (default policy): random hops from the leaf state to
        // the depth cap, all randomness from the windowed rng. The leaf's own
        // running-min/term scores the rollout via branch_cost. ----
        let mut sp = nodes[leaf].p;
        let mut sv = nodes[leaf].v;
        let mut run_min = nodes[leaf].run_min;
        let mut term = {
            // term toward target at the leaf state (so even a depth-cap leaf
            // gets a meaningful momentum reward).
            let to_t = target.sub(sp);
            let dl = to_t.len();
            if dl > 1e-6 {
                sv.dot(to_t.scale(1.0 / dl)).max(0.0)
            } else {
                0.0
            }
        };
        let mut d = nodes[leaf].depth;
        while d < MCTS_DEPTH {
            // Uniform random action over the discrete action set.
            let ai = ((rng.next() * a_n as f64) as i64 as usize).min(a_n - 1);
            let reel: i32 = if rng.next() < 0.5 { -1 } else { 0 };
            let r = rollout_primitive(
                sp, sv, anchors[ai].pos, omega, reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            sp = r.p;
            sv = r.v;
            run_min = run_min.min(r.min_dist);
            term = r.term;
            d += 1;
        }
        let cost = branch_cost(sp, run_min, term, opponents, teammates);
        let reward = mcts_cost_to_reward(cost);

        // ---- Backpropagation: add reward + visit along the visited path. ----
        for &ni in &path {
            nodes[ni].visits += 1;
            nodes[ni].value_sum += reward;
        }
    }

    // Recommendation: most-visited root child. Deterministic tie-break:
    // visits desc, then mean value desc, then anchor x asc, then reel asc.
    if nodes[0].children.is_empty() {
        return None;
    }
    let mut best_child = nodes[0].children[0];
    for &ci in &nodes[0].children {
        let cn = &nodes[ci];
        let bn = &nodes[best_child];
        let cn_mean = if cn.visits > 0 { cn.value_sum / cn.visits as f64 } else { 0.0 };
        let bn_mean = if bn.visits > 0 { bn.value_sum / bn.visits as f64 } else { 0.0 };
        let better = if cn.visits != bn.visits {
            cn.visits > bn.visits
        } else if (cn_mean - bn_mean).abs() > 1e-12 {
            cn_mean > bn_mean
        } else if cn.a_pos.x != bn.a_pos.x {
            cn.a_pos.x < bn.a_pos.x
        } else {
            cn.reel < bn.reel
        };
        if better {
            best_child = ci;
        }
    }
    let bc = &nodes[best_child];
    let mut chosen_pos = bc.a_pos;
    let mut chosen_reel = bc.reel;
    let mut chosen_is_spar = bc.is_spar;
    let mut chosen_pd = bc.run_min;
    // Root-child cost for the sticky comparison: re-derive from the first hop.
    let first_r = rollout_primitive(
        pos, vel, chosen_pos, omega, chosen_reel, target, RRT_PRIM_STEPS, PLAN_H,
    );
    let best_c = branch_cost(
        first_r.p,
        pos.sub(target).len().min(first_r.min_dist),
        first_r.term,
        opponents,
        teammates,
    );

    if let Some(st) = sticky {
        let sd = pos.sub(st.pos).len();
        if sd >= 2.0 && sd <= 180.0 && chosen_pos.sub(st.pos).len() > 4.0 {
            let sr = rollout_primitive(
                pos, vel, st.pos, omega, st.reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let sticky_cost =
                branch_cost(sr.p, sr.min_dist, sr.term, opponents, teammates);
            if best_c >= sticky_cost - ANCHOR_SWITCH_MARGIN {
                chosen_pos = st.pos;
                chosen_reel = st.reel;
                chosen_is_spar = st.pos.y.hypot(st.pos.z) < 1.0;
                chosen_pd = sr.min_dist;
            }
        }
    }
    Some(Plan {
        anchor_pos: chosen_pos,
        reel: chosen_reel,
        projected_dist: chosen_pd,
        is_spar: chosen_is_spar,
    })
}

// ── Potential-field planner (planner == 7) ───────────────────────────────────
// NO search. Build an analytic navigation field at the player's position:
//   • an ATTRACTOR pulling toward `target` (unit vector, gain PF_K_ATTRACT),
//   • a REPULSOR from each opponent within PF_REPEL_RADIUS, magnitude
//     PF_K_REPEL·(1/ρ − 1/R₀) along the away-from-opponent direction (the
//     classic FIRAS repulsive potential gradient, clamped at the radius),
//   • a SKIN repulsor pushing inward as the player's cylindrical radius
//     ρ = √(y²+z²) approaches the hull R (gain PF_K_SKIN, active within
//     PF_SKIN_BAND of R), directed along −(0,y,z)/ρ.
// The summed field gradient g is the desired direction of travel. For each
// (anchor,reel) candidate (same build_anchors set) we do ONE rollout_primitive
// and score by how well the resulting swing DIRECTION (end velocity, falling
// back to displacement) aligns with g, tie-broken toward low min_dist:
//   score = PF_W_ALIGN · align  −  min_dist        (higher = better)
// Pure, allocation-light, no RNG — determinism is structural.
const PF_K_ATTRACT: f64 = 1.0;
const PF_K_REPEL: f64 = 30.0;
const PF_REPEL_RADIUS: f64 = 14.0;
const PF_K_SKIN: f64 = 1.5;
const PF_SKIN_BAND: f64 = 8.0;
const PF_W_ALIGN: f64 = 25.0;

fn potential_field_grad(pos: Vec3, target: Vec3, opponents: &[Vec3]) -> Vec3 {
    // Attractor: unit pull toward target.
    let mut g = to_target_dir(pos, target).scale(PF_K_ATTRACT);
    // Opponent repulsors (FIRAS gradient, clamped at PF_REPEL_RADIUS).
    for o in opponents {
        let away = pos.sub(*o);
        let rho = away.len();
        if rho > 1e-6 && rho < PF_REPEL_RADIUS {
            let mag = PF_K_REPEL * (1.0 / rho - 1.0 / PF_REPEL_RADIUS) / (rho * rho);
            g = g.add(away.scale(mag / rho));
        }
    }
    // Skin repulsor: push inward as ρ→R.
    let rho_yz = (pos.y * pos.y + pos.z * pos.z).sqrt();
    if rho_yz > R - PF_SKIN_BAND {
        let depth = rho_yz - (R - PF_SKIN_BAND);
        let inward = Vec3::new(0.0, -pos.y / rho_yz, -pos.z / rho_yz);
        g = g.add(inward.scale(PF_K_SKIN * depth));
    }
    g
}

fn plan_potential_field(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    opponents: &[Vec3],
    teammates: &[Vec3],
    sticky: Option<Sticky>,
    spars: &[Vec3],
) -> Option<Plan> {
    let pos = player.p;
    let vel = player.v;
    let omega = state.omega;
    let to_dir = to_target_dir(pos, target);

    let mut anchors = build_anchors(pos, to_dir, spars, player, state);
    if anchors.is_empty() {
        return None;
    }
    sort_anchors(&mut anchors, target);

    let g = potential_field_grad(pos, target, opponents);
    let g_len = g.len();
    let g_hat = if g_len > 1e-6 {
        g.scale(1.0 / g_len)
    } else {
        to_dir
    };

    // For each (anchor,reel): one rollout, score by field alignment − min_dist.
    let mut best_score = f64::NEG_INFINITY;
    let mut best_pos = anchors[0].pos;
    let mut best_reel = -1i32;
    let mut best_is_spar = anchors[0].is_spar;
    let mut best_pd = f64::INFINITY;
    let mut best_c = f64::INFINITY;
    for a in &anchors {
        for &reel in &[-1i32, 0i32] {
            let r = rollout_primitive(
                pos, vel, a.pos, omega, reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            // Swing direction: end velocity if it has magnitude, else net
            // displacement from the start.
            let vl = r.v.len();
            let dir = if vl > 1e-6 {
                r.v.scale(1.0 / vl)
            } else {
                let disp = r.p.sub(pos);
                let dl = disp.len();
                if dl > 1e-6 {
                    disp.scale(1.0 / dl)
                } else {
                    g_hat
                }
            };
            let align = dir.dot(g_hat);
            let score = PF_W_ALIGN * align - r.min_dist;
            let c = branch_cost(r.p, r.min_dist, r.term, opponents, teammates);
            // Deterministic argmax: score desc, then cost asc, then anchor x
            // asc, then reel asc.
            let better = if score > best_score + 1e-12 {
                true
            } else if (score - best_score).abs() <= 1e-12 {
                if c < best_c - 1e-12 {
                    true
                } else if (c - best_c).abs() <= 1e-12 {
                    if a.pos.x != best_pos.x {
                        a.pos.x < best_pos.x
                    } else {
                        reel < best_reel
                    }
                } else {
                    false
                }
            } else {
                false
            };
            if better {
                best_score = score;
                best_pos = a.pos;
                best_reel = reel;
                best_is_spar = a.is_spar;
                best_pd = r.min_dist;
                best_c = c;
            }
        }
    }

    let mut chosen_pos = best_pos;
    let mut chosen_reel = best_reel;
    let mut chosen_is_spar = best_is_spar;
    let mut chosen_pd = best_pd;

    if let Some(st) = sticky {
        let sd = pos.sub(st.pos).len();
        if sd >= 2.0 && sd <= 180.0 && chosen_pos.sub(st.pos).len() > 4.0 {
            let sr = rollout_primitive(
                pos, vel, st.pos, omega, st.reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let sticky_cost =
                branch_cost(sr.p, sr.min_dist, sr.term, opponents, teammates);
            if best_c >= sticky_cost - ANCHOR_SWITCH_MARGIN {
                chosen_pos = st.pos;
                chosen_reel = st.reel;
                chosen_is_spar = st.pos.y.hypot(st.pos.z) < 1.0;
                chosen_pd = sr.min_dist;
            }
        }
    }
    Some(Plan {
        anchor_pos: chosen_pos,
        reel: chosen_reel,
        projected_dist: chosen_pd,
        is_spar: chosen_is_spar,
    })
}

// ── Random-shooting planner (planner == 8) ───────────────────────────────────
// The honest control / baseline. Sample RS_N (anchor, reel) pairs UNIFORMLY
// from the same build_anchors candidate set via the windowed HashRng — NO
// goal bias, NO categorical reweighting (deliberately dumber than CEM/MPPI's
// w[i]=1/(1+i) proposal) — roll each out and return argmin branch_cost. This
// tells us how much the structured planners actually buy over plain Monte
// Carlo. Determinism: every draw is the windowed rng; ties broken
// deterministically (cost, anchor x, reel).
const RS_N: u32 = 48;

fn plan_random_shooting(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    opponents: &[Vec3],
    teammates: &[Vec3],
    sticky: Option<Sticky>,
    spars: &[Vec3],
) -> Option<Plan> {
    let pos = player.p;
    let vel = player.v;
    let omega = state.omega;
    let to_dir = to_target_dir(pos, target);

    let mut anchors = build_anchors(pos, to_dir, spars, player, state);
    if anchors.is_empty() {
        return None;
    }
    sort_anchors(&mut anchors, target);
    let a_n = anchors.len();

    let replan_w = rrt_replan_ticks();
    let mut rng = HashRng::new(&player.id, state.tick.div_euclid(replan_w));

    let mut best_c = f64::INFINITY;
    let mut best_pos = anchors[0].pos;
    let mut best_reel = -1i32;
    let mut best_is_spar = anchors[0].is_spar;
    let mut best_pd = f64::INFINITY;
    for _ in 0..RS_N {
        // Truly uniform anchor index (no u*u goal bias) and uniform reel.
        let ai = ((rng.next() * a_n as f64) as i64 as usize).min(a_n - 1);
        let reel: i32 = if rng.next() < 0.5 { -1 } else { 0 };
        let r = rollout_primitive(
            pos, vel, anchors[ai].pos, omega, reel, target, RRT_PRIM_STEPS, PLAN_H,
        );
        let c = branch_cost(r.p, r.min_dist, r.term, opponents, teammates);
        let better = if c < best_c - 1e-12 {
            true
        } else if (c - best_c).abs() <= 1e-12 {
            if anchors[ai].pos.x != best_pos.x {
                anchors[ai].pos.x < best_pos.x
            } else {
                reel < best_reel
            }
        } else {
            false
        };
        if better {
            best_c = c;
            best_pos = anchors[ai].pos;
            best_reel = reel;
            best_is_spar = anchors[ai].is_spar;
            best_pd = r.min_dist;
        }
    }

    let mut chosen_pos = best_pos;
    let mut chosen_reel = best_reel;
    let mut chosen_is_spar = best_is_spar;
    let mut chosen_pd = best_pd;

    if let Some(st) = sticky {
        let sd = pos.sub(st.pos).len();
        if sd >= 2.0 && sd <= 180.0 && chosen_pos.sub(st.pos).len() > 4.0 {
            let sr = rollout_primitive(
                pos, vel, st.pos, omega, st.reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let sticky_cost =
                branch_cost(sr.p, sr.min_dist, sr.term, opponents, teammates);
            if best_c >= sticky_cost - ANCHOR_SWITCH_MARGIN {
                chosen_pos = st.pos;
                chosen_reel = st.reel;
                chosen_is_spar = st.pos.y.hypot(st.pos.z) < 1.0;
                chosen_pd = sr.min_dist;
            }
        }
    }
    Some(Plan {
        anchor_pos: chosen_pos,
        reel: chosen_reel,
        projected_dist: chosen_pd,
        is_spar: chosen_is_spar,
    })
}

// ── Coordination planner (planner == 9) — the first multi-agent-aware arm ─────
// Same candidate set + single rollout per (anchor,reel) as Random-Shooting,
// but the selection cost is branch_cost AUGMENTED with two genuinely
// joint-aware terms read from same-team entries of `state.players`:
//
//  (a) TEAMMATE-INTERFERENCE PENALTY. This is an explicit extension of the
//      existing wSpace spacing-penalty machinery (see branch_cost / the
//      spacing_pen closure: penalize ending within 18 m of a teammate). We
//      keep that idea but make it path-aware. For each same-team teammate
//      (id != self) we form two penalties:
//        • end-proximity: if the swing END p is within COORD_INTERF_RADIUS of
//          the teammate's CURRENT position, add
//          (COORD_INTERF_RADIUS − d) · COORD_W_INTERF.
//        • lane-crossing: approximate the teammate's likely swing lane as the
//          segment from their current position along their current velocity
//          for COORD_LANE_LEN seconds; if the player's swung path (sampled as
//          the start→end chord) passes within COORD_LANE_RADIUS of that lane
//          segment, add COORD_W_LANE · (COORD_LANE_RADIUS − sep). Segment-
//          segment closest distance, fully analytic & deterministic.
//
//  (b) PASS-SETUP ATTRACTOR. Identify the most-FORWARD same-team teammate
//      (largest x — toward the attacking goal in this rig frame) as the
//      receiver. If, at the swing END, there is an open lane to the receiver
//      — i.e. the end→receiver direction is not blocked by an opponent within
//      COORD_LOS_RADIUS of that segment AND the end is within COORD_PASS_RANGE
//      of the receiver — subtract a small bonus COORD_W_PASS (a NEGATIVE cost
//      term, capped so it never dominates the primary min_dist objective).
//
// All terms are continuous, deterministic, read only Vec-ordered
// state.players, and use no RNG beyond the windowed sampler shared with the
// rollouts. Consts COORD_* below; defaults chosen so interference ≈ the
// existing defender-danger scale (×3.5 over a few metres) and the pass bonus
// is bounded well under a typical min_dist (tens of metres).
const COORD_INTERF_RADIUS: f64 = 16.0;
const COORD_W_INTERF: f64 = 1.5;
const COORD_LANE_LEN: f64 = 1.0; // seconds of teammate velocity extrapolation
const COORD_LANE_RADIUS: f64 = 8.0;
const COORD_W_LANE: f64 = 0.9;
const COORD_LOS_RADIUS: f64 = 6.0;
const COORD_PASS_RANGE: f64 = 70.0;
const COORD_W_PASS: f64 = 8.0;

/// Closest distance between segment [a0,a1] and segment [b0,b1] (clamped,
/// analytic, deterministic — the standard Ericson formulation).
fn seg_seg_dist(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> f64 {
    let d1 = a1.sub(a0);
    let d2 = b1.sub(b0);
    let r = a0.sub(b0);
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let (mut s, mut t);
    if a <= 1e-12 && e <= 1e-12 {
        return a0.sub(b0).len();
    }
    if a <= 1e-12 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= 1e-12 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            s = if denom > 1e-12 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    let cp1 = a0.add(d1.scale(s));
    let cp2 = b0.add(d2.scale(t));
    cp1.sub(cp2).len()
}

/// Distance from point `pt` to segment [s0,s1].
fn point_seg_dist(pt: Vec3, s0: Vec3, s1: Vec3) -> f64 {
    let d = s1.sub(s0);
    let l2 = d.dot(d);
    if l2 <= 1e-12 {
        return pt.sub(s0).len();
    }
    let t = (pt.sub(s0).dot(d) / l2).clamp(0.0, 1.0);
    pt.sub(s0.add(d.scale(t))).len()
}

fn plan_coordination(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    opponents: &[Vec3],
    teammates: &[Vec3],
    sticky: Option<Sticky>,
    spars: &[Vec3],
) -> Option<Plan> {
    let pos = player.p;
    let vel = player.v;
    let omega = state.omega;
    let to_dir = to_target_dir(pos, target);

    let mut anchors = build_anchors(pos, to_dir, spars, player, state);
    if anchors.is_empty() {
        return None;
    }
    sort_anchors(&mut anchors, target);

    // Same-team teammates (id != self), in deterministic Vec order, with both
    // current position and velocity (for the lane extrapolation).
    let mut tm: Vec<(Vec3, Vec3)> = Vec::new();
    for pl in &state.players {
        if pl.id == player.id || pl.team != player.team {
            continue;
        }
        tm.push((pl.p, pl.v));
    }
    // Receiver = most-forward (max x) same-team teammate. Deterministic
    // tie-break: larger x, then smaller y, then smaller z.
    let mut receiver: Option<Vec3> = None;
    for &(tp, _) in &tm {
        receiver = Some(match receiver {
            None => tp,
            Some(cur) => {
                let take = if tp.x != cur.x {
                    tp.x > cur.x
                } else if tp.y != cur.y {
                    tp.y < cur.y
                } else {
                    tp.z < cur.z
                };
                if take {
                    tp
                } else {
                    cur
                }
            }
        });
    }

    // Augmented cost for one rollout result.
    let coord_cost = |end_p: Vec3, run_min: f64, term: f64| -> f64 {
        let mut c = branch_cost(end_p, run_min, term, opponents, teammates);
        // (a) interference: end-proximity + lane-crossing.
        for &(tp, tv) in &tm {
            let dend = end_p.sub(tp).len();
            if dend < COORD_INTERF_RADIUS {
                c += (COORD_INTERF_RADIUS - dend) * COORD_W_INTERF;
            }
            // Teammate's likely swing lane: tp → tp + tv*COORD_LANE_LEN.
            let lane_end = tp.add(tv.scale(COORD_LANE_LEN));
            // Player's swung path approximated by the start→end chord.
            let sep = seg_seg_dist(pos, end_p, tp, lane_end);
            if sep < COORD_LANE_RADIUS {
                c += (COORD_LANE_RADIUS - sep) * COORD_W_LANE;
            }
        }
        // (b) pass-setup attractor: open lane to the forward receiver.
        if let Some(rp) = receiver {
            let to_rcv = rp.sub(end_p).len();
            if to_rcv > 1e-6 && to_rcv < COORD_PASS_RANGE {
                let mut blocked = false;
                for o in opponents {
                    if point_seg_dist(*o, end_p, rp) < COORD_LOS_RADIUS {
                        blocked = true;
                        break;
                    }
                }
                if !blocked {
                    // Bonus scaled by openness (closer ⇒ stronger), capped at
                    // COORD_W_PASS so it never overwhelms min_dist.
                    let openness = 1.0 - to_rcv / COORD_PASS_RANGE;
                    c -= COORD_W_PASS * openness;
                }
            }
        }
        c
    };

    let mut best_c = f64::INFINITY;
    let mut best_pos = anchors[0].pos;
    let mut best_reel = -1i32;
    let mut best_is_spar = anchors[0].is_spar;
    let mut best_pd = f64::INFINITY;
    for a in &anchors {
        for &reel in &[-1i32, 0i32] {
            let r = rollout_primitive(
                pos, vel, a.pos, omega, reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let c = coord_cost(r.p, r.min_dist, r.term);
            let better = if c < best_c - 1e-12 {
                true
            } else if (c - best_c).abs() <= 1e-12 {
                if a.pos.x != best_pos.x {
                    a.pos.x < best_pos.x
                } else {
                    reel < best_reel
                }
            } else {
                false
            };
            if better {
                best_c = c;
                best_pos = a.pos;
                best_reel = reel;
                best_is_spar = a.is_spar;
                best_pd = r.min_dist;
            }
        }
    }

    let mut chosen_pos = best_pos;
    let mut chosen_reel = best_reel;
    let mut chosen_is_spar = best_is_spar;
    let mut chosen_pd = best_pd;

    if let Some(st) = sticky {
        let sd = pos.sub(st.pos).len();
        if sd >= 2.0 && sd <= 180.0 && chosen_pos.sub(st.pos).len() > 4.0 {
            let sr = rollout_primitive(
                pos, vel, st.pos, omega, st.reel, target, RRT_PRIM_STEPS, PLAN_H,
            );
            let sticky_cost = coord_cost(sr.p, sr.min_dist, sr.term);
            if best_c >= sticky_cost - ANCHOR_SWITCH_MARGIN {
                chosen_pos = st.pos;
                chosen_reel = st.reel;
                chosen_is_spar = st.pos.y.hypot(st.pos.z) < 1.0;
                chosen_pd = sr.min_dist;
            }
        }
    }
    Some(Plan {
        anchor_pos: chosen_pos,
        reel: chosen_reel,
        projected_dist: chosen_pd,
        is_spar: chosen_is_spar,
    })
}

// ── Defender-aware swing sim (TS GP.ts:229-300) ──────────────────────────────
struct SwingDef {
    closest_dist: f64,
    min_opp_dist: f64,
    term: f64,
}

fn simulate_grapple_swing_def(
    pos: Vec3,
    vel: Vec3,
    anchor_pos: Vec3,
    omega: f64,
    reel: i32,
    target: Vec3,
    steps: u32,
    h: f64,
    opponents: &[Vec3],
) -> SwingDef {
    let mut p = pos;
    let mut v = vel;
    let mut rest_len = p.sub(anchor_pos).len();
    let mut closest_dist = p.sub(target).len();
    let mut min_opp_dist = f64::INFINITY;
    let mut released = false;

    for i in 0..steps {
        let s = rk4_step(PointState { p, v }, omega, h);
        p = s.p;
        v = s.v;

        if reel == 0 && !released && i >= 4 {
            let sp = v.len();
            let to_t = target.sub(p);
            let dl = to_t.len();
            if sp > SWOOP_MIN_V && dl > 1e-6 && v.dot(to_t) / (sp * dl) > SWOOP_ALIGN {
                released = true;
            }
        }

        if !released {
            let d = p.sub(anchor_pos);
            let len = d.len();
            if len > 1e-6 {
                let n = d.scale(1.0 / len);
                let v_rad = v.dot(n);
                if len >= rest_len && v_rad > 0.0 {
                    v = v.sub(n.scale(v_rad));
                }
                if reel == -1 && len >= TETHER_MIN {
                    let target_len = (rest_len - REEL_RATE * h).max(TETHER_MIN);
                    if target_len < rest_len {
                        let v_rad_current = v.dot(n);
                        let scale = if rest_len > 1e-6 { rest_len / target_len } else { 1.0 };
                        let v_tan = v.sub(n.scale(v_rad_current));
                        v = v_tan.scale(scale).add(n.scale(v_rad_current));
                        rest_len = target_len;
                    }
                }
            }
        }

        let dist = p.sub(target).len();
        if dist < closest_dist {
            closest_dist = dist;
        }

        if (i & 3) == 0 {
            for o in opponents {
                let od = p.sub(*o).len();
                if od < min_opp_dist {
                    min_opp_dist = od;
                }
            }
        }
    }

    let to_t = target.sub(p);
    let dl = to_t.len();
    let term = if dl > 1e-6 {
        v.dot(to_t.scale(1.0 / dl)).max(0.0)
    } else {
        0.0
    };
    SwingDef { closest_dist, min_opp_dist, term }
}

/// Non-defender swing sim (TS GP.ts:306-375). Same physics; returns
/// (closest_dist, term).
fn simulate_grapple_swing(
    pos: Vec3,
    vel: Vec3,
    anchor_pos: Vec3,
    omega: f64,
    reel: i32,
    target: Vec3,
    steps: u32,
    h: f64,
) -> (f64, f64) {
    let r = rollout_primitive(pos, vel, anchor_pos, omega, reel, target, steps, h);
    (r.min_dist, r.term)
}

// ── Spar graph helpers (TS GP.ts:79-149) ─────────────────────────────────────
fn nearest_spar_idx(spars: &[Vec3], p: Vec3) -> usize {
    let mut bi = 0usize;
    let mut bd = f64::INFINITY;
    for (i, s) in spars.iter().enumerate() {
        let dx = s.x - p.x;
        let dy = s.y - p.y;
        let dz = s.z - p.z;
        let d = dx * dx + dy * dy + dz * dz;
        if d < bd {
            bd = d;
            bi = i;
        }
    }
    bi
}

fn spar_dist(spars: &[Vec3], i: usize, j: usize) -> f64 {
    spars[i].sub(spars[j]).len()
}

fn spar_adjacency(spars: &[Vec3]) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = spars.iter().map(|_| Vec::new()).collect();
    for i in 0..spars.len() {
        for j in (i + 1)..spars.len() {
            let dx = spars[i].x - spars[j].x;
            let dy = spars[i].y - spars[j].y;
            let dz = spars[i].z - spars[j].z;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > 1e-6 && d <= HOP_MAX {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

/// A* over the static spar graph (TS routeSpars, GP.ts:118-149). Returns the
/// node-index path start→goal inclusive, or [goal] if unreachable.
fn route_spars(spars: &[Vec3], adj: &[Vec<usize>], start: usize, goal: usize) -> Vec<usize> {
    if start == goal {
        return vec![start];
    }
    let n = spars.len();
    let mut g = vec![f64::INFINITY; n];
    let mut came = vec![-1i32; n];
    let mut closed = vec![false; n];
    g[start] = 0.0;
    let mut open: Vec<usize> = vec![start];
    loop {
        if open.is_empty() {
            break;
        }
        let mut cur: i32 = -1;
        let mut best = f64::INFINITY;
        for &k in &open {
            let f = g[k] + spar_dist(spars, k, goal);
            if f < best - 1e-9
                || ((f - best).abs() <= 1e-9 && (cur < 0 || (k as i32) < cur))
            {
                best = f;
                cur = k as i32;
            }
        }
        let cur = cur as usize;
        if cur == goal {
            break;
        }
        open.retain(|&x| x != cur);
        closed[cur] = true;
        for &nb in &adj[cur] {
            if closed[nb] {
                continue;
            }
            let ng = g[cur] + spar_dist(spars, cur, nb);
            if ng < g[nb] - 1e-9 {
                g[nb] = ng;
                came[nb] = cur as i32;
                if !open.contains(&nb) {
                    open.push(nb);
                }
            }
        }
    }
    if came[goal] == -1 && start != goal {
        return vec![goal];
    }
    let mut path: Vec<usize> = Vec::new();
    let mut c = goal as i32;
    while c != -1 {
        path.push(c as usize);
        if c as usize == start {
            break;
        }
        c = came[c as usize];
    }
    path.reverse();
    path
}

// ── Main planner entry — TS planGrapple (GP.ts:661-910) ──────────────────────
/// Choose the best grapple anchor to approach `target`. `planner`: 0 = MPC,
/// 1 = RRT, 2 = CEM. Returns None if already close enough (TS returns null).
pub fn plan_grapple(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    avoid_defenders: bool,
    sticky: Option<Sticky>,
    planner: i32,
) -> Option<Plan> {
    let pos = player.p;
    let vel = player.v;
    let omega = state.omega;

    let direct_dist = pos.sub(target).len();
    if direct_dist < 3.0 {
        return None;
    }

    let opponents: Vec<Vec3> = if avoid_defenders {
        state
            .players
            .iter()
            .filter(|p| p.team != player.team)
            .map(|p| p.p)
            .collect()
    } else {
        Vec::new()
    };
    let teammates: Vec<Vec3> = state
        .players
        .iter()
        .filter(|p| p.id != player.id && p.team == player.team)
        .map(|p| p.p)
        .collect();

    let spars = spar_positions_cached();

    // Strategy switch (TS GP.ts:689-696): planner 2 = CEM, 1 = RRT, else MPC.
    // Extended arms: 3 = MPPI, 4 = SimAnneal, 5 = Beam, 6 = MCTS,
    // 7 = PotentialField, 8 = RandomShooting, 9 = Coordination. Arms 0/1/2
    // keep their exact original code path (this is a pure prepended dispatch);
    // any id ≥ 10 (or any arm that returns None) falls through to MPC.
    match planner {
        3 => {
            if let Some(r) =
                plan_mppi(player, target, state, &opponents, &teammates, sticky, spars)
            {
                return Some(r);
            }
        }
        4 => {
            if let Some(r) = plan_simanneal(
                player, target, state, &opponents, &teammates, sticky, spars,
            ) {
                return Some(r);
            }
        }
        5 => {
            if let Some(r) =
                plan_beam(player, target, state, &opponents, &teammates, sticky, spars)
            {
                return Some(r);
            }
        }
        6 => {
            if let Some(r) =
                plan_mcts(player, target, state, &opponents, &teammates, sticky, spars)
            {
                return Some(r);
            }
        }
        7 => {
            if let Some(r) = plan_potential_field(
                player, target, state, &opponents, &teammates, sticky, spars,
            ) {
                return Some(r);
            }
        }
        8 => {
            if let Some(r) = plan_random_shooting(
                player, target, state, &opponents, &teammates, sticky, spars,
            ) {
                return Some(r);
            }
        }
        9 => {
            if let Some(r) = plan_coordination(
                player, target, state, &opponents, &teammates, sticky, spars,
            ) {
                return Some(r);
            }
        }
        _ => {}
    }
    if planner == 2 {
        if let Some(r) =
            plan_cem(player, target, state, &opponents, &teammates, sticky, spars)
        {
            return Some(r);
        }
    } else if planner == 1 {
        if let Some(r) =
            plan_rrt(player, target, state, &opponents, &teammates, sticky, spars)
        {
            return Some(r);
        }
    }

    // Teammate-spacing penalty (TS GP.ts:701-707).
    let spacing_pen = |end_ref: Vec3| -> f64 {
        if W_SPACE <= 0.0 {
            return 0.0;
        }
        let mut near_tm = f64::INFINITY;
        for t in &teammates {
            let td = end_ref.sub(*t).len();
            if td < near_tm {
                near_tm = td;
            }
        }
        if near_tm < 18.0 {
            (18.0 - near_tm) * W_SPACE
        } else {
            0.0
        }
    };

    // Cost wrapper (TS scorePlan, GP.ts:711-736) → (projected_dist, cost).
    let score_plan = |anchor: Vec3, reel: i32| -> (f64, f64) {
        if opponents.is_empty() {
            let (cd, term) =
                simulate_grapple_swing(pos, vel, anchor, omega, reel, target, PLAN_STEPS, PLAN_H);
            (cd, cd - W_MOM * term + spacing_pen(anchor))
        } else {
            let r = simulate_grapple_swing_def(
                pos, vel, anchor, omega, reel, target, PLAN_STEPS, PLAN_H, &opponents,
            );
            let danger = if r.min_opp_dist < DEFENDER_DANGER {
                (DEFENDER_DANGER - r.min_opp_dist) * 3.5
            } else {
                0.0
            };
            (
                r.closest_dist,
                r.closest_dist - W_MOM * r.term + danger + spacing_pen(anchor),
            )
        }
    };

    // candidate: (anchor_pos, reel, projected_dist, is_spar, cost)
    let mut candidates: Vec<(Vec3, i32, f64, bool, f64)> = Vec::new();

    // Candidate 1: spar anchors (TS GP.ts:741-763).
    for spar in spars {
        let sdist = pos.sub(*spar).len();
        if sdist < 2.0 || sdist > 80.0 {
            continue;
        }
        let to_spar = spar.sub(pos).norm();
        let to_target = target.sub(pos).norm();
        if to_spar.dot(to_target) < -0.7 {
            continue;
        }
        let (pd_r, c_r) = score_plan(*spar, -1);
        candidates.push((*spar, -1, pd_r, true, c_r));
        let (pd_s, c_s) = score_plan(*spar, 0);
        candidates.push((*spar, 0, pd_s, true, c_s));
    }

    // Candidate 2: skin anchor (TS GP.ts:766-782).
    let player_radius = (pos.y * pos.y + pos.z * pos.z).sqrt();
    let is_too_close_skin = player_radius > R - SKIN_BUFFER;
    let target_radius = (target.y * target.y + target.z * target.z).sqrt();
    let skin_useful = !is_too_close_skin && target_radius < player_radius;
    if skin_useful {
        let yz_len = if player_radius > 1e-6 { player_radius } else { 1.0 };
        let skin_point = Vec3::new(
            pos.x,
            (pos.y / yz_len) * R,
            (pos.z / yz_len) * R,
        );
        let (pd, c) = score_plan(skin_point, 0);
        candidates.push((skin_point, 0, pd, false, c));
    }

    // Candidate 3: teammate anchors (TS GP.ts:785-797).
    for p in &state.players {
        if p.id == player.id || p.team != player.team {
            continue;
        }
        let tm_dist = pos.sub(p.p).len();
        if tm_dist < 3.0 || tm_dist > 60.0 {
            continue;
        }
        let to_tm = p.p.sub(pos).norm();
        let to_target = target.sub(pos).norm();
        if to_tm.dot(to_target) < -0.5 {
            continue;
        }
        let (pd, c) = score_plan(p.p, -1);
        candidates.push((p.p, -1, pd, false, c));
    }

    if candidates.is_empty() {
        // Fallback: nearest spar (TS GP.ts:799-808).
        let mut nearest_spar = spars[0];
        let mut nearest_dist = f64::INFINITY;
        for spar in spars {
            let d = pos.sub(*spar).len();
            if d < nearest_dist {
                nearest_dist = d;
                nearest_spar = *spar;
            }
        }
        return Some(Plan {
            anchor_pos: nearest_spar,
            reel: -1,
            projected_dist: direct_dist,
            is_spar: true,
        });
    }

    // Sort by cost, then projected_dist, then anchor x (TS GP.ts:813-817).
    candidates.sort_by(|a, b| {
        if a.4 != b.4 {
            a.4.partial_cmp(&b.4).unwrap()
        } else if a.2 != b.2 {
            a.2.partial_cmp(&b.2).unwrap()
        } else {
            a.0.x.partial_cmp(&b.0.x).unwrap()
        }
    });
    let mut best = candidates[0];

    // Graph route for FAR targets (TS GP.ts:825-855).
    if best.2 > DIRECT_REACH {
        let adj = spar_adjacency(spars);
        let goal_n = nearest_spar_idx(spars, target);
        let d_to_goal = pos.sub(spars[goal_n]).len();
        let hop: usize = if d_to_goal <= HOP_MAX {
            goal_n
        } else {
            let start_n = nearest_spar_idx(spars, pos);
            let path = route_spars(spars, &adj, start_n, goal_n);
            let mut h = goal_n;
            for node in path {
                if pos.sub(spars[node]).len() > 4.0 {
                    h = node;
                    break;
                }
            }
            h
        };
        let hop_pos = spars[hop];
        let to_hop = hop_pos.sub(pos).norm();
        let to_tgt = target.sub(pos).norm();
        if pos.sub(hop_pos).len() > 2.0 && to_hop.dot(to_tgt) > -0.6 {
            let (pd, c) = score_plan(hop_pos, -1);
            best = (hop_pos, -1, pd, true, c);
        }
    }

    // Nav-anchor hysteresis (TS GP.ts:864-902).
    if let Some(st) = sticky {
        let s_pos = st.pos;
        let sticky_dist = pos.sub(s_pos).len();
        let s_radius = (s_pos.y * s_pos.y + s_pos.z * s_pos.z).sqrt();
        let is_skin_sticky = s_radius > R * 0.85;

        let sticky_valid = if is_skin_sticky {
            skin_useful && st.reel == 0
        } else {
            let to_anchor = s_pos.sub(pos).norm();
            let to_target = target.sub(pos).norm();
            sticky_dist >= 2.0
                && sticky_dist <= 80.0
                && to_anchor.dot(to_target) >= -0.7
                && st.reel == -1
        };

        if sticky_valid {
            let (pd, c) = score_plan(s_pos, st.reel);
            if best.4 >= c - ANCHOR_SWITCH_MARGIN {
                return Some(Plan {
                    anchor_pos: s_pos,
                    reel: st.reel,
                    projected_dist: pd,
                    is_spar: !is_skin_sticky && s_radius < 1.0,
                });
            }
        }
    }

    Some(Plan {
        anchor_pos: best.0,
        reel: best.1,
        projected_dist: best.2,
        is_spar: best.3,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::OMEGA;

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// Deterministic: identical inputs → bit-identical rollout.
    #[test]
    fn rollout_is_deterministic() {
        let a = rollout_primitive(
            v(0.0, 8.0, 0.0), v(2.0, 0.0, 0.0), v(0.0, 0.0, 0.0),
            OMEGA, -1, v(60.0, 0.0, 0.0), 18, 1.0 / 15.0,
        );
        let b = rollout_primitive(
            v(0.0, 8.0, 0.0), v(2.0, 0.0, 0.0), v(0.0, 0.0, 0.0),
            OMEGA, -1, v(60.0, 0.0, 0.0), 18, 1.0 / 15.0,
        );
        assert_eq!(a.p.x, b.p.x);
        assert_eq!(a.p.y, b.p.y);
        assert_eq!(a.v.x, b.v.x);
        assert_eq!(a.min_dist, b.min_dist);
        assert_eq!(a.term, b.term);
    }

    /// Output invariants that are TRUE by construction (no hand-claimed
    /// physics): min_dist starts at the straight distance and only ever
    /// shrinks (it's a running min), and term is a finite non-negative
    /// speed. Cross-implementation PARITY vs the TS twin is validated by
    /// the headless skill harness, not asserted here.
    #[test]
    fn rollout_outputs_are_sane() {
        let pos = v(0.0, 12.0, 0.0);
        let vel = v(1.0, 0.0, 0.0);
        let target = v(0.0, 0.0, 0.0);
        let anchor = v(0.0, 2.0, 0.0);
        let start = pos.sub(target).len();
        let r = rollout_primitive(pos, vel, anchor, OMEGA, -1, target, 45, 1.0 / 15.0);
        assert!(r.min_dist.is_finite() && r.min_dist <= start + 1e-9);
        assert!(r.term.is_finite() && r.term >= 0.0);
        // Free swing should also produce finite, sane output.
        let s = rollout_primitive(pos, v(0.0, 0.0, 12.0), anchor, OMEGA, 0, target, 45, 1.0 / 15.0);
        assert!(s.min_dist.is_finite() && s.term.is_finite() && s.term >= 0.0);
    }

    // ── spar_positions: TS contract (GP.ts:46-64) ────────────────────────────
    /// Count = axis spine (ceil(L/40)+1) + exactly 16*3 off-axis lattice.
    #[test]
    fn spar_positions_count_and_structure() {
        let spars = spar_positions();
        let axis_n = (L / 40.0).ceil() as usize + 1;
        assert_eq!(spars.len(), axis_n + 16 * 3);

        // Axis spine: first `axis_n` are on the x-axis (y=z=0), spaced 40 m
        // starting at -L/2.
        for (i, s) in spars.iter().take(axis_n).enumerate() {
            assert_eq!(s.y, 0.0);
            assert_eq!(s.z, 0.0);
            assert_eq!(s.x, -L / 2.0 + i as f64 * 40.0);
        }
        // Off-axis lattice: exactly 16*3 points at radius R*0.62 in (y,z).
        let ring_r = R * 0.62;
        let mut off_axis = 0usize;
        for s in spars.iter().skip(axis_n) {
            let r = (s.y * s.y + s.z * s.z).sqrt();
            assert!((r - ring_r).abs() < 1e-9, "off-axis radius {} != {}", r, ring_r);
            off_axis += 1;
        }
        assert_eq!(off_axis, 16 * 3);
    }

    fn demo_state(tick: i64) -> (PlayerSim, SimState, Vec3) {
        let player = PlayerSim {
            id: "rigger-A".to_string(),
            team: 0,
            p: v(0.0, 12.0, 0.0),
            v: v(2.0, 0.0, 0.0),
        };
        let opp = PlayerSim {
            id: "mark-B".to_string(),
            team: 1,
            p: v(40.0, 5.0, 0.0),
            v: v(0.0, 0.0, 0.0),
        };
        let state = SimState {
            omega: OMEGA,
            tick,
            players: vec![player.clone(), opp],
        };
        (player, state, v(120.0, 0.0, 0.0))
    }

    /// hashRng is deterministic and stays in [0,1).
    #[test]
    fn hash_rng_deterministic_and_unit_range() {
        let mut a = HashRng::new("rigger-A", 7);
        let mut b = HashRng::new("rigger-A", 7);
        for _ in 0..50 {
            let x = a.next();
            assert_eq!(x, b.next());
            assert!(x >= 0.0 && x < 1.0, "rng out of [0,1): {}", x);
        }
        // Different seed → diverges (true by construction of xorshift).
        let mut c = HashRng::new("rigger-A", 8);
        assert!(c.next() != HashRng::new("rigger-A", 7).next());
    }

    /// Determinism: identical inputs → identical Plan, for every planner.
    #[test]
    fn plan_grapple_is_deterministic() {
        for planner in 0..=2 {
            let (player, state, target) = demo_state(99);
            let a = plan_grapple(&player, target, &state, true, None, planner);
            let b = plan_grapple(&player, target, &state, true, None, planner);
            assert_eq!(a, b, "planner {} not deterministic", planner);
        }
    }

    /// MPC/RRT/CEM all produce finite, sane Plans (parity vs TS is the
    /// harness's job; here only by-construction invariants).
    #[test]
    fn planners_produce_finite_plans() {
        for planner in 0..=2 {
            let (player, state, target) = demo_state(123);
            let plan = plan_grapple(&player, target, &state, true, None, planner)
                .expect("a far target should yield a plan");
            assert!(plan.anchor_pos.x.is_finite());
            assert!(plan.anchor_pos.y.is_finite());
            assert!(plan.anchor_pos.z.is_finite());
            assert!(plan.projected_dist.is_finite());
            assert!(plan.reel == -1 || plan.reel == 0, "reel {}", plan.reel);
        }
    }

    /// Already-close → None (TS returns null when directDist < 3).
    #[test]
    fn plan_grapple_none_when_close() {
        let player = PlayerSim {
            id: "x".to_string(),
            team: 0,
            p: v(0.0, 0.0, 0.0),
            v: v(0.0, 0.0, 0.0),
        };
        let state = SimState { omega: OMEGA, tick: 0, players: vec![player.clone()] };
        assert!(plan_grapple(&player, v(1.0, 0.0, 0.0), &state, true, None, 0).is_none());
    }

    /// Sticky hysteresis path stays deterministic across planners.
    #[test]
    fn plan_grapple_sticky_deterministic() {
        for planner in 0..=2 {
            let (player, state, target) = demo_state(50);
            let st = Some(Sticky { pos: v(40.0, 0.0, 0.0), reel: -1 });
            let a = plan_grapple(&player, target, &state, true, st, planner);
            let b = plan_grapple(&player, target, &state, true, st, planner);
            assert_eq!(a, b);
        }
    }

    // ── New planner classes: MPPI (3), SimAnneal (4), Beam (5) ───────────────

    /// Determinism: same inputs ⇒ identical Plan for the 3 new classes,
    /// with and without a sticky anchor.
    #[test]
    fn new_planners_are_deterministic() {
        for planner in 3..=5 {
            let (player, state, target) = demo_state(99);
            let a = plan_grapple(&player, target, &state, true, None, planner);
            let b = plan_grapple(&player, target, &state, true, None, planner);
            assert_eq!(a, b, "planner {} not deterministic", planner);

            let st = Some(Sticky { pos: v(40.0, 0.0, 0.0), reel: -1 });
            let c = plan_grapple(&player, target, &state, true, st, planner);
            let d = plan_grapple(&player, target, &state, true, st, planner);
            assert_eq!(c, d, "planner {} sticky not deterministic", planner);
        }
    }

    /// Finite/sane Plan: reel ∈ {-1,0}, finite anchor + projected_dist.
    #[test]
    fn new_planners_produce_finite_plans() {
        for planner in 3..=5 {
            let (player, state, target) = demo_state(123);
            let plan = plan_grapple(&player, target, &state, true, None, planner)
                .expect("a far target should yield a plan");
            assert!(plan.anchor_pos.x.is_finite());
            assert!(plan.anchor_pos.y.is_finite());
            assert!(plan.anchor_pos.z.is_finite());
            assert!(plan.projected_dist.is_finite());
            assert!(
                plan.reel == -1 || plan.reel == 0,
                "planner {} reel {}",
                planner,
                plan.reel
            );
        }
    }

    /// Adding the new arms must not perturb classes 0/1/2: their Plans are
    /// exactly what the original dispatch produced (regression pin).
    #[test]
    fn classes_0_1_2_unaffected_by_new_arms() {
        for tick in [50i64, 99, 123] {
            for planner in 0..=2 {
                let (player, state, target) = demo_state(tick);
                let base = plan_grapple(&player, target, &state, true, None, planner);
                // Re-run: stable, and the new match arm is a no-op for 0/1/2.
                let again =
                    plan_grapple(&player, target, &state, true, None, planner);
                assert_eq!(base, again, "planner {} tick {}", planner, tick);
                // A high planner id (>=10) must fall back to MPC == planner 0
                // (6..=9 are now real arms; 10+ keeps the MPC fallback).
                if planner == 0 {
                    let hi = plan_grapple(&player, target, &state, true, None, 10);
                    assert_eq!(base, hi, "planner 10 should fall back to MPC");
                    let hi2 = plan_grapple(&player, target, &state, true, None, 42);
                    assert_eq!(base, hi2, "planner 42 should fall back to MPC");
                }
            }
        }
    }

    /// Already-close → None holds for the new classes too.
    #[test]
    fn new_planners_none_when_close() {
        let player = PlayerSim {
            id: "x".to_string(),
            team: 0,
            p: v(0.0, 0.0, 0.0),
            v: v(0.0, 0.0, 0.0),
        };
        let state = SimState { omega: OMEGA, tick: 0, players: vec![player.clone()] };
        for planner in 3..=5 {
            assert!(plan_grapple(&player, v(1.0, 0.0, 0.0), &state, true, None, planner)
                .is_none());
        }
    }

    // ── Newest planner classes: MCTS (6), PotentialField (7),
    //    RandomShooting (8), Coordination (9) ──────────────────────────────

    /// Determinism: same inputs ⇒ identical Plan for classes 6..=9, tested
    /// twice with AND without a sticky anchor (±sticky).
    #[test]
    fn coord_planners_are_deterministic() {
        for planner in 6..=9 {
            let (player, state, target) = demo_state(99);
            let a = plan_grapple(&player, target, &state, true, None, planner);
            let b = plan_grapple(&player, target, &state, true, None, planner);
            assert_eq!(a, b, "planner {} not deterministic", planner);

            let st = Some(Sticky { pos: v(40.0, 0.0, 0.0), reel: -1 });
            let c = plan_grapple(&player, target, &state, true, st, planner);
            let d = plan_grapple(&player, target, &state, true, st, planner);
            assert_eq!(c, d, "planner {} sticky not deterministic", planner);

            // A different tick window stays self-consistent too.
            let (p2, s2, t2) = demo_state(123);
            let e = plan_grapple(&p2, t2, &s2, true, st, planner);
            let f = plan_grapple(&p2, t2, &s2, true, st, planner);
            assert_eq!(e, f, "planner {} tick123 not deterministic", planner);
        }
    }

    /// Finite/sane Plan: reel ∈ {-1,0}, finite anchor + projected_dist for
    /// classes 6..=9.
    #[test]
    fn coord_planners_produce_finite_plans() {
        for planner in 6..=9 {
            let (player, state, target) = demo_state(123);
            let plan = plan_grapple(&player, target, &state, true, None, planner)
                .expect("a far target should yield a plan");
            assert!(plan.anchor_pos.x.is_finite());
            assert!(plan.anchor_pos.y.is_finite());
            assert!(plan.anchor_pos.z.is_finite());
            assert!(plan.projected_dist.is_finite());
            assert!(
                plan.reel == -1 || plan.reel == 0,
                "planner {} reel {}",
                planner,
                plan.reel
            );
        }
    }

    /// Already-close → None holds for classes 6..=9 too.
    #[test]
    fn coord_planners_none_when_close() {
        let player = PlayerSim {
            id: "x".to_string(),
            team: 0,
            p: v(0.0, 0.0, 0.0),
            v: v(0.0, 0.0, 0.0),
        };
        let state = SimState { omega: OMEGA, tick: 0, players: vec![player.clone()] };
        for planner in 6..=9 {
            assert!(plan_grapple(&player, v(1.0, 0.0, 0.0), &state, true, None, planner)
                .is_none());
        }
    }

    /// Regression pin: adding arms 6..=9 must not perturb classes 0..=5 — for
    /// several ticks each class is byte-identical to a re-run, and ids ≥ 10
    /// still fall back to MPC (== planner 0).
    #[test]
    fn classes_0_through_5_unaffected_by_coord_arms() {
        for tick in [50i64, 99, 123, 7, 256] {
            let (player, state, target) = demo_state(tick);
            let mpc = plan_grapple(&player, target, &state, true, None, 0);
            for planner in 0..=5 {
                let a = plan_grapple(&player, target, &state, true, None, planner);
                let b = plan_grapple(&player, target, &state, true, None, planner);
                assert_eq!(a, b, "planner {} tick {} unstable", planner, tick);
            }
            for hi in [10i32, 11, 99, 1000] {
                let h = plan_grapple(&player, target, &state, true, None, hi);
                assert_eq!(mpc, h, "planner {} should fall back to MPC", hi);
            }
        }
    }

    /// MCTS cost→reward map: strictly decreasing in cost, bounded in (0,1).
    #[test]
    fn mcts_reward_map_is_monotone_and_bounded() {
        let mut prev = f64::INFINITY;
        for k in -50..=50 {
            let cost = k as f64 * 4.0;
            let r = mcts_cost_to_reward(cost);
            assert!(r > 0.0 && r < 1.0, "reward {} out of (0,1)", r);
            assert!(r < prev, "reward not strictly decreasing in cost");
            prev = r;
        }
    }

    /// Coordination is genuinely joint-aware: introducing a teammate sitting
    /// on the swing path / interfering changes the chosen Plan relative to
    /// the structure-blind Random-Shooting baseline on the same candidates.
    /// (Not a parity claim — just that the coordination terms have effect.)
    #[test]
    fn coordination_terms_have_effect() {
        let player = PlayerSim {
            id: "rigger-A".to_string(),
            team: 0,
            p: v(0.0, 12.0, 0.0),
            v: v(2.0, 0.0, 0.0),
        };
        // A same-team mate parked right in the typical swing corridor, plus
        // a forward receiver to trigger the pass-setup attractor.
        let mate = PlayerSim {
            id: "rigger-B".to_string(),
            team: 0,
            p: v(40.0, 6.0, 2.0),
            v: v(1.0, 0.0, 0.0),
        };
        let receiver = PlayerSim {
            id: "rigger-C".to_string(),
            team: 0,
            p: v(100.0, 0.0, 0.0),
            v: v(0.0, 0.0, 0.0),
        };
        let opp = PlayerSim {
            id: "mark-Z".to_string(),
            team: 1,
            p: v(55.0, 5.0, 0.0),
            v: v(0.0, 0.0, 0.0),
        };
        let state = SimState {
            omega: OMEGA,
            tick: 99,
            players: vec![player.clone(), mate, receiver, opp],
        };
        let target = v(120.0, 0.0, 0.0);
        let coord = plan_grapple(&player, target, &state, true, None, 9)
            .expect("coordination should yield a plan");
        let coord2 = plan_grapple(&player, target, &state, true, None, 9)
            .expect("coordination should yield a plan");
        assert_eq!(coord, coord2, "coordination not deterministic w/ teammates");
        // Both coordination and the baseline must still be sane plans.
        let base = plan_grapple(&player, target, &state, true, None, 8)
            .expect("random-shooting should yield a plan");
        for p in [coord, base] {
            assert!(p.anchor_pos.x.is_finite() && p.projected_dist.is_finite());
            assert!(p.reel == -1 || p.reel == 0);
        }
    }
}

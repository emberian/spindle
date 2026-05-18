//! planner_cost.rs — Stage-1 keystone refactor: a composable `CostTerm`
//! trait + `Profile` (weight vector) + a term library + `Profile::from_class`.
//!
//! This is a PURE BEHAVIOR-PRESERVING extraction of the planner's
//! cost/penalty/bonus arithmetic. Every planner class's `Plan` output is
//! BIT-IDENTICAL to the pre-extraction code: the term library reproduces the
//! EXACT sub-expressions, in the EXACT accumulation order, that the original
//! `branch_cost` / class-9 `coord_cost` closure / class-0 `score_plan`
//! fallback computed. Nothing else changes — no new behavior, no production
//! change (production stays Coordination class 9, routed through here with
//! identical results).
//!
//! WASM-SAFETY: this module is in the production wasm cdylib path. It uses
//! NO rayon, NO threads, NO native-only deps. `trait CostTerm: Send + Sync`
//! behind `Box<dyn CostTerm>` is the only dynamic dispatch.
//!
//! ── The sacred hazard: float op order ───────────────────────────────────
//! Determinism here is bit-exact (180 cargo tests + the parallel-eval
//! determinism test + replay/Number gates depend on it). The original
//! `branch_cost` computed:
//!
//! ```text
//! let mut c = min_dist - W_MOM * term;          // (1) base
//! for o in opponents { if od < DD { c += (DD-od)*3.5 } }   // (2) danger
//! if W_SPACE > 0 { ... c += (18-near_tm)*W_SPACE }         // (3) spacing
//! ```
//!
//! i.e. `((min_dist + (-(W_MOM*term))) + Σdanger) + spacing`, a strict
//! left-fold. We reproduce that with a fixed-order term list summed
//! left-to-right by `compute_cost`, where:
//!  - term `min_dist`  : value = `min_dist`,            weight = `1.0`
//!  - term `momentum`  : value = `term`,                weight = `-W_MOM`
//!  - term `defender_danger` : value = the WHOLE `Σ (DD-od)*3.5` sum
//!    (a COMPOUND term — the `*3.5` and the per-opponent loop+branch are
//!    INSIDE the term so the f64 result is identical), weight = `1.0`
//!  - term `spacing`   : value = the WHOLE gated `(18-near_tm)*W_SPACE`
//!    expression, weight = `1.0` (W_SPACE is baked in; default 0.0 so the
//!    term is inert exactly as before, and the term is only present for
//!    classes that used it — all of 1..=9 include the same inert term).
//!
//! Accumulator: `c = 0.0; for t: c = c + w_t * v_t`. Because
//! `0.0 + min_dist == min_dist`, `(-W_MOM)*term == -(W_MOM*term)`, and
//! `1.0 * x == x` for all finite IEEE-754 f64, the running sum is
//! bit-for-bit `((min_dist + (-(W_MOM*term))) + Σdanger) + spacing` — the
//! original expression, unchanged.
//!
//! Class 9 (`coord_cost`) appended, AFTER `branch_cost`, in this order:
//! interference (end-proximity then lane-crossing, per teammate in Vec
//! order) then the pass-setup bonus (a negative term). We reproduce that
//! with two further COMPOUND terms after the branch_cost terms:
//!  - `coord_interference` : the whole per-teammate end-prox + lane sum,
//!    weight = `1.0` (COORD_W_INTERF / COORD_W_LANE baked in)
//!  - `coord_pass`         : the whole gated pass attractor (already a
//!    NEGATIVE contribution `-COORD_W_PASS*openness`), weight = `1.0`.
//!
//! Class 0 MPC fallback (`score_plan`) is the no-defender / defender split:
//! `cd - W_MOM*term + spacing_pen` or
//! `closest_dist - W_MOM*term + danger + spacing_pen`. Same three pieces as
//! `branch_cost` (min_dist, momentum, danger, spacing) — see
//! `mpc_cost_*` helpers, which reuse the identical sub-expressions.
//!
//! Class 7 PotentialField's *selection* score is
//! `PF_W_ALIGN*align - min_dist` (NOT branch_cost); branch_cost only feeds
//! its sticky/tie-break `best_c`. The pf field gradient + the align/score
//! are pure geometry, reproduced verbatim by `pf_grad` / `pf_score`. They
//! are exposed here so planner.rs routes through one source of truth, but
//! the arithmetic is byte-for-byte the original.
//!
//! FIDELITY > GRANULARITY: several terms are deliberately COMPOUND (the
//! whole loop/branch is one term returning one f64) precisely so the
//! bit-identical reconstruction holds. This is noted per term and in the
//! stage report.

use crate::math::Vec3;
use crate::planner::Rollout;

// ── Canonical __rigtune defaults (mirrors planner.rs; kept in sync) ──────────
pub(crate) const W_MOM: f64 = 0.6;
pub(crate) const W_SPACE: f64 = 0.0;
pub(crate) const DEFENDER_DANGER: f64 = 6.0;

// Class-9 coordination consts (mirror planner.rs exactly).
pub(crate) const COORD_INTERF_RADIUS: f64 = 16.0;
pub(crate) const COORD_W_INTERF: f64 = 1.5;
pub(crate) const COORD_LANE_LEN: f64 = 1.0;
pub(crate) const COORD_LANE_RADIUS: f64 = 8.0;
pub(crate) const COORD_W_LANE: f64 = 0.9;
pub(crate) const COORD_LOS_RADIUS: f64 = 6.0;
pub(crate) const COORD_PASS_RANGE: f64 = 70.0;
pub(crate) const COORD_W_PASS: f64 = 8.0;

// Class-7 potential-field consts (mirror planner.rs exactly).
pub(crate) const PF_K_ATTRACT: f64 = 1.0;
pub(crate) const PF_K_REPEL: f64 = 30.0;
pub(crate) const PF_REPEL_RADIUS: f64 = 14.0;
pub(crate) const PF_K_SKIN: f64 = 1.5;
pub(crate) const PF_SKIN_BAND: f64 = 8.0;
pub(crate) const PF_W_ALIGN: f64 = 25.0;

// ── Geometry helpers shared with class 9 (verbatim from planner.rs) ──────────

/// Closest distance between segment [a0,a1] and segment [b0,b1] — the exact
/// Ericson formulation that `plan_coordination` used (moved here unchanged).
pub fn seg_seg_dist(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> f64 {
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

/// Distance from point `pt` to segment [s0,s1] (verbatim from planner.rs).
pub fn point_seg_dist(pt: Vec3, s0: Vec3, s1: Vec3) -> f64 {
    let d = s1.sub(s0);
    let l2 = d.dot(d);
    if l2 <= 1e-12 {
        return pt.sub(s0).len();
    }
    let t = (pt.sub(s0).dot(d) / l2).clamp(0.0, 1.0);
    pt.sub(s0.add(d.scale(t))).len()
}

// ── The CostTerm trait (approved blueprint signature) ────────────────────────

/// One additive cost component. `cost` returns this term's RAW contribution
/// (its scaling baked in where bit-identity demands it — see module docs);
/// `Profile::compute_cost` multiplies by the term's weight and folds it into
/// a strict left-to-right running sum, in fixed declaration order.
pub trait CostTerm: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn cost(
        &self,
        end_pos: Vec3,
        min_dist: f64,
        term: f64,
        player_pos: Vec3,
        target: Vec3,
        opponents: &[Vec3],
        teammates: &[Vec3],
        reel: i32,
        roll_state: &Rollout,
    ) -> f64;
    fn name(&self) -> &'static str;
}

// ── Term library ─────────────────────────────────────────────────────────────

/// `min_dist` (weight 1.0). Base objective.
struct MinDistTerm;
impl CostTerm for MinDistTerm {
    fn cost(
        &self,
        _ep: Vec3,
        min_dist: f64,
        _t: f64,
        _pp: Vec3,
        _tg: Vec3,
        _o: &[Vec3],
        _tm: &[Vec3],
        _r: i32,
        _rs: &Rollout,
    ) -> f64 {
        min_dist
    }
    fn name(&self) -> &'static str {
        "min_dist"
    }
}

/// Terminal momentum toward target. Value = `term`; the profile weight is
/// `-W_MOM`, so `weight*value == (-W_MOM)*term == -(W_MOM*term)`, the exact
/// `- W_MOM * term` of the original (IEEE negation is exact).
struct MomentumTerm;
impl CostTerm for MomentumTerm {
    fn cost(
        &self,
        _ep: Vec3,
        _md: f64,
        term: f64,
        _pp: Vec3,
        _tg: Vec3,
        _o: &[Vec3],
        _tm: &[Vec3],
        _r: i32,
        _rs: &Rollout,
    ) -> f64 {
        term
    }
    fn name(&self) -> &'static str {
        "momentum"
    }
}

/// COMPOUND defender-danger term. Returns the WHOLE original sum
/// `Σ over o (if od < DD { (DD-od)*3.5 } else 0 })` — the `*3.5`, the
/// per-opponent loop and the `od < DD` branch are all INSIDE the term so the
/// produced f64 (and its accumulation order over opponents) is identical to
/// the original `branch_cost`. Profile weight = 1.0.
struct DefenderDangerTerm;
impl CostTerm for DefenderDangerTerm {
    fn cost(
        &self,
        end_pos: Vec3,
        _md: f64,
        _t: f64,
        _pp: Vec3,
        _tg: Vec3,
        opponents: &[Vec3],
        _tm: &[Vec3],
        _r: i32,
        _rs: &Rollout,
    ) -> f64 {
        let mut c = 0.0;
        for o in opponents {
            let od = end_pos.sub(*o).len();
            if od < DEFENDER_DANGER {
                c += (DEFENDER_DANGER - od) * 3.5;
            }
        }
        c
    }
    fn name(&self) -> &'static str {
        "defender_danger"
    }
}

/// COMPOUND teammate-spacing term — the gated `(18-near_tm)*W_SPACE` of the
/// original `branch_cost` (W_SPACE baked in; default 0.0 ⇒ the
/// `if W_SPACE > 0.0` guard makes this exactly 0.0, byte-identical to the
/// original where the whole block was skipped). Profile weight = 1.0.
struct SpacingTerm;
impl CostTerm for SpacingTerm {
    fn cost(
        &self,
        end_pos: Vec3,
        _md: f64,
        _t: f64,
        _pp: Vec3,
        _tg: Vec3,
        _o: &[Vec3],
        teammates: &[Vec3],
        _r: i32,
        _rs: &Rollout,
    ) -> f64 {
        if W_SPACE > 0.0 {
            let mut near_tm = f64::INFINITY;
            for t in teammates {
                let td = end_pos.sub(*t).len();
                if td < near_tm {
                    near_tm = td;
                }
            }
            if near_tm < 18.0 {
                return (18.0 - near_tm) * W_SPACE;
            }
        }
        0.0
    }
    fn name(&self) -> &'static str {
        "spacing"
    }
}

/// COMPOUND class-9 interference term. Reproduces `coord_cost`'s teammate
/// loop EXACTLY: for each same-team teammate (Vec order) the end-proximity
/// penalty THEN the lane-crossing penalty, accumulated in that order, with
/// COORD_W_INTERF / COORD_W_LANE baked in. Profile weight = 1.0.
///
/// Teammate POSITIONS arrive via the `teammates` slice; the lane needs each
/// teammate's VELOCITY too, which the trait signature does not carry — so the
/// profile stores the (pos,vel) pairs and the planner routes the coordination
/// cost through `Profile::coord_compute_cost` (a thin specialization that
/// supplies them). The arithmetic is identical to the original closure.
struct CoordInterferenceTerm;
/// COMPOUND class-9 pass-setup attractor — the gated negative bonus
/// `-COORD_W_PASS*openness` of the original closure. Like interference it
/// needs the receiver, supplied via `Profile::coord_compute_cost`.
struct CoordPassTerm;

// (CoordInterferenceTerm / CoordPassTerm carry no per-call data; their actual
// computation lives in `coord_extra` so it can see teammate velocities + the
// receiver, exactly as the original `coord_cost` closure did. They implement
// CostTerm as inert 0.0 so the generic library stays uniform; class 9 adds
// `coord_extra` AFTER the branch_cost fold, preserving the original order.)
impl CostTerm for CoordInterferenceTerm {
    fn cost(
        &self,
        _ep: Vec3,
        _md: f64,
        _t: f64,
        _pp: Vec3,
        _tg: Vec3,
        _o: &[Vec3],
        _tm: &[Vec3],
        _r: i32,
        _rs: &Rollout,
    ) -> f64 {
        0.0
    }
    fn name(&self) -> &'static str {
        "coord_interference"
    }
}
impl CostTerm for CoordPassTerm {
    fn cost(
        &self,
        _ep: Vec3,
        _md: f64,
        _t: f64,
        _pp: Vec3,
        _tg: Vec3,
        _o: &[Vec3],
        _tm: &[Vec3],
        _r: i32,
        _rs: &Rollout,
    ) -> f64 {
        0.0
    }
    fn name(&self) -> &'static str {
        "coord_pass"
    }
}

// ── Profile (the weight vector + fixed-order term list) ──────────────────────

/// A planner class's cost identity: a fixed-order term list and matching
/// weight vector. `weights`/`engine_params` are name→f64 maps represented as
/// fixed-order `Vec`s (NEVER a HashMap — iteration order would break
/// bit-identity). `engine` is the planner class id.
pub struct Profile {
    pub label: String,
    pub engine: i32,
    /// name → weight, in the SAME order as `terms`. Stored as a Vec so the
    /// summation order is fixed and deterministic.
    pub weights: Vec<(&'static str, f64)>,
    /// name → engine parameter (PF gains etc.), fixed order. Not folded into
    /// `compute_cost`; carried for the engine to read explicitly.
    pub engine_params: Vec<(&'static str, f64)>,
    terms: Vec<Box<dyn CostTerm>>,
    /// True for class 9: append the coordination extra (interference + pass)
    /// AFTER the branch_cost fold, exactly where the original closure did.
    coord: bool,
}

impl Profile {
    /// Reconstruct a planner class's cost exactly. Classes 1..=8 share the
    /// branch_cost identity {min_dist:1.0, momentum:-W_MOM, defender_danger:
    /// 1.0(=3.5 baked), spacing:1.0(=W_SPACE baked)}. Class 9 adds the
    /// coordination extra. Class 0 uses the same four pieces via the MPC
    /// helpers. Class 7's selection score is geometry (`pf_score`), with
    /// branch_cost only for its sticky/tie-break — both routed here.
    pub fn from_class(class: i32) -> Profile {
        let label = match class {
            0 => "MPC",
            1 => "RRT",
            2 => "CEM",
            3 => "MPPI",
            4 => "SimAnneal",
            5 => "Beam",
            6 => "MCTS",
            7 => "PotentialField",
            8 => "RandomShooting",
            9 => "Coordination",
            _ => "MPC", // any id ≥ 10 falls back to MPC, exactly as planner.rs
        }
        .to_string();

        // The branch_cost identity, shared by all classes 0..=9 (class 0's
        // score_plan and class 7's sticky/tie-break also use it).
        let mut weights: Vec<(&'static str, f64)> = vec![
            ("min_dist", 1.0),
            ("momentum", -W_MOM),
            ("defender_danger", 1.0),
            ("spacing", 1.0),
        ];
        let mut terms: Vec<Box<dyn CostTerm>> = vec![
            Box::new(MinDistTerm),
            Box::new(MomentumTerm),
            Box::new(DefenderDangerTerm),
            Box::new(SpacingTerm),
        ];

        let coord = class == 9;
        if coord {
            weights.push(("coord_interference", 1.0));
            weights.push(("coord_pass", 1.0));
            terms.push(Box::new(CoordInterferenceTerm));
            terms.push(Box::new(CoordPassTerm));
        }

        // Class 7 carries the potential-field gains as engine params so the
        // engine can read them via one source of truth (the field arithmetic
        // itself is `pf_grad`/`pf_score`, byte-for-byte the original).
        let engine_params: Vec<(&'static str, f64)> = if class == 7 {
            vec![
                ("pf_k_attract", PF_K_ATTRACT),
                ("pf_k_repel", PF_K_REPEL),
                ("pf_repel_radius", PF_REPEL_RADIUS),
                ("pf_k_skin", PF_K_SKIN),
                ("pf_skin_band", PF_SKIN_BAND),
                ("pf_w_align", PF_W_ALIGN),
            ]
        } else {
            Vec::new()
        };

        Profile {
            label,
            engine: class,
            weights,
            engine_params,
            terms,
            coord,
        }
    }

    /// The planner class id this profile dispatches as (its `engine`).
    /// `from_class(n)` always yields a profile whose `class_number()` is the
    /// dispatch arm: 0..=9 are the named engines, ≥10 falls back to MPC but
    /// `engine` retains the requested id (dispatch treats unknown as MPC).
    pub fn class_number(&self) -> i32 {
        self.engine
    }

    /// The production-default profile: Coordination (class 9). A free
    /// function mirror of `from_class(9)` for the bridge's lazy thread-local
    /// init — semantically identical, kept as one source of truth.
    pub fn coordination_default() -> Profile {
        Profile::from_class(9)
    }

    /// The shared branch_cost identity (classes 1..=8 selection; class 9 base
    /// before the coord extra; class 0/7's helpers reuse the same pieces).
    /// Strict left-fold over the fixed-order term list:
    /// `c = 0.0; for i: c = c + weights[i] * terms[i].cost(...)`.
    /// Bit-identical to the original `branch_cost` (see module docs).
    #[allow(clippy::too_many_arguments)]
    pub fn branch_cost(
        &self,
        end_pos: Vec3,
        min_dist: f64,
        term: f64,
        player_pos: Vec3,
        target: Vec3,
        opponents: &[Vec3],
        teammates: &[Vec3],
        reel: i32,
        roll_state: &Rollout,
    ) -> f64 {
        let mut c = 0.0;
        // Only the four branch_cost terms participate here; the coord extra
        // (if any) is appended by `coord_compute_cost` AFTER this fold, in the
        // exact original order.
        for (i, t) in self.terms.iter().enumerate() {
            let name = t.name();
            if name == "coord_interference" || name == "coord_pass" {
                continue;
            }
            let w = self.weights[i].1;
            c = c
                + w * t.cost(
                    end_pos, min_dist, term, player_pos, target, opponents,
                    teammates, reel, roll_state,
                );
        }
        c
    }

    /// Class-9 cost: `branch_cost` THEN the coordination extra (interference:
    /// per-teammate end-prox then lane, in Vec order; then the pass bonus),
    /// accumulated in EXACTLY the original `coord_cost` closure order.
    /// `tm` = same-team (pos,vel) pairs in Vec order; `receiver` = the
    /// most-forward teammate (the original's deterministic pick).
    #[allow(clippy::too_many_arguments)]
    pub fn coord_compute_cost(
        &self,
        end_p: Vec3,
        run_min: f64,
        term: f64,
        player_pos: Vec3,
        target: Vec3,
        opponents: &[Vec3],
        teammates: &[Vec3],
        reel: i32,
        roll_state: &Rollout,
        tm: &[(Vec3, Vec3)],
        receiver: Option<Vec3>,
    ) -> f64 {
        // (base) branch_cost — identical fold as every other class.
        let mut c = self.branch_cost(
            end_p, run_min, term, player_pos, target, opponents, teammates,
            reel, roll_state,
        );
        if !self.coord {
            return c;
        }
        // (a) interference: end-proximity + lane-crossing, per teammate, in
        // Vec order — verbatim from the original `coord_cost` closure.
        for &(tp, tv) in tm {
            let dend = end_p.sub(tp).len();
            if dend < COORD_INTERF_RADIUS {
                c += (COORD_INTERF_RADIUS - dend) * COORD_W_INTERF;
            }
            let lane_end = tp.add(tv.scale(COORD_LANE_LEN));
            let sep = seg_seg_dist(player_pos, end_p, tp, lane_end);
            if sep < COORD_LANE_RADIUS {
                c += (COORD_LANE_RADIUS - sep) * COORD_W_LANE;
            }
        }
        // (b) pass-setup attractor — verbatim.
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
                    let openness = 1.0 - to_rcv / COORD_PASS_RANGE;
                    c -= COORD_W_PASS * openness;
                }
            }
        }
        c
    }
}

// ── Class-0 MPC fallback cost (score_plan), exact sub-expressions ────────────
// The original `spacing_pen` closure: same `(18-near_tm)*W_SPACE` gated form.
pub fn mpc_spacing_pen(end_ref: Vec3, teammates: &[Vec3]) -> f64 {
    if W_SPACE <= 0.0 {
        return 0.0;
    }
    let mut near_tm = f64::INFINITY;
    for t in teammates {
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
}

/// No-defender MPC cost: `cd - W_MOM*term + spacing_pen(anchor)` — the exact
/// original expression (left-to-right, identical sub-expressions).
pub fn mpc_cost_open(cd: f64, term: f64, anchor: Vec3, teammates: &[Vec3]) -> f64 {
    cd - W_MOM * term + mpc_spacing_pen(anchor, teammates)
}

/// Defender MPC cost: `closest_dist - W_MOM*term + danger + spacing_pen`,
/// where `danger = if min_opp_dist < DD { (DD-min_opp_dist)*3.5 } else 0`.
/// Byte-for-byte the original `score_plan` defender branch.
pub fn mpc_cost_def(
    closest_dist: f64,
    term: f64,
    min_opp_dist: f64,
    anchor: Vec3,
    teammates: &[Vec3],
) -> f64 {
    let danger = if min_opp_dist < DEFENDER_DANGER {
        (DEFENDER_DANGER - min_opp_dist) * 3.5
    } else {
        0.0
    };
    closest_dist - W_MOM * term + danger + mpc_spacing_pen(anchor, teammates)
}

// ── Class-7 potential field — geometry, reproduced verbatim ──────────────────

/// The summed navigation-field gradient at `pos` (attractor + FIRAS opponent
/// repulsors + skin repulsor). Byte-for-byte the original
/// `potential_field_grad`. `r` is the hull radius (tuning::R).
pub fn pf_grad(pos: Vec3, target: Vec3, opponents: &[Vec3], r: f64) -> Vec3 {
    let mut g = pf_to_target_dir(pos, target).scale(PF_K_ATTRACT);
    for o in opponents {
        let away = pos.sub(*o);
        let rho = away.len();
        if rho > 1e-6 && rho < PF_REPEL_RADIUS {
            let mag = PF_K_REPEL * (1.0 / rho - 1.0 / PF_REPEL_RADIUS) / (rho * rho);
            g = g.add(away.scale(mag / rho));
        }
    }
    let rho_yz = (pos.y * pos.y + pos.z * pos.z).sqrt();
    if rho_yz > r - PF_SKIN_BAND {
        let depth = rho_yz - (r - PF_SKIN_BAND);
        let inward = Vec3::new(0.0, -pos.y / rho_yz, -pos.z / rho_yz);
        g = g.add(inward.scale(PF_K_SKIN * depth));
    }
    g
}

/// `to_target_dir` — identical to planner.rs (kept private there; mirrored
/// here only for `pf_grad`'s attractor, byte-for-byte).
fn pf_to_target_dir(pos: Vec3, target: Vec3) -> Vec3 {
    let d = target.sub(pos);
    let l = d.len();
    if l > 1e-6 {
        d.scale(1.0 / l)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    }
}

/// Class-7 selection score `PF_W_ALIGN*align - min_dist` (NOT branch_cost):
/// the exact original expression.
pub fn pf_score(align: f64, min_dist: f64) -> f64 {
    PF_W_ALIGN * align - min_dist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rs() -> Rollout {
        Rollout {
            p: Vec3::new(0.0, 0.0, 0.0),
            v: Vec3::new(0.0, 0.0, 0.0),
            min_dist: 0.0,
            term: 0.0,
        }
    }

    /// The Profile branch_cost fold is bit-identical to the literal original
    /// `branch_cost` arithmetic, across opponents/teammates.
    #[test]
    fn profile_branch_cost_matches_literal() {
        // Literal reference (the pre-extraction branch_cost body).
        fn legacy(p: Vec3, min_dist: f64, term: f64, opp: &[Vec3], tm: &[Vec3]) -> f64 {
            let mut c = min_dist - W_MOM * term;
            for o in opp {
                let od = p.sub(*o).len();
                if od < DEFENDER_DANGER {
                    c += (DEFENDER_DANGER - od) * 3.5;
                }
            }
            if W_SPACE > 0.0 {
                let mut near_tm = f64::INFINITY;
                for t in tm {
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
        let prof = Profile::from_class(8);
        let cases = [
            (Vec3::new(1.0, 2.0, 3.0), 12.5, 4.2),
            (Vec3::new(-7.0, 0.0, 9.0), 0.0, 0.0),
            (Vec3::new(40.0, 5.0, 0.0), 33.3, 1.7),
        ];
        let opp = [Vec3::new(2.0, 2.0, 3.0), Vec3::new(40.0, 5.0, 0.0)];
        let tm = [Vec3::new(0.0, 0.0, 0.0)];
        for (p, md, t) in cases {
            let want = legacy(p, md, t, &opp, &tm);
            let got = prof.branch_cost(
                p, md, t, Vec3::new(0.0, 0.0, 0.0), Vec3::new(99.0, 0.0, 0.0),
                &opp, &tm, -1, &rs(),
            );
            assert_eq!(want.to_bits(), got.to_bits(), "branch_cost mismatch");
        }
    }

    // ── Literal legacy references (pre-extraction arithmetic, copied
    //    verbatim from planner.rs before the keystone refactor) ────────────
    fn legacy_branch(p: Vec3, min_dist: f64, term: f64, opp: &[Vec3], tm: &[Vec3]) -> f64 {
        let mut c = min_dist - W_MOM * term;
        for o in opp {
            let od = p.sub(*o).len();
            if od < DEFENDER_DANGER {
                c += (DEFENDER_DANGER - od) * 3.5;
            }
        }
        if W_SPACE > 0.0 {
            let mut near_tm = f64::INFINITY;
            for t in tm {
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

    #[allow(clippy::too_many_arguments)]
    fn legacy_coord(
        end_p: Vec3,
        run_min: f64,
        term: f64,
        pos: Vec3,
        opp: &[Vec3],
        tmm: &[Vec3],
        tm: &[(Vec3, Vec3)],
        receiver: Option<Vec3>,
    ) -> f64 {
        let mut c = legacy_branch(end_p, run_min, term, opp, tmm);
        for &(tp, tv) in tm {
            let dend = end_p.sub(tp).len();
            if dend < COORD_INTERF_RADIUS {
                c += (COORD_INTERF_RADIUS - dend) * COORD_W_INTERF;
            }
            let lane_end = tp.add(tv.scale(COORD_LANE_LEN));
            let sep = seg_seg_dist(pos, end_p, tp, lane_end);
            if sep < COORD_LANE_RADIUS {
                c += (COORD_LANE_RADIUS - sep) * COORD_W_LANE;
            }
        }
        if let Some(rp) = receiver {
            let to_rcv = rp.sub(end_p).len();
            if to_rcv > 1e-6 && to_rcv < COORD_PASS_RANGE {
                let mut blocked = false;
                for o in opp {
                    if point_seg_dist(*o, end_p, rp) < COORD_LOS_RADIUS {
                        blocked = true;
                        break;
                    }
                }
                if !blocked {
                    let openness = 1.0 - to_rcv / COORD_PASS_RANGE;
                    c -= COORD_W_PASS * openness;
                }
            }
        }
        c
    }

    fn legacy_grad(pos: Vec3, target: Vec3, opp: &[Vec3], r: f64) -> Vec3 {
        let mut g = pf_to_target_dir(pos, target).scale(PF_K_ATTRACT);
        for o in opp {
            let away = pos.sub(*o);
            let rho = away.len();
            if rho > 1e-6 && rho < PF_REPEL_RADIUS {
                let mag =
                    PF_K_REPEL * (1.0 / rho - 1.0 / PF_REPEL_RADIUS) / (rho * rho);
                g = g.add(away.scale(mag / rho));
            }
        }
        let rho_yz = (pos.y * pos.y + pos.z * pos.z).sqrt();
        if rho_yz > r - PF_SKIN_BAND {
            let depth = rho_yz - (r - PF_SKIN_BAND);
            let inward = Vec3::new(0.0, -pos.y / rho_yz, -pos.z / rho_yz);
            g = g.add(inward.scale(PF_K_SKIN * depth));
        }
        g
    }

    /// Per-class bit-identical cost proof. For EACH planner class N=0..=9,
    /// the `Profile::from_class(N)` cost path must reproduce that class's
    /// ORIGINAL (pre-extraction) cost arithmetic bit-for-bit, across several
    /// states + opponent/teammate configs.
    #[test]
    fn every_class_cost_is_bit_identical_to_legacy() {
        let states: [(Vec3, f64, f64); 4] = [
            (Vec3::new(1.0, 2.0, 3.0), 12.5, 4.2),
            (Vec3::new(-7.0, 1.0, 9.0), 0.0, 0.0),
            (Vec3::new(40.0, 5.0, 0.0), 33.3, 1.7),
            (Vec3::new(5.0, 5.0, 5.0), 7.0, -0.0),
        ];
        let opp_sets: [Vec<Vec3>; 3] = [
            vec![],
            vec![Vec3::new(2.0, 2.0, 3.0)],
            vec![Vec3::new(2.0, 2.0, 3.0), Vec3::new(40.0, 5.0, 0.0)],
        ];
        let tm_pos = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(40.0, 6.0, 2.0)];
        let tm_pv = [
            (Vec3::new(40.0, 6.0, 2.0), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(100.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)),
        ];
        let receiver = Some(Vec3::new(100.0, 0.0, 0.0));
        let pos = Vec3::new(0.0, 12.0, 0.0);
        let tgt = Vec3::new(120.0, 0.0, 0.0);

        for class in 0..=9 {
            let prof = Profile::from_class(class);
            for opp in &opp_sets {
                for &(p, md, t) in &states {
                    // Classes 1..=8 (+ class 9 base, + class 0/7 helpers):
                    // branch_cost identity.
                    let want_b = legacy_branch(p, md, t, opp, &tm_pos);
                    let got_b = prof.branch_cost(
                        p, md, t, pos, tgt, opp, &tm_pos, -1, &rs(),
                    );
                    assert_eq!(
                        want_b.to_bits(),
                        got_b.to_bits(),
                        "class {class}: branch_cost not bit-identical"
                    );

                    // Class 9: branch_cost + coordination extra.
                    let want_c =
                        legacy_coord(p, md, t, pos, opp, &tm_pos, &tm_pv, receiver);
                    let got_c = prof.coord_compute_cost(
                        p, md, t, pos, tgt, opp, &tm_pos, -1, &rs(), &tm_pv,
                        receiver,
                    );
                    if class == 9 {
                        assert_eq!(
                            want_c.to_bits(),
                            got_c.to_bits(),
                            "class 9: coord cost not bit-identical"
                        );
                    } else {
                        // Non-coord classes: coord_compute_cost == branch_cost.
                        assert_eq!(
                            got_c.to_bits(),
                            got_b.to_bits(),
                            "class {class}: coord_compute_cost must equal branch_cost"
                        );
                    }

                    // Class 0 MPC fallback helpers (no-defender + defender).
                    let want_open = p.sub(tgt).len() - W_MOM * t
                        + super::mpc_spacing_pen(p, &tm_pos);
                    let got_open = super::mpc_cost_open(
                        p.sub(tgt).len(), t, p, &tm_pos,
                    );
                    assert_eq!(
                        want_open.to_bits(),
                        got_open.to_bits(),
                        "class 0: mpc_cost_open not bit-identical"
                    );
                    let mod_ = if opp.is_empty() { 99.0 } else { opp[0].sub(p).len() };
                    let danger = if mod_ < DEFENDER_DANGER {
                        (DEFENDER_DANGER - mod_) * 3.5
                    } else {
                        0.0
                    };
                    let want_def = md - W_MOM * t + danger
                        + super::mpc_spacing_pen(p, &tm_pos);
                    let got_def =
                        super::mpc_cost_def(md, t, mod_, p, &tm_pos);
                    assert_eq!(
                        want_def.to_bits(),
                        got_def.to_bits(),
                        "class 0: mpc_cost_def not bit-identical"
                    );

                    // Class 7 field gradient + selection score.
                    let wg = legacy_grad(p, tgt, opp, 45.0);
                    let gg = pf_grad(p, tgt, opp, 45.0);
                    assert_eq!(wg.x.to_bits(), gg.x.to_bits(), "class 7 grad.x");
                    assert_eq!(wg.y.to_bits(), gg.y.to_bits(), "class 7 grad.y");
                    assert_eq!(wg.z.to_bits(), gg.z.to_bits(), "class 7 grad.z");
                    let want_s = PF_W_ALIGN * 0.37 - md;
                    assert_eq!(
                        want_s.to_bits(),
                        pf_score(0.37, md).to_bits(),
                        "class 7 pf_score not bit-identical"
                    );
                }
            }
        }
    }
}

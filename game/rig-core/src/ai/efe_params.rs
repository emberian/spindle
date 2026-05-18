//! `EfeParams` — the tunable parameter surface of the EFE + w-maxing
//! controller (`ai::efe` + `ai::rigger_ai`'s `wmax_coverage_target` /
//! `decide_throw`).
//!
//! WHY THIS EXISTS. The coordination learner (`crate::coord_learner`)
//! optimizes the active-inference controller *for team coordination*. The
//! knobs it turns are EXACTLY the EFE term weights, the prediction
//! horizons, the epistemic/coverage gains and the w-maxing acceptance
//! thresholds — i.e. the controller's own decision shape, NOT a meta
//! ranker. This struct is that surface, made data so it can be evolved.
//!
//! BEHAVIOR-PRESERVING DEFAULT. `EfeParams::default()` is the *verbatim*
//! set of constants that were hardcoded in `rigger_ai.rs` before this
//! refactor. With the default installed (the production case) the
//! controller is bit-identical to before — proved by
//! `default_is_behavior_preserving` in `rigger_ai`'s tests and the
//! existing determinism gates.
//!
//! DETERMINISM. Like the planner-profile knob in `plan_bridge`, the
//! active params live in a `thread_local!` cell so the native parallel
//! self-play loop can evaluate independent parameter sets on rayon
//! threads with zero cross-thread clobber. No RNG, no wall-clock, no
//! HashMap-iteration folds are introduced here. Production wasm is
//! single-threaded so the thread-local default is observationally
//! identical to a global.

use std::cell::RefCell;
use std::rc::Rc;

/// The evolvable parameter set of the EFE + w-maxing controller.
///
/// Every field below replaces a literal that used to be hardcoded in
/// `rigger_ai::wmax_coverage_target` or `rigger_ai::decide_throw`. The
/// `default()` values ARE those literals — do not change them; the
/// learner explores around this point, production keeps it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EfeParams {
    // ── wmax_coverage_target: navigation EFE (off-ball coverage) ──────────
    /// Off-ball offense anchor: how far ahead of the bell (along attack)
    /// the coverage fan is centred. (was `+ sgn * 40.0`)
    pub cov_offense_anchor_ahead: f64,
    /// Defense (Zone) anchor offset from our own ring. (was `+ sgn * 30.0`)
    pub cov_defense_anchor_ahead: f64,
    /// Pragmatic close-velocity assumed when scoring response gap, m/s.
    /// (was `22.0` in `expected_response_gap`)
    pub cov_close_v: f64,
    /// Forward-pull gain that softens the pragmatic term toward attack so
    /// offense still flows up-field. (was `6.0`)
    pub cov_fwd_pull_gain: f64,
    /// EPISTEMIC crowding-penalty weight (anti-clump). (was `10.0`)
    pub cov_crowd_w: f64,
    /// EPISTEMIC axis-redundancy weight (don't all stack at one depth).
    /// (was `6.0`)
    pub cov_redun_w: f64,
    /// W-MAXING breadth bonus: the negative EFE nudge given to the broad
    /// central depth candidate so the weakest-sufficient (least committal)
    /// region wins ties. More negative ⇒ stronger generalization bias.
    /// (was `-1.5`)
    pub cov_breadth_bonus: f64,

    // ── decide_throw: the w-maxing active-inference throw controller ──────
    /// Pragmatic acceptance threshold: a candidate launch is "fit" only if
    /// its predicted outcome quality ≥ this. LOWER ⇒ throws more often /
    /// weaker preconditions (the Bennett anti-brittle direction). (was
    /// `0.55` = `good_enough`)
    pub throw_good_enough: f64,
    /// Generative roll horizon (s) used to score a launch's outcome.
    /// (was `3.0`)
    pub throw_eval_horizon: f64,
    /// Teammate-future prediction horizon (s) for "catchable by where
    /// they're going". (was `0.9`)
    pub throw_tm_horizon: f64,
    /// Catch-envelope radius (m) for the reachable-by-teammate reward.
    /// Wider = more generous / weaker precondition. (was `38.0`)
    pub throw_reach_radius: f64,
    /// W-maxing tolerance-band error magnitude scale: the aim-error fan is
    /// perturbed by `base + throw_variance * this`. LARGER ⇒ "robust"
    /// means robust to bigger error ⇒ favours wider, less brittle throws.
    /// (was `60.0`)
    pub throw_err_mag_scale: f64,
    /// Stall-pressure tick count past which a (weak, robust) release is
    /// forced rather than carrying a dead ball. (was `540.0`)
    pub throw_stall_force_ticks: f64,
    /// Opponent distance (m) under which the carrier is "pressured" and
    /// will take a robust outlet. (was `14.0`)
    pub throw_pressure_dist: f64,
}

impl Default for EfeParams {
    /// VERBATIM the constants that were inline in `rigger_ai.rs`. Changing
    /// any value here is a production behavior change — don't. The learner
    /// searches around this point.
    fn default() -> Self {
        EfeParams {
            cov_offense_anchor_ahead: 40.0,
            cov_defense_anchor_ahead: 30.0,
            cov_close_v: 22.0,
            cov_fwd_pull_gain: 6.0,
            cov_crowd_w: 10.0,
            cov_redun_w: 6.0,
            cov_breadth_bonus: -1.5,
            throw_good_enough: 0.55,
            throw_eval_horizon: 3.0,
            throw_tm_horizon: 0.9,
            throw_reach_radius: 38.0,
            throw_err_mag_scale: 60.0,
            throw_stall_force_ticks: 540.0,
            throw_pressure_dist: 14.0,
        }
    }
}

impl EfeParams {
    /// Fixed-order (name, value) view — the GENOME the learner mutates /
    /// crosses over. Order is the immutable scaffold (never reordered) so
    /// the seeded evolution is bit-reproducible. `from_genes` is its exact
    /// inverse.
    pub fn to_genes(&self) -> [(&'static str, f64); 14] {
        [
            ("cov_offense_anchor_ahead", self.cov_offense_anchor_ahead),
            ("cov_defense_anchor_ahead", self.cov_defense_anchor_ahead),
            ("cov_close_v", self.cov_close_v),
            ("cov_fwd_pull_gain", self.cov_fwd_pull_gain),
            ("cov_crowd_w", self.cov_crowd_w),
            ("cov_redun_w", self.cov_redun_w),
            ("cov_breadth_bonus", self.cov_breadth_bonus),
            ("throw_good_enough", self.throw_good_enough),
            ("throw_eval_horizon", self.throw_eval_horizon),
            ("throw_tm_horizon", self.throw_tm_horizon),
            ("throw_reach_radius", self.throw_reach_radius),
            ("throw_err_mag_scale", self.throw_err_mag_scale),
            ("throw_stall_force_ticks", self.throw_stall_force_ticks),
            ("throw_pressure_dist", self.throw_pressure_dist),
        ]
    }

    /// Rebuild from a fixed-order gene vector (the exact inverse of
    /// `to_genes`). Asserts the scaffold order so a mis-ordered genome is
    /// a hard bug, never silent drift.
    pub fn from_genes(g: &[(&'static str, f64)]) -> Self {
        let scaffold = Self::default().to_genes();
        debug_assert_eq!(g.len(), scaffold.len(), "genome length mismatch");
        for (a, b) in g.iter().zip(scaffold.iter()) {
            debug_assert_eq!(a.0, b.0, "genome term-order must match scaffold");
        }
        EfeParams {
            cov_offense_anchor_ahead: g[0].1,
            cov_defense_anchor_ahead: g[1].1,
            cov_close_v: g[2].1,
            cov_fwd_pull_gain: g[3].1,
            cov_crowd_w: g[4].1,
            cov_redun_w: g[5].1,
            cov_breadth_bonus: g[6].1,
            throw_good_enough: g[7].1,
            throw_eval_horizon: g[8].1,
            throw_tm_horizon: g[9].1,
            throw_reach_radius: g[10].1,
            throw_err_mag_scale: g[11].1,
            throw_stall_force_ticks: g[12].1,
            throw_pressure_dist: g[13].1,
        }
    }

    /// Per-gene sane bounds (lo, hi) in the SAME fixed order as
    /// `to_genes`. Keeps the controller finite and the search in a
    /// behaviorally meaningful region (the default point is interior to
    /// every interval). The learner clips every mutated gene to these.
    pub fn bounds() -> [(f64, f64); 14] {
        [
            (0.0, 120.0),   // cov_offense_anchor_ahead
            (0.0, 90.0),    // cov_defense_anchor_ahead
            (6.0, 45.0),    // cov_close_v
            (0.0, 30.0),    // cov_fwd_pull_gain
            (0.0, 40.0),    // cov_crowd_w
            (0.0, 30.0),    // cov_redun_w
            (-12.0, 0.0),   // cov_breadth_bonus (≤0: it is a bonus)
            (0.15, 0.95),   // throw_good_enough (a quality fraction)
            (1.0, 6.0),     // throw_eval_horizon (s)
            (0.3, 2.5),     // throw_tm_horizon (s)
            (12.0, 90.0),   // throw_reach_radius (m)
            (10.0, 140.0),  // throw_err_mag_scale
            (120.0, 1200.0),// throw_stall_force_ticks
            (4.0, 40.0),    // throw_pressure_dist (m)
        ]
    }

    /// Clip every gene to `bounds()` in place (used by the learner after
    /// mutation/crossover). Deterministic, no RNG.
    pub fn clipped(mut self) -> Self {
        let b = Self::bounds();
        let mut g = self.to_genes();
        for i in 0..g.len() {
            let (lo, hi) = b[i];
            if g[i].1 < lo {
                g[i].1 = lo;
            } else if g[i].1 > hi {
                g[i].1 = hi;
            }
        }
        self = Self::from_genes(&g);
        self
    }

    /// Bennett w-maxing "weakness" of this parameter set: a deterministic,
    /// monotone scalar that is LARGER when the controller's preconditions
    /// are WEAKER (more generalizing / less brittle) — lower throw
    /// acceptance threshold, wider catch envelope, larger robustness fan,
    /// stronger breadth bonus, more forgiving stall/pressure. The learner
    /// uses this ONLY as a tiny tie-break toward the weakest-sufficient
    /// parameterization among near-equal game outcomes (the generalization
    /// criterion — keeps it from overfitting one matchup). It is NOT the
    /// fitness and NOT a meta-instrument.
    pub fn weakness(&self) -> f64 {
        // Normalise each "weaker ⇒ larger" contribution into ~[0,1] by its
        // own bound, then average. Pure arithmetic, order-fixed.
        let b = Self::bounds();
        let norm = |v: f64, i: usize| {
            let (lo, hi) = b[i];
            ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
        };
        // lower good_enough = weaker ⇒ invert
        let w_thresh = 1.0 - norm(self.throw_good_enough, 7);
        let w_reach = norm(self.throw_reach_radius, 10);
        let w_errfan = norm(self.throw_err_mag_scale, 11);
        // more negative breadth bonus = stronger generalization ⇒ invert
        let w_breadth = 1.0 - norm(self.cov_breadth_bonus, 6);
        let w_stall = norm(self.throw_stall_force_ticks, 12);
        (w_thresh + w_reach + w_errfan + w_breadth + w_stall) / 5.0
    }
}

thread_local! {
    static EFE_PARAMS: RefCell<Option<Rc<EfeParams>>> =
        const { RefCell::new(None) };
}

/// Install an explicit `EfeParams` on the calling thread (the self-play
/// learner). `None` restores the lazily-cached production default. Set
/// before an episode; constant during it. Thread-local ⇒ only the calling
/// thread is affected (parallel-safe, exactly like the planner knob).
pub fn set_efe_params(p: Option<Rc<EfeParams>>) {
    EFE_PARAMS.with(|c| *c.borrow_mut() = p);
}

/// The active `EfeParams` on the calling thread. If none installed (the
/// production / test case) returns the per-thread lazily-cached default,
/// built at most once per thread and thereafter a cheap `Rc::clone` — NO
/// per-tick allocation (mirrors `plan_bridge::planner_profile`).
pub fn efe_params() -> Rc<EfeParams> {
    EFE_PARAMS.with(|c| {
        if let Some(p) = c.borrow().as_ref() {
            return Rc::clone(p);
        }
        let def = Rc::new(EfeParams::default());
        *c.borrow_mut() = Some(Rc::clone(&def));
        def
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genes_round_trip_bit_identical() {
        let p = EfeParams::default();
        let g = p.to_genes();
        let p2 = EfeParams::from_genes(&g);
        assert_eq!(p, p2);
        // every field bit-identical through the round-trip
        for (a, b) in p.to_genes().iter().zip(p2.to_genes().iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1.to_bits(), b.1.to_bits());
        }
    }

    #[test]
    fn default_is_interior_to_bounds() {
        let g = EfeParams::default().to_genes();
        let b = EfeParams::bounds();
        for (i, (name, v)) in g.iter().enumerate() {
            let (lo, hi) = b[i];
            assert!(
                *v >= lo && *v <= hi,
                "default {name}={v} outside bound [{lo},{hi}]"
            );
        }
    }

    #[test]
    fn clipped_keeps_in_bounds_and_is_idempotent() {
        let mut g = EfeParams::default().to_genes();
        for gene in g.iter_mut() {
            gene.1 = 1e9; // force every gene out of range
        }
        let c = EfeParams::from_genes(&g).clipped();
        let c2 = c.clipped();
        assert_eq!(c, c2, "clip must be idempotent");
        let b = EfeParams::bounds();
        for (i, (_, v)) in c.to_genes().iter().enumerate() {
            let (lo, hi) = b[i];
            assert!(*v >= lo && *v <= hi);
        }
    }

    #[test]
    fn thread_local_default_and_isolation() {
        std::thread::spawn(|| {
            // untouched thread ⇒ the production default
            assert_eq!(*efe_params(), EfeParams::default());
            let mut custom = EfeParams::default();
            custom.throw_good_enough = 0.3;
            set_efe_params(Some(Rc::new(custom)));
            assert_eq!(efe_params().throw_good_enough, 0.3);
            set_efe_params(None);
            assert_eq!(*efe_params(), EfeParams::default());
        })
        .join()
        .unwrap();
        // a sibling thread is unaffected
        std::thread::spawn(|| {
            assert_eq!(*efe_params(), EfeParams::default());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn weaker_params_have_higher_weakness() {
        let strong = EfeParams::default();
        let mut weak = EfeParams::default();
        weak.throw_good_enough = 0.2; // accept more (weaker precondition)
        weak.throw_reach_radius = 85.0; // wider envelope
        weak.throw_err_mag_scale = 130.0; // robust to bigger error
        weak.cov_breadth_bonus = -10.0; // stronger generalization bias
        assert!(
            weak.weakness() > strong.weakness(),
            "weaker preconditions must score higher weakness ({} vs {})",
            weak.weakness(),
            strong.weakness()
        );
    }
}

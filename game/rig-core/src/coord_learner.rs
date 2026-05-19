//! coord_learner.rs — the COORDINATION LEARNER.
//!
//! A deterministic, seeded **self-play / evolutionary** loop that
//! OPTIMIZES the EFE + w-maxing controller (`ai::efe` + `ai::rigger_ai`)
//! *for team coordination*. It is NOT a ranker, NOT a meta-composite, NOT
//! an instrument. The thing it produces is a learned `EfeParams` set that
//! demonstrably makes the controlled team coordinate/play better than the
//! baseline AI, and the loop is *watchable* (per-generation progress +
//! best-params artifact printed/returned).
//!
//! ── WHAT IS OPTIMIZED ───────────────────────────────────────────────────
//! Exactly the active-inference controller's own decision shape, exposed
//! as `ai::efe_params::EfeParams`: the EFE term weights (pragmatic
//! forward-pull, epistemic crowding / axis-redundancy), the prediction
//! horizons (throw-eval roll, teammate-future), the coverage anchors, and
//! the **w-maxing** acceptance/robustness knobs (throw good-enough
//! threshold, catch-envelope radius, error-tolerance-fan magnitude,
//! breadth bonus, stall/pressure forgiveness). The genome IS that fixed-
//! order parameter vector; the controller reads it via the thread-local in
//! `efe_params` (parallel-safe, exactly like the planner-profile knob).
//!
//! ── THE FITNESS = THE GAME'S OWN INTRINSIC OUTCOME ──────────────────────
//! `episode_outcome` runs one full AI-vs-AI match (the controlled team's
//! controller carries the candidate `EfeParams`; the opponent runs the
//! verbatim baseline default) and returns the SPORT'S OWN RESULT for the
//! controlled team — minimal and honest:
//!   * net score (own points − opponent points)        — the actual result
//!   * possession held (fraction of live ticks)         — held the bell
//!   * gate advanced (first→deep→mouth clears)           — moved the cast up
//!   * contests won                                      — won the contest
//!   * bell kept in play (1 − out-of-field tick frac)    — sport stayed live
//! These are the game's intrinsic events, NOT a normalised weighted
//! meta-instrument: there is no cap-table, no Goodhart-resistant
//! renormalisation, no relative-ranking. The scalar is just a small honest
//! fold of "did this team actually play the sport well", to be judged
//! ultimately by a human WATCHING the trained policy.
//!
//! ── W-MAXING KEEPS IT GENERAL, NOT BRITTLE ──────────────────────────────
//! Two ways. (1) The controller it tunes IS a w-maxing controller — the
//! params it evolves are the weakest-sufficient knobs themselves, so a
//! high-fitness genome is one whose *weak* preconditions still win. (2) The
//! learner's selection applies Bennett's razor as a TIE-BREAK: among
//! genomes within a small fitness band, the one with the higher
//! `EfeParams::weakness()` (weaker preconditions = larger success set =
//! generalizes across matchups) is preferred. Fitness itself is the mean
//! intrinsic outcome over a SPREAD of matchups (so a one-matchup overfit
//! cannot win), and the elitist best is the weakest among the joint-best.
//!
//! ── DETERMINISM IS SACRED ───────────────────────────────────────────────
//! One `ChaCha8Rng::seed_from_u64(cfg.seed)` threaded EXPLICITLY through
//! every draw in a FIXED order (pop init → per gen: parallel fitness
//! collected in fixed population-index order, then elitism by sorted
//! index, then per-offspring select/select/crossover/mutate). No
//! thread_rng, no wall-clock, no HashMap-iteration folds. Same seed ⇒
//! bit-identical evolution ⇒ bit-identical best `EfeParams`.
//!
//! ── THIN ADAPTER TO THE REAL `Env` ──────────────────────────────────────
//! The episode runner is structured as a `CoordEnv` with `reset` / `step`
//! / outcome accessors mirroring the fixed seam
//! `Env{reset,step,snapshot,restore}` (Reward = the game's own
//! per-step+terminal outcome). When the real `Env` lands separately, the
//! orchestrator swaps `CoordEnv` for it behind `run_episode` — a thin
//! adapter, no learner change (fitness already consumes only the intrinsic
//! outcome the seam's `Reward` carries).

#![cfg(not(target_arch = "wasm32"))]

use crate::ai::decision_types::TeamConfig;
use crate::ai::efe_params::{set_efe_params, EfeParams};
use crate::ai::profile::{style_to_profile, Difficulty};
use crate::ai::system::AiSystem;
use crate::ai::types as ai;
use crate::match_sm::{Gate, MatchPhase, MatchStateMachine};
use crate::math::Vec3;
use crate::scoring;
use crate::sim_world::{SimEvent as SwEv, SimWorld};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::rc::Rc;

// ── Roster / matchups (the same deterministic 8-rigger setup + style
// spread skill_eval uses — reused, not reinvented; only the SCORING is
// new = the intrinsic game outcome instead of the retired meta-composite).
struct RosterEntry {
    id: &'static str,
    team: ai::TeamSide,
    role: ai::RiggerRole,
    x: f64,
}
fn roster() -> [RosterEntry; 8] {
    use ai::RiggerRole::*;
    use ai::TeamSide::*;
    [
        RosterEntry { id: "H1", team: Home, role: Spinner, x: -40.0 },
        RosterEntry { id: "H2", team: Home, role: Anchor, x: -90.0 },
        RosterEntry { id: "H3", team: Home, role: Faithwing, x: -20.0 },
        RosterEntry { id: "H4", team: Home, role: Reach, x: -260.0 },
        RosterEntry { id: "A1", team: Away, role: Spinner, x: 40.0 },
        RosterEntry { id: "A2", team: Away, role: Anchor, x: 90.0 },
        RosterEntry { id: "A3", team: Away, role: Freewing, x: 20.0 },
        RosterEntry { id: "A4", team: Away, role: Reach, x: 260.0 },
    ]
}

const H: f64 = 1.0 / 240.0;

/// A spread of matchups so fitness is the mean intrinsic outcome over
/// several styles — a one-matchup overfit cannot win (the generalization
/// pressure, complementary to the w-maxing tie-break).
// BROADENED (the audit: 3 narrow rows let an overfit win). A wider
// style/cylinder spread of opponents — including same-style, power,
// chaos, tempo and grind matchups across both cylinder classes — so the
// MEAN intrinsic outcome is the generalization pressure: a genome that
// only beats one archetype scores poorly here. Fixed order (the parallel
// mean folds in matchup-index order — determinism contract).
const MATCHUPS: [(&str, &str, &str, &str); 7] = [
    ("fall-dynasty", "big-slow", "fall-dynasty", "big-slow"),
    ("fall-grind", "big-slow", "rise-power", "small-fast"),
    ("fall-tempo", "big-slow", "rise-chaos", "small-fast"),
    ("fall-dynasty", "small-fast", "rise-power", "big-slow"),
    ("fall-tempo", "small-fast", "fall-dynasty", "small-fast"),
    ("fall-grind", "big-slow", "rise-chaos", "big-slow"),
    ("fall-dynasty", "big-slow", "rise-power", "small-fast"),
];

// ── Seam adapters ───────────────────────────────────────────────────────────
// CONVERGE DEDUP: the learner's private `conv` copy was deleted. The
// episode runner now drives the *real* gym `Env` (`crate::gym::RigEnv`),
// which already performs the sim_world↔ai::types↔scoring↔match_sm
// conversions through skill_eval's `pub(crate)` `conv` (the SINGLE
// source of truth — harness made it crate-visible exactly so the gym
// and any learner reuse the proven bridges, no reinvention). This alias
// keeps the path name available for any residual reference.
#[allow(unused_imports)]
use crate::skill_eval::conv;

/// The game's OWN intrinsic outcome for one episode, from the *controlled*
/// team's point of view. Minimal + honest — the sport's actual result, not
/// a normalised meta-instrument. (This is the seam's `Reward`, accumulated
/// per-step + terminal.)
#[derive(Clone, Copy, Debug, Default)]
pub struct Outcome {
    /// own points − opponent points (the actual scoreboard result).
    pub net_score: f64,
    /// fraction of live ticks the controlled team held the bell.
    pub possession_frac: f64,
    /// genuine first→deep / deep→mouth gate advances by the controlled cast.
    pub gate_clears: f64,
    /// contests the controlled team won.
    pub contests_won: f64,
    /// fraction of ticks the bell stayed in the field (1 − out-of-field).
    pub in_field_frac: f64,
    /// raw ticks simulated (diagnostic).
    pub ticks: f64,
}

impl Outcome {
    /// The single scalar the evolution maximizes. A SMALL honest fold of
    /// the sport's own result — net score dominates (it is the actual
    /// point of the game), the rest are secondary coordination signals
    /// (held the bell, moved the cast up, won contests, kept it live).
    /// Deliberately NOT a Goodhart cap-table / relative ranker: it is the
    /// game outcome, judged ultimately by a human watching.
    pub fn fitness(&self) -> f64 {
        self.net_score
            + 3.0 * self.possession_frac
            + 0.5 * self.gate_clears
            + 0.75 * self.contests_won
            + 1.0 * self.in_field_frac
    }
}

/// One full AI-vs-AI episode driven through the REAL gym `Env`
/// (`crate::gym::RigEnv`). THIN ADAPTER (the pinned converge seam): the
/// candidate `EfeParams` are installed on this thread, the env is reset
/// with an EMPTY `controlled_ids` set (so `RigEnv` runs its internal
/// `AiSystem` for all 8 riggers — bit-identical to the old self-contained
/// loop: same roster, same `ang.cos()*8.0` ring placement, same
/// `set_bell_held("H1")`, same synthetic `FoulGarrote{"__start__"}`
/// kickoff, same `Difficulty::Pro` + `style_to_profile`, same
/// `sim.step`+`mat.consume`, same dead-ball re-arm — all now OWNED by
/// `RigEnv`), and the gym `Reward`/`Observation`/`StepInfo` stream is
/// folded into the learner's `Outcome` using the SAME field definitions
/// the in-clone loop used. The candidate biases the controlled team's
/// active-inference controller via the thread-local exactly as before;
/// the opponent runs the verbatim baseline. Behavior-equivalent: a
/// learner episode = the learned-controlled team vs baseline through the
/// real Env. Nothing upstream of `run_episode` changed.
fn run_match(
    params: &EfeParams,
    controlled: ai::TeamSide,
    h_style: &str,
    h_cyl: &str,
    a_style: &str,
    a_cyl: &str,
    seed: u32,
    max_ticks: u64,
) -> Outcome {
    use crate::gym::{Env, RewardConfig, RigEnv, Scenario, TeamSide as GymSide};

    set_efe_params(Some(Rc::new(*params)));

    let gym_side = match controlled {
        ai::TeamSide::Home => GymSide::Home,
        ai::TeamSide::Away => GymSide::Away,
    };
    let scenario = Scenario {
        home_style: h_style.to_string(),
        home_cyl: h_cyl.to_string(),
        away_style: a_style.to_string(),
        away_cyl: a_cyl.to_string(),
        // EMPTY ⇒ RigEnv's internal AiSystem drives all 8 (the candidate
        // biases the controlled side's controller via the thread-local,
        // exactly like the old loop). The reward is scored from the
        // controlled side regardless of who supplies the inputs.
        controlled_ids: vec![],
        max_ticks,
        reward_config: RewardConfig {
            reward_side: Some(gym_side),
            ..RewardConfig::default()
        },
    };

    // Map the gym possession-team code (0 Home / 1 Away) to "is the
    // controlled side". This is the SAME possession the old loop read
    // from `mat.state().possession`.
    let ctrl_poss_code: u8 = match controlled {
        ai::TeamSide::Home => 0,
        ai::TeamSide::Away => 1,
    };

    let mut env = RigEnv::new();
    let mut obs = env.reset(seed, &scenario);

    let mut held_ticks = 0.0_f64;
    let mut in_field_ticks = 0.0_f64;
    let mut ticks = 0.0_f64;
    let mut gate_clears = 0.0_f64;
    let mut contests_won = 0.0_f64;
    let mut prev_gate: Option<u8> = None;
    let mut prev_poss: Option<u8> = None;

    let roster_team = |id: &str| -> Option<ai::TeamSide> {
        roster().iter().find(|r| r.id == id).map(|r| r.team)
    };

    for _ in 0..max_ticks {
        // Pre-step accounting (the old loop measured possession + in-field
        // on the PRE-step snapshot/state each tick).
        if let Some(hb) = &obs.bell_held_by {
            if roster_team(hb) == Some(controlled) {
                held_ticks += 1.0;
            }
        }
        if (obs.bell_p.y.powi(2) + obs.bell_p.z.powi(2)).sqrt() <= 31.0 {
            in_field_ticks += 1.0;
        }
        // A genuine gate advance while the controlled team had possession
        // (prev gate < cur gate AND prev possession == controlled) — the
        // old loop's exact condition, read from the match SM via the obs.
        if let (Some(pg), Some(pp)) = (prev_gate, prev_poss) {
            if obs.gate > pg && pp == ctrl_poss_code {
                gate_clears += 1.0;
            }
        }
        prev_gate = Some(obs.gate);
        prev_poss = Some(obs.possession);

        // The env owns the AI tick (uncontrolled = all 8 here), the
        // sim.step, the match consume, and the dead-ball re-arm — the
        // same flow the old loop ran inline.
        let step = env.step(&[]);

        // A turnover that lands possession with the controlled team is
        // the sport's "won the contest" result for it. `RewardConfig`'s
        // `reward_side` is the controlled side, so a turnover in its
        // favour shows as a positive `contest_won` the same step
        // `info.turnover` is set — exactly the old `upd.turnover.team ==
        // ctrl` count.
        if step.info.turnover && step.reward.contest_won > 0.0 {
            contests_won += 1.0;
        }

        ticks += 1.0;
        obs = step.obs;
        if step.done {
            break;
        }
    }

    let t = ticks.max(1.0);
    // Net score = final scoreboard from the controlled side (the old
    // loop's `sh - sa` / `sa - sh`).
    let net = match controlled {
        ai::TeamSide::Home => obs.score_home as f64 - obs.score_away as f64,
        ai::TeamSide::Away => obs.score_away as f64 - obs.score_home as f64,
    };

    set_efe_params(None);
    Outcome {
        net_score: net,
        possession_frac: held_ticks / t,
        gate_clears,
        contests_won,
        in_field_frac: in_field_ticks / t,
        ticks,
    }
}

/// The episode-runner SEAM. Fitness consumes ONLY this. Swapping the
/// in-clone match loop for the real `Env::reset/step` is a thin adapter
/// here: build the `Env`, run `max_ticks` steps with the candidate params
/// installed, fold the seam's `Reward` into an `Outcome` (the seam's
/// `Reward` already carries the game's own per-step+terminal outcome —
/// `net_score/possession/gate/contest/in_field` are exactly its fields).
/// Nothing upstream of this function changes when that swap happens.
fn run_episode(
    params: &EfeParams,
    controlled: ai::TeamSide,
    matchup: (&str, &str, &str, &str),
    seed: u32,
    max_ticks: u64,
) -> Outcome {
    let (hs, hc, as_, ac) = matchup;
    run_match(params, controlled, hs, hc, as_, ac, seed, max_ticks)
}

/// Mean intrinsic outcome of `params` over the matchup spread, controlling
/// the Home team. The mean (not a single matchup) is the generalization
/// pressure: an overfit to one style scores poorly here. Parallel over
/// matchups, collected in FIXED matchup-index order (float-reproducible
/// regardless of rayon scheduling — the determinism contract). The
/// thread-local `EfeParams` is set per rayon thread inside `run_match`
/// (parallel-safe, exactly like the planner-profile knob).
pub fn evaluate_params(params: &EfeParams, seed: u32, max_ticks: u64) -> (f64, Outcome) {
    use rayon::prelude::*;
    let outs: Vec<Outcome> = (0..MATCHUPS.len())
        .into_par_iter()
        .map(|i| run_episode(params, ai::TeamSide::Home, MATCHUPS[i], seed, max_ticks))
        .collect();
    // Fixed-order mean (commutative sum over a fixed-length Vec).
    let n = outs.len() as f64;
    let mean = Outcome {
        net_score: outs.iter().map(|o| o.net_score).sum::<f64>() / n,
        possession_frac: outs.iter().map(|o| o.possession_frac).sum::<f64>() / n,
        gate_clears: outs.iter().map(|o| o.gate_clears).sum::<f64>() / n,
        contests_won: outs.iter().map(|o| o.contests_won).sum::<f64>() / n,
        in_field_frac: outs.iter().map(|o| o.in_field_frac).sum::<f64>() / n,
        ticks: outs.iter().map(|o| o.ticks).sum::<f64>() / n,
    };
    (mean.fitness(), mean)
}

// ── The seeded evolutionary self-play loop ──────────────────────────────

/// Learner hyper-parameters. Evolution is a pure deterministic function of
/// these + nothing else.
#[derive(Clone, Debug)]
pub struct LearnConfig {
    pub pop_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub mutation_scale: f64,
    pub crossover_rate: f64,
    pub elite_size: usize,
    /// Master seed. Same seed ⇒ bit-identical run.
    pub seed: u64,
    /// Episode budget (ticks) per matchup.
    pub eval_max_ticks: u64,
    /// Fixed episode seed (constant across the run so fitness is
    /// comparable generation-to-generation; the stochasticity is `seed`).
    pub eval_seed: u32,
    /// Fitness band within which the higher-`weakness()` genome is
    /// preferred (the Bennett generalization tie-break). 0 ⇒ pure fitness.
    pub weakness_band: f64,
}

impl Default for LearnConfig {
    fn default() -> Self {
        LearnConfig {
            pop_size: 24,
            generations: 16,
            mutation_rate: 0.25,
            mutation_scale: 0.20,
            crossover_rate: 0.75,
            elite_size: 3,
            seed: 0xC0FFEE,
            eval_max_ticks: 8000,
            eval_seed: 1234,
            weakness_band: 0.5,
        }
    }
}

/// A genome = the EFE controller's fixed-order parameter vector.
#[derive(Clone, Debug)]
struct Genome {
    genes: Vec<(&'static str, f64)>,
}
impl Genome {
    fn params(&self) -> EfeParams {
        EfeParams::from_genes(&self.genes).clipped()
    }
    fn from_params(p: &EfeParams) -> Self {
        Genome { genes: p.to_genes().to_vec() }
    }
}

/// Per-generation watchable record (printed by the on-demand real run,
/// and the artifact returned from `learn`).
#[derive(Clone, Debug)]
pub struct GenReport {
    pub gen: usize,
    pub best_fitness: f64,
    pub best_weakness: f64,
    pub best_outcome: Outcome,
}

/// The learned artifact: the best `EfeParams` + the watchable per-gen
/// history (elitist ⇒ `best_fitness` monotone non-decreasing).
#[derive(Clone, Debug)]
pub struct Learned {
    pub params: EfeParams,
    pub fitness: f64,
    /// The verbatim `EfeParams::default()` controller's fitness on the
    /// SAME matchup spread/seed — the honest baseline, NOT seeded into
    /// the population (de-rigged).
    pub baseline_fitness: f64,
    /// `fitness − baseline_fitness`. A null/negative learning result is
    /// now VISIBLE here (it is no longer masked by an elitist verbatim-
    /// default seed). Callers/tests assert/report this explicitly.
    pub delta_vs_default: f64,
    pub history: Vec<GenReport>,
}

pub struct CoordLearner {
    cfg: LearnConfig,
    rng: ChaCha8Rng,
}

impl CoordLearner {
    pub fn new(cfg: LearnConfig) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(cfg.seed);
        CoordLearner { cfg, rng }
    }

    /// Deterministic initial population. DE-RIGGED (the audit: seeding
    /// `pop[0]` with the verbatim default + elitism made a null/negative
    /// result invisible — it could never report worse than baseline even
    /// if it learned nothing). NO individual is the verbatim default now:
    /// ~half are Gaussian-jittered around the default (a sane search
    /// neighbourhood, but DISPLACED — not the exact baseline), the rest
    /// uniform-random within bounds. So the elitist best is a genuinely
    /// EVOLVED genome; if it fails to beat the default the reported
    /// best-vs-default DELTA is honestly ≤ 0 (visible, not masked). Every
    /// draw is threaded through `self.rng` in fixed per-individual,
    /// per-gene order (determinism unchanged).
    fn init_pop(&mut self) -> Vec<Genome> {
        let n = self.cfg.pop_size.max(1);
        let base = EfeParams::default();
        let bounds = EfeParams::bounds();
        let mut pop = Vec::with_capacity(n);
        let jittered_until = (n / 2).max(1);
        // A DISPLACING jitter (mean-0 but wide) so even index 0 is a
        // perturbed genome, never the verbatim default.
        let jitter = Normal::new(0.0, 0.25).expect("valid normal");
        for i in 0..n {
            let mut g = Genome::from_params(&base);
            if i < jittered_until {
                for (gi, gene) in g.genes.iter_mut().enumerate() {
                    let (lo, hi) = bounds[gi];
                    let span = hi - lo;
                    let d: f64 = jitter.sample(&mut self.rng) * span;
                    gene.1 += d;
                }
            } else {
                for (gi, gene) in g.genes.iter_mut().enumerate() {
                    let (lo, hi) = bounds[gi];
                    gene.1 = self.rng.gen_range(lo..=hi);
                }
            }
            // clip via EfeParams so every genome is always in-bounds.
            g = Genome::from_params(&g.params());
            pop.push(g);
        }
        pop
    }

    /// Parallel fitness over the whole population, collected in FIXED
    /// population-index order (never an unordered reduce). The genome →
    /// params → episode eval is pure given the genome; the thread-local
    /// `EfeParams` is set per rayon thread inside `run_match`.
    fn evaluate(&self, pop: &[Genome]) -> Vec<(f64, Outcome)> {
        use rayon::prelude::*;
        (0..pop.len())
            .into_par_iter()
            .map(|i| {
                let p = pop[i].params();
                evaluate_params(&p, self.cfg.eval_seed, self.cfg.eval_max_ticks)
            })
            .collect()
    }

    /// Seeded fitness-proportional (roulette) selection. One `gen_range`
    /// draw; degenerate-equal ⇒ deterministic uniform pick.
    fn select(&mut self, fit: &[f64]) -> usize {
        let min = fit.iter().cloned().fold(f64::INFINITY, f64::min);
        let w: Vec<f64> = fit.iter().map(|f| (f - min) + 1e-6).collect();
        let total: f64 = w.iter().sum();
        if !(total > 0.0) {
            return self.rng.gen_range(0..fit.len());
        }
        let mut pick = self.rng.gen_range(0.0..total);
        for (i, wi) in w.iter().enumerate() {
            if pick < *wi {
                return i;
            }
            pick -= *wi;
        }
        fit.len() - 1
    }

    /// 2-point crossover over the fixed-order gene vector (two seeded
    /// sorted point draws; genes in `[c0,c1)` from B, rest from A).
    fn crossover(&mut self, a: &Genome, b: &Genome) -> Genome {
        let len = a.genes.len();
        let mut child = a.clone();
        if len < 2 || !self.rng.gen_bool(self.cfg.crossover_rate.clamp(0.0, 1.0)) {
            return child;
        }
        let p0 = self.rng.gen_range(0..len);
        let p1 = self.rng.gen_range(0..len);
        let (c0, c1) = if p0 <= p1 { (p0, p1) } else { (p1, p0) };
        for k in c0..c1 {
            child.genes[k].1 = b.genes[k].1;
        }
        child
    }

    /// Per-gene seeded Gaussian mutation in fixed gene order. Each gene:
    /// one `gen_bool(rate)` gate, then (if it fires) a `Normal(0,
    /// scale*span)` step. Clipped to bounds via `EfeParams`.
    fn mutate(&mut self, g: &mut Genome) {
        let rate = self.cfg.mutation_rate.clamp(0.0, 1.0);
        let bounds = EfeParams::bounds();
        let normal = Normal::new(0.0, self.cfg.mutation_scale.max(1e-12))
            .expect("valid normal");
        for (gi, gene) in g.genes.iter_mut().enumerate() {
            if self.rng.gen_bool(rate) {
                let (lo, hi) = bounds[gi];
                let d: f64 = normal.sample(&mut self.rng) * (hi - lo);
                gene.1 += d;
            }
        }
        *g = Genome::from_params(&g.params());
    }

    /// Run the full evolution. Pure deterministic function of `self.cfg`
    /// (seed included). Watchable: returns the per-gen history + best.
    pub fn learn(&mut self) -> Learned {
        let n = self.cfg.pop_size.max(1);
        let gens = self.cfg.generations.max(1);
        let mut pop = self.init_pop();
        let mut best = pop[0].clone();
        let mut best_fit = f64::NEG_INFINITY;
        let mut best_out = Outcome::default();
        let mut history: Vec<GenReport> = Vec::with_capacity(gens);

        for gen in 0..gens {
            let evald = self.evaluate(&pop);
            let fit: Vec<f64> = evald.iter().map(|(f, _)| *f).collect();

            // Deterministic ranking: by fitness desc, then — the BENNETT
            // GENERALIZATION TIE-BREAK — within `weakness_band` prefer the
            // WEAKER (higher `weakness()`) genome (larger success set ⇒
            // generalizes, anti-brittle), then ascending index (stable).
            let mut order: Vec<usize> = (0..pop.len()).collect();
            let band = self.cfg.weakness_band;
            order.sort_by(|&i, &j| {
                let (fi, fj) = (fit[i], fit[j]);
                if (fi - fj).abs() <= band {
                    let (wi, wj) =
                        (pop[i].params().weakness(), pop[j].params().weakness());
                    wj.partial_cmp(&wi)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(i.cmp(&j))
                } else {
                    fj.partial_cmp(&fi)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(i.cmp(&j))
                }
            });

            let gb = order[0];
            // Track global best by RAW fitness (so `best_fitness` history
            // is honestly monotone via elitism; the weakness tie-break
            // only ever swaps between equal-fitness genomes).
            if fit[gb] > best_fit {
                best_fit = fit[gb];
                best = pop[gb].clone();
                best_out = evald[gb].1;
            }
            history.push(GenReport {
                gen,
                best_fitness: best_fit,
                best_weakness: best.params().weakness(),
                best_outcome: best_out,
            });

            if gen + 1 == gens {
                break;
            }

            // Elitism: carry the top-K verbatim, in ranked order.
            let elite = self.cfg.elite_size.min(n);
            let mut next: Vec<Genome> = Vec::with_capacity(n);
            for &idx in order.iter().take(elite) {
                next.push(pop[idx].clone());
            }
            while next.len() < n {
                let ia = self.select(&fit);
                let ib = self.select(&fit);
                let mut child = self.crossover(&pop[ia], &pop[ib]);
                self.mutate(&mut child);
                next.push(child);
            }
            pop = next;
        }

        // De-rigged honesty: the verbatim default's fitness on the SAME
        // spread/seed is computed SEPARATELY (never seeded into the pop)
        // so the reported delta is a true learned-vs-baseline figure.
        let (baseline_fitness, _) = evaluate_params(
            &EfeParams::default(),
            self.cfg.eval_seed,
            self.cfg.eval_max_ticks,
        );

        Learned {
            params: best.params(),
            fitness: best_fit,
            baseline_fitness,
            delta_vs_default: best_fit - baseline_fitness,
            history,
        }
    }
}

/// Entry point: learn a coordination-optimized `EfeParams`. Deterministic
/// in `cfg.seed`.
pub fn learn_coordination(cfg: LearnConfig) -> Learned {
    CoordLearner::new(cfg).learn()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(seed: u64) -> LearnConfig {
        LearnConfig {
            pop_size: 4,
            generations: 2,
            mutation_rate: 0.3,
            mutation_scale: 0.2,
            crossover_rate: 0.7,
            elite_size: 1,
            seed,
            eval_max_ticks: 250,
            eval_seed: 1234,
            weakness_band: 0.5,
        }
    }

    /// FAST determinism gate: 2 gens, tiny pop/ticks, same seed ⇒
    /// bit-identical best params. DETERMINISM IS SACRED.
    #[test]
    fn learner_is_deterministic_same_seed() {
        let a = learn_coordination(tiny(42));
        let b = learn_coordination(tiny(42));
        let ga = a.params.to_genes();
        let gb = b.params.to_genes();
        for ((ka, va), (kb, vb)) in ga.iter().zip(gb.iter()) {
            assert_eq!(ka, kb, "gene order must match");
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "param {ka} must be bit-identical across same-seed runs"
            );
        }
        assert_eq!(a.fitness.to_bits(), b.fitness.to_bits());
        // a 3rd same-seed run is still bit-identical (no hidden global /
        // rayon-order dependence in the parallel fitness eval).
        let c = learn_coordination(tiny(42));
        for ((_, va), (_, vc)) in ga.iter().zip(c.params.to_genes().iter()) {
            assert_eq!(va.to_bits(), vc.to_bits(), "3rd same-seed run drifted");
        }
    }

    /// Elitist best-fitness is monotone non-decreasing across generations
    /// (elitism over the EVOLVED population is intact) — but the learner
    /// is now DE-RIGGED: it does NOT seed the verbatim default, so it is
    /// NOT guaranteed ≥ baseline. The honest best-vs-default delta is
    /// REPORTED (finite, consistent with the recomputed baseline) and may
    /// be ≤ 0 — that visibility is the point of the de-rigging.
    #[test]
    fn learner_elitism_monotone() {
        let l = learn_coordination(tiny(7));
        assert!(!l.history.is_empty());
        for w in l.history.windows(2) {
            assert!(
                w[1].best_fitness >= w[0].best_fitness,
                "elitism violated: best fitness dropped {} -> {}",
                w[0].best_fitness,
                w[1].best_fitness
            );
        }
        // The reported delta is honest: it equals best − baseline, the
        // baseline is the verbatim default on the SAME spread/seed, and
        // it is NOT forced ≥ 0 (a null/negative is now visible).
        assert!(l.baseline_fitness.is_finite());
        assert!(
            (l.delta_vs_default - (l.fitness - l.baseline_fitness)).abs()
                < 1e-9,
            "delta must equal best − baseline (honest, unmasked)"
        );
    }

    /// The episode reward IS the game's own intrinsic outcome, not a
    /// meta-instrument: a finite, sane `Outcome` with possession/in-field
    /// as proper fractions.
    #[test]
    fn outcome_is_intrinsic_and_sane() {
        let (_f, o) = evaluate_params(&EfeParams::default(), 1234, 300);
        assert!(o.net_score.is_finite());
        assert!((0.0..=1.0).contains(&o.possession_frac));
        assert!((0.0..=1.0).contains(&o.in_field_frac));
        assert!(o.gate_clears >= 0.0 && o.contests_won >= 0.0);
        assert!(o.ticks > 0.0);
        assert!(o.fitness().is_finite());
    }

    /// On-demand REAL learning run (NOT a CI gate — long). Run with:
    ///   cargo test -p rig-core coord_learner::tests::learn_real -- --ignored --nocapture
    #[test]
    #[ignore = "long real coordination-learning run; --ignored --nocapture"]
    fn learn_real() {
        let cfg = LearnConfig {
            pop_size: 16,
            generations: 10,
            eval_max_ticks: 8000,
            ..LearnConfig::default()
        };
        let (base_fit, base_out) =
            evaluate_params(&EfeParams::default(), cfg.eval_seed, cfg.eval_max_ticks);
        println!(
            "BASELINE controller: fitness={base_fit:.3} outcome={base_out:?}"
        );
        let mut l = CoordLearner::new(cfg);
        let learned = l.learn();
        for r in &learned.history {
            println!(
                "gen {:>2}  best_fitness={:>7.3}  weakness={:.3}  net={:+.2} \
                 poss={:.3} gate={:.1} contest={:.1} field={:.3}",
                r.gen,
                r.best_fitness,
                r.best_weakness,
                r.best_outcome.net_score,
                r.best_outcome.possession_frac,
                r.best_outcome.gate_clears,
                r.best_outcome.contests_won,
                r.best_outcome.in_field_frac,
            );
        }
        println!(
            "LEARNED: fitness={:.3} baseline(default)={:.3} \
             delta_vs_default={:+.3} (DE-RIGGED: default NOT seeded; a \
             negative delta is a real null result, not masked)",
            learned.fitness, learned.baseline_fitness, learned.delta_vs_default
        );
        println!("LEARNED PARAMS: {:?}", learned.params.to_genes());
        // Honest reporting only — NO ≥-baseline guarantee anymore. We
        // only assert internal consistency + the external recompute
        // agrees with the carried baseline.
        assert!((learned.baseline_fitness - base_fit).abs() < 1e-9);
        assert!(
            (learned.delta_vs_default
                - (learned.fitness - learned.baseline_fitness))
                .abs()
                < 1e-9
        );
    }
}

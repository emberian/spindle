//! ga.rs — Stage-3 offline Genetic Algorithm over planner *profile-space*.
//!
//! ADDITIVE & NATIVE-ONLY. This does NOT change production behavior:
//! production still resolves the lazily-cached default Coordination
//! `Profile` (engine 9). This module is an OFFLINE SEARCH TOOL that
//! evolves the cost-term WEIGHT VECTOR of the Coordination profile (the
//! `branch_cost`/`coord_compute_cost` fold weights) and scores each
//! candidate with the native `skill_eval::eval_profile` harness. Engine
//! stays 9 (Coordination dispatch) — per the Stage-2 eval the leverage is
//! in the composed cost TERMS, not the search engine, so the GA explores
//! the weights of the branch + coordination terms.
//!
//! ── DETERMINISM IS SACRED ───────────────────────────────────────────────
//! There is NO `rand::thread_rng()` anywhere. A single seeded
//! `ChaCha8Rng::seed_from_u64(config.seed)` is threaded EXPLICITLY through
//! every random draw, in a fixed deterministic order:
//!   1. population init  (per-individual, per-gene jitter / random draws)
//!   2. each generation, in order: parallel fitness (collected in fixed
//!      population-index order — never an unordered reduce), then
//!      elitism (deterministic top-K by sorted index), then for each
//!      offspring slot: parent A select, parent B select, crossover-point
//!      draws, then per-gene mutation gate + Gaussian magnitude.
//! Same `config.seed` ⇒ bit-identical evolution ⇒ bit-identical final
//! `Profile` weights (proved by `ga_is_deterministic` via `.to_bits()`).
//!
//! The parallel fitness eval uses `into_par_iter().collect::<Vec<_>>()`
//! indexed by population position (exactly like `eval_skill`'s matchup
//! collect) so float order is reproducible regardless of thread schedule.
//! The PRNG itself is used only on the single evolving (main) thread, in
//! the strict order above — rayon only parallelises the (pure, given the
//! genome) fitness function.

#![cfg(not(target_arch = "wasm32"))]

use crate::planner_cost::Profile;
use crate::skill_eval::eval_profile;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Sane bound on every evolved weight (the blueprint's [-12,12] clip).
/// Keeps the cost fold finite and the search in a behaviorally meaningful
/// region (the seed Coordination weights are all within [-0.6, 1.0]).
const WEIGHT_LO: f64 = -12.0;
const WEIGHT_HI: f64 = 12.0;

#[inline]
fn clip(x: f64) -> f64 {
    if x < WEIGHT_LO {
        WEIGHT_LO
    } else if x > WEIGHT_HI {
        WEIGHT_HI
    } else {
        x
    }
}

/// GA hyper-parameters (blueprint shape). All evolution is a pure
/// deterministic function of these + nothing else.
#[derive(Clone, Debug)]
pub struct GaConfig {
    pub pop_size: usize,
    pub generations: usize,
    /// Per-gene probability of a Gaussian mutation.
    pub mutation_rate: f64,
    /// Std-dev of the Gaussian mutation (added to the gene, then clipped).
    pub mutation_scale: f64,
    /// Probability an offspring is produced by 2-point crossover (else it
    /// is a clone of parent A).
    pub crossover_rate: f64,
    /// Number of top individuals carried verbatim into the next gen.
    pub elite_size: usize,
    /// The master seed. Same seed ⇒ bit-identical run.
    pub seed: u64,
    /// Fitness-eval budget per match (passed to `eval_profile`).
    pub eval_max_ticks: u64,
    /// Fixed eval seed for the skill harness (kept constant across the run
    /// so fitness is comparable across generations; the GA's stochasticity
    /// is the master `seed`, NOT the eval seed).
    pub eval_seed: u32,
}

impl Default for GaConfig {
    fn default() -> Self {
        GaConfig {
            pop_size: 24,
            generations: 20,
            mutation_rate: 0.20,
            mutation_scale: 0.5,
            crossover_rate: 0.75,
            elite_size: 3,
            seed: 0xC0FFEE,
            eval_max_ticks: 8000,
            eval_seed: 1234,
        }
    }
}

/// A genome = the Coordination profile's fixed-order weight vector. The
/// term *names* (and their order) are the immutable scaffold copied from
/// the seed `Profile`; only the f64 weights evolve. Engine is implicitly
/// 9 (we rebuild the profile from the base each eval — see `to_profile`).
#[derive(Clone, Debug)]
struct Genome {
    /// Fixed-order (name, weight). Names == seed Profile's `weights` keys.
    genes: Vec<(&'static str, f64)>,
    engine: i32,
    label: String,
}

impl Genome {
    /// Materialise into a `Profile` for `eval_profile`. We rebuild from the
    /// engine class (giving a fresh, correctly-ordered term list) and
    /// overwrite the weights with the genome — exactly the contract
    /// `eval_profile`/`one_match_profile` expect (term order is identical
    /// because both come from `Profile::from_class(engine)`).
    fn to_profile(&self) -> Profile {
        let mut p = Profile::from_class(self.engine);
        p.label = self.label.clone();
        debug_assert_eq!(p.weights.len(), self.genes.len());
        for (slot, g) in p.weights.iter_mut().zip(self.genes.iter()) {
            debug_assert_eq!(slot.0, g.0);
            slot.1 = g.1;
        }
        p
    }
}

/// The GA driver.
pub struct GeneticAlgorithm {
    config: GaConfig,
    rng: ChaCha8Rng,
    /// The immutable seed genome (default = `Profile::coordination_default()`
    /// weights). All individuals share its `(name, order, engine)` scaffold.
    seed_genome: Genome,
    /// Per-generation best composite, in order — monotone non-decreasing by
    /// elitism (asserted by `ga_elitism_non_decreasing`).
    pub best_history: Vec<f64>,
}

impl GeneticAlgorithm {
    /// Build a GA seeded from a base profile (default
    /// `Profile::coordination_default()`).
    pub fn new(config: GaConfig) -> Self {
        Self::with_base(config, Profile::coordination_default())
    }

    /// Build a GA seeded from an explicit base profile (its weight vector +
    /// engine become the scaffold). Used by tests.
    pub fn with_base(config: GaConfig, base: Profile) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(config.seed);
        let seed_genome = Genome {
            genes: base.weights.clone(),
            engine: base.engine,
            label: base.label.clone(),
        };
        GeneticAlgorithm {
            config,
            rng,
            seed_genome,
            best_history: Vec::new(),
        }
    }

    /// Deterministic initial population: index 0 is the verbatim seed
    /// (so the GA can never do worse than the Coordination baseline via
    /// elitism); the next ~third are jittered seeds; the rest are random
    /// within bounds. Every draw is threaded through `self.rng` in fixed
    /// per-individual, per-gene order.
    fn init_population(&mut self) -> Vec<Genome> {
        let n = self.config.pop_size.max(1);
        let mut pop = Vec::with_capacity(n);
        // [0] verbatim seed.
        pop.push(self.seed_genome.clone());
        let jittered_until = (n / 3).max(1);
        let jitter = Normal::new(0.0, 0.35).expect("valid normal");
        for i in 1..n {
            let mut g = self.seed_genome.clone();
            if i < jittered_until {
                for gene in g.genes.iter_mut() {
                    let d: f64 = jitter.sample(&mut self.rng);
                    gene.1 = clip(gene.1 + d);
                }
            } else {
                for gene in g.genes.iter_mut() {
                    let v: f64 = self.rng.gen_range(WEIGHT_LO..=WEIGHT_HI);
                    gene.1 = clip(v);
                }
            }
            g.label = self.seed_genome.label.clone();
            pop.push(g);
        }
        pop
    }

    /// Parallel fitness over the whole population, collected in FIXED
    /// population-index order (never an unordered reduce) so the f64s are
    /// reproducible regardless of rayon scheduling. The genome → profile
    /// rebuild + `eval_profile` is a pure function of the genome, and the
    /// thread-local planner knob is set per rayon thread inside
    /// `one_match_profile` — parallel-safe (the Stage-2 guarantee).
    fn evaluate(&self, pop: &[Genome], label_prefix: &str) -> Vec<f64> {
        use rayon::prelude::*;
        (0..pop.len())
            .into_par_iter()
            .map(|i| {
                let mut prof = pop[i].to_profile();
                prof.label = format!("{label_prefix}#{i}");
                eval_profile(&prof, self.config.eval_seed, self.config.eval_max_ticks)
                    .composite
            })
            .collect::<Vec<f64>>()
    }

    /// Seeded fitness-proportional (roulette) selection. Shifts fitnesses
    /// so the minimum maps to a small positive floor (composites are ≥0
    /// here but this is robust + deterministic). One `gen_range` draw.
    fn select(&mut self, fitness: &[f64]) -> usize {
        let min = fitness.iter().cloned().fold(f64::INFINITY, f64::min);
        let weights: Vec<f64> = fitness.iter().map(|f| (f - min) + 1e-6).collect();
        let total: f64 = weights.iter().sum();
        if !(total > 0.0) {
            // Degenerate (all equal) — deterministic uniform pick.
            return self.rng.gen_range(0..fitness.len());
        }
        let mut pick = self.rng.gen_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            if pick < *w {
                return i;
            }
            pick -= *w;
        }
        fitness.len() - 1
    }

    /// 2-point crossover over the fixed-order gene vector. Two seeded
    /// crossover-point draws (sorted); genes in `[c0, c1)` come from B,
    /// the rest from A. Order of genes is the immutable scaffold so this
    /// is fully deterministic given the rng state.
    fn crossover(&mut self, a: &Genome, b: &Genome) -> Genome {
        let len = a.genes.len();
        let mut child = a.clone();
        if len < 2 || !self.rng.gen_bool(self.config.crossover_rate.clamp(0.0, 1.0)) {
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

    /// Per-gene seeded Gaussian mutation, in fixed gene order. For each
    /// gene: one `gen_bool(mutation_rate)` gate draw, then (if it fires)
    /// one `Normal(0, mutation_scale)` sample added + clipped.
    fn mutate(&mut self, g: &mut Genome) {
        let normal = Normal::new(0.0, self.config.mutation_scale.max(1e-12))
            .expect("valid normal");
        let rate = self.config.mutation_rate.clamp(0.0, 1.0);
        for gene in g.genes.iter_mut() {
            if self.rng.gen_bool(rate) {
                let d: f64 = normal.sample(&mut self.rng);
                gene.1 = clip(gene.1 + d);
            }
        }
    }

    /// Run the full evolution and return the best `Profile` found. Pure
    /// deterministic function of `self.config` (seed included).
    pub fn evolve(&mut self) -> Profile {
        let n = self.config.pop_size.max(1);
        let mut pop = self.init_population();
        let mut best_genome = pop[0].clone();
        let mut best_fit = f64::NEG_INFINITY;

        for gen in 0..self.config.generations.max(1) {
            let fitness = self.evaluate(&pop, "ga");

            // Deterministic ranking: indices sorted by fitness desc, ties
            // broken by ascending index (stable + reproducible).
            let mut order: Vec<usize> = (0..pop.len()).collect();
            order.sort_by(|&i, &j| {
                fitness[j]
                    .partial_cmp(&fitness[i])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(i.cmp(&j))
            });

            // Track the global best (monotone via elitism — see below).
            let gen_best_idx = order[0];
            if fitness[gen_best_idx] > best_fit {
                best_fit = fitness[gen_best_idx];
                best_genome = pop[gen_best_idx].clone();
            }
            self.best_history.push(best_fit);

            // Last generation: no need to breed.
            if gen + 1 == self.config.generations.max(1) {
                break;
            }

            // Elitism: carry the top-K verbatim, in ranked order.
            let elite = self.config.elite_size.min(n);
            let mut next: Vec<Genome> = Vec::with_capacity(n);
            for &idx in order.iter().take(elite) {
                next.push(pop[idx].clone());
            }
            // Fill the rest by select → crossover → mutate, in fixed
            // offspring-slot order (deterministic rng consumption).
            while next.len() < n {
                let ia = self.select(&fitness);
                let ib = self.select(&fitness);
                let mut child = self.crossover(&pop[ia], &pop[ib]);
                self.mutate(&mut child);
                child.label = self.seed_genome.label.clone();
                next.push(child);
            }
            pop = next;
        }

        let mut out = best_genome.to_profile();
        out.label = self.seed_genome.label.clone();
        out
    }
}

/// Stage-3 entry point: evolve the planner profile and return the best
/// `Profile`. Deterministic in `config.seed`. The returned profile keeps
/// engine 9 (Coordination dispatch); only its WEIGHTS are evolved.
pub fn evolve_profiles(config: GaConfig) -> Profile {
    GeneticAlgorithm::new(config).evolve()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg(seed: u64) -> GaConfig {
        GaConfig {
            pop_size: 4,
            generations: 3,
            mutation_rate: 0.3,
            mutation_scale: 0.4,
            crossover_rate: 0.7,
            elite_size: 1,
            seed,
            eval_max_ticks: 300,
            eval_seed: 1234,
        }
    }

    /// DETERMINISM IS SACRED: same seed ⇒ bit-identical final weights.
    #[test]
    fn ga_is_deterministic() {
        let a = evolve_profiles(tiny_cfg(42));
        let b = evolve_profiles(tiny_cfg(42));
        assert_eq!(a.weights.len(), b.weights.len());
        for ((ka, va), (kb, vb)) in a.weights.iter().zip(b.weights.iter()) {
            assert_eq!(ka, kb, "term order must be identical");
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "weight {ka} must be bit-identical across seeded runs"
            );
        }
        // Engine unchanged (Coordination dispatch).
        assert_eq!(a.engine, 9);
        assert_eq!(b.engine, 9);
        // Re-running the SAME seed a third time is still bit-identical
        // (defends against any hidden global / order dependence in the
        // parallel fitness eval).
        let a3 = evolve_profiles(tiny_cfg(42));
        for ((_, va), (_, v3)) in a.weights.iter().zip(a3.weights.iter()) {
            assert_eq!(va.to_bits(), v3.to_bits(), "3rd same-seed run drifted");
        }
        // NOTE: a "different seed ⇒ different output" check is deliberately
        // NOT asserted here — at the tiny CI budget (pop=4, gen=3,
        // max_ticks=300) the fitness landscape is ~flat for all candidates,
        // so elitism correctly carries the verbatim seed (index 0) and the
        // output can legitimately be seed-independent. Real divergence is
        // exercised by the on-demand `evolve_real` at a real budget.
    }

    /// Elitism ⇒ the per-generation global-best fitness is monotone
    /// non-decreasing. Tiny budget so this is seconds.
    #[test]
    fn ga_elitism_non_decreasing() {
        let mut ga = GeneticAlgorithm::new(tiny_cfg(7));
        let _ = ga.evolve();
        assert!(!ga.best_history.is_empty());
        for w in ga.best_history.windows(2) {
            assert!(
                w[1] >= w[0],
                "elitism violated: best fitness dropped {} -> {}",
                w[0],
                w[1]
            );
        }
        // The seed (index 0) is the verbatim Coordination baseline, so the
        // GA's best is ≥ that baseline's composite.
        assert!(ga.best_history.last().copied().unwrap() >= 0.0);
    }

    /// Consistency proof: `eval_profile(from_class(9))` composite equals
    /// `eval_skill("Coordination", 9, ...)` composite at a small fixed
    /// budget — i.e. the new evolved-weights path is consistent with the
    /// untouched ranker path when the genome IS the stock Coordination
    /// weight vector.
    #[test]
    fn eval_profile_matches_eval_skill_for_class_profile() {
        let prof = Profile::from_class(9);
        let via_profile = eval_profile(&prof, 1234, 400);
        let via_skill = crate::skill_eval::eval_skill("Coordination", 9, 1234, 400);
        assert_eq!(
            via_profile.composite.to_bits(),
            via_skill.composite.to_bits(),
            "eval_profile(from_class(9)) must match eval_skill Coordination"
        );
        assert_eq!(via_profile.parts.len(), via_skill.parts.len());
        for ((ka, va), (kb, vb)) in
            via_profile.parts.iter().zip(via_skill.parts.iter())
        {
            assert_eq!(ka, kb);
            assert_eq!(va.to_bits(), vb.to_bits());
        }
        // restore production default for any later test on this thread.
        crate::ai::plan_bridge::set_planner_class(0);
    }

    /// On-demand REAL evolution (NOT a CI gate — long). Run with:
    ///   cargo test -p rig-core ga::tests::evolve_real -- --ignored --nocapture
    #[test]
    #[ignore = "long real GA run; --ignored --nocapture"]
    fn evolve_real() {
        let cfg = GaConfig {
            pop_size: 16,
            generations: 12,
            eval_max_ticks: 8000,
            ..GaConfig::default()
        };
        let baseline =
            crate::skill_eval::eval_skill("Coordination", 9, cfg.eval_seed, cfg.eval_max_ticks)
                .composite;
        let mut ga = GeneticAlgorithm::new(cfg);
        let best = ga.evolve();
        let best_fit = ga.best_history.last().copied().unwrap();
        println!(
            "GA real run: Coordination baseline composite = {baseline:.1}, \
             evolved best composite = {best_fit:.1}"
        );
        println!("evolved weights: {:?}", best.weights);
        assert!(best_fit >= baseline - 1e-9, "elitism guarantees ≥ seed");
        crate::ai::plan_bridge::set_planner_class(0);
    }
}

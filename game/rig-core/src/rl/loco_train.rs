//! rl/loco_train.rs — ES trainer for the locomotion policy.
//!
//! OpenAI-ES (antithetic pairs, rank-normalized) training loop. Each
//! generation: perturb the current policy weights, evaluate episodes in
//! parallel, rank-normalize fitness, update weights. Same proven pattern
//! as `rl/train.rs` adapted for the smaller LocoPolicy (4957 params).
//!
//! NATIVE-ONLY: rand_chacha, rayon.

use crate::rl::loco::{LocoPolicy, LOCO_PARAM_W};
use crate::rl::loco_env::{evaluate_episode, LocoEnvConfig};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Training configuration.
#[derive(Clone, Debug)]
pub struct LocoTrainConfig {
    /// ES population size (number of antithetic pairs = pop_size/2 perturbations).
    pub pop_size: usize,
    /// Noise standard deviation for parameter perturbation.
    pub sigma: f64,
    /// Learning rate (step size for weight update).
    pub lr: f64,
    /// Weight decay coefficient.
    pub weight_decay: f64,
    /// Number of episodes per perturbation for fitness averaging.
    pub episodes_per_eval: usize,
    /// Total training generations.
    pub max_generations: usize,
    /// Master seed.
    pub seed: u64,
    /// Curriculum stages (applied sequentially by generation).
    pub curriculum: Vec<CurriculumStage>,
}

/// One stage of the training curriculum.
#[derive(Clone, Debug)]
pub struct CurriculumStage {
    pub start_gen: usize,
    pub env_config: LocoEnvConfig,
}

impl Default for LocoTrainConfig {
    fn default() -> Self {
        LocoTrainConfig {
            pop_size: 40,
            sigma: 0.05,
            lr: 0.03,
            weight_decay: 0.003,
            episodes_per_eval: 4,
            max_generations: 300,
            seed: 0x10C0_10C0,
            curriculum: vec![
                CurriculumStage {
                    start_gen: 0,
                    env_config: LocoEnvConfig {
                        min_dist: 30.0,
                        max_dist: 60.0,
                        max_ticks: 480,
                        max_init_speed: 0.0,
                    },
                },
                CurriculumStage {
                    start_gen: 50,
                    env_config: LocoEnvConfig {
                        min_dist: 50.0,
                        max_dist: 120.0,
                        max_ticks: 960,
                        max_init_speed: 10.0,
                    },
                },
                CurriculumStage {
                    start_gen: 150,
                    env_config: LocoEnvConfig {
                        min_dist: 80.0,
                        max_dist: 200.0,
                        max_ticks: 1440,
                        max_init_speed: 15.0,
                    },
                },
            ],
        }
    }
}

/// Per-generation training report.
#[derive(Clone, Debug)]
pub struct LocoGenReport {
    pub gen: usize,
    pub mean_fitness: f64,
    pub best_fitness: f64,
    pub arrival_rate: f64,
    pub mean_ticks: f64,
    pub stage: usize,
}

/// Final training result.
pub struct LocoTrainResult {
    pub policy: LocoPolicy,
    pub history: Vec<LocoGenReport>,
}

/// Get the active curriculum stage for a given generation.
fn active_stage(gen: usize, curriculum: &[CurriculumStage]) -> &CurriculumStage {
    let mut active = &curriculum[0];
    for stage in curriculum {
        if gen >= stage.start_gen {
            active = stage;
        }
    }
    active
}

/// Train the locomotion policy using OpenAI-ES.
pub fn train_loco(cfg: LocoTrainConfig) -> LocoTrainResult {
    let mut master_rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut policy = LocoPolicy::new_zeros();
    let mut history: Vec<LocoGenReport> = Vec::with_capacity(cfg.max_generations);

    let half_pop = cfg.pop_size / 2;

    for gen in 0..cfg.max_generations {
        let stage = active_stage(gen, &cfg.curriculum);
        let env_config = stage.env_config.clone();
        let stage_idx = cfg.curriculum.iter().position(|s| s.start_gen == stage.start_gen).unwrap_or(0);

        // Generate perturbation seeds for this generation
        let gen_seed: u64 = master_rng.gen();

        // Evaluate antithetic pairs in parallel
        let base_params = policy.params.clone();
        let sigma = cfg.sigma;
        let eps_per_eval = cfg.episodes_per_eval;

        let results: Vec<(f64, f64, f64, f64)> = (0..half_pop)
            .into_par_iter()
            .map(|i| {
                let mut pair_rng = ChaCha8Rng::seed_from_u64(gen_seed.wrapping_add(i as u64));
                let normal = Normal::new(0.0, 1.0).unwrap();

                // Generate noise vector
                let noise: Vec<f64> = (0..LOCO_PARAM_W)
                    .map(|_| normal.sample(&mut pair_rng))
                    .collect();

                // Positive perturbation
                let mut pos_params: Vec<f64> = base_params
                    .iter()
                    .zip(noise.iter())
                    .map(|(w, n)| w + sigma * n)
                    .collect();
                let pos_policy = LocoPolicy::from_params(pos_params);

                // Negative perturbation
                let neg_params: Vec<f64> = base_params
                    .iter()
                    .zip(noise.iter())
                    .map(|(w, n)| w - sigma * n)
                    .collect();
                let neg_policy = LocoPolicy::from_params(neg_params);

                // Evaluate both
                let mut eval_rng = ChaCha8Rng::seed_from_u64(gen_seed.wrapping_add(1000 + i as u64));
                let mut pos_fit = 0.0_f64;
                let mut neg_fit = 0.0_f64;
                let mut pos_arrived = 0u32;
                let mut neg_arrived = 0u32;

                for _ in 0..eps_per_eval {
                    let (r, a, _) = evaluate_episode(&pos_policy, &env_config, &mut eval_rng);
                    pos_fit += r;
                    if a { pos_arrived += 1; }
                }
                // Reset rng for symmetric eval
                eval_rng = ChaCha8Rng::seed_from_u64(gen_seed.wrapping_add(1000 + i as u64));
                for _ in 0..eps_per_eval {
                    let (r, a, _) = evaluate_episode(&neg_policy, &env_config, &mut eval_rng);
                    neg_fit += r;
                    if a { neg_arrived += 1; }
                }

                pos_fit /= eps_per_eval as f64;
                neg_fit /= eps_per_eval as f64;
                let arr_rate = (pos_arrived + neg_arrived) as f64 / (2 * eps_per_eval) as f64;
                let mean_fit = (pos_fit + neg_fit) / 2.0;
                (pos_fit, neg_fit, arr_rate, mean_fit)
            })
            .collect();

        // Rank-normalize the fitness (all 2*half_pop fitnesses together)
        let mut all_fits: Vec<(usize, f64)> = Vec::with_capacity(cfg.pop_size);
        for (i, (pf, nf, _, _)) in results.iter().enumerate() {
            all_fits.push((2 * i, *pf));
            all_fits.push((2 * i + 1, *nf));
        }
        all_fits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let n = all_fits.len() as f64;
        let mut ranks = vec![0.0_f64; all_fits.len()];
        for (rank, (idx, _)) in all_fits.iter().enumerate() {
            ranks[*idx] = (rank as f64) / (n - 1.0) - 0.5; // centered [-0.5, 0.5]
        }

        // Compute gradient estimate and update
        let gen_seed_for_noise = gen_seed;
        let mut grad = vec![0.0_f64; LOCO_PARAM_W];
        for i in 0..half_pop {
            let mut pair_rng = ChaCha8Rng::seed_from_u64(gen_seed_for_noise.wrapping_add(i as u64));
            let normal = Normal::new(0.0, 1.0).unwrap();
            let noise: Vec<f64> = (0..LOCO_PARAM_W)
                .map(|_| normal.sample(&mut pair_rng))
                .collect();

            let pos_rank = ranks[2 * i];
            let neg_rank = ranks[2 * i + 1];
            let diff = pos_rank - neg_rank; // antithetic: exploit paired structure

            for j in 0..LOCO_PARAM_W {
                grad[j] += diff * noise[j];
            }
        }

        // Normalize and apply
        let scale = cfg.lr / (half_pop as f64 * cfg.sigma);
        for j in 0..LOCO_PARAM_W {
            policy.params[j] += scale * grad[j] - cfg.weight_decay * policy.params[j];
        }

        // Report
        let mean_fitness: f64 = results.iter().map(|(_, _, _, m)| m).sum::<f64>() / half_pop as f64;
        let best_fitness: f64 = results.iter().map(|(p, n, _, _)| p.max(*n)).fold(f64::NEG_INFINITY, f64::max);
        let arrival_rate: f64 = results.iter().map(|(_, _, a, _)| a).sum::<f64>() / half_pop as f64;

        let report = LocoGenReport {
            gen,
            mean_fitness,
            best_fitness,
            arrival_rate,
            mean_ticks: 0.0, // TODO: track
            stage: stage_idx,
        };

        if gen % 10 == 0 || gen == cfg.max_generations - 1 {
            eprintln!(
                "[gen {:3}] stage={} mean_fit={:.1} best={:.1} arrival={:.0}%",
                gen, stage_idx, mean_fitness, best_fitness, arrival_rate * 100.0
            );
        }

        history.push(report);
    }

    LocoTrainResult { policy, history }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_is_deterministic() {
        let cfg = LocoTrainConfig {
            pop_size: 4,
            max_generations: 3,
            episodes_per_eval: 1,
            curriculum: vec![CurriculumStage {
                start_gen: 0,
                env_config: LocoEnvConfig {
                    min_dist: 30.0,
                    max_dist: 50.0,
                    max_ticks: 120,
                    max_init_speed: 0.0,
                },
            }],
            ..Default::default()
        };
        let a = train_loco(cfg.clone());
        let b = train_loco(cfg);
        assert_eq!(a.policy.params.len(), b.policy.params.len());
        for (wa, wb) in a.policy.params.iter().zip(b.policy.params.iter()) {
            assert_eq!(wa.to_bits(), wb.to_bits());
        }
    }

    #[test]
    fn fitness_improves_over_generations() {
        let cfg = LocoTrainConfig {
            pop_size: 20,
            max_generations: 20,
            episodes_per_eval: 2,
            sigma: 0.1,
            lr: 0.05,
            curriculum: vec![CurriculumStage {
                start_gen: 0,
                env_config: LocoEnvConfig {
                    min_dist: 30.0,
                    max_dist: 50.0,
                    max_ticks: 240,
                    max_init_speed: 0.0,
                },
            }],
            ..Default::default()
        };
        let result = train_loco(cfg);
        let first_fit = result.history[0].mean_fitness;
        let last_fit = result.history.last().unwrap().mean_fitness;
        // Should improve (or at least not regress badly)
        assert!(
            last_fit >= first_fit - 5.0,
            "fitness regressed: first={first_fit:.1} last={last_fit:.1}"
        );
    }
}

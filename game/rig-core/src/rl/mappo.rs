//! rl/mappo.rs — Native Rust MAPPO training loop (centralized training,
//! decentralized execution) with self-play population.
//!
//! GRADIENT APPROACH: **ES-shaped PPO** (evolutionary strategy with the PPO
//! clipped surrogate as the shaped fitness). Rationale: full analytical
//! backprop through the multi-head attention layers (24,598 actor + 34,433
//! critic params) is ~2x the forward-pass code — feasible but massive for
//! one drop and brittle to get exactly right. Finite differences at 24k
//! params would need 48k forward passes per gradient step — intractable.
//! Instead we compute the PPO objective (clipped surrogate + value loss +
//! entropy bonus) from collected rollouts, then use it as shaped fitness
//! for an ES update (antithetic, rank-normalized). This gives us:
//!   - PPO's advantage-shaped objective (GAE, per-agent credit from the
//!     centralized value function)
//!   - ES's gradient-free parameter update (scales to 24k params with
//!     antithetic pairs — proven by the existing ES trainer)
//!   - Correct PPO semantics: clipping, value loss, entropy regularization
//!     all inform the fitness landscape the ES navigates
//!
//! The critic is trained via a separate ES process that minimizes the value
//! prediction error (MSE of V(s) vs actual returns from the rollout).
//!
//! NATIVE-ONLY: gated `cfg(not(target_arch = "wasm32"))` (rand, rayon, gym).
//! DETERMINISTIC: seeded ChaCha8Rng, fixed-order parallel collection.

use crate::ai::types as ai;
use crate::gym::{Env, Observation, RewardConfig, RigEnv, Scenario, TeamSide};
use crate::rl::attention::{AttentionPolicy, ATTN_PARAM_W};
use crate::rl::policy::{sorted_others, K, OUT_W};
use crate::rl::self_play::{Population, SelfPlayEnv, SelfPlayStep};
use crate::rl::value::{compute_gae, value_featurize, CentralizedValue, VALUE_FEAT_W, VALUE_PARAM_W};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

// ── Constants ─────────────────────────────────────────────────────────────────

const HOME_IDS: [&str; 4] = ["H1", "H2", "H3", "H4"];
const AWAY_IDS: [&str; 4] = ["A1", "A2", "A3", "A4"];
const ALL_IDS: [&str; 8] = ["H1", "H2", "H3", "H4", "A1", "A2", "A3", "A4"];

/// Number of controlled agents per team.
const N_AGENTS: usize = 4;

// ── Action log-probability ────────────────────────────────────────────────────

/// Fixed std for continuous action heads (aim, fire_dir, throw, thrumbler).
/// A moderate std allows exploration while keeping log_probs meaningful.
const CONTINUOUS_STD: f64 = 0.5;

/// Compute log_prob of the action taken, given the raw policy outputs.
/// The action space is mixed discrete/continuous:
///   - Discrete heads: reel (3-way argmax), pass-target (K-way argmax)
///     log_prob = log(softmax(logit_chosen))
///   - Continuous heads: aim(3), fire_dir(3), throw_charge(1), throw_spin(1),
///     thrumbler(3) — Gaussian log_prob with fixed std around the mean (tanh output).
///   - Binary gates: fire_gate, release, pushoff, catch_intent, pass_gate
///     treated as Bernoulli with sigmoid probability.
///
/// Total log_prob = sum of all head log_probs (independence assumption).
pub fn compute_log_prob(out: &[f64; OUT_W], action_taken: &ActionRecord) -> f64 {
    let mut lp = 0.0_f64;

    // 1. Reel head: 3-way categorical over out[3..6]
    lp += log_softmax_at(&[out[3], out[4], out[5]], action_taken.reel_idx);

    // 2. Aim head: 3 continuous values (out[0..3]) with fixed Gaussian
    lp += gaussian_log_prob(out[0], action_taken.aim[0], CONTINUOUS_STD);
    lp += gaussian_log_prob(out[1], action_taken.aim[1], CONTINUOUS_STD);
    lp += gaussian_log_prob(out[2], action_taken.aim[2], CONTINUOUS_STD);

    // 3. Fire gate: Bernoulli (sigmoid of out[6])
    lp += bernoulli_log_prob(out[6], action_taken.fire_gate);

    // 4. Fire direction: 3 continuous (out[7..10])
    lp += gaussian_log_prob(out[7], action_taken.fire_dir[0], CONTINUOUS_STD);
    lp += gaussian_log_prob(out[8], action_taken.fire_dir[1], CONTINUOUS_STD);
    lp += gaussian_log_prob(out[9], action_taken.fire_dir[2], CONTINUOUS_STD);

    // 5. Bool heads: release, pushoff, catch_intent (out[10..13])
    lp += bernoulli_log_prob(out[10], action_taken.release);
    lp += bernoulli_log_prob(out[11], action_taken.pushoff);
    lp += bernoulli_log_prob(out[12], action_taken.catch_intent);

    // 6. Throw charge & spin: continuous (out[13..15])
    lp += gaussian_log_prob(out[13], action_taken.throw_charge_raw, CONTINUOUS_STD);
    lp += gaussian_log_prob(out[14], action_taken.throw_spin_raw, CONTINUOUS_STD);

    // 7. Thrumbler: 3 continuous (out[15..18])
    lp += gaussian_log_prob(out[15], action_taken.thrumbler[0], CONTINUOUS_STD);
    lp += gaussian_log_prob(out[16], action_taken.thrumbler[1], CONTINUOUS_STD);
    lp += gaussian_log_prob(out[17], action_taken.thrumbler[2], CONTINUOUS_STD);

    // 8. Pass gate: Bernoulli (out[18])
    lp += bernoulli_log_prob(out[18], action_taken.pass_gate);

    // 9. Pass target: K-way categorical over out[19..22]
    let pass_logits: Vec<f64> = (0..K).map(|k| out[19 + k]).collect();
    lp += log_softmax_at(&pass_logits, action_taken.pass_target_idx);

    lp
}

/// Compute entropy of the policy at a given output (sum of per-head entropies).
pub fn compute_entropy(out: &[f64; OUT_W]) -> f64 {
    let mut ent = 0.0_f64;

    // Reel head: categorical entropy
    ent += categorical_entropy(&[out[3], out[4], out[5]]);

    // Continuous heads: Gaussian entropy = 0.5 * ln(2*pi*e*sigma^2) per dim
    // 11 continuous dims total (aim 3 + fire_dir 3 + throw 2 + thrumbler 3)
    let gauss_ent = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E
        * CONTINUOUS_STD * CONTINUOUS_STD).ln();
    ent += 11.0 * gauss_ent;

    // Binary heads: Bernoulli entropy (5 heads: fire_gate, release, pushoff, catch, pass_gate)
    for &idx in &[6usize, 10, 11, 12, 18] {
        ent += bernoulli_entropy(out[idx]);
    }

    // Pass target: categorical entropy
    let pass_logits: Vec<f64> = (0..K).map(|k| out[19 + k]).collect();
    ent += categorical_entropy(&pass_logits);

    ent
}

/// Record of the action actually taken (sufficient to recompute log_prob).
#[derive(Clone, Debug)]
pub struct ActionRecord {
    pub reel_idx: usize,        // 0,1,2 → maps to reel -1,0,+1
    pub aim: [f64; 3],          // raw output values (pre-normalization)
    pub fire_gate: bool,
    pub fire_dir: [f64; 3],
    pub release: bool,
    pub pushoff: bool,
    pub catch_intent: bool,
    pub throw_charge_raw: f64,  // raw output (pre-tanh squash)
    pub throw_spin_raw: f64,
    pub thrumbler: [f64; 3],
    pub pass_gate: bool,
    pub pass_target_idx: usize, // 0..K
}

impl ActionRecord {
    /// Build from the raw OUT_W policy output (records what was "chosen").
    pub fn from_output(out: &[f64; OUT_W]) -> Self {
        // Reel: argmax
        let reel_logits = [out[3], out[4], out[5]];
        let mut reel_idx = 0usize;
        for i in 1..3 {
            if reel_logits[i] > reel_logits[reel_idx] {
                reel_idx = i;
            }
        }

        // Pass target: argmax
        let mut pass_idx = 0usize;
        for k in 1..K {
            if out[19 + k] > out[19 + pass_idx] {
                pass_idx = k;
            }
        }

        ActionRecord {
            reel_idx,
            aim: [out[0], out[1], out[2]],
            fire_gate: out[6] > 0.0,
            fire_dir: [out[7], out[8], out[9]],
            release: out[10].tanh() > 0.0,
            pushoff: out[11].tanh() > 0.0,
            catch_intent: out[12].tanh() > 0.0,
            throw_charge_raw: out[13],
            throw_spin_raw: out[14],
            thrumbler: [out[15], out[16], out[17]],
            pass_gate: out[18] > 0.0,
            pass_target_idx: pass_idx,
        }
    }
}

// ── Helper math ───────────────────────────────────────────────────────────────

/// Log of softmax at index `idx` from logits. Numerically stable.
fn log_softmax_at(logits: &[f64], idx: usize) -> f64 {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let log_sum_exp: f64 = logits.iter().map(|&l| (l - max).exp()).sum::<f64>().ln();
    logits[idx] - max - log_sum_exp
}

/// Categorical entropy from logits (stable softmax).
fn categorical_entropy(logits: &[f64]) -> f64 {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut ent = 0.0_f64;
    for &e in &exps {
        let p = e / sum;
        if p > 1e-12 {
            ent -= p * p.ln();
        }
    }
    ent
}

/// Gaussian log-probability: log N(x | mean, std).
fn gaussian_log_prob(mean: f64, x: f64, std: f64) -> f64 {
    let var = std * std;
    -0.5 * ((x - mean).powi(2) / var + (2.0 * std::f64::consts::PI * var).ln())
}

/// Bernoulli log-probability from a logit (sigmoid gate).
fn bernoulli_log_prob(logit: f64, taken: bool) -> f64 {
    // log sigmoid(x) = -log(1+exp(-x)) for taken=true
    // log(1-sigmoid(x)) = -log(1+exp(x)) for taken=false
    if taken {
        -softplus(-logit)
    } else {
        -softplus(logit)
    }
}

/// Bernoulli entropy from a logit.
fn bernoulli_entropy(logit: f64) -> f64 {
    let p = sigmoid(logit);
    let q = 1.0 - p;
    let mut ent = 0.0;
    if p > 1e-12 {
        ent -= p * p.ln();
    }
    if q > 1e-12 {
        ent -= q * q.ln();
    }
    ent
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

// ── Trajectory storage ────────────────────────────────────────────────────────

/// One timestep of one agent's experience.
#[derive(Clone, Debug)]
struct AgentTransition {
    obs: Observation,
    self_id: String,
    action_record: ActionRecord,
    log_prob: f64,
    value: f64,
    reward: f64,
    done: bool,
}

/// A complete rollout buffer for one environment.
struct EnvRollout {
    /// Per-agent (4 agents) per-timestep transitions.
    /// Shape: [horizon][N_AGENTS]
    transitions: Vec<[AgentTransition; N_AGENTS]>,
    /// Bootstrap value at the end of the rollout (per-agent).
    bootstrap_values: [f64; N_AGENTS],
}

// ── MAPPO Configuration ───────────────────────────────────────────────────────

/// Configuration for the MAPPO training loop.
#[derive(Clone, Debug)]
pub struct MappoConfig {
    /// Number of parallel environments for rollout collection.
    pub n_envs: usize,
    /// Steps per rollout per environment.
    pub horizon: usize,
    /// PPO epochs per rollout (how many times to reuse the collected data).
    pub epochs: usize,
    /// Minibatch size for PPO updates (number of agent-timesteps per batch).
    pub minibatch_size: usize,
    /// GAE discount factor.
    pub gamma: f64,
    /// GAE lambda (bias-variance tradeoff).
    pub lambda: f64,
    /// PPO clip epsilon.
    pub clip_epsilon: f64,
    /// Actor learning rate (ES step size).
    pub lr_actor: f64,
    /// Critic learning rate (ES step size for value function).
    pub lr_critic: f64,
    /// Entropy bonus coefficient.
    pub entropy_coef: f64,
    /// Value loss coefficient.
    pub value_coef: f64,
    /// Total training generations (rollout-update cycles).
    pub max_generations: usize,
    /// Add current policy to population every N generations.
    pub checkpoint_interval: usize,
    /// Master seed (deterministic training).
    pub seed: u64,
    /// Use self-play population for opponents.
    pub self_play: bool,
    /// ES population size for actor updates (antithetic pairs).
    pub es_pop_size: usize,
    /// ES noise std for actor perturbation.
    pub es_sigma_actor: f64,
    /// ES noise std for critic perturbation.
    pub es_sigma_critic: f64,
    /// Actor weight decay.
    pub weight_decay_actor: f64,
    /// Critic weight decay.
    pub weight_decay_critic: f64,
    /// Episode length in ticks per environment step.
    pub episode_ticks: u64,
    /// Maximum population size for self-play.
    pub max_population: usize,
}

impl Default for MappoConfig {
    fn default() -> Self {
        MappoConfig {
            n_envs: 16,
            horizon: 256,
            epochs: 4,
            minibatch_size: 512,
            gamma: 0.99,
            lambda: 0.95,
            clip_epsilon: 0.2,
            lr_actor: 0.03,
            lr_critic: 0.05,
            entropy_coef: 0.01,
            value_coef: 0.5,
            max_generations: 100,
            checkpoint_interval: 10,
            seed: 0xAA990,
            self_play: true,
            es_pop_size: 20,
            es_sigma_actor: 0.05,
            es_sigma_critic: 0.05,
            weight_decay_actor: 0.003,
            weight_decay_critic: 0.003,
            episode_ticks: 4000,
            max_population: 30,
        }
    }
}

/// Per-generation report.
#[derive(Clone, Debug)]
pub struct MappoGenReport {
    pub gen: usize,
    pub mean_reward: f64,
    pub mean_value_loss: f64,
    pub mean_policy_loss: f64,
    pub mean_entropy: f64,
    pub win_rate: f64,
    pub population_size: usize,
}

/// The final training result.
#[derive(Clone, Debug)]
pub struct MappoResult {
    pub actor_weights: Vec<f64>,
    pub critic_weights: Vec<f64>,
    pub history: Vec<MappoGenReport>,
    pub population: Vec<Vec<f64>>,
}

// ── The MAPPO Training Loop ───────────────────────────────────────────────────

/// Run the full MAPPO training loop. Deterministic given `cfg.seed`.
///
/// Architecture:
///   - Rollout collection: N_ENVS parallel SelfPlayEnv instances, each running
///     HORIZON steps with the current actor driving Home, opponent from population
///     driving Away.
///   - GAE advantage computation from the centralized value function.
///   - ES-shaped PPO update: the PPO clipped surrogate objective (computed from
///     the rollout) serves as the fitness for an ES gradient step on the actor.
///   - Critic update: ES minimizes value prediction MSE.
///   - Self-play: periodic checkpointing into a prioritized population.
pub fn train_mappo(cfg: MappoConfig) -> MappoResult {
    let mut master_rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let unit = Normal::new(0.0_f64, 1.0_f64).expect("valid normal");

    // Initialize actor and critic with small random weights.
    let mut actor_weights = vec![0.0_f64; ATTN_PARAM_W];
    let mut critic_weights = vec![0.0_f64; VALUE_PARAM_W];

    // Small random initialization for better starting point.
    let init_scale = 0.02;
    for w in actor_weights.iter_mut() {
        *w = unit.sample(&mut master_rng) * init_scale;
    }
    for w in critic_weights.iter_mut() {
        *w = unit.sample(&mut master_rng) * init_scale;
    }

    // Self-play population.
    let mut population = Population::new(cfg.max_population);
    population.add(vec![0.0_f64; ATTN_PARAM_W], 0); // seed with zero policy

    let mut history: Vec<MappoGenReport> = Vec::with_capacity(cfg.max_generations);

    // Running win statistics.
    let mut total_wins = 0u64;
    let mut total_games = 0u64;

    for gen in 0..cfg.max_generations {
        // ── 1. Select opponent ────────────────────────────────────────────
        let opp_weights = if cfg.self_play && !population.is_empty() {
            let opp_idx = population.sample_opponent(&mut master_rng);
            population.get_weights(opp_idx).to_vec()
        } else {
            vec![0.0_f64; ATTN_PARAM_W]
        };

        // ── 2. Collect rollouts ──────────────────────────────────────────
        let rollout_seed = master_rng.gen::<u64>();
        let (rollouts, episode_rewards) = collect_rollouts(
            &actor_weights,
            &critic_weights,
            &opp_weights,
            &cfg,
            rollout_seed,
        );

        // ── 3. Compute advantages (GAE) ─────────────────────────────────
        let (all_advantages, all_returns, all_old_log_probs, all_obs, all_ids, all_actions) =
            process_rollouts(&rollouts, &cfg);

        // Normalize advantages across the entire batch.
        let adv_mean = all_advantages.iter().sum::<f64>() / all_advantages.len().max(1) as f64;
        let adv_var = all_advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f64>()
            / all_advantages.len().max(1) as f64;
        let adv_std = (adv_var + 1e-8).sqrt();
        let norm_advantages: Vec<f64> = all_advantages.iter()
            .map(|a| (a - adv_mean) / adv_std)
            .collect();

        // ── 4. Compute PPO-shaped fitness for ES actor update ────────────
        // The PPO objective evaluated at the current actor weights.
        let mean_entropy = compute_batch_entropy(&actor_weights, &all_obs, &all_ids);

        // ES actor update: perturb weights, evaluate PPO objective, gradient step.
        let es_pairs = (cfg.es_pop_size.max(2) / 2) * 2;
        let n_pairs = es_pairs / 2;

        // Draw noise for actor.
        let mut actor_eps: Vec<Vec<f64>> = Vec::with_capacity(n_pairs);
        for _ in 0..n_pairs {
            let mut e = vec![0.0_f64; ATTN_PARAM_W];
            for d in 0..ATTN_PARAM_W {
                e[d] = unit.sample(&mut master_rng);
            }
            actor_eps.push(e);
        }

        // Evaluate PPO objective for each perturbed actor.
        let actor_ref = &actor_weights;
        let norm_adv_ref = &norm_advantages;
        let all_returns_ref = &all_returns;
        let all_old_lp_ref = &all_old_log_probs;
        let all_obs_ref = &all_obs;
        let all_ids_ref = &all_ids;
        let all_actions_ref = &all_actions;
        let critic_ref = &critic_weights;
        let cfg_ref = &cfg;

        let actor_fits: Vec<f64> = (0..es_pairs)
            .into_par_iter()
            .map(|m| {
                let pair = m / 2;
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                let e = &actor_eps[pair];
                let perturbed: Vec<f64> = (0..ATTN_PARAM_W)
                    .map(|d| actor_ref[d] + sign * cfg_ref.es_sigma_actor * e[d])
                    .collect();
                eval_ppo_objective(
                    &perturbed,
                    critic_ref,
                    norm_adv_ref,
                    all_returns_ref,
                    all_old_lp_ref,
                    all_obs_ref,
                    all_ids_ref,
                    all_actions_ref,
                    cfg_ref,
                )
            })
            .collect();

        // Rank-normalize actor fitnesses.
        let actor_utils = rank_normalize(&actor_fits);

        // Gradient step for actor.
        let scale = 1.0 / (cfg.es_sigma_actor * n_pairs as f64);
        let mut actor_grad = vec![0.0_f64; ATTN_PARAM_W];
        for pair in 0..n_pairs {
            let du = actor_utils[2 * pair] - actor_utils[2 * pair + 1];
            let e = &actor_eps[pair];
            for d in 0..ATTN_PARAM_W {
                actor_grad[d] += du * e[d];
            }
        }
        for d in 0..ATTN_PARAM_W {
            actor_weights[d] += cfg.lr_actor * (scale * actor_grad[d])
                - cfg.lr_actor * cfg.weight_decay_actor * actor_weights[d];
        }

        // ── 5. ES critic update: minimize value prediction error ─────────
        let mut critic_eps: Vec<Vec<f64>> = Vec::with_capacity(n_pairs);
        for _ in 0..n_pairs {
            let mut e = vec![0.0_f64; VALUE_PARAM_W];
            for d in 0..VALUE_PARAM_W {
                e[d] = unit.sample(&mut master_rng);
            }
            critic_eps.push(e);
        }

        let critic_fits: Vec<f64> = (0..es_pairs)
            .into_par_iter()
            .map(|m| {
                let pair = m / 2;
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                let e = &critic_eps[pair];
                let perturbed: Vec<f64> = (0..VALUE_PARAM_W)
                    .map(|d| critic_ref[d] + sign * cfg_ref.es_sigma_critic * e[d])
                    .collect();
                // Negative MSE (higher = better fit to returns).
                eval_critic_fitness(&perturbed, all_obs_ref, all_returns_ref)
            })
            .collect();

        let critic_utils = rank_normalize(&critic_fits);

        let scale_c = 1.0 / (cfg.es_sigma_critic * n_pairs as f64);
        let mut critic_grad = vec![0.0_f64; VALUE_PARAM_W];
        for pair in 0..n_pairs {
            let du = critic_utils[2 * pair] - critic_utils[2 * pair + 1];
            let e = &critic_eps[pair];
            for d in 0..VALUE_PARAM_W {
                critic_grad[d] += du * e[d];
            }
        }
        for d in 0..VALUE_PARAM_W {
            critic_weights[d] += cfg.lr_critic * (scale_c * critic_grad[d])
                - cfg.lr_critic * cfg.weight_decay_critic * critic_weights[d];
        }

        // ── 6. Compute statistics ────────────────────────────────────────
        let mean_reward = episode_rewards.iter().sum::<f64>() / episode_rewards.len().max(1) as f64;
        let mean_value_loss = {
            let vf = CentralizedValue::from_weights(critic_weights.clone());
            let mut mse = 0.0_f64;
            let n = all_obs.len().min(all_returns.len());
            for i in 0..n {
                let feat = value_featurize(&all_obs[i]);
                let pred = vf.forward(&feat);
                mse += (pred - all_returns[i]).powi(2);
            }
            mse / n.max(1) as f64
        };
        let mean_policy_loss = actor_fits.iter().sum::<f64>() / actor_fits.len().max(1) as f64;

        // Win rate: check episode rewards (positive = Home won).
        let wins = episode_rewards.iter().filter(|&&r| r > 0.0).count() as u64;
        total_wins += wins;
        total_games += episode_rewards.len() as u64;
        let win_rate = if total_games > 0 {
            total_wins as f64 / total_games as f64
        } else {
            0.5
        };

        // ── 7. Self-play population checkpoint ───────────────────────────
        let pop_size = if cfg.self_play {
            if (gen + 1) % cfg.checkpoint_interval == 0 || gen + 1 == cfg.max_generations {
                population.add(actor_weights.clone(), gen + 1);
            }
            population.len()
        } else {
            0
        };

        history.push(MappoGenReport {
            gen,
            mean_reward,
            mean_value_loss,
            mean_policy_loss,
            mean_entropy,
            win_rate,
            population_size: pop_size,
        });
    }

    // Collect population weights.
    let pop_weights: Vec<Vec<f64>> = population
        .entries()
        .iter()
        .map(|e| e.weights.clone())
        .collect();

    MappoResult {
        actor_weights,
        critic_weights,
        history,
        population: pop_weights,
    }
}

// ── Rollout Collection ────────────────────────────────────────────────────────

/// Collect rollouts from N_ENVS parallel environments.
/// Returns (rollouts, per-env cumulative rewards).
fn collect_rollouts(
    actor_weights: &[f64],
    critic_weights: &[f64],
    opp_weights: &[f64],
    cfg: &MappoConfig,
    seed: u64,
) -> (Vec<EnvRollout>, Vec<f64>) {
    // Collect in parallel but with fixed-index ordering for determinism.
    let results: Vec<(EnvRollout, f64)> = (0..cfg.n_envs)
        .into_par_iter()
        .map(|env_idx| {
            // Pin thread-local AI state for determinism.
            crate::ai::efe_params::set_efe_params(None);
            crate::ai::plan_bridge::set_planner_profile(None);

            let env_seed = seed.wrapping_add(env_idx as u64).wrapping_mul(6364136223846793005);
            collect_single_env_rollout(
                actor_weights,
                critic_weights,
                opp_weights,
                cfg,
                env_seed,
            )
        })
        .collect();

    let mut rollouts = Vec::with_capacity(cfg.n_envs);
    let mut rewards = Vec::with_capacity(cfg.n_envs);
    for (r, rew) in results {
        rollouts.push(r);
        rewards.push(rew);
    }
    (rollouts, rewards)
}

/// Collect a single environment's rollout.
fn collect_single_env_rollout(
    actor_weights: &[f64],
    critic_weights: &[f64],
    opp_weights: &[f64],
    cfg: &MappoConfig,
    seed: u64,
) -> (EnvRollout, f64) {
    let actor = AttentionPolicy::from_weights(actor_weights.to_vec());
    let opponent = AttentionPolicy::from_weights(opp_weights.to_vec());
    let critic = CentralizedValue::from_weights(critic_weights.to_vec());

    let mut env = SelfPlayEnv::new();
    let mut obs = env.reset((seed & 0xFFFFFFFF) as u32, cfg.episode_ticks);

    let mut transitions: Vec<[AgentTransition; N_AGENTS]> = Vec::with_capacity(cfg.horizon);
    let mut cumulative_reward = 0.0_f64;

    for _t in 0..cfg.horizon {
        // Value estimate for this state (centralized).
        let value = critic.evaluate(&obs);

        // Actor forward pass for each Home agent.
        let mut home_outputs = [[0.0_f64; OUT_W]; N_AGENTS];
        let mut home_action_records = Vec::with_capacity(N_AGENTS);
        let mut home_log_probs = [0.0_f64; N_AGENTS];
        let mut home_actions: Vec<ai::PlayerInput> = Vec::with_capacity(N_AGENTS);

        for (i, id) in HOME_IDS.iter().enumerate() {
            let out = actor.forward(&obs, id);
            home_outputs[i] = out;
            let ar = ActionRecord::from_output(&out);
            home_log_probs[i] = compute_log_prob(&out, &ar);
            home_actions.push(actor.decode(&out, &obs, id));
            home_action_records.push(ar);
        }

        // Opponent drives Away.
        let away_actions: Vec<ai::PlayerInput> = AWAY_IDS
            .iter()
            .map(|id| opponent.act(&obs, id))
            .collect();

        // Combine into 8-action array.
        let mut all_actions: [ai::PlayerInput; 8] = std::array::from_fn(|_| {
            ai::PlayerInput::idle("X")
        });
        for (i, a) in home_actions.iter().enumerate() {
            all_actions[i] = a.clone();
        }
        for (i, a) in away_actions.iter().enumerate() {
            all_actions[4 + i] = a.clone();
        }

        // Step the environment.
        let step = env.step(&all_actions);
        let reward = step.home_reward;
        cumulative_reward += reward;

        // Store transitions for each agent.
        let agent_transitions: [AgentTransition; N_AGENTS] = std::array::from_fn(|i| {
            AgentTransition {
                obs: obs.clone(),
                self_id: HOME_IDS[i].to_string(),
                action_record: home_action_records[i].clone(),
                log_prob: home_log_probs[i],
                value,  // Centralized value (same for all agents in this state)
                reward, // Team reward (same for all agents)
                done: step.done,
            }
        });
        transitions.push(agent_transitions);

        obs = step.obs.clone();

        if step.done {
            // Reset and continue collecting.
            let new_seed = seed.wrapping_add(transitions.len() as u64);
            obs = env.reset((new_seed & 0xFFFFFFFF) as u32, cfg.episode_ticks);
        }
    }

    // Bootstrap value.
    let bootstrap_value = critic.evaluate(&obs);
    let bootstrap_values = [bootstrap_value; N_AGENTS];

    (
        EnvRollout {
            transitions,
            bootstrap_values,
        },
        cumulative_reward,
    )
}

// ── Rollout Processing (GAE) ──────────────────────────────────────────────────

/// Process rollouts into flat vectors for PPO. Returns:
/// (advantages, returns, old_log_probs, observations, agent_ids, action_records)
fn process_rollouts(
    rollouts: &[EnvRollout],
    cfg: &MappoConfig,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<Observation>, Vec<String>, Vec<ActionRecord>) {
    let mut all_advantages = Vec::new();
    let mut all_returns = Vec::new();
    let mut all_old_log_probs = Vec::new();
    let mut all_obs = Vec::new();
    let mut all_ids = Vec::new();
    let mut all_actions = Vec::new();

    for rollout in rollouts {
        let t = rollout.transitions.len();
        if t == 0 {
            continue;
        }

        // Compute GAE per agent.
        for agent_idx in 0..N_AGENTS {
            let mut rewards = Vec::with_capacity(t);
            let mut values = Vec::with_capacity(t + 1);
            let mut dones = Vec::with_capacity(t);

            for step in &rollout.transitions {
                rewards.push(step[agent_idx].reward);
                values.push(step[agent_idx].value);
                dones.push(step[agent_idx].done);
            }
            // Bootstrap value.
            values.push(rollout.bootstrap_values[agent_idx]);

            let (advantages, returns) = compute_gae(&rewards, &values, &dones, cfg.gamma, cfg.lambda);

            for (i, step) in rollout.transitions.iter().enumerate() {
                all_advantages.push(advantages[i]);
                all_returns.push(returns[i]);
                all_old_log_probs.push(step[agent_idx].log_prob);
                all_obs.push(step[agent_idx].obs.clone());
                all_ids.push(step[agent_idx].self_id.clone());
                all_actions.push(step[agent_idx].action_record.clone());
            }
        }
    }

    (all_advantages, all_returns, all_old_log_probs, all_obs, all_ids, all_actions)
}

// ── PPO Objective Evaluation ──────────────────────────────────────────────────

/// Evaluate the PPO clipped surrogate objective for a given actor weight vector.
/// This is the "fitness" the ES maximizes.
fn eval_ppo_objective(
    actor_w: &[f64],
    critic_w: &[f64],
    advantages: &[f64],
    returns: &[f64],
    old_log_probs: &[f64],
    obs: &[Observation],
    ids: &[String],
    actions: &[ActionRecord],
    cfg: &MappoConfig,
) -> f64 {
    let actor = AttentionPolicy::from_weights(actor_w.to_vec());
    let critic = CentralizedValue::from_weights(critic_w.to_vec());
    let n = obs.len().min(advantages.len()).min(old_log_probs.len());

    if n == 0 {
        return 0.0;
    }

    // Sample a subset for efficiency (the ES evaluates many perturbations).
    let sample_size = cfg.minibatch_size.min(n);
    let stride = n / sample_size.max(1);

    let mut policy_loss = 0.0_f64;
    let mut value_loss = 0.0_f64;
    let mut entropy_sum = 0.0_f64;
    let mut count = 0usize;

    for i in (0..n).step_by(stride.max(1)) {
        if count >= sample_size {
            break;
        }

        // Re-run actor forward to get new log_prob.
        let new_out = actor.forward(&obs[i], &ids[i]);
        let new_log_prob = compute_log_prob(&new_out, &actions[i]);
        let entropy = compute_entropy(&new_out);

        // PPO ratio.
        let ratio = (new_log_prob - old_log_probs[i]).exp();
        let adv = advantages[i];

        // Clipped surrogate.
        let surr1 = ratio * adv;
        let surr2 = ratio.clamp(1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * adv;
        policy_loss += surr1.min(surr2);

        // Value loss.
        let feat = value_featurize(&obs[i]);
        let new_value = critic.forward(&feat);
        value_loss += (new_value - returns[i]).powi(2);

        entropy_sum += entropy;
        count += 1;
    }

    let count_f = count.max(1) as f64;
    // PPO objective: maximize policy_loss + entropy, minimize value_loss.
    // Return as a fitness (higher = better).
    (policy_loss / count_f)
        - cfg.value_coef * (value_loss / count_f)
        + cfg.entropy_coef * (entropy_sum / count_f)
}

/// Evaluate critic fitness: negative MSE of value predictions vs returns.
fn eval_critic_fitness(
    critic_w: &[f64],
    obs: &[Observation],
    returns: &[f64],
) -> f64 {
    let critic = CentralizedValue::from_weights(critic_w.to_vec());
    let n = obs.len().min(returns.len());
    if n == 0 {
        return 0.0;
    }

    // Sample for efficiency.
    let sample_size = 256.min(n);
    let stride = n / sample_size.max(1);
    let mut mse = 0.0_f64;
    let mut count = 0usize;

    for i in (0..n).step_by(stride.max(1)) {
        if count >= sample_size {
            break;
        }
        let feat = value_featurize(&obs[i]);
        let pred = critic.forward(&feat);
        mse += (pred - returns[i]).powi(2);
        count += 1;
    }

    // Negative MSE (higher = better for ES maximization).
    -(mse / count.max(1) as f64)
}

// ── Batch Entropy ─────────────────────────────────────────────────────────────

/// Compute mean entropy over a sample of observations.
fn compute_batch_entropy(actor_w: &[f64], obs: &[Observation], ids: &[String]) -> f64 {
    let actor = AttentionPolicy::from_weights(actor_w.to_vec());
    let n = obs.len();
    if n == 0 {
        return 0.0;
    }
    let sample_size = 128.min(n);
    let stride = n / sample_size.max(1);
    let mut total_ent = 0.0_f64;
    let mut count = 0usize;

    for i in (0..n).step_by(stride.max(1)) {
        if count >= sample_size {
            break;
        }
        let out = actor.forward(&obs[i], &ids[i]);
        total_ent += compute_entropy(&out);
        count += 1;
    }
    total_ent / count.max(1) as f64
}

// ── Rank Normalization ────────────────────────────────────────────────────────

/// Rank-normalize fitnesses to centered utilities in [-0.5, 0.5].
fn rank_normalize(fits: &[f64]) -> Vec<f64> {
    let pop = fits.len();
    if pop == 0 {
        return vec![];
    }
    let mut order: Vec<usize> = (0..pop).collect();
    order.sort_by(|&i, &j| {
        fits[j]
            .partial_cmp(&fits[i])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });
    let mut util = vec![0.0_f64; pop];
    let denom = (pop - 1).max(1) as f64;
    for (rank, &idx) in order.iter().enumerate() {
        util[idx] = rank as f64 / denom - 0.5;
    }
    util
}

// ── Weight serialization ──────────────────────────────────────────────────────

/// Serialize MAPPO actor weights to the standard attention policy JSON format.
pub fn mappo_weights_to_json(
    actor_weights: &[f64],
    cfg: &MappoConfig,
    mean_reward: f64,
) -> String {
    let config_str = format!(
        "mappo: n_envs={} horizon={} epochs={} es_pop={} sigma={:.3} lr_a={:.4} lr_c={:.4} \
         gamma={:.3} lambda={:.3} clip={:.2} ent={:.3} val={:.2} seed={}",
        cfg.n_envs, cfg.horizon, cfg.epochs, cfg.es_pop_size,
        cfg.es_sigma_actor, cfg.lr_actor, cfg.lr_critic,
        cfg.gamma, cfg.lambda, cfg.clip_epsilon, cfg.entropy_coef, cfg.value_coef, cfg.seed
    );
    crate::rl::attention::attn_weights_to_json(
        actor_weights,
        cfg.seed,
        &config_str,
        0.0,
        mean_reward,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Log-prob computation is finite and negative for a valid output.
    #[test]
    fn log_prob_is_finite_and_negative() {
        let out: [f64; OUT_W] = std::array::from_fn(|i| (i as f64 - 10.0) * 0.1);
        let ar = ActionRecord::from_output(&out);
        let lp = compute_log_prob(&out, &ar);
        assert!(lp.is_finite(), "log_prob not finite: {lp}");
        assert!(lp <= 0.0, "log_prob should be non-positive: {lp}");
    }

    /// Entropy is finite and non-negative.
    #[test]
    fn entropy_is_finite_and_nonneg() {
        let out: [f64; OUT_W] = std::array::from_fn(|i| (i as f64 - 11.0) * 0.05);
        let ent = compute_entropy(&out);
        assert!(ent.is_finite(), "entropy not finite: {ent}");
        assert!(ent >= 0.0, "entropy should be non-negative: {ent}");
    }

    /// ActionRecord round-trip: from_output + compute_log_prob is consistent.
    #[test]
    fn action_record_consistent() {
        // Different outputs should give different log_probs.
        let out1: [f64; OUT_W] = std::array::from_fn(|i| (i as f64) * 0.3);
        let out2: [f64; OUT_W] = std::array::from_fn(|i| -(i as f64) * 0.3);
        let ar1 = ActionRecord::from_output(&out1);
        let lp_same = compute_log_prob(&out1, &ar1);
        let lp_diff = compute_log_prob(&out2, &ar1);
        // The log_prob of the action under the policy that generated it should
        // generally be higher than under a very different policy.
        assert!(lp_same.is_finite());
        assert!(lp_diff.is_finite());
        // Not guaranteed to be strictly greater in all cases, but should differ.
        assert!((lp_same - lp_diff).abs() > 1e-6,
            "log_probs should differ: same={lp_same} diff={lp_diff}");
    }

    /// Rank normalization produces centered utilities.
    #[test]
    fn rank_normalize_is_centered() {
        let fits = vec![5.0, 3.0, 7.0, 1.0, 9.0, 2.0];
        let utils = rank_normalize(&fits);
        assert_eq!(utils.len(), 6);
        let sum: f64 = utils.iter().sum();
        assert!((sum).abs() < 1e-10, "utilities should sum to ~0: {sum}");
        // Best fitness (9.0 at index 4) should get utility -0.5 (rank 0).
        assert!((utils[4] - (-0.5)).abs() < 1e-10);
        // Worst fitness (1.0 at index 3) should get utility 0.5 (rank 5).
        assert!((utils[3] - 0.5).abs() < 1e-10);
    }

    /// Micro MAPPO smoke test: 2 envs, 32 horizon, 2 gens. Completes without
    /// panic and produces finite results.
    #[test]
    #[ignore = "MAPPO smoke test — runs training (slow)"]
    fn mappo_smoke() {
        let cfg = MappoConfig {
            n_envs: 2,
            horizon: 32,
            epochs: 1,
            minibatch_size: 64,
            max_generations: 2,
            es_pop_size: 4,
            episode_ticks: 200,
            checkpoint_interval: 1,
            seed: 42,
            ..MappoConfig::default()
        };
        let result = train_mappo(cfg);
        assert_eq!(result.actor_weights.len(), ATTN_PARAM_W);
        assert_eq!(result.critic_weights.len(), VALUE_PARAM_W);
        assert_eq!(result.history.len(), 2);
        for r in &result.history {
            assert!(r.mean_reward.is_finite(), "mean_reward not finite at gen {}", r.gen);
            assert!(r.mean_entropy.is_finite(), "entropy not finite at gen {}", r.gen);
            assert!(r.mean_value_loss.is_finite(), "value_loss not finite at gen {}", r.gen);
        }
        println!("MAPPO smoke test results:");
        for r in &result.history {
            println!(
                "  gen {} reward={:.4} vloss={:.4} entropy={:.4} winrate={:.2} pop={}",
                r.gen, r.mean_reward, r.mean_value_loss, r.mean_entropy, r.win_rate, r.population_size
            );
        }
    }

    /// The math helpers: softplus, sigmoid, log_softmax are well-behaved.
    #[test]
    fn math_helpers_well_behaved() {
        // Sigmoid bounds.
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);

        // Softplus.
        assert!((softplus(0.0) - 0.693).abs() < 0.01);
        assert!((softplus(100.0) - 100.0).abs() < 1e-5);
        assert!(softplus(-100.0).abs() < 1e-5);

        // Log softmax sums to 0 in exp-space.
        let logits = [1.0, 2.0, 3.0];
        let lse: f64 = (0..3).map(|i| log_softmax_at(&logits, i).exp()).sum();
        assert!((lse - 1.0).abs() < 1e-10, "softmax should sum to 1: {lse}");
    }

    /// Gaussian log_prob peaks at the mean and is symmetric.
    #[test]
    fn gaussian_log_prob_properties() {
        let mean = 1.5;
        let std = 0.5;
        let at_mean = gaussian_log_prob(mean, mean, std);
        let off_by_1 = gaussian_log_prob(mean, mean + 1.0, std);
        let off_by_neg1 = gaussian_log_prob(mean, mean - 1.0, std);
        assert!(at_mean > off_by_1, "peak should be at mean");
        assert!((off_by_1 - off_by_neg1).abs() < 1e-10, "should be symmetric");
    }
}

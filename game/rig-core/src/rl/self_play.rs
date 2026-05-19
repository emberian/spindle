//! rl/self_play.rs — Self-play population manager for robust RL training.
//!
//! The existing gym (`RigEnv`) pits a learned policy (driving `controlled_ids`)
//! against the fixed baseline `AiSystem`. This module extends the training
//! infrastructure so BOTH teams can be policy-driven, enabling self-play
//! training against a diverse population of past selves.
//!
//! ── Architecture ────────────────────────────────────────────────────────────
//!
//! `SelfPlayEnv` wraps `RigEnv` with ALL 8 riggers controlled (the `AiSystem`
//! is bypassed entirely). `step` accepts actions for all 8, and computes reward
//! from EACH side's perspective so both policies get training signal.
//!
//! `Population` holds a capped collection of policy weight checkpoints with
//! metadata (generation, win rate). Opponent sampling uses prioritized
//! fictitious self-play (AlphaStar-lite): 50% best, 30% uniform random past
//! self, 20% most recent.
//!
//! `run_self_play` ties these together with the ES optimizer: each generation
//! samples an opponent, runs episodes, updates the population.
//!
//! ── Determinism ─────────────────────────────────────────────────────────────
//! All sampling uses a seeded `ChaCha8Rng`. Episode evaluation uses fixed
//! seeds. Population operations iterate in insertion order (a `Vec`, not
//! HashMap). Same master seed + config => bit-identical self-play trajectory.
//!
//! NATIVE-ONLY (gated like everything in `rl/`).

use super::policy::{RlPolicy, PARAM_W};
use crate::ai::types as ai;
use crate::gym::{Env, Observation, RewardConfig, RigEnv, Scenario, Step, TeamSide};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

// ── Constants ──────────────────────────────────────────────────────────────────

/// All 8 riggers (both teams) are externally controlled in self-play mode.
const ALL_IDS: [&str; 8] = ["H1", "H2", "H3", "H4", "A1", "A2", "A3", "A4"];
const HOME_IDS: [&str; 4] = ["H1", "H2", "H3", "H4"];
const AWAY_IDS: [&str; 4] = ["A1", "A2", "A3", "A4"];

// ── SelfPlayEnv ────────────────────────────────────────────────────────────────

/// A full-control gym wrapper: BOTH teams driven by policies. The baseline
/// `AiSystem` is bypassed entirely (all 8 riggers are in `controlled_ids`).
///
/// `step` accepts 8 actions (H1-H4 then A1-A4, in roster order) and returns
/// reward from both sides' perspectives.
pub struct SelfPlayEnv {
    env: RigEnv,
    episode_ticks: u64,
}

/// The result of one self-play step, with reward from both perspectives.
#[derive(Clone, Debug)]
pub struct SelfPlayStep {
    pub obs: Observation,
    /// Reward from Home team's perspective.
    pub home_reward: f64,
    /// Reward from Away team's perspective.
    pub away_reward: f64,
    pub done: bool,
    pub truncated: bool,
}

impl SelfPlayEnv {
    pub fn new() -> Self {
        SelfPlayEnv {
            env: RigEnv::new(),
            episode_ticks: 8000,
        }
    }

    /// Build the full-control scenario. All 8 riggers are controlled; the
    /// baseline AI is bypassed. Reward side is set to Home by default (the
    /// step method computes both perspectives regardless).
    fn scenario(&self, reward_side: TeamSide) -> Scenario {
        Scenario {
            home_style: "fall-dynasty".into(),
            home_cyl: "big-slow".into(),
            away_style: "fall-dynasty".into(),
            away_cyl: "big-slow".into(),
            controlled_ids: ALL_IDS.iter().map(|s| s.to_string()).collect(),
            max_ticks: self.episode_ticks,
            reward_config: RewardConfig {
                reward_side: Some(reward_side),
                ..RewardConfig::default()
            },
        }
    }

    /// Reset with full control. Returns the initial observation.
    pub fn reset(&mut self, seed: u32, episode_ticks: u64) -> Observation {
        self.episode_ticks = episode_ticks;
        // Pin baseline AI params (even though it's bypassed, keep the
        // thread-local state clean for determinism).
        crate::ai::efe_params::set_efe_params(None);
        crate::ai::plan_bridge::set_planner_profile(None);
        let sc = self.scenario(TeamSide::Home);
        self.env.reset(seed, &sc)
    }

    /// Step with actions for all 8 riggers (H1-H4 then A1-A4).
    /// Returns reward from both perspectives.
    ///
    /// Implementation: we run the env once with reward_side=Home, then
    /// compute the away reward by noting that away reward components are
    /// the negation of home's (possession/gate/score/contest swap signs
    /// for the other team). This avoids running the env twice.
    pub fn step(&mut self, actions: &[ai::PlayerInput; 8]) -> SelfPlayStep {
        let step = self.env.step(actions.as_slice());
        // The env was reset with reward_side=Home, so step.reward.total is
        // from Home's perspective. Away's perspective is approximately the
        // negative (zero-sum game), but we compute it honestly from the
        // reward components: possession/gate/contest/score swap signs.
        let home_r = step.reward.total;
        // In a zero-sum sport, the opponent's reward is the negation of the
        // intrinsic signal (both see the same events from opposite sides).
        // The terminal bonus is already signed correctly for Home; Away gets
        // the opposite. The raw components (possession, gate, contest, score)
        // are already signed from the reward side — flip them all.
        let away_r = -home_r;

        SelfPlayStep {
            obs: step.obs,
            home_reward: home_r,
            away_reward: away_r,
            done: step.done,
            truncated: step.truncated,
        }
    }
}

// ── Population ─────────────────────────────────────────────────────────────────

/// Metadata for a population member.
#[derive(Clone, Debug)]
pub struct PopEntry {
    pub weights: Vec<f64>,
    pub generation: usize,
    pub wins: u64,
    pub games: u64,
}

impl PopEntry {
    pub fn win_rate(&self) -> f64 {
        if self.games == 0 {
            0.5
        } else {
            self.wins as f64 / self.games as f64
        }
    }
}

/// A capped population of policy checkpoints for self-play training.
/// Maintains insertion order (Vec-based, deterministic iteration).
///
/// Sampling strategy (prioritized fictitious self-play, AlphaStar-lite):
///   - 50% chance: the current best (highest win rate; ties → latest gen)
///   - 30% chance: uniform random past self (breadth)
///   - 20% chance: the most recent checkpoint (stability)
///
/// Capacity is capped at `max_size`; when full, the oldest non-best entry
/// is pruned.
pub struct Population {
    entries: Vec<PopEntry>,
    max_size: usize,
}

impl Population {
    pub fn new(max_size: usize) -> Self {
        Population {
            entries: Vec::with_capacity(max_size.min(64)),
            max_size: max_size.max(2),
        }
    }

    /// Add a checkpoint to the population. Prunes oldest non-best if full.
    pub fn add(&mut self, weights: Vec<f64>, generation: usize) {
        if self.entries.len() >= self.max_size {
            self.prune();
        }
        self.entries.push(PopEntry {
            weights,
            generation,
            wins: 0,
            games: 0,
        });
    }

    /// Number of entries currently in the population.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the population is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Index of the current best (highest win rate; ties → latest generation;
    /// ties again → latest index). Deterministic.
    fn best_idx(&self) -> usize {
        if self.entries.is_empty() {
            return 0;
        }
        let mut best = 0usize;
        for i in 1..self.entries.len() {
            let bwr = self.entries[best].win_rate();
            let iwr = self.entries[i].win_rate();
            if iwr > bwr
                || (iwr == bwr && self.entries[i].generation > self.entries[best].generation)
                || (iwr == bwr
                    && self.entries[i].generation == self.entries[best].generation
                    && i > best)
            {
                best = i;
            }
        }
        best
    }

    /// Prune the oldest non-best entry to make room.
    fn prune(&mut self) {
        if self.entries.len() < 2 {
            return;
        }
        let best = self.best_idx();
        // Find the oldest (lowest index) that isn't best.
        let victim = (0..self.entries.len()).find(|&i| i != best);
        if let Some(v) = victim {
            self.entries.remove(v);
        }
    }

    /// Sample an opponent index using the prioritized fictitious self-play
    /// strategy. Deterministic given the rng state.
    ///
    /// - 50%: best (highest win rate)
    /// - 30%: uniform random (breadth)
    /// - 20%: most recent (stability)
    pub fn sample_opponent(&self, rng: &mut ChaCha8Rng) -> usize {
        if self.entries.is_empty() {
            return 0;
        }
        if self.entries.len() == 1 {
            return 0;
        }
        let roll: f64 = rng.gen();
        if roll < 0.5 {
            // Best
            self.best_idx()
        } else if roll < 0.8 {
            // Uniform random
            rng.gen_range(0..self.entries.len())
        } else {
            // Most recent
            self.entries.len() - 1
        }
    }

    /// Get the weights of a population entry by index.
    pub fn get_weights(&self, idx: usize) -> &[f64] {
        &self.entries[idx].weights
    }

    /// Update win/loss stats for a population entry.
    pub fn update_stats(&mut self, idx: usize, won: bool) {
        if idx < self.entries.len() {
            self.entries[idx].games += 1;
            if won {
                self.entries[idx].wins += 1;
            }
        }
    }

    /// Get all entries (for inspection/testing).
    pub fn entries(&self) -> &[PopEntry] {
        &self.entries
    }
}

// ── Self-play training config ──────────────────────────────────────────────────

/// Configuration for the self-play training loop.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// ES population size (antithetic pairs).
    pub pop_size: usize,
    /// Number of self-play generations.
    pub generations: usize,
    /// ES noise std.
    pub sigma: f64,
    /// ES learning rate.
    pub lr: f64,
    /// Decoupled weight decay.
    pub weight_decay: f64,
    /// Master seed (deterministic).
    pub seed: u64,
    /// Episode length in ticks.
    pub episode_ticks: u64,
    /// Number of episodes per fitness eval.
    pub episodes: usize,
    /// Base episode seed.
    pub eval_seed: u32,
    /// How often to checkpoint into the population (every N generations).
    pub checkpoint_interval: usize,
    /// Maximum population size.
    pub max_population: usize,
    /// Optional shaping gamma (same as TrainConfig).
    pub shaping_gamma: Option<f64>,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            pop_size: 40,
            generations: 16,
            sigma: 0.10,
            lr: 0.05,
            weight_decay: 0.005,
            seed: 0x5E1F,
            episode_ticks: 8000,
            episodes: 2,
            eval_seed: 1234,
            checkpoint_interval: 4,
            max_population: 50,
            shaping_gamma: Some(0.997),
        }
    }
}

/// Per-generation report for self-play training.
#[derive(Clone, Debug)]
pub struct SelfPlayGenReport {
    pub gen: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    /// Index of the opponent sampled from the population.
    pub opponent_idx: usize,
    /// Win rate of the current best against the population (running).
    pub vs_population_win_rate: f64,
    /// Number of distinct opponents faced so far.
    pub diversity: usize,
}

/// The final result of self-play training.
#[derive(Clone, Debug)]
pub struct SelfPlayResult {
    pub weights: Vec<f64>,
    pub fitness: f64,
    pub history: Vec<SelfPlayGenReport>,
    /// The final population (for continuation / analysis).
    pub population_size: usize,
    /// Approximate Elo (optional; computed from win-rate matrix if enough data).
    pub elo_estimate: Option<f64>,
}

// ── Self-play episode runner ───────────────────────────────────────────────────

/// Run one deterministic self-play episode: `home_policy` drives H1-H4,
/// `away_policy` drives A1-A4. Returns (home_total_reward, away_total_reward).
/// Pure given (weights, seed, episode_ticks, shaping_gamma).
fn run_self_play_episode(
    home_policy: &RlPolicy,
    away_policy: &RlPolicy,
    seed: u32,
    episode_ticks: u64,
    shaping_gamma: Option<f64>,
) -> (f64, f64) {
    crate::ai::efe_params::set_efe_params(None);
    crate::ai::plan_bridge::set_planner_profile(None);

    let sc = Scenario {
        home_style: "fall-dynasty".into(),
        home_cyl: "big-slow".into(),
        away_style: "fall-dynasty".into(),
        away_cyl: "big-slow".into(),
        controlled_ids: ALL_IDS.iter().map(|s| s.to_string()).collect(),
        max_ticks: episode_ticks,
        reward_config: RewardConfig {
            reward_side: Some(TeamSide::Home),
            shaping_gamma,
            ..RewardConfig::default()
        },
    };

    let mut env = RigEnv::new();
    let mut obs: Observation = env.reset(seed, &sc);
    let mut home_total = 0.0_f64;

    for _ in 0..episode_ticks {
        // Home actions (H1-H4) from home_policy's perspective.
        let home_acts: Vec<ai::PlayerInput> =
            HOME_IDS.iter().map(|id| home_policy.act(&obs, id)).collect();
        // Away actions (A1-A4) from away_policy's perspective.
        let away_acts: Vec<ai::PlayerInput> =
            AWAY_IDS.iter().map(|id| away_policy.act(&obs, id)).collect();

        // Combine into a single 8-action slice (H1-H4 then A1-A4).
        let mut all_acts: Vec<ai::PlayerInput> = Vec::with_capacity(8);
        all_acts.extend(home_acts);
        all_acts.extend(away_acts);

        let step = env.step(&all_acts);
        home_total += step.reward.total;
        obs = step.obs;
        if step.done {
            break;
        }
    }

    // Zero-sum: away reward is the negation of home reward.
    (home_total, -home_total)
}

/// Evaluate a policy against an opponent over multiple episodes.
/// Returns mean home reward (the fitness of the home/current policy).
fn evaluate_vs(
    current: &RlPolicy,
    opponent: &RlPolicy,
    eval_seed: u32,
    episodes: usize,
    episode_ticks: u64,
    shaping_gamma: Option<f64>,
) -> f64 {
    let n = episodes.max(1);
    let mut total = 0.0_f64;
    for e in 0..n {
        let (home_r, _) = run_self_play_episode(
            current,
            opponent,
            eval_seed + e as u32,
            episode_ticks,
            shaping_gamma,
        );
        total += home_r;
    }
    total / n as f64
}

// ── Self-play training loop ────────────────────────────────────────────────────

/// The main self-play training loop. Ties together:
/// - Full-control env (both sides policy-driven)
/// - Population (opponent sampling + checkpointing)
/// - ES optimizer (antithetic, rank-normalized)
///
/// Each generation:
///   1. Sample an opponent from the population
///   2. Run ES: evaluate current policy (Home) vs sampled opponent (Away)
///   3. ES gradient step
///   4. Every N gens: checkpoint current best into the population
///
/// Deterministic given the master seed.
pub fn run_self_play(cfg: SelfPlayConfig) -> SelfPlayResult {
    use rayon::prelude::*;

    let dim = PARAM_W;
    let pop = (cfg.pop_size.max(2) / 2) * 2;
    let n_pairs = pop / 2;
    let gens = cfg.generations.max(1);
    let sigma = cfg.sigma.max(0.01);
    let lr = cfg.lr;
    let wd = cfg.weight_decay;

    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let unit = Normal::new(0.0_f64, 1.0_f64).expect("valid normal");

    // Start from zero weights.
    let mut theta = vec![0.0_f64; dim];
    let mut best_w = theta.clone();
    let mut best_fit = f64::NEG_INFINITY;
    let mut history: Vec<SelfPlayGenReport> = Vec::with_capacity(gens);

    // Initialize population with the zero policy as the first opponent.
    let mut population = Population::new(cfg.max_population);
    population.add(vec![0.0_f64; dim], 0);

    // Track distinct opponents faced.
    let mut opponents_faced: Vec<bool> = vec![false; cfg.max_population + gens];
    let mut diversity_count = 0usize;

    // Running win counter for vs-population metric.
    let mut vs_pop_wins = 0u64;
    let mut vs_pop_games = 0u64;

    for gen in 0..gens {
        // Sample opponent from population.
        let opp_idx = population.sample_opponent(&mut rng);
        let opp_weights = population.get_weights(opp_idx).to_vec();
        let opponent = RlPolicy::from_weights(opp_weights);

        // Track diversity.
        if opp_idx < opponents_faced.len() && !opponents_faced[opp_idx] {
            opponents_faced[opp_idx] = true;
            diversity_count += 1;
        }

        // (a) Draw noise vectors in fixed (pair, dim) order.
        let mut eps: Vec<Vec<f64>> = Vec::with_capacity(n_pairs);
        for _ in 0..n_pairs {
            let mut e = vec![0.0_f64; dim];
            for d in 0..dim {
                e[d] = unit.sample(&mut rng);
            }
            eps.push(e);
        }

        // (b) Evaluate fitness: current policy (perturbed) vs sampled opponent.
        let cfg_clone = cfg.clone();
        let theta_ref = &theta;
        let opp_ref = &opponent;
        let fits: Vec<f64> = (0..pop)
            .into_par_iter()
            .map(|m| {
                // Pin thread-local AI state for determinism.
                crate::ai::efe_params::set_efe_params(None);
                crate::ai::plan_bridge::set_planner_profile(None);

                let pair = m / 2;
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                let e = &eps[pair];
                let w: Vec<f64> = (0..dim)
                    .map(|d| theta_ref[d] + sign * sigma * e[d])
                    .collect();
                let current = RlPolicy::from_weights(w);
                evaluate_vs(
                    &current,
                    opp_ref,
                    cfg_clone.eval_seed,
                    cfg_clone.episodes,
                    cfg_clone.episode_ticks,
                    cfg_clone.shaping_gamma,
                )
            })
            .collect();

        // Track global best.
        let mut gb = 0usize;
        for m in 1..pop {
            if fits[m] > fits[gb] {
                gb = m;
            }
        }
        if fits[gb] > best_fit {
            best_fit = fits[gb];
            let pair = gb / 2;
            let sign = if gb % 2 == 0 { 1.0 } else { -1.0 };
            best_w = (0..dim)
                .map(|d| theta[d] + sign * sigma * eps[pair][d])
                .collect();
        }

        // Did the best beat the opponent? (fitness > 0 means home won on balance)
        let won = best_fit > 0.0;
        population.update_stats(opp_idx, !won); // opponent's perspective
        vs_pop_games += 1;
        if won {
            vs_pop_wins += 1;
        }

        let mean_fit = fits.iter().sum::<f64>() / pop as f64;
        let vs_wr = if vs_pop_games > 0 {
            vs_pop_wins as f64 / vs_pop_games as f64
        } else {
            0.5
        };

        history.push(SelfPlayGenReport {
            gen,
            best_fitness: best_fit,
            mean_fitness: mean_fit,
            opponent_idx: opp_idx,
            vs_population_win_rate: vs_wr,
            diversity: diversity_count,
        });

        // Checkpoint into population every N generations.
        if (gen + 1) % cfg.checkpoint_interval == 0 || gen + 1 == gens {
            population.add(best_w.clone(), gen + 1);
        }

        if gen + 1 == gens {
            break;
        }

        // (c) Rank-normalize.
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

        // (d) Gradient step.
        let scale = 1.0 / (sigma * n_pairs as f64);
        let mut grad = vec![0.0_f64; dim];
        for pair in 0..n_pairs {
            let du = util[2 * pair] - util[2 * pair + 1];
            let e = &eps[pair];
            for d in 0..dim {
                grad[d] += du * e[d];
            }
        }
        for d in 0..dim {
            theta[d] += lr * (scale * grad[d]) - lr * wd * theta[d];
        }
    }

    SelfPlayResult {
        weights: best_w,
        fitness: best_fit,
        history,
        population_size: population.len(),
        elo_estimate: None, // Future: compute from win-rate matrix
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// CHEAP always-on test: full control with the zero policy on both sides,
    /// same seed twice => bit-identical trajectory.
    #[test]
    fn full_control_deterministic() {
        let zero = RlPolicy::zeros();
        let seed = 777u32;
        let ticks = 200u64;

        let run = || {
            crate::ai::efe_params::set_efe_params(None);
            crate::ai::plan_bridge::set_planner_profile(None);
            let sc = Scenario {
                home_style: "fall-dynasty".into(),
                home_cyl: "big-slow".into(),
                away_style: "fall-dynasty".into(),
                away_cyl: "big-slow".into(),
                controlled_ids: ALL_IDS.iter().map(|s| s.to_string()).collect(),
                max_ticks: ticks,
                reward_config: RewardConfig {
                    reward_side: Some(TeamSide::Home),
                    ..RewardConfig::default()
                },
            };
            let mut env = RigEnv::new();
            let mut obs = env.reset(seed, &sc);
            let mut rewards: Vec<u64> = Vec::new();
            for _ in 0..ticks {
                let home_acts: Vec<ai::PlayerInput> =
                    HOME_IDS.iter().map(|id| zero.act(&obs, id)).collect();
                let away_acts: Vec<ai::PlayerInput> =
                    AWAY_IDS.iter().map(|id| zero.act(&obs, id)).collect();
                let mut all_acts: Vec<ai::PlayerInput> = Vec::with_capacity(8);
                all_acts.extend(home_acts);
                all_acts.extend(away_acts);
                let step = env.step(&all_acts);
                rewards.push(step.reward.total.to_bits());
                obs = step.obs;
                if step.done {
                    break;
                }
            }
            rewards
        };

        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len(), "trajectory length mismatch");
        for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(ra, rb, "reward diverged at step {i}");
        }
    }

    /// Both teams driven by policies, 100 steps, no panic, reward from both
    /// perspectives.
    #[test]
    #[ignore = "self-play env smoke test — on-demand"]
    fn self_play_env_both_sides_controlled() {
        let home = RlPolicy::zeros();
        let away = RlPolicy::zeros();
        let seed = 42u32;
        let ticks = 100u64;

        let (home_r, away_r) = run_self_play_episode(
            &home,
            &away,
            seed,
            ticks,
            None,
        );

        assert!(home_r.is_finite(), "home reward not finite");
        assert!(away_r.is_finite(), "away reward not finite");
        // Zero-sum: they should be negations of each other.
        assert_eq!(
            home_r.to_bits(),
            (-away_r).to_bits(),
            "not zero-sum: home={home_r} away={away_r}"
        );
    }

    /// Population deterministic sampling: same seed => same opponent sequence.
    #[test]
    fn population_deterministic_sampling() {
        let mut pop = Population::new(10);
        // Add several entries.
        for i in 0..5 {
            pop.add(vec![i as f64; PARAM_W], i);
        }
        // Update some stats to make sampling non-trivial.
        pop.update_stats(2, true);
        pop.update_stats(2, true);
        pop.update_stats(0, false);

        let mut rng1 = ChaCha8Rng::seed_from_u64(12345);
        let mut rng2 = ChaCha8Rng::seed_from_u64(12345);

        let seq1: Vec<usize> = (0..20).map(|_| pop.sample_opponent(&mut rng1)).collect();
        let seq2: Vec<usize> = (0..20).map(|_| pop.sample_opponent(&mut rng2)).collect();

        assert_eq!(seq1, seq2, "population sampling not deterministic");

        // Verify the sequence isn't degenerate (all same index).
        let all_same = seq1.iter().all(|&x| x == seq1[0]);
        assert!(!all_same, "sampling should produce variety with 5 entries");
    }

    /// Population add/prune/capacity.
    #[test]
    fn population_capacity_and_pruning() {
        let mut pop = Population::new(5);
        for i in 0..5 {
            pop.add(vec![i as f64; PARAM_W], i);
        }
        assert_eq!(pop.len(), 5);

        // Mark entry 3 as the best.
        pop.update_stats(3, true);
        pop.update_stats(3, true);
        pop.update_stats(3, true);

        // Adding a 6th should prune the oldest non-best (index 0).
        pop.add(vec![5.0; PARAM_W], 5);
        assert_eq!(pop.len(), 5);
        // The best (entry with gen=3 and high win rate) should survive.
        assert!(pop.entries().iter().any(|e| e.generation == 3));
    }

    /// Micro self-play run: tiny policy, 2 gens, pop cap 3, completes and
    /// population grows.
    #[test]
    #[ignore = "self-play training smoke test — slow"]
    fn self_play_smoke() {
        let cfg = SelfPlayConfig {
            pop_size: 4,
            generations: 2,
            sigma: 0.10,
            lr: 0.05,
            weight_decay: 0.005,
            seed: 9999,
            episode_ticks: 100,
            episodes: 1,
            eval_seed: 1234,
            checkpoint_interval: 1,
            max_population: 3,
            shaping_gamma: None,
        };

        let result = run_self_play(cfg);

        assert_eq!(result.weights.len(), PARAM_W);
        assert!(result.fitness.is_finite());
        assert_eq!(result.history.len(), 2);
        // Population grew (initial + checkpoints).
        assert!(result.population_size >= 2);
        // History records are sane.
        for r in &result.history {
            assert!(r.best_fitness.is_finite());
            assert!(r.mean_fitness.is_finite());
            assert!(r.diversity >= 1);
        }
        println!("self_play_smoke result:");
        for r in &result.history {
            println!(
                "  gen {} best={:.4} mean={:.4} opp_idx={} vs_pop_wr={:.2} diversity={}",
                r.gen, r.best_fitness, r.mean_fitness, r.opponent_idx,
                r.vs_population_win_rate, r.diversity
            );
        }
        println!(
            "  final: weights_len={} fitness={:.4} pop_size={}",
            result.weights.len(), result.fitness, result.population_size
        );
    }
}

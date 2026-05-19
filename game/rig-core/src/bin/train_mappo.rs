//! train_mappo — Binary that runs the MAPPO training loop.
//!
//! Usage: cargo run --release --bin train_mappo [-- --gens N --envs N ...]
//! Outputs per-generation stats and saves trained weights to policy-mappo.json.
//!
//! NATIVE-ONLY (not compiled for wasm32).

use rig_core::rl::mappo::{train_mappo, MappoConfig, mappo_weights_to_json};

use std::env;
use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse simple CLI args (--key value pairs).
    let mut cfg = MappoConfig::default();
    let mut output_path = "policy-mappo.json".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gens" | "--generations" => {
                i += 1;
                cfg.max_generations = args[i].parse().unwrap_or(cfg.max_generations);
            }
            "--envs" | "--n-envs" => {
                i += 1;
                cfg.n_envs = args[i].parse().unwrap_or(cfg.n_envs);
            }
            "--horizon" => {
                i += 1;
                cfg.horizon = args[i].parse().unwrap_or(cfg.horizon);
            }
            "--epochs" => {
                i += 1;
                cfg.epochs = args[i].parse().unwrap_or(cfg.epochs);
            }
            "--minibatch" => {
                i += 1;
                cfg.minibatch_size = args[i].parse().unwrap_or(cfg.minibatch_size);
            }
            "--gamma" => {
                i += 1;
                cfg.gamma = args[i].parse().unwrap_or(cfg.gamma);
            }
            "--lambda" => {
                i += 1;
                cfg.lambda = args[i].parse().unwrap_or(cfg.lambda);
            }
            "--clip" => {
                i += 1;
                cfg.clip_epsilon = args[i].parse().unwrap_or(cfg.clip_epsilon);
            }
            "--lr-actor" => {
                i += 1;
                cfg.lr_actor = args[i].parse().unwrap_or(cfg.lr_actor);
            }
            "--lr-critic" => {
                i += 1;
                cfg.lr_critic = args[i].parse().unwrap_or(cfg.lr_critic);
            }
            "--entropy-coef" => {
                i += 1;
                cfg.entropy_coef = args[i].parse().unwrap_or(cfg.entropy_coef);
            }
            "--value-coef" => {
                i += 1;
                cfg.value_coef = args[i].parse().unwrap_or(cfg.value_coef);
            }
            "--seed" => {
                i += 1;
                cfg.seed = args[i].parse().unwrap_or(cfg.seed);
            }
            "--es-pop" => {
                i += 1;
                cfg.es_pop_size = args[i].parse().unwrap_or(cfg.es_pop_size);
            }
            "--es-sigma" => {
                i += 1;
                let s: f64 = args[i].parse().unwrap_or(cfg.es_sigma_actor);
                cfg.es_sigma_actor = s;
                cfg.es_sigma_critic = s;
            }
            "--episode-ticks" => {
                i += 1;
                cfg.episode_ticks = args[i].parse().unwrap_or(cfg.episode_ticks);
            }
            "--checkpoint-interval" => {
                i += 1;
                cfg.checkpoint_interval = args[i].parse().unwrap_or(cfg.checkpoint_interval);
            }
            "--no-self-play" => {
                cfg.self_play = false;
            }
            "--output" | "-o" => {
                i += 1;
                output_path = args[i].clone();
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              MAPPO Training — Rig Self-Play                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Envs: {:>4}  Horizon: {:>4}  Epochs: {:>2}  ES pop: {:>3}       ║",
        cfg.n_envs, cfg.horizon, cfg.epochs, cfg.es_pop_size);
    println!("║ Gens: {:>4}  Gamma: {:.3}  Lambda: {:.3}  Clip: {:.2}       ║",
        cfg.max_generations, cfg.gamma, cfg.lambda, cfg.clip_epsilon);
    println!("║ LR actor: {:.4}  LR critic: {:.4}  Entropy: {:.4}      ║",
        cfg.lr_actor, cfg.lr_critic, cfg.entropy_coef);
    println!("║ ES sigma actor: {:.4}  critic: {:.4}  WD: {:.4}        ║",
        cfg.es_sigma_actor, cfg.es_sigma_critic, cfg.weight_decay_actor);
    println!("║ Seed: {:>10}  Self-play: {:>5}  Ep ticks: {:>5}      ║",
        cfg.seed, cfg.self_play, cfg.episode_ticks);
    println!("║ Rollout shape: {} envs × {} steps × 4 agents = {} samples ║",
        cfg.n_envs, cfg.horizon, cfg.n_envs * cfg.horizon * 4);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();
    let result = train_mappo(cfg.clone());
    let elapsed = start.elapsed();

    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("Training complete in {:.1}s ({} generations)", elapsed.as_secs_f64(), cfg.max_generations);
    println!("══════════════════════════════════════════════════════════════");
    println!();

    // Print per-generation stats.
    println!("{:<5} {:>10} {:>10} {:>10} {:>10} {:>8} {:>5}",
        "Gen", "Reward", "VLoss", "PLoss", "Entropy", "WinRate", "Pop");
    println!("{}", "-".repeat(65));
    for r in &result.history {
        println!("{:<5} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>8.3} {:>5}",
            r.gen, r.mean_reward, r.mean_value_loss, r.mean_policy_loss,
            r.mean_entropy, r.win_rate, r.population_size);
    }
    println!();

    // Summary.
    if let (Some(first), Some(last)) = (result.history.first(), result.history.last()) {
        let reward_delta = last.mean_reward - first.mean_reward;
        println!("Reward delta (first→last): {:+.4}", reward_delta);
        println!("Final win rate: {:.3}", last.win_rate);
        println!("Final entropy: {:.4}", last.mean_entropy);
        println!("Final value loss: {:.4}", last.mean_value_loss);
        println!("Population size: {}", last.population_size);
    }

    // Save weights.
    let mean_reward = result.history.last().map(|r| r.mean_reward).unwrap_or(0.0);
    let json = mappo_weights_to_json(&result.actor_weights, &cfg, mean_reward);
    match fs::write(&output_path, &json) {
        Ok(()) => println!("\nWeights saved to: {output_path} ({} bytes)", json.len()),
        Err(e) => eprintln!("\nFailed to save weights: {e}"),
    }

    println!("\nActor params: {}  Critic params: {}", result.actor_weights.len(), result.critic_weights.len());
    println!("Done.");
}

fn print_help() {
    println!("train_mappo — MAPPO training loop for Rig self-play");
    println!();
    println!("USAGE: cargo run --release --bin train_mappo [-- OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --gens N              Number of training generations (default: 100)");
    println!("  --envs N              Parallel environments (default: 16)");
    println!("  --horizon N           Steps per rollout (default: 256)");
    println!("  --epochs N            PPO epochs per rollout (default: 4)");
    println!("  --minibatch N         Minibatch size (default: 512)");
    println!("  --gamma F             GAE discount (default: 0.99)");
    println!("  --lambda F            GAE lambda (default: 0.95)");
    println!("  --clip F              PPO clip epsilon (default: 0.2)");
    println!("  --lr-actor F          Actor learning rate (default: 0.03)");
    println!("  --lr-critic F         Critic learning rate (default: 0.05)");
    println!("  --entropy-coef F      Entropy bonus weight (default: 0.01)");
    println!("  --value-coef F        Value loss weight (default: 0.5)");
    println!("  --seed N              Master seed (default: 699280)");
    println!("  --es-pop N            ES population size (default: 20)");
    println!("  --es-sigma F          ES noise std (default: 0.05)");
    println!("  --episode-ticks N     Episode length in ticks (default: 4000)");
    println!("  --checkpoint-interval N  Population checkpoint interval (default: 10)");
    println!("  --no-self-play        Disable self-play population");
    println!("  --output PATH         Output JSON path (default: policy-mappo.json)");
    println!("  --help                Show this help");
}

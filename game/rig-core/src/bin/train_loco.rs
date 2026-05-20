//! Binary to train the locomotion policy.
//!
//! Usage: cargo run --release --bin train_loco [-- --gens N --pop N --out path.json]

use rig_core::rl::loco::{loco_weights_to_json, LOCO_PARAM_W};
use rig_core::rl::loco_train::{train_loco, LocoTrainConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = LocoTrainConfig::default();
    let mut out_path = "loco_weights.json".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gens" => {
                i += 1;
                cfg.max_generations = args[i].parse().expect("invalid --gens");
            }
            "--pop" => {
                i += 1;
                cfg.pop_size = args[i].parse().expect("invalid --pop");
            }
            "--sigma" => {
                i += 1;
                cfg.sigma = args[i].parse().expect("invalid --sigma");
            }
            "--lr" => {
                i += 1;
                cfg.lr = args[i].parse().expect("invalid --lr");
            }
            "--out" => {
                i += 1;
                out_path = args[i].clone();
            }
            "--seed" => {
                i += 1;
                cfg.seed = args[i].parse().expect("invalid --seed");
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("=== LOCOMOTION POLICY TRAINER ===");
    eprintln!("  params: {}", LOCO_PARAM_W);
    eprintln!("  pop_size: {}", cfg.pop_size);
    eprintln!("  generations: {}", cfg.max_generations);
    eprintln!("  sigma: {}", cfg.sigma);
    eprintln!("  lr: {}", cfg.lr);
    eprintln!("  curriculum stages: {}", cfg.curriculum.len());
    eprintln!();

    let result = train_loco(cfg);

    eprintln!();
    eprintln!("=== TRAINING COMPLETE ===");
    if let Some(last) = result.history.last() {
        eprintln!("  final mean_fitness: {:.2}", last.mean_fitness);
        eprintln!("  final arrival_rate: {:.0}%", last.arrival_rate * 100.0);
    }

    let json = loco_weights_to_json(&result.policy);
    std::fs::write(&out_path, &json).expect("failed to write weights");
    eprintln!("  weights saved to: {}", out_path);
}

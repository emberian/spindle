//! Skill Drill CLI — isolated AI behavior benchmarks.
//!
//! Usage:
//!   cargo run --release --bin skill_drill
//!   cargo run --release --bin skill_drill -- --drill catch
//!   cargo run --release --bin skill_drill -- --drill pursuit --verbose

use rig_core::drill;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let filter: Option<&str> = args
        .iter()
        .position(|a| a == "--drill" || a == "-d")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    println!("=== RIG SKILL DRILL ===\n");

    if let Some(name) = filter {
        match drill::run_one(name, verbose) {
            Some(r) => print_result(&r),
            None => {
                eprintln!("Unknown drill: '{}'. Available: catch, pursuit, throw, nav, chain", name);
                std::process::exit(1);
            }
        }
    } else {
        let suite = drill::run_all(verbose);
        for r in &suite.results {
            print_result(r);
        }
        println!("\n--- SUMMARY ---");
        println!(
            "  Overall: {}/{} ({:.0}%)",
            suite.results.iter().map(|r| r.successes).sum::<u32>(),
            suite.results.iter().map(|r| r.trials).sum::<u32>(),
            suite.overall_rate() * 100.0,
        );
    }
}

fn print_result(r: &drill::DrillResult) {
    println!(
        "  {:<20} {}/{} ({:.0}%)  avg_t={:.0} ({:.2}s)",
        r.name,
        r.successes,
        r.trials,
        r.rate() * 100.0,
        r.avg_ticks,
        r.avg_ticks / 240.0,
    );
    for d in &r.details {
        println!("    {}", d);
    }
}

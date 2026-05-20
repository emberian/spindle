//! Curriculum Learning Gate System
//!
//! Defines progression stages, evaluates readiness via drill pass rates and
//! match composite scores, and provides orchestration logic for RL training.
//! Each stage gates on specific drill benchmarks; agents graduate when ALL
//! gates for a stage are met, unlocking training scenarios at the next level.
//!
//! NATIVE-ONLY (uses drill + skill_eval infrastructure, gated identically).

use crate::drill;
use crate::skill_eval::{eval_skill, Composite};

// ── Stage Enum ──────────────────────────────────────────────────────────────

/// Training curriculum stages, ordered from basic to advanced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Fundamentals,  // catch, throw, basic movement
    Mechanics,     // pursuit, grapple chains, gate scoring
    Tactics,       // pressure-release, defense, positioning
    Coordination,  // multi-pass, team spacing, role rotation
    Integration,   // full match play, self-play (terminal)
}

impl Stage {
    pub const ALL: &'static [Stage] = &[
        Stage::Fundamentals,
        Stage::Mechanics,
        Stage::Tactics,
        Stage::Coordination,
        Stage::Integration,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Stage::Fundamentals => "Fundamentals",
            Stage::Mechanics => "Mechanics",
            Stage::Tactics => "Tactics",
            Stage::Coordination => "Coordination",
            Stage::Integration => "Integration",
        }
    }

    /// The next stage, or None if this is Integration.
    pub fn next(&self) -> Option<Stage> {
        match self {
            Stage::Fundamentals => Some(Stage::Mechanics),
            Stage::Mechanics => Some(Stage::Tactics),
            Stage::Tactics => Some(Stage::Coordination),
            Stage::Coordination => Some(Stage::Integration),
            Stage::Integration => None,
        }
    }
}

// ── Gate Structures ─────────────────────────────────────────────────────────

/// A single drill that must meet a minimum pass rate to graduate.
#[derive(Clone, Debug)]
pub struct DrillGate {
    pub drill_name: String,
    pub min_rate: f64,
}

/// A match composite score gate (via skill_eval).
#[derive(Clone, Debug)]
pub struct MatchGate {
    pub min_composite: f64,
}

/// All requirements for a single stage.
#[derive(Clone, Debug)]
pub struct StageRequirement {
    pub stage: Stage,
    pub drill_gates: Vec<DrillGate>,
    pub match_gate: Option<MatchGate>,
}

// ── Requirement Definitions ─────────────────────────────────────────────────

fn gate(name: &str, rate: f64) -> DrillGate {
    DrillGate { drill_name: name.to_string(), min_rate: rate }
}

pub fn requirements() -> Vec<StageRequirement> {
    vec![
        StageRequirement {
            stage: Stage::Fundamentals,
            drill_gates: vec![
                gate("static_catch", 0.90),
                gate("short_pass", 0.90),
                gate("swing_distance", 0.80),
                gate("momentum_chain", 0.80),
                gate("hold_and_advance", 0.60),
            ],
            match_gate: None,
        },
        StageRequirement {
            stage: Stage::Mechanics,
            drill_gates: vec![
                gate("pursuit", 0.60),
                gate("catch_angles", 0.75),
                gate("pursuit_reacquire", 0.70),
                gate("gate_score", 0.60),
                gate("throw_accuracy", 0.50),
                gate("speed_challenge", 0.50),
            ],
            match_gate: None,
        },
        StageRequirement {
            stage: Stage::Tactics,
            drill_gates: vec![
                gate("outlet_under_pressure", 0.80),
                gate("defensive_intercept", 0.50),
                gate("under_pressure", 0.40),
                gate("support_positioning", 0.50),
                gate("loose_ball_race", 0.40),
                gate("deny_advance", 0.30),
            ],
            match_gate: None,
        },
        StageRequirement {
            stage: Stage::Coordination,
            drill_gates: vec![
                gate("two_pass", 0.50),
                gate("spacing_maintenance", 0.80),
                gate("rotation_after_catch", 0.50),
                gate("advance_under_defense", 0.50),
                gate("fast_break", 0.40),
            ],
            match_gate: Some(MatchGate { min_composite: 15.0 }),
        },
        StageRequirement {
            stage: Stage::Integration,
            drill_gates: vec![
                gate("possession_chain", 0.50),
            ],
            match_gate: Some(MatchGate { min_composite: 30.0 }),
        },
    ]
}

// ── Evaluation ──────────────────────────────────────────────────────────────

/// Result of evaluating a single stage's gates.
#[derive(Clone, Debug)]
pub struct StageEvaluation {
    pub stage: Stage,
    pub passed: bool,
    /// (drill_name, actual_rate, required_rate, passed)
    pub drill_results: Vec<(String, f64, f64, bool)>,
    /// (actual_composite, required_composite, passed)
    pub match_result: Option<(f64, f64, bool)>,
}

/// Evaluate whether all gates for a given stage are met.
/// Runs the relevant drills (non-verbose) and optionally a match eval.
pub fn evaluate_stage(stage: &Stage) -> StageEvaluation {
    let reqs = requirements();
    let req = reqs.iter().find(|r| r.stage == *stage).unwrap();

    let mut drill_results = Vec::new();
    let mut all_passed = true;

    for g in &req.drill_gates {
        let result = drill::run_one(&g.drill_name, false);
        let actual = result.map(|r| r.rate()).unwrap_or(0.0);
        let passed = actual >= g.min_rate;
        if !passed {
            all_passed = false;
        }
        drill_results.push((g.drill_name.clone(), actual, g.min_rate, passed));
    }

    let match_result = if let Some(mg) = &req.match_gate {
        let comp = eval_skill("curriculum", 0, 42, 14400);
        let passed = comp.composite >= mg.min_composite;
        if !passed {
            all_passed = false;
        }
        Some((comp.composite, mg.min_composite, passed))
    } else {
        None
    };

    StageEvaluation {
        stage: *stage,
        passed: all_passed,
        drill_results,
        match_result,
    }
}

// ── Current Stage Detection ─────────────────────────────────────────────────

/// Detect the current curriculum stage. Evaluates from Fundamentals upward;
/// returns the highest stage the agent has graduated INTO (i.e. if Fundamentals
/// passes, the agent is working on Mechanics). Returns the active stage and
/// its evaluation.
pub fn detect_current_stage() -> (Stage, StageEvaluation) {
    let stages = Stage::ALL;
    // Walk up the ladder: if stage[i] passes, the agent has graduated past it
    // and is working on stage[i+1].
    let mut last_eval = evaluate_stage(&stages[0]);
    if !last_eval.passed {
        return (stages[0], last_eval);
    }
    for i in 1..stages.len() {
        let eval = evaluate_stage(&stages[i]);
        if !eval.passed {
            return (stages[i], eval);
        }
        last_eval = eval;
    }
    // All stages pass — still in Integration (it never graduates).
    (Stage::Integration, last_eval)
}

// ── Training Scenario Selection ─────────────────────────────────────────────

/// A training scenario to be sampled during RL.
#[derive(Clone, Debug)]
pub struct TrainingScenario {
    pub name: String,
    pub drill_name: Option<String>,
    pub match_config: Option<MatchConfig>,
    pub weight: f64,
}

/// Configuration for a full match scenario.
#[derive(Clone, Debug)]
pub struct MatchConfig {
    pub h_style: String,
    pub a_style: String,
    pub max_ticks: u64,
}

/// Return the training scenarios appropriate for the given stage.
/// Weights emphasize frontier drills (those near the gate threshold).
pub fn training_scenarios_for_stage(stage: &Stage) -> Vec<TrainingScenario> {
    let reqs = requirements();
    let mut scenarios = Vec::new();

    // Gather drills from THIS stage and all prior stages.
    let stage_idx = Stage::ALL.iter().position(|s| s == stage).unwrap_or(0);

    // Consolidation: drills from prior stages (lower weight).
    for i in 0..stage_idx {
        let req = &reqs[i];
        for g in &req.drill_gates {
            scenarios.push(TrainingScenario {
                name: format!("consolidate:{}", g.drill_name),
                drill_name: Some(g.drill_name.clone()),
                match_config: None,
                weight: 0.5,
            });
        }
    }

    // Frontier: drills at the current stage (high weight).
    let current_req = &reqs[stage_idx];
    // Run drills to determine proximity to gate.
    for g in &current_req.drill_gates {
        let result = drill::run_one(&g.drill_name, false);
        let actual = result.map(|r| r.rate()).unwrap_or(0.0);
        // Weight higher if the drill is near the threshold (improvement zone).
        let gap = g.min_rate - actual;
        let weight = if gap <= 0.0 {
            1.0 // already passing, still train but lower priority
        } else if gap < 0.2 {
            5.0 // close to passing — high priority
        } else {
            3.0 // still far, moderate priority
        };
        scenarios.push(TrainingScenario {
            name: format!("frontier:{}", g.drill_name),
            drill_name: Some(g.drill_name.clone()),
            match_config: None,
            weight,
        });
    }

    // Exposure: a couple drills from the NEXT stage (low weight).
    if let Some(next) = stage.next() {
        let next_idx = Stage::ALL.iter().position(|s| *s == next).unwrap_or(stage_idx + 1);
        if next_idx < reqs.len() {
            let next_req = &reqs[next_idx];
            for g in next_req.drill_gates.iter().take(2) {
                scenarios.push(TrainingScenario {
                    name: format!("exposure:{}", g.drill_name),
                    drill_name: Some(g.drill_name.clone()),
                    match_config: None,
                    weight: 0.3,
                });
            }
        }
    }

    // Match play at Coordination+ stages.
    if stage_idx >= 3 {
        scenarios.push(TrainingScenario {
            name: "match:coordination_vs_coordination".to_string(),
            drill_name: None,
            match_config: Some(MatchConfig {
                h_style: "Coordination".to_string(),
                a_style: "Coordination".to_string(),
                max_ticks: 14400,
            }),
            weight: 2.0,
        });
        scenarios.push(TrainingScenario {
            name: "match:coordination_vs_aggro".to_string(),
            drill_name: None,
            match_config: Some(MatchConfig {
                h_style: "Coordination".to_string(),
                a_style: "Aggressive".to_string(),
                max_ticks: 14400,
            }),
            weight: 1.5,
        });
    }

    scenarios
}

// ── Report / Print ──────────────────────────────────────────────────────────

/// Run all stage evaluations and print a curriculum status report.
pub fn print_curriculum_status() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              CURRICULUM STATUS REPORT                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut current_found = false;
    let mut current_stage = Stage::Fundamentals;

    for &stage in Stage::ALL {
        let eval = evaluate_stage(&stage);
        let status = if eval.passed { "PASSED" } else { "ACTIVE" };
        let marker = if !eval.passed && !current_found {
            current_found = true;
            current_stage = stage;
            " ← CURRENT"
        } else {
            ""
        };

        println!("── {} [{}]{} ──", stage.name(), status, marker);

        for (name, actual, required, passed) in &eval.drill_results {
            let icon = if *passed { "+" } else { "-" };
            println!(
                "  [{}] {:<25} {:.0}% / {:.0}% required",
                icon,
                name,
                actual * 100.0,
                required * 100.0
            );
        }

        if let Some((actual, required, passed)) = &eval.match_result {
            let icon = if *passed { "+" } else { "-" };
            println!(
                "  [{}] {:<25} {:.1} / {:.1} required",
                icon, "match_composite", actual, required
            );
        }

        println!();
    }

    if !current_found {
        current_stage = Stage::Integration;
        println!("All stages passed — operating in Integration (continuous improvement).");
    }

    // Recommended focus.
    println!("── RECOMMENDED FOCUS ──");
    let eval = evaluate_stage(&current_stage);
    let mut focus: Vec<_> = eval
        .drill_results
        .iter()
        .filter(|(_, _, _, passed)| !passed)
        .collect();
    // Sort by gap (smallest gap first = closest to passing).
    focus.sort_by(|a, b| {
        let gap_a = a.2 - a.1;
        let gap_b = b.2 - b.1;
        gap_a.partial_cmp(&gap_b).unwrap_or(std::cmp::Ordering::Equal)
    });
    if focus.is_empty() {
        println!("  All drill gates met for current stage.");
    } else {
        for (name, actual, required, _) in focus.iter().take(3) {
            let gap = required - actual;
            println!(
                "  -> {}: need +{:.0}pp ({:.0}% -> {:.0}%)",
                name,
                gap * 100.0,
                actual * 100.0,
                required * 100.0
            );
        }
    }
    if let Some((actual, required, passed)) = &eval.match_result {
        if !passed {
            println!(
                "  -> match composite: need +{:.1} ({:.1} -> {:.1})",
                required - actual,
                actual,
                required
            );
        }
    }
    println!();
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn curriculum_status() {
        print_curriculum_status();
    }
}

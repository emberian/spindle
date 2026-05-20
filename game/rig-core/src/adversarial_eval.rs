//! Adversarial evaluation — pit the real AI against DEGENERATE strategies
//! (trivially bad input generators) to expose exploits and stalemate traps.
//!
//! Each degenerate strategy is a pure function from (players, bell, tick) to
//! a Vec<PlayerInput> for the degenerate (Away) team. If the production AI
//! cannot dominate these, that is a measurable weakness.
//!
//! NATIVE-ONLY (uses rayon + the skill_eval conv bridges, same gate as
//! skill_eval/ga/gym).

use crate::ai::decision_types::TeamConfig;
use crate::ai::plan_bridge::set_planner_class;
use crate::ai::profile::{style_to_profile, Difficulty};
use crate::ai::system::AiSystem;
use crate::ai::types::{self as ai, BellState, InputFrame, PlayerInput, PlayerSim};
use crate::match_sm::{MatchPhase, MatchStateMachine};
use crate::math::Vec3;
use crate::scoring;
use crate::sim_world::SimWorld;
use crate::skill_eval::conv;

// ── Roster (mirrors skill_eval) ─────────────────────────────────────────────

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

// ── Degenerate Strategies ───────────────────────────────────────────────────

/// All players emit zero inputs every tick. No movement, no throws.
/// The AI should score freely against stationary opponents.
pub fn strategy_do_nothing(players: &[PlayerSim], _bell: &BellState, _tick: u64) -> Vec<PlayerInput> {
    players
        .iter()
        .filter(|p| p.team == ai::TeamSide::Away)
        .map(|p| PlayerInput::idle(&p.id))
        .collect()
}

/// Whenever a player holds the ball, immediately throw it backward (toward
/// their OWN ring at -x for Away). All players otherwise idle.
/// Tests if the AI exploits bad turnovers.
pub fn strategy_fling_backward(players: &[PlayerSim], bell: &BellState, _tick: u64) -> Vec<PlayerInput> {
    players
        .iter()
        .filter(|p| p.team == ai::TeamSide::Away)
        .map(|p| {
            let mut inp = PlayerInput::idle(&p.id);
            // Away attacks toward -x, so "backward" is +x (their own ring).
            if bell.held_by.as_deref() == Some(&p.id) {
                inp.throw_charge = 1.0;
                inp.throw_released = true;
                inp.throw_spin = 0.0;
                // Aim backward (+x = toward own ring for Away)
                inp.aim = Vec3::new(1.0, 0.0, 0.0);
            }
            inp
        })
        .collect()
}

/// Every player fires grapple toward the ball's position and reels in.
/// No throwing, no strategy. Tests if concentrated ball-hawking can
/// accidentally win.
pub fn strategy_all_chase_ball(players: &[PlayerSim], bell: &BellState, _tick: u64) -> Vec<PlayerInput> {
    players
        .iter()
        .filter(|p| p.team == ai::TeamSide::Away)
        .map(|p| {
            let mut inp = PlayerInput::idle(&p.id);
            inp.fire_line_at = Some(bell.p);
            inp.reel = -1; // reel in
            inp.catch_intent = true;
            inp
        })
        .collect()
}

/// When a player catches, they NEVER throw (hold forever). Others chase
/// the ball. Tests if the AI can strip/contest.
pub fn strategy_hold_forever(players: &[PlayerSim], bell: &BellState, _tick: u64) -> Vec<PlayerInput> {
    players
        .iter()
        .filter(|p| p.team == ai::TeamSide::Away)
        .map(|p| {
            let mut inp = PlayerInput::idle(&p.id);
            if bell.held_by.as_deref() == Some(&p.id) {
                // Hold: do nothing (never throw)
            } else {
                // Chase the ball
                inp.fire_line_at = Some(bell.p);
                inp.reel = -1;
                inp.catch_intent = true;
            }
            inp
        })
        .collect()
}

/// Every 120 ticks, the ball carrier throws in a random (seeded) direction
/// at max power. Tests if chaos creates problems.
pub fn strategy_random_fling(players: &[PlayerSim], bell: &BellState, tick: u64) -> Vec<PlayerInput> {
    players
        .iter()
        .filter(|p| p.team == ai::TeamSide::Away)
        .map(|p| {
            let mut inp = PlayerInput::idle(&p.id);
            if bell.held_by.as_deref() == Some(&p.id) {
                if tick % 120 == 0 {
                    // Simple LCG seeded by tick
                    let mut s = (tick as u32).wrapping_mul(1103515245).wrapping_add(12345);
                    let x_val = ((s >> 16) & 0x7fff) as f64 / 32768.0 * 2.0 - 1.0;
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    let y_val = ((s >> 16) & 0x7fff) as f64 / 32768.0 * 2.0 - 1.0;
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    let z_val = ((s >> 16) & 0x7fff) as f64 / 32768.0 * 2.0 - 1.0;
                    inp.aim = Vec3::new(x_val, y_val, z_val);
                    inp.throw_charge = 1.0;
                    inp.throw_released = true;
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    inp.throw_spin = ((s >> 16) & 0x7fff) as f64 / 32768.0 * 2.0 - 1.0;
                }
            } else {
                // Chase the ball
                inp.fire_line_at = Some(bell.p);
                inp.reel = -1;
                inp.catch_intent = true;
            }
            inp
        })
        .collect()
}

// ── Result / Verdict ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AdversarialResult {
    pub strategy: String,
    pub home_score: f64,
    pub away_score: f64,
    pub home_possession_frac: f64,
    pub home_gate_clears: u32,
    pub verdict: &'static str,
}

fn compute_verdict(home_score: f64, away_score: f64, home_gate_clears: u32) -> &'static str {
    if away_score >= home_score {
        "EXPLOIT"
    } else if home_score > away_score && home_gate_clears > 0 {
        "PASS"
    } else if home_score + away_score == 0.0 && home_gate_clears == 0 {
        "STALEMATE"
    } else {
        // home_score > away_score but no gate clears — partial pass
        "PASS"
    }
}

// ── Match Runner ────────────────────────────────────────────────────────────

/// Run one adversarial match: HOME = real AI (Pro, planner_class=0),
/// AWAY = degenerate strategy function.
pub fn run_adversarial_match(
    strategy_name: &str,
    strategy_fn: fn(&[PlayerSim], &BellState, u64) -> Vec<PlayerInput>,
    seed: u32,
    max_ticks: u64,
) -> AdversarialResult {
    set_planner_class(0); // MPC baseline (Pro difficulty)

    let mut sim = SimWorld::new(seed);
    let rs = roster();
    for (k, r) in rs.iter().enumerate() {
        let ang = (k as f64 / rs.len() as f64) * std::f64::consts::PI * 2.0;
        let sw_team = match r.team {
            ai::TeamSide::Home => crate::sim_world::TeamSide::Home,
            ai::TeamSide::Away => crate::sim_world::TeamSide::Away,
        };
        let sw_role = match r.role {
            ai::RiggerRole::Anchor => crate::sim_world::RiggerRole::Anchor,
            ai::RiggerRole::Spinner => crate::sim_world::RiggerRole::Spinner,
            ai::RiggerRole::Faithwing => crate::sim_world::RiggerRole::Faithwing,
            ai::RiggerRole::Freewing => crate::sim_world::RiggerRole::Freewing,
            ai::RiggerRole::Reach => crate::sim_world::RiggerRole::Reach,
        };
        sim.add_player(
            r.id,
            sw_team,
            sw_role,
            Vec3::new(r.x, ang.cos() * 8.0, ang.sin() * 8.0),
        );
    }
    sim.set_bell_held("H1");

    let mut mat = MatchStateMachine::new(scoring::RingEnd::PlusX, scoring::TeamSide::Home);
    let snap0 = sim.snapshot();
    let s0 = conv::ai_to_scoring(&conv::snap_to_ai(&snap0));
    mat.consume(
        &[scoring::SimEvent::FoulGarrote { by: "__start__".to_string() }],
        &s0,
    );

    // Only Home gets an AI config — Away is driven by the degenerate fn.
    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let mut home_held_ticks: u64 = 0;
    let mut total_ticks: u64 = 0;
    let mut home_gate_clears: u32 = 0;
    let mut prev_gate_ord: Option<i32> = None;

    for tick_idx in 0..max_ticks {
        if mat.state().winner.is_some() {
            break;
        }
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = conv::msm_to_ai(mat.state());

        // Track home possession
        if let Some(ref hb) = snap.bell.held_by {
            let is_home = rs.iter().any(|r| r.id == hb && r.team == ai::TeamSide::Home);
            if is_home {
                home_held_ticks += 1;
            }
        }

        // HOME inputs from AiSystem
        let home_frame = aisys.tick(&ai_sim, &ai_match, &cfgs, seed);

        // AWAY inputs from the degenerate strategy
        let away_players: Vec<&PlayerSim> = ai_sim
            .players
            .iter()
            .filter(|p| p.team == ai::TeamSide::Away)
            .collect();
        let _ = away_players; // just for clarity; strategy_fn filters internally
        let away_inputs = strategy_fn(&ai_sim.players, &ai_sim.bell, tick_idx);

        // Merge into one InputFrame
        let mut all_inputs = home_frame.players;
        all_inputs.extend(away_inputs);
        let combined_frame = InputFrame {
            tick: ai_sim.tick,
            players: all_inputs,
        };

        // Step the sim
        let sw_frame = conv::frame_ai_to_sw(&combined_frame);
        let evs = sim.step(&sw_frame, H);

        // Feed events to match state machine
        let scoring_evs: Vec<scoring::SimEvent> =
            evs.iter().map(conv::ev_sw_to_scoring).collect();
        let post = conv::ai_to_scoring(&conv::snap_to_ai(&sim.snapshot()));
        mat.consume(&scoring_evs, &post);

        // Track gate clears (home only — home attacks toward +x)
        let g_ord = match mat.state().cast.gate {
            crate::match_sm::Gate::First => 0,
            crate::match_sm::Gate::Deep => 1,
            crate::match_sm::Gate::Mouth => 2,
        };
        if let Some(pg) = prev_gate_ord {
            if g_ord > pg {
                home_gate_clears += 1;
            }
        }
        prev_gate_ord = Some(g_ord);

        // Re-arm dead ball
        if mat.state().winner.is_none() && mat.state().phase != MatchPhase::Live {
            let poss = mat.state().possession;
            if let Some(r) = rs
                .iter()
                .find(|r| conv::team_ai_to_scoring(r.team) == poss)
            {
                sim.set_bell_held(r.id);
            }
            mat.resume_live();
        }

        total_ticks += 1;
    }

    let home_score = mat.state().score_home as f64;
    let away_score = mat.state().score_away as f64;
    let home_possession_frac = if total_ticks > 0 {
        home_held_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };
    let verdict = compute_verdict(home_score, away_score, home_gate_clears);

    AdversarialResult {
        strategy: strategy_name.to_string(),
        home_score,
        away_score,
        home_possession_frac,
        home_gate_clears,
        verdict,
    }
}

// ── Run All ─────────────────────────────────────────────────────────────────

/// Run all 5 degenerate strategies and return results.
pub fn run_all_adversarial(seed: u32, max_ticks: u64) -> Vec<AdversarialResult> {
    let strategies: Vec<(&str, fn(&[PlayerSim], &BellState, u64) -> Vec<PlayerInput>)> = vec![
        ("do_nothing", strategy_do_nothing as fn(&[PlayerSim], &BellState, u64) -> Vec<PlayerInput>),
        ("fling_backward", strategy_fling_backward),
        ("all_chase_ball", strategy_all_chase_ball),
        ("hold_forever", strategy_hold_forever),
        ("random_fling", strategy_random_fling),
    ];

    strategies
        .into_iter()
        .map(|(name, f)| run_adversarial_match(name, f, seed, max_ticks))
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "adversarial eval suite; run with --ignored --nocapture"]
    fn run_adversarial_eval_suite() {
        let results = run_all_adversarial(1234, 8000);
        println!("\n=== ADVERSARIAL EVALUATION RESULTS (8000 ticks = {:.1}s) ===\n", 8000.0 / 240.0);
        println!("{:<18} {:>6} {:>6} {:>8} {:>6} {:>10}",
            "STRATEGY", "H_SCR", "A_SCR", "H_POSS", "GATES", "VERDICT");
        println!("{}", "-".repeat(60));
        for r in &results {
            println!("{:<18} {:>6.0} {:>6.0} {:>7.1}% {:>6} {:>10}",
                r.strategy,
                r.home_score,
                r.away_score,
                r.home_possession_frac * 100.0,
                r.home_gate_clears,
                r.verdict,
            );
        }
        println!();
        let exploits: Vec<&AdversarialResult> = results.iter().filter(|r| r.verdict == "EXPLOIT").collect();
        let stalemates: Vec<&AdversarialResult> = results.iter().filter(|r| r.verdict == "STALEMATE").collect();
        if !exploits.is_empty() {
            println!("WARNING: {} strategy/strategies EXPLOITED the AI:", exploits.len());
            for e in &exploits {
                println!("  - {} (away scored {} vs home {})", e.strategy, e.away_score, e.home_score);
            }
        }
        if !stalemates.is_empty() {
            println!("WARNING: {} strategy/strategies caused STALEMATE:", stalemates.len());
            for s in &stalemates {
                println!("  - {}", s.strategy);
            }
        }
        if exploits.is_empty() && stalemates.is_empty() {
            println!("ALL PASS: AI dominates all degenerate strategies.");
        }
    }
}

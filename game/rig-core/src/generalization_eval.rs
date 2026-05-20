//! Robustness / generalization evaluation — tests whether the AI works
//! across DIVERSE conditions (styles, skill mismatches, novel formations),
//! not just the 3 default matchups it was tuned against.
//!
//! Key insight: a strong AI should have LOW variance across style matchups,
//! dominate weaker opponents proportionally, and handle arbitrary starting
//! positions without catastrophic degradation.
//!
//! NATIVE-ONLY (rayon + skill_eval infrastructure).

use crate::ai::decision_types::TeamConfig;
use crate::ai::plan_bridge::set_planner_class;
use crate::ai::profile::{style_to_profile, Difficulty};
use crate::ai::system::AiSystem;
use crate::ai::types as ai;
use crate::match_sm::{MatchPhase, MatchStateMachine};
use crate::math::Vec3;
use crate::scoring;
use crate::sim_world::SimWorld;
use crate::skill_eval::conv;
use crate::skill_eval::{assemble_composite, roster, run_scored_match, Composite};
use rayon::prelude::*;
use std::collections::HashMap;

const H: f64 = 1.0 / 240.0;

// ── Cross-Style matchups ────────────────────────────────────────────────────
// All meaningful style pairings including mirrors and cross-family.
const CROSS_STYLE_MATCHUPS: [(&str, &str, &str, &str); 8] = [
    ("fall-dynasty", "big-slow", "rise-power", "small-fast"),
    ("fall-dynasty", "big-slow", "rise-chaos", "small-fast"),
    ("fall-grind", "big-slow", "rise-power", "small-fast"),
    ("fall-grind", "big-slow", "rise-chaos", "small-fast"),
    ("fall-tempo", "big-slow", "fall-dynasty", "big-slow"),
    ("rise-power", "small-fast", "rise-chaos", "small-fast"),
    ("fall-dynasty", "big-slow", "fall-dynasty", "big-slow"), // mirror
    ("rise-power", "small-fast", "rise-power", "small-fast"), // mirror
];

// ── 1. Cross-Style Evaluation ───────────────────────────────────────────────

/// Run the AI against ALL style combinations and return a Composite for each.
/// High variance across matchups indicates the AI is style-dependent.
pub fn eval_cross_style(planner_class: i32, seed: u32, max_ticks: u64) -> Vec<Composite> {
    let results: Vec<Composite> = (0..CROSS_STYLE_MATCHUPS.len())
        .into_par_iter()
        .map(|i| {
            let (hs, hc, as_, ac) = CROSS_STYLE_MATCHUPS[i];
            set_planner_class(planner_class);
            let (sub, raw) = run_scored_match(hs, hc, as_, ac, seed, max_ticks);
            let label = format!("{} vs {}", hs, as_);
            let results = vec![(sub, raw)];
            assemble_composite(&label, results)
        })
        .collect();

    // Print variance summary.
    let composites: Vec<f64> = results.iter().map(|c| c.composite).collect();
    let n = composites.len() as f64;
    let mean = composites.iter().sum::<f64>() / n;
    let variance = composites.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    println!("=== Cross-Style Evaluation ===");
    for r in &results {
        println!("  {:<35} composite={:.1}", r.label, r.composite);
    }
    println!("  mean={:.1}  std_dev={:.2}  (high = style-dependent)", mean, std_dev);
    println!();

    results
}

// ── 2. Asymmetric Skill Evaluation ──────────────────────────────────────────

/// Result of a single asymmetric match.
#[derive(Clone, Debug)]
pub struct MatchResult {
    pub label: String,
    pub home_score: f64,
    pub away_score: f64,
    pub home_possession: f64,
    pub away_possession: f64,
    pub home_gates: f64,
    pub away_gates: f64,
}

/// Report from the asymmetric evaluation suite.
#[derive(Clone, Debug)]
pub struct AsymmetricReport {
    pub pro_vs_rookie: MatchResult,
    pub pro_vs_standard: MatchResult,
    pub four_vs_three: MatchResult,
}

/// Run a full match with custom difficulties for each side.
/// Returns (home_score, away_score, home_possession_frac, away_possession_frac,
/// home_gate_clears, away_gate_clears).
fn run_asymmetric_match(
    h_style: &str,
    h_cyl: &str,
    h_diff: Difficulty,
    a_style: &str,
    a_cyl: &str,
    a_diff: Difficulty,
    seed: u32,
    max_ticks: u64,
    skip_a4: bool,
) -> (f64, f64, f64, f64, f64, f64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(seed);
    let rs = roster();
    let team_of: HashMap<String, ai::TeamSide> =
        rs.iter().map(|r| (r.id.to_string(), r.team)).collect();

    for (k, r) in rs.iter().enumerate() {
        if skip_a4 && r.id == "A4" {
            continue;
        }
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

    let cfgs = vec![
        TeamConfig {
            side: ai::TeamSide::Home,
            profile: style_to_profile(h_style, h_cyl),
            difficulty: h_diff,
        },
        TeamConfig {
            side: ai::TeamSide::Away,
            profile: style_to_profile(a_style, a_cyl),
            difficulty: a_diff,
        },
    ];
    let mut aisys = AiSystem::new();

    let mut home_held: f64 = 0.0;
    let mut away_held: f64 = 0.0;
    let mut home_gates: f64 = 0.0;
    let mut away_gates: f64 = 0.0;
    let mut prev_gate_ord: Option<i32> = None;

    for _ in 0..max_ticks {
        if mat.state().winner.is_some() {
            break;
        }
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = conv::msm_to_ai(mat.state());

        // Track possession by team.
        if let Some(hb) = &snap.bell.held_by {
            match team_of.get(hb) {
                Some(ai::TeamSide::Home) => home_held += 1.0,
                Some(ai::TeamSide::Away) => away_held += 1.0,
                _ => {}
            }
        }

        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, seed);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        let evs = sim.step(&sw_frame, H);
        let scoring_evs: Vec<scoring::SimEvent> =
            evs.iter().map(conv::ev_sw_to_scoring).collect();
        let post = conv::ai_to_scoring(&conv::snap_to_ai(&sim.snapshot()));
        mat.consume(&scoring_evs, &post);

        // Track gate clears per possessing team.
        let g_ord = match mat.state().cast.gate {
            crate::match_sm::Gate::First => 0,
            crate::match_sm::Gate::Deep => 1,
            crate::match_sm::Gate::Mouth => 2,
        };
        if let Some(pg) = prev_gate_ord {
            if g_ord > pg {
                match mat.state().possession {
                    scoring::TeamSide::Home => home_gates += 1.0,
                    scoring::TeamSide::Away => away_gates += 1.0,
                }
            }
        }
        prev_gate_ord = Some(g_ord);

        // Re-arm dead ball.
        if mat.state().winner.is_none() && mat.state().phase != MatchPhase::Live {
            let poss = mat.state().possession;
            let active_roster: Vec<&crate::skill_eval::RosterEntry> = rs
                .iter()
                .filter(|r| !(skip_a4 && r.id == "A4"))
                .collect();
            if let Some(r) = active_roster
                .iter()
                .find(|r| conv::team_ai_to_scoring(r.team) == poss)
            {
                sim.set_bell_held(r.id);
            }
            mat.resume_live();
        }
    }

    let total = (home_held + away_held).max(1.0);
    (
        mat.state().score_home as f64,
        mat.state().score_away as f64,
        home_held / total,
        away_held / total,
        home_gates,
        away_gates,
    )
}

/// Run matches with skill mismatches to test how difficulty asymmetry manifests.
pub fn eval_asymmetric(seed: u32, max_ticks: u64) -> AsymmetricReport {
    // All three scenarios are independent — run in parallel.
    let scenarios: Vec<(&str, Difficulty, Difficulty, bool)> = vec![
        ("Pro vs Rookie", Difficulty::Pro, Difficulty::Rookie, false),
        // "Standard" maps to Legend vs Pro (Legend is the highest tier).
        ("Legend vs Pro", Difficulty::Legend, Difficulty::Pro, false),
        ("4v3 (Pro vs Pro)", Difficulty::Pro, Difficulty::Pro, true),
    ];

    let results: Vec<(f64, f64, f64, f64, f64, f64)> = scenarios
        .par_iter()
        .map(|(_, h_diff, a_diff, skip_a4)| {
            run_asymmetric_match(
                "fall-dynasty",
                "big-slow",
                *h_diff,
                "fall-dynasty",
                "big-slow",
                *a_diff,
                seed,
                max_ticks,
                *skip_a4,
            )
        })
        .collect();

    let make_result = |i: usize, label: &str| -> MatchResult {
        let (hs, as_, hp, ap, hg, ag) = results[i];
        MatchResult {
            label: label.to_string(),
            home_score: hs,
            away_score: as_,
            home_possession: hp,
            away_possession: ap,
            home_gates: hg,
            away_gates: ag,
        }
    };

    let report = AsymmetricReport {
        pro_vs_rookie: make_result(0, "Pro vs Rookie"),
        pro_vs_standard: make_result(1, "Legend vs Pro"),
        four_vs_three: make_result(2, "4v3 (Pro vs Pro)"),
    };

    println!("=== Asymmetric Skill Evaluation ===");
    for mr in [&report.pro_vs_rookie, &report.pro_vs_standard, &report.four_vs_three] {
        println!(
            "  {:<20} score={:.0}-{:.0}  poss={:.1}%-{:.1}%  gates={:.0}-{:.0}",
            mr.label,
            mr.home_score,
            mr.away_score,
            mr.home_possession * 100.0,
            mr.away_possession * 100.0,
            mr.home_gates,
            mr.away_gates,
        );
    }
    println!();

    report
}

// ── 3. Novel Formations Evaluation ──────────────────────────────────────────

/// Formation descriptor: how to place home-team players for this scenario.
struct Formation {
    label: &'static str,
    /// (id_filter, x, y, z) — if id_filter is Some, only apply to that id.
    /// Otherwise apply to all home players in order.
    positions: Vec<(Option<&'static str>, f64, f64, f64)>,
}

/// Run a match where the home team starts in a novel formation.
fn run_formation_match(
    formation: &Formation,
    planner_class: i32,
    seed: u32,
    max_ticks: u64,
) -> Composite {
    set_planner_class(planner_class);
    let mut sim = SimWorld::new(seed);
    let rs = roster();

    // Place players — home side gets novel positions, away gets standard.
    let home_ids: Vec<&str> = rs.iter().filter(|r| r.team == ai::TeamSide::Home).map(|r| r.id).collect();

    for (k, r) in rs.iter().enumerate() {
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

        let pos = if r.team == ai::TeamSide::Home {
            // Find this player's novel position.
            let home_idx = home_ids.iter().position(|&id| id == r.id).unwrap_or(0);
            if let Some(fp) = formation.positions.iter().find(|(id, _, _, _)| {
                id.map_or(false, |fid| fid == r.id)
            }) {
                Vec3::new(fp.1, fp.2, fp.3)
            } else if home_idx < formation.positions.iter().filter(|(id, _, _, _)| id.is_none()).count() {
                // Use positional entries (no id filter) in order.
                let no_id_entries: Vec<&(Option<&str>, f64, f64, f64)> =
                    formation.positions.iter().filter(|(id, _, _, _)| id.is_none()).collect();
                let entry = no_id_entries[home_idx];
                Vec3::new(entry.1, entry.2, entry.3)
            } else {
                // Fallback: standard position.
                let ang = (k as f64 / rs.len() as f64) * std::f64::consts::PI * 2.0;
                Vec3::new(r.x, ang.cos() * 8.0, ang.sin() * 8.0)
            }
        } else {
            // Away team: standard positions.
            let ang = (k as f64 / rs.len() as f64) * std::f64::consts::PI * 2.0;
            Vec3::new(r.x, ang.cos() * 8.0, ang.sin() * 8.0)
        };

        sim.add_player(r.id, sw_team, sw_role, pos);
    }
    sim.set_bell_held("H1");

    let mut mat = MatchStateMachine::new(scoring::RingEnd::PlusX, scoring::TeamSide::Home);
    let snap0 = sim.snapshot();
    let s0 = conv::ai_to_scoring(&conv::snap_to_ai(&snap0));
    mat.consume(
        &[scoring::SimEvent::FoulGarrote { by: "__start__".to_string() }],
        &s0,
    );

    let cfgs = vec![
        TeamConfig {
            side: ai::TeamSide::Home,
            profile: style_to_profile("fall-dynasty", "big-slow"),
            difficulty: Difficulty::Pro,
        },
        TeamConfig {
            side: ai::TeamSide::Away,
            profile: style_to_profile("fall-dynasty", "big-slow"),
            difficulty: Difficulty::Pro,
        },
    ];
    let mut aisys = AiSystem::new();

    // Reuse the skill_eval scoring infrastructure inline (same signals).
    let team_of: HashMap<String, ai::TeamSide> =
        rs.iter().map(|r| (r.id.to_string(), r.team)).collect();
    let (mut passes, mut intercepts, mut gate_clears, mut thrash) = (0.0, 0.0, 0.0, 0.0);
    let (mut streak, mut streak_max, mut held_ticks, mut ticks, mut skin_ticks) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut scores: HashMap<&str, f64> = HashMap::new();
    let mut throws: HashMap<String, f64> = HashMap::new();
    let mut prev_held: Option<String> = None;
    let mut last_thrown_by: Option<String> = None;
    let mut prev_gate_ord: Option<i32> = None;
    let mut prev_fire: HashMap<String, Vec3> = HashMap::new();

    for _ in 0..max_ticks {
        if mat.state().winner.is_some() {
            break;
        }
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = conv::msm_to_ai(mat.state());

        if snap.bell.held_by.is_some() {
            held_ticks += 1.0;
        }
        if (snap.bell.p.y.powi(2) + snap.bell.p.z.powi(2)).sqrt() > 31.0 {
            skin_ticks += 1.0;
        }

        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, seed);
        for inp in &frame.players {
            if inp.throw_released {
                *throws.entry(inp.id.clone()).or_insert(0.0) += 1.0;
            }
            if let Some(f) = inp.fire_line_at {
                if let Some(pf) = prev_fire.get(&inp.id) {
                    if ((f.x - pf.x).powi(2) + (f.y - pf.y).powi(2) + (f.z - pf.z).powi(2))
                        .sqrt()
                        > 8.0
                    {
                        thrash += 1.0;
                    }
                }
                prev_fire.insert(inp.id.clone(), f);
            }
        }

        // Pass / intercept on grip transition.
        let b = &snap.bell;
        if b.held_by.is_some() && prev_held.is_none() {
            if let (Some(lt), Some(hb)) = (&last_thrown_by, &b.held_by) {
                if lt != hb {
                    match (team_of.get(lt), team_of.get(hb)) {
                        (Some(a), Some(c)) if a == c => {
                            passes += 1.0;
                            streak += 1.0;
                            if streak > streak_max {
                                streak_max = streak;
                            }
                        }
                        _ => {
                            intercepts += 1.0;
                            streak = 0.0;
                        }
                    }
                }
            }
        }
        if b.held_by.is_none() {
            if let Some(tb) = &b.thrown_by {
                last_thrown_by = Some(tb.clone());
            }
        } else {
            last_thrown_by = None;
        }
        prev_held = b.held_by.clone();

        let sw_frame = conv::frame_ai_to_sw(&frame);
        let evs = sim.step(&sw_frame, H);
        for e in &evs {
            match e {
                crate::sim_world::SimEvent::BellCaught { .. } => {
                    *scores.entry("caught").or_insert(0.0) += 1.0
                }
                crate::sim_world::SimEvent::BellClatter { .. }
                | crate::sim_world::SimEvent::BellBobble { .. } => {
                    *scores.entry("clatter").or_insert(0.0) += 1.0
                }
                _ => {}
            }
        }
        let scoring_evs: Vec<scoring::SimEvent> =
            evs.iter().map(conv::ev_sw_to_scoring).collect();
        let post = conv::ai_to_scoring(&conv::snap_to_ai(&sim.snapshot()));
        let upd = mat.consume(&scoring_evs, &post);
        if let Some(sr) = &upd.scored {
            let key: &str = match sr.kind {
                scoring::ScoreKind::Fall => "fall",
                scoring::ScoreKind::Rise => "rise",
                scoring::ScoreKind::Loop => "loop",
                scoring::ScoreKind::Curl => "curl",
                scoring::ScoreKind::Ground => "ground",
            };
            *scores.entry(key).or_insert(0.0) += 1.0;
            streak = 0.0;
        } else if upd.turnover.is_some() {
            *scores.entry("turnover").or_insert(0.0) += 1.0;
            streak = 0.0;
        }

        let g_ord = match mat.state().cast.gate {
            crate::match_sm::Gate::First => 0,
            crate::match_sm::Gate::Deep => 1,
            crate::match_sm::Gate::Mouth => 2,
        };
        if let Some(pg) = prev_gate_ord {
            if g_ord > pg {
                gate_clears += 1.0;
            }
        }
        prev_gate_ord = Some(g_ord);

        // Re-arm dead ball.
        if mat.state().winner.is_none() && mat.state().phase != MatchPhase::Live {
            let poss = mat.state().possession;
            if let Some(r) = rs.iter().find(|r| conv::team_ai_to_scoring(r.team) == poss) {
                sim.set_bell_held(r.id);
            }
            mat.resume_live();
        }
        ticks += 1.0;
    }

    // Compute normalized sub-signals (same caps/weights as skill_eval).
    use crate::skill_eval::{CAP, WT};
    fn clamp01(x: f64) -> f64 {
        x.clamp(0.0, 1.0)
    }
    let kinds = ["fall", "rise", "loop", "curl", "ground"];
    fn score_pts(k: scoring::ScoreKind) -> f64 {
        use scoring::ScoreKind::*;
        match k {
            Fall => 2.0,
            Rise => 5.0,
            Loop => 7.0,
            Curl => 5.0,
            Ground => 1.0,
        }
    }
    let score_pts_total: f64 = kinds
        .iter()
        .map(|k| {
            let n = *scores.get(k).unwrap_or(&0.0);
            let kk = match *k {
                "fall" => scoring::ScoreKind::Fall,
                "rise" => scoring::ScoreKind::Rise,
                "loop" => scoring::ScoreKind::Loop,
                "curl" => scoring::ScoreKind::Curl,
                _ => scoring::ScoreKind::Ground,
            };
            score_pts(kk) * n
        })
        .sum();
    let score_events: f64 = kinds.iter().map(|k| *scores.get(k).unwrap_or(&0.0)).sum();
    let total_throws: f64 = throws.values().sum();
    let shot_conv = if total_throws > 0.0 { score_events / total_throws } else { 0.0 };
    let t = ticks.max(1.0);
    let held_frac = held_ticks / t;
    let skin_pct = 100.0 * skin_ticks / t;
    let thrash_k = (thrash / t) * 1000.0;

    let mut sub: HashMap<&str, f64> = HashMap::new();
    sub.insert("prog", clamp01(gate_clears / CAP.gate));
    sub.insert("pass", clamp01(passes / CAP.pass));
    sub.insert("intc", clamp01(intercepts / CAP.intc));
    sub.insert("score", clamp01(score_pts_total / CAP.score));
    sub.insert("poss", clamp01(held_frac / CAP.poss));
    sub.insert("calm", 1.0 - clamp01(thrash_k / CAP.thrash_k));
    sub.insert("field", 1.0 - clamp01(skin_pct / CAP.skin_pct));
    sub.insert("shotConv", clamp01(shot_conv / CAP.shot_conv));
    sub.insert("chain", clamp01(streak_max / CAP.chain));

    let mut raw: HashMap<&str, f64> = HashMap::new();
    raw.insert("gateClears", gate_clears);
    raw.insert("passes", passes);
    raw.insert("intercepts", intercepts);
    raw.insert("scorePts", score_pts_total);
    raw.insert("heldFrac", (held_frac * 1000.0).round() / 1000.0);
    raw.insert("thrashK", thrash_k.round());
    raw.insert("skinPct", skin_pct.round());
    raw.insert("shotConv", (shot_conv * 100.0).round() / 100.0);
    raw.insert("chainMax", streak_max);
    raw.insert("ticks", ticks);

    assemble_composite(formation.label, vec![(sub, raw)])
}

/// Run matches with unusual starting formations. The AI should converge to
/// good play regardless of initial player placement.
pub fn eval_novel_formations(seed: u32, max_ticks: u64) -> Vec<Composite> {
    let formations = vec![
        Formation {
            label: "bunched-center",
            positions: vec![
                (None, 0.0, 8.0, 0.0),
                (None, 0.0, 8.0, 0.0),
                (None, 0.0, 8.0, 0.0),
                (None, 0.0, 8.0, 0.0),
            ],
        },
        Formation {
            label: "all-deep",
            positions: vec![
                (None, -250.0, 8.0, 0.0),
                (None, -250.0, 8.0, 2.0),
                (None, -250.0, 8.0, -2.0),
                (None, -250.0, 8.0, 4.0),
            ],
        },
        Formation {
            label: "inverted",
            // Home players in away positions, away in home positions.
            positions: vec![
                (Some("H1"), 40.0, 8.0, 0.0),
                (Some("H2"), 90.0, 8.0, 0.0),
                (Some("H3"), 20.0, 8.0, 0.0),
                (Some("H4"), 260.0, 8.0, 0.0),
            ],
        },
        Formation {
            label: "wide-spread",
            positions: vec![
                (None, -20.0, 8.0, 0.0),
                (None, -100.0, 8.0, 0.0),
                (None, -200.0, 8.0, 0.0),
                (None, -300.0, 8.0, 0.0),
            ],
        },
    ];

    let planner_class = 0; // MPC default for baseline comparison.
    let results: Vec<Composite> = formations
        .par_iter()
        .map(|f| run_formation_match(f, planner_class, seed, max_ticks))
        .collect();

    println!("=== Novel Formations Evaluation ===");
    for r in &results {
        println!("  {:<20} composite={:.1}", r.label, r.composite);
    }
    println!();

    results
}

// ── 4. Summary Report ───────────────────────────────────────────────────────

/// Full generalization report — aggregates all sub-evaluations.
#[derive(Clone, Debug)]
pub struct GeneralizationReport {
    pub cross_style_results: Vec<Composite>,
    pub cross_style_mean: f64,
    pub cross_style_std_dev: f64,
    pub asymmetric: AsymmetricReport,
    pub exploitation_score: f64,
    pub novel_formations: Vec<Composite>,
    pub formation_robustness: f64,
    pub baseline_composite: f64,
}

/// Runs all generalization evaluations and produces a unified summary.
pub fn full_generalization_report(seed: u32) -> GeneralizationReport {
    let max_ticks: u64 = 8000;

    // Run baseline (normal conditions) for reference.
    let baseline = {
        set_planner_class(0);
        let (sub, raw) = run_scored_match(
            "fall-dynasty", "big-slow", "fall-dynasty", "big-slow", seed, max_ticks,
        );
        assemble_composite("baseline", vec![(sub, raw)])
    };

    // Run all sub-evaluations.
    let cross_style = eval_cross_style(0, seed, max_ticks);
    let asymmetric = eval_asymmetric(seed, max_ticks);
    let novel_formations = eval_novel_formations(seed, max_ticks);

    // Compute summary statistics.
    let composites: Vec<f64> = cross_style.iter().map(|c| c.composite).collect();
    let n = composites.len() as f64;
    let mean = composites.iter().sum::<f64>() / n;
    let variance = composites.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // Exploitation score: how much does Pro dominate Rookie?
    // (home_score - away_score) normalized. Higher = difficulty works correctly.
    let exploitation_score =
        asymmetric.pro_vs_rookie.home_score - asymmetric.pro_vs_rookie.away_score;

    // Formation robustness: min composite across formations / baseline.
    let min_formation = novel_formations
        .iter()
        .map(|c| c.composite)
        .fold(f64::INFINITY, f64::min);
    let formation_robustness = if baseline.composite > 0.0 {
        min_formation / baseline.composite
    } else {
        0.0
    };

    let report = GeneralizationReport {
        cross_style_results: cross_style,
        cross_style_mean: mean,
        cross_style_std_dev: std_dev,
        asymmetric,
        exploitation_score,
        novel_formations,
        formation_robustness,
        baseline_composite: baseline.composite,
    };

    println!("=== FULL GENERALIZATION REPORT ===");
    println!("  Baseline composite:      {:.1}", report.baseline_composite);
    println!("  Cross-style mean:        {:.1}", report.cross_style_mean);
    println!("  Cross-style std dev:     {:.2}  (lower = more robust)", report.cross_style_std_dev);
    println!(
        "  Exploitation (Pro-Rookie): {:.0}  (positive = difficulty works)",
        report.exploitation_score
    );
    println!(
        "  Formation robustness:    {:.2}  (min_formation/baseline, 1.0 = no degradation)",
        report.formation_robustness
    );
    println!();

    report
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "full generalization eval; run with --ignored --nocapture"]
    fn generalization_report() {
        let report = full_generalization_report(1234);
        // Sanity checks — the report should produce finite, reasonable values.
        assert!(report.baseline_composite.is_finite());
        assert!(report.cross_style_mean.is_finite());
        assert!(report.cross_style_std_dev.is_finite());
        assert!(report.cross_style_std_dev >= 0.0);
        assert!(report.formation_robustness.is_finite());
        for c in &report.cross_style_results {
            assert!(c.composite.is_finite());
            assert!((0.0..=100.0).contains(&c.composite));
        }
        for c in &report.novel_formations {
            assert!(c.composite.is_finite());
            assert!((0.0..=100.0).contains(&c.composite));
        }
        set_planner_class(0);
    }
}

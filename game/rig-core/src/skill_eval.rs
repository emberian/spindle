//! Native skill-eval harness — the fast measuring instrument.
//!
//! Runs a full AI-vs-AI match in-process (SimWorld + MatchStateMachine +
//! AiSystem, all Rust) and scores it with the same Goodhart-resistant
//! skill telemetry as the old TS `src/eval/headless.ts` (which was
//! deleted in the cutover — this is its native replacement, ~1000× faster
//! and the keystone for ranking the planner algorithm zoo).
//!
//! Scoring is RELATIVE by design (rank algorithm classes / knobs against
//! each other), normalised + weighted to sum ≤ 1, multi-matchup, so no
//! single signal can be gamed. Absolute parity with the old TS numbers is
//! explicitly NOT a goal (that paradigm was retired with the TS).
//!
//! The 7 module-boundary seams (sim_world ↔ scoring/match_sm ↔ ai::types)
//! are bridged by the private `conv` adapters below.

use crate::ai::decision_types::TeamConfig;
use crate::ai::plan_bridge::set_planner_class;
use crate::ai::profile::{style_to_profile, Difficulty};
use crate::ai::system::AiSystem;
use crate::ai::types as ai;
use crate::match_sm::{Gate, MatchPhase, MatchStateMachine};
use crate::math::Vec3;
use crate::scoring;
use crate::sim_world::{RingEnd as SwRing, SimEvent as SwEv, SimWorld, Snapshot};
use std::collections::HashMap;

// ── Roster (headless.ts ROSTER) ──────────────────────────────────────────────
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

// Normalisation caps + weights — verbatim from headless.ts (CAP/WT). The
// 9 weights sum to 1.0; each sub-signal is clamped to [0,1] so no metric
// can be Goodharted past its cap.
struct Caps {
    gate: f64,
    pass: f64,
    intc: f64,
    score: f64,
    poss: f64,
    thrash_k: f64,
    skin_pct: f64,
    shot_conv: f64,
    chain: f64,
}
const CAP: Caps = Caps {
    gate: 6.0,
    pass: 6.0,
    intc: 8.0,
    score: 30.0,
    poss: 0.3,
    thrash_k: 300.0,
    skin_pct: 80.0,
    shot_conv: 0.4,
    chain: 4.0,
};
// WT (Σ = 1.00). This is a skill model, not a planner-comfort model:
// gate progression, retained passes and scoring are the core sport signals.
// Possession/calm/field remain useful diagnostics, but they are deliberately
// low-weight because they are easy to Goodhart via holding, low-event play or
// conservative anti-thrash behavior.
const WT: [(&str, f64); 9] = [
    ("prog", 0.22),
    ("pass", 0.14),
    ("intc", 0.08),
    ("score", 0.18),
    ("poss", 0.04),
    ("calm", 0.08),
    ("field", 0.04),
    ("shotConv", 0.12),
    ("chain", 0.10),
];

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

/// (home_style, home_cyl, away_style, away_cyl) — three deterministic
/// matchups spanning the style space (Fall/big-slow, Rise/small-fast,
/// Ground/mid), the spirit of headless.ts MATCHUPS without the 32-team
/// TS table (the eval is relative; exact rows aren't sacred).
const MATCHUPS: [(&str, &str, &str, &str); 3] = [
    ("fall-dynasty", "big-slow", "fall-dynasty", "big-slow"),
    ("fall-grind", "big-slow", "rise-power", "small-fast"),
    ("fall-tempo", "big-slow", "rise-chaos", "small-fast"),
];

#[inline]
fn clamp01(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

#[derive(Clone, Debug)]
pub struct Composite {
    pub label: String,
    pub composite: f64,
    pub parts: Vec<(String, f64)>,
    pub raw: Vec<(String, f64)>,
}

// ── Seam adapters (the 7 conversions from the seam map) ──────────────────────
// `pub(crate)` so the native-only `gym` Env reuses these EXACT bridges
// (no reinvention — same sim_world↔scoring↔ai::types conversions the
// ranker path uses, so a gym match and a skill_eval match see identical
// seam semantics).
pub(crate) mod conv {
    use super::*;

    pub fn team_ai_to_scoring(t: ai::TeamSide) -> scoring::TeamSide {
        match t {
            ai::TeamSide::Home => scoring::TeamSide::Home,
            ai::TeamSide::Away => scoring::TeamSide::Away,
        }
    }
    pub fn team_scoring_to_ai(t: scoring::TeamSide) -> ai::TeamSide {
        match t {
            scoring::TeamSide::Home => ai::TeamSide::Home,
            scoring::TeamSide::Away => ai::TeamSide::Away,
        }
    }
    pub fn role_ai_to_scoring(r: ai::RiggerRole) -> scoring::RiggerRole {
        use ai::RiggerRole as A;
        use scoring::RiggerRole as S;
        match r {
            A::Anchor => S::Anchor,
            A::Spinner => S::Spinner,
            A::Faithwing => S::Faithwing,
            A::Freewing => S::Freewing,
            A::Reach => S::Reach,
        }
    }
    fn sw_team_to_ai(t: crate::sim_world::TeamSide) -> ai::TeamSide {
        match t {
            crate::sim_world::TeamSide::Home => ai::TeamSide::Home,
            crate::sim_world::TeamSide::Away => ai::TeamSide::Away,
        }
    }
    fn sw_role_to_ai(r: crate::sim_world::RiggerRole) -> ai::RiggerRole {
        use ai::RiggerRole as A;
        use crate::sim_world::RiggerRole as W;
        match r {
            W::Anchor => A::Anchor,
            W::Spinner => A::Spinner,
            W::Faithwing => A::Faithwing,
            W::Freewing => A::Freewing,
            W::Reach => A::Reach,
        }
    }

    /// SEAM 1: sim_world::Snapshot → ai::types::SimState (line restructure).
    pub fn snap_to_ai(s: &Snapshot) -> ai::SimState {
        ai::SimState {
            tick: s.tick as f64,
            omega: s.omega,
            bell: ai::BellState {
                p: s.bell.p,
                v: s.bell.v,
                q: s.bell.q,
                w: s.bell.w,
                chime: s.bell.chime,
                held_by: s.bell.held_by.clone(),
                thrown_by: s.bell.thrown_by.clone(),
                touched_since_throw: s.bell.touched_since_throw,
                release_pos: s.bell.release_pos,
                release_tick: s.bell.release_tick as f64,
                pass_chain: s.bell.pass_chain.clone(),
            },
            players: s
                .players
                .iter()
                .map(|p| ai::PlayerSim {
                    id: p.id.clone(),
                    team: sw_team_to_ai(p.team),
                    role: sw_role_to_ai(p.role),
                    p: p.p,
                    v: p.v,
                    q: p.q,
                    line: p.line_anchor.map(|a| ai::GrappleState {
                        anchor_type: ai::AnchorType::Spar,
                        anchor_ref: None,
                        anchor_pos: a,
                        rest_len: p.line_rest_len.unwrap_or(0.0),
                        taut: p.line_taut.unwrap_or(false),
                    }),
                    dv_budget: p.dv_budget,
                    contact_ref: p.contact_ref.clone(),
                    grounded: p.grounded,
                })
                .collect(),
        }
    }

    /// SEAM 3: ai::types::SimState → scoring::SimState (minimal view).
    pub fn ai_to_scoring(s: &ai::SimState) -> scoring::SimState {
        scoring::SimState {
            tick: s.tick as u64,
            omega: s.omega,
            bell: scoring::BellState {
                p: s.bell.p,
                v: s.bell.v,
                held_by: s.bell.held_by.clone(),
                thrown_by: s.bell.thrown_by.clone(),
                touched_since_throw: s.bell.touched_since_throw,
                pass_chain: s.bell.pass_chain.clone(),
            },
            players: s
                .players
                .iter()
                .map(|p| scoring::PlayerSim {
                    id: p.id.clone(),
                    team: team_ai_to_scoring(p.team),
                    role: role_ai_to_scoring(p.role),
                    p: p.p,
                    v: p.v,
                })
                .collect(),
        }
    }

    /// SEAM 2: sim_world::SimEvent → scoring::SimEvent.
    pub fn ev_sw_to_scoring(e: &SwEv) -> scoring::SimEvent {
        use scoring::SimEvent as S;
        let ring = |r: &SwRing| match r {
            SwRing::PlusX => scoring::RingEnd::PlusX,
            SwRing::MinusX => scoring::RingEnd::MinusX,
        };
        match e {
            SwEv::BellThroughRing { end, touched, loop_tier } => S::BellThroughRing {
                end: ring(end),
                touched: *touched,
                loop_tier: match loop_tier {
                    crate::sim_world::LoopTierOut::Loop => scoring::LoopTier::Loop,
                    crate::sim_world::LoopTierOut::Curl => scoring::LoopTier::Curl,
                    crate::sim_world::LoopTierOut::None => scoring::LoopTier::None,
                },
            },
            SwEv::BellMissed { end } => S::BellMissed { end: ring(end) },
            SwEv::BellCaught { by } => S::BellCaught { by: by.clone() },
            SwEv::BellBobble { by } => S::BellBobble { by: by.clone() },
            SwEv::BellClatter { by } => S::BellClatter { by: by.clone() },
            SwEv::BellSkin => S::BellSkin,
            SwEv::PlayerSkinned { id } => S::PlayerSkinned { id: id.clone() },
            SwEv::ContestStarted { thrower, contester } => S::ContestStarted {
                thrower: thrower.clone(),
                contester: contester.clone(),
            },
            SwEv::FoulGarrote { by } => S::FoulGarrote { by: by.clone() },
        }
    }

    /// SEAM 4: match_sm::MatchState → ai::types::MatchState.
    pub fn msm_to_ai(m: &crate::match_sm::MatchState) -> ai::MatchState {
        ai::MatchState {
            inning: m.inning as f64,
            spine: m.spine,
            possession: team_scoring_to_ai(m.possession),
            faith_end: match m.faith_end {
                scoring::RingEnd::PlusX => ai::FaithEnd::PlusX,
                scoring::RingEnd::MinusX => ai::FaithEnd::MinusX,
            },
            cast: ai::Cast {
                throws_left: m.cast.throws_left,
                gate: match m.cast.gate {
                    Gate::First => ai::Gate::First,
                    Gate::Deep => ai::Gate::Deep,
                    Gate::Mouth => ai::Gate::Mouth,
                },
                spot_x: m.cast.spot_x,
            },
            contest: m.contest.as_ref().map(|c| ai::MatchContest {
                thrower: c.thrower.clone(),
                contester: c.contester.clone(),
                count: c.count,
                radius: c.radius,
                direction: match c.direction {
                    crate::contest::ContestDirection::Fair => ai::ContestDirection::Fair,
                    crate::contest::ContestDirection::Cross => ai::ContestDirection::Cross,
                },
            }),
            score_home: m.score_home as f64,
            score_away: m.score_away as f64,
            phase: match m.phase {
                MatchPhase::Set => ai::MatchPhase::Set,
                MatchPhase::Live => ai::MatchPhase::Live,
                MatchPhase::Contest => ai::MatchPhase::Contest,
                MatchPhase::Dead => ai::MatchPhase::Dead,
                MatchPhase::InningBreak => ai::MatchPhase::InningBreak,
                MatchPhase::Spine => ai::MatchPhase::Spine,
                MatchPhase::Final => ai::MatchPhase::Final,
            },
            message: m.message.clone(),
            winner: m.winner.map(team_scoring_to_ai),
        }
    }

    /// SEAM 5: ai::types::InputFrame → sim_world::InputFrame.
    pub fn frame_ai_to_sw(f: &ai::InputFrame) -> crate::sim_world::InputFrame {
        crate::sim_world::InputFrame {
            tick: f.tick as u64,
            players: f
                .players
                .iter()
                .map(|p| crate::sim_world::PlayerInput {
                    id: p.id.clone(),
                    aim: p.aim,
                    fire_line_at: p.fire_line_at,
                    reel: p.reel,
                    release: p.release,
                    pushoff: p.pushoff,
                    throw_charge: p.throw_charge,
                    throw_released: p.throw_released,
                    throw_spin: p.throw_spin,
                    thrumbler: p.thrumbler,
                    catch_intent: p.catch_intent,
                })
                .collect(),
        }
    }
}

fn gate_ord(g: ai::Gate) -> i32 {
    match g {
        ai::Gate::First => 0,
        ai::Gate::Deep => 1,
        ai::Gate::Mouth => 2,
    }
}

/// One full AI-vs-AI match. Returns (sub-signals[name→0..1], raw[name→val]).
///
/// The planner knob (thread-local) MUST already be installed by the caller:
/// `one_match` (the ranker path) installs it via `set_planner_class` BEFORE
/// calling this; the GA path (`one_match_profile`) installs an evolved
/// `Rc<Profile>` via `set_planner_profile`. Everything below — the sim loop,
/// every seam, and ALL scoring math — is shared verbatim so the two paths
/// are guaranteed consistent and the documented ranker composites are
/// untouched (the only difference between the paths is WHICH planner profile
/// the thread-local holds when `aisys.tick` → `plan_grapple` reads it).
fn run_scored_match(
    h_style: &str,
    h_cyl: &str,
    a_style: &str,
    a_cyl: &str,
    seed: u32,
    max_ticks: u64,
) -> (HashMap<&'static str, f64>, HashMap<&'static str, f64>) {
    let mut sim = SimWorld::new(seed);
    let rs = roster();
    let team_of: HashMap<String, ai::TeamSide> =
        rs.iter().map(|r| (r.id.to_string(), r.team)).collect();
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
    // headless kicks the match to live with a synthetic foul.
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
            difficulty: Difficulty::Pro,
        },
        TeamConfig {
            side: ai::TeamSide::Away,
            profile: style_to_profile(a_style, a_cyl),
            difficulty: Difficulty::Pro,
        },
    ];
    let mut aisys = AiSystem::new();

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
                    if ((f.x - pf.x).powi(2) + (f.y - pf.y).powi(2) + (f.z - pf.z).powi(2)).sqrt()
                        > 8.0
                    {
                        thrash += 1.0;
                    }
                }
                prev_fire.insert(inp.id.clone(), f);
            }
        }

        // pass / intercept on a grip transition (headless pre-step logic).
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

        // step the sim, then feed events + post-step state to the match SM.
        let sw_frame = conv::frame_ai_to_sw(&frame);
        let evs = sim.step(&sw_frame, H);
        for e in &evs {
            match e {
                SwEv::BellCaught { .. } => *scores.entry("caught").or_insert(0.0) += 1.0,
                SwEv::BellClatter { .. } | SwEv::BellBobble { .. } => {
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

        // Fix A: count a gate clear AFTER mat.consume() has processed
        // this tick's events — consume is where cast.gate advances
        // (on_throw_spent), so the old pre-consume read never observed
        // a same-tick advance. Read post-consume, before the dead-ball
        // re-arm; inning/turnover resets are gate DROPS (lower ord) so
        // they correctly don't count as clears, and each genuine
        // first→deep / deep→mouth advance is counted exactly once.
        let g_ord = match mat.state().cast.gate {
            Gate::First => 0,
            Gate::Deep => 1,
            Gate::Mouth => 2,
        };
        if let Some(pg) = prev_gate_ord {
            if g_ord > pg {
                gate_clears += 1.0;
            }
        }
        prev_gate_ord = Some(g_ord);

        // re-arm a dead / inning-break ball (headless).
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
        ticks += 1.0;
    }

    // derived signals (headless formulas).
    let kinds = ["fall", "rise", "loop", "curl", "ground"];
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
    let shot_conv = if total_throws > 0.0 {
        score_events / total_throws
    } else {
        0.0
    };
    let t = ticks.max(1.0);
    let held_frac = held_ticks / t;
    let skin_pct = 100.0 * skin_ticks / t;
    let thrash_k = (thrash / t) * 1000.0;

    let mut sub = HashMap::new();
    sub.insert("prog", clamp01(gate_clears / CAP.gate));
    sub.insert("pass", clamp01(passes / CAP.pass));
    sub.insert("intc", clamp01(intercepts / CAP.intc));
    sub.insert("score", clamp01(score_pts_total / CAP.score));
    sub.insert("poss", clamp01(held_frac / CAP.poss));
    sub.insert("calm", 1.0 - clamp01(thrash_k / CAP.thrash_k));
    sub.insert("field", 1.0 - clamp01(skin_pct / CAP.skin_pct));
    sub.insert("shotConv", clamp01(shot_conv / CAP.shot_conv));
    sub.insert("chain", clamp01(streak_max / CAP.chain));

    let mut raw = HashMap::new();
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
    (sub, raw)
}

/// The ORIGINAL ranker entry: install the class on the thread-local knob via
/// the `set_planner_class` back-compat shim, then run the shared scored
/// match. STRUCTURALLY + BEHAVIORALLY UNCHANGED from before the Stage-3
/// refactor — `set_planner_class(class)` then the exact same match/scoring
/// body (now in `run_scored_match`). This keeps `eval_skill` and the
/// documented ranker composites bit-identical.
fn one_match(
    planner_class: i32,
    h_style: &str,
    h_cyl: &str,
    a_style: &str,
    a_cyl: &str,
    seed: u32,
    max_ticks: u64,
) -> (HashMap<&'static str, f64>, HashMap<&'static str, f64>) {
    set_planner_class(planner_class);
    run_scored_match(h_style, h_cyl, a_style, a_cyl, seed, max_ticks)
}

/// The GA path's match: install an EVOLVED `Profile` (not an engine class)
/// on the calling rayon thread's thread-local knob, then run the SAME shared
/// scored match. Because `PLANNER_PROFILE` is thread-local, installing here
/// affects only this thread — parallel-safe exactly like `set_planner_class`
/// (the whole point of Stage 2). `aisys.tick` → `plan_grapple` →
/// `planner_profile()` reads back THIS profile, so the GA's evolved weights
/// (`branch_cost`/`coord_compute_cost` fold) actually drive the match.
fn one_match_profile(
    profile: &crate::planner_cost::Profile,
    h_style: &str,
    h_cyl: &str,
    a_style: &str,
    a_cyl: &str,
    seed: u32,
    max_ticks: u64,
) -> (HashMap<&'static str, f64>, HashMap<&'static str, f64>) {
    use crate::ai::plan_bridge::set_planner_profile;
    use crate::planner_cost::Profile;
    use std::rc::Rc;
    // `Profile` owns a `Vec<Box<dyn CostTerm>>` (no `Clone`), and the GA
    // only ever evolves the WEIGHT VECTOR (engine stays 9 = Coordination
    // dispatch). So reconstruct a fresh term-bearing `Profile` for this
    // thread from the genome's engine, then overwrite its `weights` with
    // the evolved fixed-order vector. `from_class` builds the SAME
    // fixed-order term list whose names line up 1:1 with `weights`, so the
    // `branch_cost`/`coord_compute_cost` fold uses the evolved weights
    // exactly. Built once per match (3 per eval), never per tick.
    let mut p = Profile::from_class(profile.engine);
    p.label = profile.label.clone();
    debug_assert_eq!(
        p.weights.len(),
        profile.weights.len(),
        "GA genome must match the engine's term-vector length"
    );
    for (slot, g) in p.weights.iter_mut().zip(profile.weights.iter()) {
        debug_assert_eq!(slot.0, g.0, "GA genome term-order must match engine");
        slot.1 = g.1;
    }
    set_planner_profile(Some(Rc::new(p)));
    run_scored_match(h_style, h_cyl, a_style, a_cyl, seed, max_ticks)
}

/// Shared matchup reduction: average each sub-signal / raw in FIXED
/// match-index order and assemble the `Composite` EXACTLY as `eval_skill`
/// does (same `avg`, same WT fold, same rounding). Used by `eval_profile`;
/// `eval_skill` keeps its own inline copy verbatim so its path is provably
/// untouched (this helper is byte-equivalent — verified by
/// `eval_profile_matches_eval_skill_for_class_profile`).
fn assemble_composite(
    label: &str,
    results: Vec<(HashMap<&str, f64>, HashMap<&str, f64>)>,
) -> Composite {
    let mut subs: Vec<HashMap<&str, f64>> = Vec::with_capacity(results.len());
    let mut raws: Vec<HashMap<&str, f64>> = Vec::with_capacity(results.len());
    for (s, r) in results {
        subs.push(s);
        raws.push(r);
    }
    let avg = |k: &str, arr: &[HashMap<&str, f64>]| -> f64 {
        arr.iter().map(|m| *m.get(k).unwrap_or(&0.0)).sum::<f64>() / arr.len() as f64
    };
    let mut parts = Vec::new();
    let mut composite = 0.0;
    for (k, w) in WT.iter() {
        let v = avg(k, &subs);
        parts.push((k.to_string(), (v * 1000.0).round() / 1000.0));
        composite += w * v;
    }
    let raw_keys = [
        "gateClears", "passes", "intercepts", "scorePts", "heldFrac", "thrashK", "skinPct",
        "shotConv", "chainMax", "ticks",
    ];
    let raw = raw_keys
        .iter()
        .map(|k| (k.to_string(), (avg(k, &raws) * 10.0).round() / 10.0))
        .collect();
    Composite {
        label: label.to_string(),
        composite: (1000.0 * composite).round() / 10.0,
        parts,
        raw,
    }
}

/// STAGE-3 GA fitness: evaluate an EVOLVED `Profile`'s weights (NOT an
/// engine class) over the same three matchups, parallel + fixed-order
/// collect, scored identically to `eval_skill`. The GA's fitness is
/// `eval_profile(profile, seed, max_ticks).composite`. This is the
/// corrected blueprint wiring: it injects the profile via
/// `set_planner_profile` (per-rayon-thread, thread-local ⇒ parallel-safe)
/// so the GA actually optimizes the coordination/branch WEIGHTS rather than
/// re-scoring stock Coordination (engine stays 9 for dispatch).
pub fn eval_profile(
    profile: &crate::planner_cost::Profile,
    seed: u32,
    max_ticks: u64,
) -> Composite {
    use rayon::prelude::*;
    // Same parallel structure as `eval_skill`: independent matchups, but
    // collected into a Vec indexed by matchup position so every downstream
    // float reduction happens in fixed match-index order (determinism).
    let results: Vec<(HashMap<&str, f64>, HashMap<&str, f64>)> = (0..MATCHUPS.len())
        .into_par_iter()
        .map(|i| {
            let (hs, hc, as_, ac) = MATCHUPS[i];
            one_match_profile(profile, hs, hc, as_, ac, seed, max_ticks)
        })
        .collect();
    assemble_composite(&profile.label, results)
}

/// Full skill eval for one planner class, averaged over the matchups.
/// Native and fast — this is what makes the algorithm zoo rankable.
pub fn eval_skill(label: &str, planner_class: i32, seed: u32, max_ticks: u64) -> Composite {
    use rayon::prelude::*;
    // Matches are fully independent (distinct matchup, shared seed/class).
    // Run them in PARALLEL but collect into a Vec indexed by matchup
    // position via `into_par_iter().collect()` — this preserves the
    // serial matchup order EXACTLY, so all downstream float reductions
    // (the `avg` sums below) happen in fixed match-index order. The
    // thread-local PLANNER_CLASS (set by `one_match`'s `set_planner_class`)
    // makes each task self-contained; no global is clobbered.
    let results: Vec<(HashMap<&str, f64>, HashMap<&str, f64>)> = (0..MATCHUPS.len())
        .into_par_iter()
        .map(|i| {
            let (hs, hc, as_, ac) = MATCHUPS[i];
            one_match(planner_class, hs, hc, as_, ac, seed, max_ticks)
        })
        .collect();
    let mut subs: Vec<HashMap<&str, f64>> = Vec::with_capacity(results.len());
    let mut raws: Vec<HashMap<&str, f64>> = Vec::with_capacity(results.len());
    for (s, r) in results {
        subs.push(s);
        raws.push(r);
    }
    let avg = |k: &str, arr: &[HashMap<&str, f64>]| -> f64 {
        arr.iter().map(|m| *m.get(k).unwrap_or(&0.0)).sum::<f64>() / arr.len() as f64
    };
    let mut parts = Vec::new();
    let mut composite = 0.0;
    for (k, w) in WT.iter() {
        let v = avg(k, &subs);
        parts.push((k.to_string(), (v * 1000.0).round() / 1000.0));
        composite += w * v;
    }
    let raw_keys = [
        "gateClears", "passes", "intercepts", "scorePts", "heldFrac", "thrashK", "skinPct",
        "shotConv", "chainMax", "ticks",
    ];
    let raw = raw_keys
        .iter()
        .map(|k| (k.to_string(), (avg(k, &raws) * 10.0).round() / 10.0))
        .collect();
    Composite {
        label: label.to_string(),
        composite: (1000.0 * composite).round() / 10.0,
        parts,
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        // Tiny budget — just prove the loop + seams + scoring are
        // deterministic (same seed/class ⇒ identical composite).
        let a = eval_skill("MPC", 0, 1234, 400);
        let b = eval_skill("MPC", 0, 1234, 400);
        assert_eq!(a.composite.to_bits(), b.composite.to_bits());
        assert!(a.composite.is_finite());
        for ((ka, va), (kb, vb)) in a.parts.iter().zip(b.parts.iter()) {
            assert_eq!(ka, kb);
            assert_eq!(va.to_bits(), vb.to_bits());
        }
        // restore the production default for any later test.
        set_planner_class(0);
    }

    #[test]
    fn parallel_eval_is_bit_identical_and_stable() {
        // Determinism proof for the rayon parallelisation: a parallel
        // `eval_skill` run must equal a SECOND parallel run bit-for-bit
        // (composite .to_bits() + every part), regardless of thread
        // scheduling. The fix that makes this hold: PLANNER_CLASS is
        // thread-local (no cross-thread clobber) and the matchup results
        // are collected in fixed match-index order (float sums never
        // reorder). Cheap budget so it stays a CI gate.
        let a = eval_skill("Coord", 9, 1234, 400);
        let b = eval_skill("Coord", 9, 1234, 400);
        assert_eq!(a.composite.to_bits(), b.composite.to_bits());
        assert!(a.composite.is_finite());
        assert_eq!(a.parts.len(), b.parts.len());
        for ((ka, va), (kb, vb)) in a.parts.iter().zip(b.parts.iter()) {
            assert_eq!(ka, kb);
            assert_eq!(va.to_bits(), vb.to_bits());
        }
        assert_eq!(a.raw.len(), b.raw.len());
        for ((ka, va), (kb, vb)) in a.raw.iter().zip(b.raw.iter()) {
            assert_eq!(ka, kb);
            assert_eq!(va.to_bits(), vb.to_bits());
        }
        // Stable across a third run AND across a different class run
        // interleaved (proves concurrent-different-class safety: the
        // thread-local means class 0 here cannot poison class 9 above).
        let c0 = eval_skill("MPC", 0, 1234, 400);
        let c = eval_skill("Coord", 9, 1234, 400);
        assert_eq!(a.composite.to_bits(), c.composite.to_bits());
        assert!(c0.composite.is_finite());
        set_planner_class(0);
    }

    // On-demand exploration tool, NOT a CI gate (the full 6-class zoo at
    // a real budget is ~155 s — like the old TS headless it must not tax
    // every CI run). Run it deliberately:
    //   cargo test -p rig-core ranks_the_algorithm_zoo -- --ignored --nocapture
    // The fast `deterministic_same_seed` test above stays always-on as
    // the harness regression guard.
    #[test]
    #[ignore = "exploration tool; run with --ignored --nocapture"]
    fn ranks_the_algorithm_zoo() {
        // The instrument in action: rank the full planner menagerie
        // (0..=9). Budget = 8000 ticks — the OFFENSE-RESOLVING regime
        // (the old TS headless used 8000 and offense demonstrably
        // emerged there; at 2000 the eval was near-sighted, scoring
        // ~0 for everyone). ~17 min for all 10 — strictly an on-demand
        // tool. We do NOT hard-assert an ordering (that's the
        // explorable part); we DO print parts+raw so the offense
        // signals (gateClears/scorePts/shotConv) are visible.
        use rayon::prelude::*;
        let budget = 8000;
        // The 10 classes are independent (thread-local planner class ⇒
        // `eval_skill` is safe to call concurrently for different classes).
        // Parallelise across them, but collect indexed so print order is
        // fixed (class 0..=9). Nested rayon (this par + eval_skill's
        // matchup par) is fine — rayon's work-stealing pool handles it.
        let menagerie: [(&str, i32); 10] = [
            ("MPC", 0),
            ("RRT", 1),
            ("CEM", 2),
            ("MPPI", 3),
            ("SA", 4),
            ("Beam", 5),
            ("MCTS", 6),
            ("PotFld", 7),
            ("RandSh", 8),
            ("Coord", 9),
        ];
        let out: Vec<Composite> = (0..menagerie.len())
            .into_par_iter()
            .map(|i| {
                let (label, class) = menagerie[i];
                eval_skill(label, class, 1234, budget)
            })
            .collect();
        for r in &out {
            println!(
                "SKILL {:<4} composite={:>5}  parts={:?}  raw={:?}",
                r.label, r.composite, r.parts, r.raw
            );
            assert!(r.composite.is_finite());
            assert!((0.0..=100.0).contains(&r.composite));
        }
        set_planner_class(0);
    }
}

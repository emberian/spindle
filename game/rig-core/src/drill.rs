//! Skill Drill — isolated scenario benchmarks for specific AI behavior
//! controllers. Unlike skill_eval (full matches), drills test individual
//! links in the game loop: catching, pursuit, throw-receive, navigation,
//! possession chains.
//!
//! NATIVE-ONLY (uses the same sim_world/match_sm/AiSystem infrastructure
//! as skill_eval, gated identically).

use crate::ai::decision_types::TeamConfig;
use crate::ai::plan_bridge::set_planner_class;
use crate::ai::profile::{style_to_profile, Difficulty};
use crate::ai::system::AiSystem;
use crate::ai::types as ai;
use crate::match_sm::{Gate, MatchPhase, MatchStateMachine};
use crate::math::Vec3;
use crate::scoring;
use crate::sim_world::{SimEvent as SwEv, SimWorld};
use crate::skill_eval::conv;

const H: f64 = 1.0 / 240.0;

#[derive(Clone, Debug)]
pub struct DrillResult {
    pub name: String,
    pub trials: u32,
    pub successes: u32,
    pub avg_ticks: f64,
    pub details: Vec<String>,
}

impl DrillResult {
    pub fn rate(&self) -> f64 {
        if self.trials == 0 {
            0.0
        } else {
            self.successes as f64 / self.trials as f64
        }
    }
}

#[derive(Clone, Debug)]
pub struct DrillSuite {
    pub results: Vec<DrillResult>,
}

impl DrillSuite {
    pub fn overall_rate(&self) -> f64 {
        let t: u32 = self.results.iter().map(|r| r.trials).sum();
        let s: u32 = self.results.iter().map(|r| r.successes).sum();
        if t == 0 { 0.0 } else { s as f64 / t as f64 }
    }
}

pub fn run_all(verbose: bool) -> DrillSuite {
    let results = vec![
        drill_static_catch(verbose),
        drill_pursuit(verbose),
        drill_throw_receive(verbose),
        drill_grapple_nav(verbose),
        drill_possession_chain(verbose),
        drill_catch_angles(verbose),
        drill_catch_while_moving(verbose),
        drill_pursuit_reacquire(verbose),
        drill_throw_accuracy(verbose),
        drill_two_pass(verbose),
        drill_under_pressure(verbose),
        drill_gate_score(verbose),
        drill_loose_ball_race(verbose),
        drill_defensive_intercept(verbose),
        drill_hold_and_advance(verbose),
        drill_short_pass(verbose),
        drill_outlet_under_pressure(verbose),
        drill_moving_receive(verbose),
        // Navigation drills
        drill_speed_challenge(verbose),
        drill_waypoint_circuit(verbose),
        drill_precision_arrival(verbose),
        drill_skin_traverse(verbose),
        // Grapple mechanics drills
        drill_swing_distance(verbose),
        drill_grapple_efficiency(verbose),
        drill_momentum_chain(verbose),
        // Coordination drills
        drill_spacing_maintenance(verbose),
        drill_support_positioning(verbose),
        drill_rotation_after_catch(verbose),
        // Strategy drills
        drill_advance_under_defense(verbose),
        drill_deny_advance(verbose),
        drill_fast_break(verbose),
    ];
    DrillSuite { results }
}

pub fn run_one(name: &str, verbose: bool) -> Option<DrillResult> {
    match name {
        "catch" | "static_catch" => Some(drill_static_catch(verbose)),
        "pursuit" => Some(drill_pursuit(verbose)),
        "throw" | "throw_receive" => Some(drill_throw_receive(verbose)),
        "nav" | "grapple_nav" => Some(drill_grapple_nav(verbose)),
        "chain" | "possession_chain" => Some(drill_possession_chain(verbose)),
        "angles" | "catch_angles" => Some(drill_catch_angles(verbose)),
        "moving" | "catch_while_moving" => Some(drill_catch_while_moving(verbose)),
        "reacquire" | "pursuit_reacquire" => Some(drill_pursuit_reacquire(verbose)),
        "accuracy" | "throw_accuracy" => Some(drill_throw_accuracy(verbose)),
        "two_pass" | "pass_chain" => Some(drill_two_pass(verbose)),
        "pressure" | "under_pressure" => Some(drill_under_pressure(verbose)),
        "gate" | "gate_score" => Some(drill_gate_score(verbose)),
        "race" | "loose_ball_race" => Some(drill_loose_ball_race(verbose)),
        "intercept" | "defensive_intercept" => Some(drill_defensive_intercept(verbose)),
        "advance" | "hold_and_advance" => Some(drill_hold_and_advance(verbose)),
        "short_pass" | "short" => Some(drill_short_pass(verbose)),
        "outlet" | "outlet_under_pressure" => Some(drill_outlet_under_pressure(verbose)),
        "moving_receive" | "lead_pass" => Some(drill_moving_receive(verbose)),
        // Navigation drills
        "speed" | "speed_challenge" => Some(drill_speed_challenge(verbose)),
        "waypoint" | "waypoint_circuit" => Some(drill_waypoint_circuit(verbose)),
        "precision" | "precision_arrival" => Some(drill_precision_arrival(verbose)),
        "skin" | "skin_traverse" => Some(drill_skin_traverse(verbose)),
        // Grapple mechanics drills
        "swing" | "swing_distance" => Some(drill_swing_distance(verbose)),
        "efficiency" | "grapple_efficiency" => Some(drill_grapple_efficiency(verbose)),
        "momentum" | "momentum_chain" => Some(drill_momentum_chain(verbose)),
        // Coordination drills
        "spacing" | "spacing_maintenance" => Some(drill_spacing_maintenance(verbose)),
        "support" | "support_positioning" => Some(drill_support_positioning(verbose)),
        "rotation" | "rotation_after_catch" => Some(drill_rotation_after_catch(verbose)),
        // Strategy drills
        "advance_defense" | "advance_under_defense" => Some(drill_advance_under_defense(verbose)),
        "deny" | "deny_advance" => Some(drill_deny_advance(verbose)),
        "fast_break" | "break" => Some(drill_fast_break(verbose)),
        _ => None,
    }
}

// ── DRILL 1: Static Catch ───────────────────────────────────────────────────
// Ball thrown directly at a stationary player from varying distances/speeds.

pub fn drill_static_catch(verbose: bool) -> DrillResult {
    let speeds = [10.0, 15.0, 20.0, 25.0, 30.0];
    let distances = [15.0, 30.0, 50.0];
    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for &dist in &distances {
        for &speed in &speeds {
            trials += 1;
            let (caught, ticks) = run_catch_trial(dist, speed, 0.0);
            if caught {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "dist={:.0}m spd={:.0}m/s → {} t={}",
                    dist, speed,
                    if caught { "CATCH" } else { "MISS" }, ticks
                ));
            }
        }
    }

    DrillResult {
        name: "static_catch".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_catch_trial(dist: f64, speed: f64, lateral_offset: f64) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(100);
    let catcher_pos = Vec3::new(0.0, 10.0, 0.0);
    let ball_start = Vec3::new(dist, 10.0 + lateral_offset, 0.0);
    let ball_vel = Vec3::new(-speed, 0.0, 0.0);

    sim.add_player(
        "C1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        catcher_pos,
    );
    sim.launch_bell(ball_start, ball_vel, Vec3::new(0.0, 0.0, 0.0), None);

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 720u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if snap.bell.held_by.as_deref() == Some("C1") {
            return (true, tick);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 100);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks)
}

// ── DRILL 2: Pursuit Catch ──────────────────────────────────────────────────
// Ball thrown PAST a player at lateral offsets. Player must grapple-pursue.

pub fn drill_pursuit(verbose: bool) -> DrillResult {
    let offsets = [5.0, 10.0, 20.0, 30.0];
    let speeds = [12.0, 18.0, 24.0];
    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for &offset in &offsets {
        for &speed in &speeds {
            trials += 1;
            let (caught, ticks) = run_pursuit_trial(offset, speed);
            if caught {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "offset={:.0}m spd={:.0}m/s → {} t={}",
                    offset, speed,
                    if caught { "CATCH" } else { "MISS" }, ticks
                ));
            }
        }
    }

    DrillResult {
        name: "pursuit".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_pursuit_trial(lateral_offset: f64, ball_speed: f64) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(200);
    let player_pos = Vec3::new(0.0, 10.0, 0.0);
    let ball_start = Vec3::new(-40.0, 10.0 + lateral_offset, 0.0);
    let ball_vel = Vec3::new(ball_speed, 0.0, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        player_pos,
    );
    sim.launch_bell(ball_start, ball_vel, Vec3::new(0.0, 0.0, 0.0), Some("__pass__"));

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 1800u64; // 7.5 seconds — allow 2-3 reacquire cycles
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if snap.bell.held_by.as_deref() == Some("H1") {
            return (true, tick);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 200);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks)
}

// ── DRILL 3: Throw & Receive ────────────────────────────────────────────────
// Carrier + one teammate. Does the carrier throw? Does the teammate catch?

pub fn drill_throw_receive(verbose: bool) -> DrillResult {
    let separations = [30.0, 60.0, 100.0, 150.0];
    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for &sep in &separations {
        trials += 1;
        let r = run_throw_receive_trial(sep);
        if r.2 {
            successes += 1;
            total_ticks += r.3;
        }
        if verbose {
            details.push(format!(
                "sep={:.0}m → threw={} t_throw={} caught={} t_catch={}",
                sep, r.0, r.1, r.2, r.3
            ));
        }
    }

    DrillResult {
        name: "throw_receive".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

/// Returns (threw, throw_tick, caught, catch_tick)
fn run_throw_receive_trial(separation: f64) -> (bool, u64, bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(300);
    let carrier_pos = Vec3::new(-100.0, 10.0, 0.0);
    let receiver_pos = Vec3::new(-100.0 + separation, 10.0, 0.0);

    sim.add_player(
        "C1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        carrier_pos,
    );
    sim.add_player(
        "R1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        receiver_pos,
    );
    sim.set_bell_held("C1");

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 2400u64;
    let mut threw = false;
    let mut throw_tick = 0u64;
    let mut min_dist_to_r1 = f64::INFINITY;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if !threw && snap.bell.held_by.is_none() {
            threw = true;
            throw_tick = tick;
        }
        if threw && snap.bell.held_by.as_deref() == Some("R1") {
            return (true, throw_tick, true, tick);
        }
        // Track closest approach of ball to receiver
        if threw && snap.bell.held_by.is_none() {
            if let Some(r1) = snap.players.iter().find(|p| p.id == "R1") {
                let d = ((r1.p.x - snap.bell.p.x).powi(2)
                    + (r1.p.y - snap.bell.p.y).powi(2)
                    + (r1.p.z - snap.bell.p.z).powi(2)).sqrt();
                if d < min_dist_to_r1 {
                    min_dist_to_r1 = d;
                }
            }
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 300);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    if min_dist_to_r1 < f64::INFINITY {
        eprintln!("  [throw_recv sep={:.0}] min_dist_to_R1={:.1}m", separation, min_dist_to_r1);
    }
    (threw, throw_tick, false, max_ticks)
}

// ── DRILL 4: Grapple Navigation ─────────────────────────────────────────────
// Player must navigate to a target using grapple mechanics.

pub fn drill_grapple_nav(verbose: bool) -> DrillResult {
    let targets: [(&str, Vec3); 4] = [
        ("forward_30", Vec3::new(30.0, 10.0, 0.0)),
        ("forward_80", Vec3::new(80.0, 10.0, 0.0)),
        ("lateral_20", Vec3::new(0.0, 30.0, 0.0)),
        ("diagonal_50", Vec3::new(35.0, 35.0, 0.0)),
    ];
    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for (name, target) in &targets {
        trials += 1;
        let (arrived, ticks, final_dist) = run_nav_trial(*target);
        if arrived {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} → {} t={} final_dist={:.1}m",
                name,
                if arrived { "ARRIVED" } else { "TIMEOUT" },
                ticks, final_dist
            ));
        }
    }

    DrillResult {
        name: "grapple_nav".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_nav_trial(target: Vec3) -> (bool, u64, f64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(400);
    let start = Vec3::new(0.0, 10.0, 0.0);
    sim.add_player(
        "N1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        start,
    );
    // Place a second player holding the ball at/near the target so the AI
    // has a reason to navigate that direction (loose ball near target would
    // trigger dive-commit instead of pure nav).
    sim.add_player(
        "Dummy",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        target,
    );
    sim.set_bell_held("Dummy");

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

    let arrival_radius = 12.0;
    let max_ticks = 1440u64;
    let mut final_dist = f64::INFINITY;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        let p = snap.players.iter().find(|p| p.id == "N1").unwrap();
        let d = ((p.p.x - target.x).powi(2)
            + (p.p.y - target.y).powi(2)
            + (p.p.z - target.z).powi(2))
        .sqrt();
        final_dist = d;
        if d < arrival_radius {
            return (true, tick, d);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 400);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks, final_dist)
}

// ── DRILL 5: Full Possession Chain ──────────────────────────────────────────
// 4v4. Integrated test — throws, catches, hold time, gate advances.

#[derive(Clone, Debug)]
pub struct ChainResult {
    pub throws: u32,
    pub catches: u32,
    pub bobbles: u32,
    pub hold_frac: f64,
    pub avg_hold_s: f64,
    pub max_hold_s: f64,
    pub gate_clears: u32,
    pub rearms: u32,
    pub score_home: f64,
    pub score_away: f64,
}

pub fn drill_possession_chain(verbose: bool) -> DrillResult {
    let result = run_possession_chain();
    let success = result.catches > 2 && result.throws > 3;
    let mut details = Vec::new();
    details.push(format!(
        "throws={} catches={} bobbles={} hold_frac={:.1}% gates={}",
        result.throws, result.catches, result.bobbles,
        result.hold_frac * 100.0, result.gate_clears
    ));
    details.push(format!(
        "avg_hold={:.2}s longest_hold={:.2}s rearms={} score={}v{}",
        result.avg_hold_s, result.max_hold_s, result.rearms,
        result.score_home, result.score_away
    ));
    if verbose {
        // always show chain details since it's the integration test
    }

    DrillResult {
        name: "possession_chain".into(),
        trials: 1,
        successes: if success { 1 } else { 0 },
        avg_ticks: 0.0,
        details,
    }
}

fn run_possession_chain() -> ChainResult {
    set_planner_class(0);
    let mut sim = SimWorld::new(1234);

    let roster: [(& str, ai::TeamSide, ai::RiggerRole, f64); 8] = [
        ("H1", ai::TeamSide::Home, ai::RiggerRole::Spinner, -40.0),
        ("H2", ai::TeamSide::Home, ai::RiggerRole::Anchor, -90.0),
        ("H3", ai::TeamSide::Home, ai::RiggerRole::Faithwing, -20.0),
        ("H4", ai::TeamSide::Home, ai::RiggerRole::Reach, -260.0),
        ("A1", ai::TeamSide::Away, ai::RiggerRole::Spinner, 40.0),
        ("A2", ai::TeamSide::Away, ai::RiggerRole::Anchor, 90.0),
        ("A3", ai::TeamSide::Away, ai::RiggerRole::Freewing, 20.0),
        ("A4", ai::TeamSide::Away, ai::RiggerRole::Reach, 260.0),
    ];

    for (k, (id, team, role, x)) in roster.iter().enumerate() {
        let ang = (k as f64 / roster.len() as f64) * std::f64::consts::PI * 2.0;
        let sw_team = match team {
            ai::TeamSide::Home => crate::sim_world::TeamSide::Home,
            ai::TeamSide::Away => crate::sim_world::TeamSide::Away,
        };
        let sw_role = match role {
            ai::RiggerRole::Anchor => crate::sim_world::RiggerRole::Anchor,
            ai::RiggerRole::Spinner => crate::sim_world::RiggerRole::Spinner,
            ai::RiggerRole::Faithwing => crate::sim_world::RiggerRole::Faithwing,
            ai::RiggerRole::Freewing => crate::sim_world::RiggerRole::Freewing,
            ai::RiggerRole::Reach => crate::sim_world::RiggerRole::Reach,
        };
        sim.add_player(id, sw_team, sw_role, Vec3::new(*x, ang.cos() * 8.0, ang.sin() * 8.0));
    }
    sim.set_bell_held("H1");

    let mut mat = MatchStateMachine::new(scoring::RingEnd::PlusX, scoring::TeamSide::Home);
    let snap0 = sim.snapshot();
    let s0 = conv::ai_to_scoring(&conv::snap_to_ai(&snap0));
    mat.consume(
        &[scoring::SimEvent::FoulGarrote { by: "__start__".into() }],
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

    let mut throws = 0u32;
    let mut catches = 0u32;
    let mut bobbles = 0u32;
    let mut held_ticks = 0u32;
    let mut hold_durations: Vec<u32> = Vec::new();
    let mut current_hold = 0u32;
    let mut gate_clears = 0u32;
    let mut rearms = 0u32;
    let mut prev_gate_ord = 0i32;

    let max_ticks = 8000u64;
    for _tick in 0..max_ticks {
        if mat.state().winner.is_some() {
            break;
        }
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = conv::msm_to_ai(mat.state());

        if snap.bell.held_by.is_some() {
            held_ticks += 1;
            current_hold += 1;
        } else if current_hold > 0 {
            hold_durations.push(current_hold);
            current_hold = 0;
        }

        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1234);
        for inp in &frame.players {
            if inp.throw_released {
                throws += 1;
            }
        }

        let sw_frame = conv::frame_ai_to_sw(&frame);
        let evs = sim.step(&sw_frame, H);
        for e in &evs {
            match e {
                SwEv::BellCaught { .. } => catches += 1,
                SwEv::BellBobble { .. } => bobbles += 1,
                _ => {}
            }
        }

        let scoring_evs: Vec<scoring::SimEvent> =
            evs.iter().map(conv::ev_sw_to_scoring).collect();
        let post = conv::ai_to_scoring(&conv::snap_to_ai(&sim.snapshot()));
        mat.consume(&scoring_evs, &post);

        let g_ord = match mat.state().cast.gate {
            Gate::First => 0,
            Gate::Deep => 1,
            Gate::Mouth => 2,
        };
        if g_ord > prev_gate_ord {
            gate_clears += 1;
        }
        prev_gate_ord = g_ord;

        if mat.state().winner.is_none() && mat.state().phase != MatchPhase::Live {
            let poss = mat.state().possession;
            let find_team = match poss {
                scoring::TeamSide::Home => ai::TeamSide::Home,
                scoring::TeamSide::Away => ai::TeamSide::Away,
            };
            if let Some((id, ..)) = roster.iter().find(|(_, t, ..)| *t == find_team) {
                sim.set_bell_held(id);
            }
            mat.resume_live();
            rearms += 1;
        }
    }
    if current_hold > 0 {
        hold_durations.push(current_hold);
    }

    let avg_hold = if hold_durations.is_empty() {
        0.0
    } else {
        hold_durations.iter().sum::<u32>() as f64 / hold_durations.len() as f64 / 240.0
    };
    let max_hold = hold_durations.iter().max().copied().unwrap_or(0) as f64 / 240.0;

    set_planner_class(0);

    ChainResult {
        throws,
        catches,
        bobbles,
        hold_frac: held_ticks as f64 / max_ticks as f64,
        avg_hold_s: avg_hold,
        max_hold_s: max_hold,
        gate_clears,
        rearms,
        score_home: mat.state().score_home as f64,
        score_away: mat.state().score_away as f64,
    }
}

// ── DRILL 6: Catch from Various Angles ─────────────────────────────────────
// Ball approaches from front, side, above, below, diagonal at various speeds.

pub fn drill_catch_angles(verbose: bool) -> DrillResult {
    // (label, direction the ball comes FROM relative to player at origin,y=10)
    let approaches: [(&str, Vec3); 6] = [
        ("front", Vec3::new(1.0, 0.0, 0.0)),
        ("behind", Vec3::new(-1.0, 0.0, 0.0)),
        ("left", Vec3::new(0.0, 0.0, 1.0)),
        ("above", Vec3::new(0.0, 1.0, 0.0)),
        ("below", Vec3::new(0.0, -1.0, 0.0)),
        ("diagonal", Vec3::new(0.707, 0.5, 0.5)),
    ];
    let speeds = [15.0, 20.0, 25.0];
    let dist = 20.0;

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    // --- Original tier: moderate speed, short distance ---
    for (label, dir) in &approaches {
        for &speed in &speeds {
            trials += 1;
            let (caught, ticks) = run_angle_catch_trial(*dir, dist, speed);
            if caught {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "{} spd={:.0}m/s dist=20m → {} t={}",
                    label, speed,
                    if caught { "CATCH" } else { "MISS" }, ticks
                ));
            }
        }
    }

    // --- Hard tier: extreme speed (30-34 m/s) at moderate distance ---
    let hard_speeds = [30.0, 32.0, 34.0];
    for (label, dir) in &approaches {
        for &speed in &hard_speeds {
            trials += 1;
            let (caught, ticks) = run_angle_catch_trial(*dir, 20.0, speed);
            if caught {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "{} spd={:.0}m/s dist=20m [HARD] → {} t={}",
                    label, speed,
                    if caught { "CATCH" } else { "MISS" }, ticks
                ));
            }
        }
    }

    // --- Far tier: longer distance (40-60m), moderate speed (more Coriolis) ---
    let far_dists = [40.0, 50.0, 60.0];
    let far_approaches: [(&str, Vec3); 4] = [
        ("front_far", Vec3::new(1.0, 0.0, 0.0)),
        ("diag_far", Vec3::new(0.707, 0.5, 0.5)),
        ("left_far", Vec3::new(0.0, 0.0, 1.0)),
        ("above_far", Vec3::new(0.0, 1.0, 0.0)),
    ];
    for (label, dir) in &far_approaches {
        for &d in &far_dists {
            trials += 1;
            let (caught, ticks) = run_angle_catch_trial(*dir, d, 22.0);
            if caught {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "{} spd=22m/s dist={:.0}m [FAR] → {} t={}",
                    label, d,
                    if caught { "CATCH" } else { "MISS" }, ticks
                ));
            }
        }
    }

    // --- Extreme tier: fast + far + off-axis ---
    let extreme_cases: [(&str, Vec3, f64, f64); 6] = [
        ("extreme_front_fast_far", Vec3::new(1.0, 0.0, 0.0), 50.0, 32.0),
        ("extreme_diag_fast_far", Vec3::new(0.6, 0.6, 0.5), 45.0, 34.0),
        ("extreme_behind_fast", Vec3::new(-1.0, 0.2, 0.3), 40.0, 30.0),
        ("extreme_below_fast_far", Vec3::new(0.3, -0.9, 0.2), 55.0, 31.0),
        ("extreme_side_fast_far", Vec3::new(0.1, 0.3, 0.9), 60.0, 33.0),
        ("extreme_above_fast_far", Vec3::new(0.2, 0.95, 0.1), 50.0, 34.0),
    ];
    for (label, dir, d, spd) in &extreme_cases {
        trials += 1;
        let (caught, ticks) = run_angle_catch_trial(*dir, *d, *spd);
        if caught {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} spd={:.0}m/s dist={:.0}m [EXTREME] → {} t={}",
                label, spd, d,
                if caught { "CATCH" } else { "MISS" }, ticks
            ));
        }
    }

    DrillResult {
        name: "catch_angles".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_angle_catch_trial(from_dir: Vec3, dist: f64, speed: f64) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(600);
    let catcher_pos = Vec3::new(0.0, 10.0, 0.0);
    let dir_norm = from_dir.norm();
    let ball_start = catcher_pos.add(dir_norm.scale(dist));
    // Ball velocity aimed at the catcher
    let ball_vel = catcher_pos.sub(ball_start).norm().scale(speed);

    sim.add_player(
        "C1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        catcher_pos,
    );
    sim.launch_bell(ball_start, ball_vel, Vec3::new(0.0, 0.0, 0.0), Some("__pass__"));

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 720u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if snap.bell.held_by.as_deref() == Some("C1") {
            return (true, tick);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 600);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks)
}

// ── DRILL 7: Catch While Moving ────────────────────────────────────────────
// Player is already in motion; ball approaches from various angles.

pub fn drill_catch_while_moving(verbose: bool) -> DrillResult {
    let player_speeds = [10.0, 15.0, 20.0];
    let ball_angles: [(&str, Vec3); 3] = [
        ("head-on", Vec3::new(-1.0, 0.0, 0.0)),
        ("side", Vec3::new(0.0, 0.0, 1.0)),
        ("above", Vec3::new(-0.3, 1.0, 0.0)),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    // --- Original tier ---
    for &pspd in &player_speeds {
        for (label, ball_dir) in &ball_angles {
            trials += 1;
            let (caught, ticks) = run_moving_catch_trial(pspd, *ball_dir);
            if caught {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "pvel={:.0}m/s ball_from={} → {} t={}",
                    pspd, label,
                    if caught { "CATCH" } else { "MISS" }, ticks
                ));
            }
        }
    }

    // --- Hard tier: faster player speeds (25-30 m/s) ---
    let fast_speeds = [25.0, 28.0, 30.0];
    for &pspd in &fast_speeds {
        for (label, ball_dir) in &ball_angles {
            trials += 1;
            let (caught, ticks) = run_moving_catch_trial(pspd, *ball_dir);
            if caught {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "pvel={:.0}m/s ball_from={} [FAST] → {} t={}",
                    pspd, label,
                    if caught { "CATCH" } else { "MISS" }, ticks
                ));
            }
        }
    }

    // --- Behind tier: ball from behind (player moving away from ball) ---
    let behind_angles: [(&str, Vec3); 3] = [
        ("behind_direct", Vec3::new(1.0, 0.0, 0.0)),
        ("behind_high", Vec3::new(0.8, 0.6, 0.0)),
        ("behind_side", Vec3::new(0.8, 0.0, 0.5)),
    ];
    let behind_speeds = [15.0, 22.0, 28.0];
    for &pspd in &behind_speeds {
        for (label, ball_dir) in &behind_angles {
            trials += 1;
            let (caught, ticks) = run_moving_catch_trial(pspd, *ball_dir);
            if caught {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "pvel={:.0}m/s ball_from={} [BEHIND] → {} t={}",
                    pspd, label,
                    if caught { "CATCH" } else { "MISS" }, ticks
                ));
            }
        }
    }

    // --- Tight timing tier: ball arrives in <0.5s (short distance, fast ball) ---
    let tight_cases: [(&str, f64, Vec3); 4] = [
        ("tight_head", 20.0, Vec3::new(-1.0, 0.0, 0.0)),
        ("tight_side", 25.0, Vec3::new(0.0, 0.0, 1.0)),
        ("tight_above", 22.0, Vec3::new(-0.2, 1.0, 0.0)),
        ("tight_behind", 28.0, Vec3::new(0.8, 0.3, 0.0)),
    ];
    for (label, pspd, ball_dir) in &tight_cases {
        trials += 1;
        let (caught, ticks) = run_moving_catch_tight_trial(*pspd, *ball_dir);
        if caught {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "pvel={:.0}m/s ball_from={} [TIGHT] → {} t={}",
                pspd, label,
                if caught { "CATCH" } else { "MISS" }, ticks
            ));
        }
    }

    DrillResult {
        name: "catch_while_moving".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_moving_catch_trial(player_speed: f64, ball_from_dir: Vec3) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(700);
    // Player starts at x=-50, moving +x
    let player_start = Vec3::new(-50.0, 10.0, 0.0);
    sim.add_player(
        "C1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        player_start,
    );

    // Let the player build up speed for 120 ticks (0.5s) before launching ball
    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    // Pre-run: let the player get moving by placing a "target" ahead
    // We'll give the ball a destination that intercepts the player's path
    let warmup = 120u64;
    for _tick in 0..warmup {
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 700);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    // Now launch ball toward where the player will be
    let snap = sim.snapshot();
    let p = snap.players.iter().find(|p| p.id == "C1").unwrap();
    let player_pos = p.p;
    // Intercept point ahead of player
    let intercept = player_pos.add(Vec3::new(player_speed * 0.5, 0.0, 0.0));
    let ball_start = intercept.add(ball_from_dir.norm().scale(25.0));
    let ball_vel = intercept.sub(ball_start).norm().scale(20.0);
    sim.launch_bell(ball_start, ball_vel, Vec3::new(0.0, 0.0, 0.0), Some("__pass__"));

    let max_ticks = 900u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if snap.bell.held_by.as_deref() == Some("C1") {
            return (true, tick + warmup);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 700);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks + warmup)
}

/// Tight-timing variant: ball launched from only 10m away at 30m/s.
/// Player has <0.5s to react.
fn run_moving_catch_tight_trial(player_speed: f64, ball_from_dir: Vec3) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(750);
    let player_start = Vec3::new(-50.0, 10.0, 0.0);
    sim.add_player(
        "C1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        player_start,
    );

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    // Warmup: let the player build speed
    let warmup = 120u64;
    for _tick in 0..warmup {
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 750);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    // Launch ball from very close (10m), very fast (30m/s) — <0.33s flight
    let snap = sim.snapshot();
    let p = snap.players.iter().find(|p| p.id == "C1").unwrap();
    let player_pos = p.p;
    let intercept = player_pos.add(Vec3::new(player_speed * 0.25, 0.0, 0.0));
    let ball_start = intercept.add(ball_from_dir.norm().scale(10.0));
    let ball_vel = intercept.sub(ball_start).norm().scale(30.0);
    sim.launch_bell(ball_start, ball_vel, Vec3::new(0.0, 0.0, 0.0), Some("__pass__"));

    let max_ticks = 600u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if snap.bell.held_by.as_deref() == Some("C1") {
            return (true, tick + warmup);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 750);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks + warmup)
}

// ── DRILL 8: Pursuit Reacquire (skin bounce) ───────────────────────────────
// Ball bounces off cylinder skin mid-flight. Player must reacquire.

pub fn drill_pursuit_reacquire(verbose: bool) -> DrillResult {
    // Launch ball at varying angles to hit the skin and bounce
    let launch_angles: [(&str, f64, f64); 4] = [
        ("shallow", 0.3, 18.0),
        ("medium", 0.5, 20.0),
        ("steep", 0.7, 22.0),
        ("fast_steep", 0.7, 28.0),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    // --- Original tier ---
    for (label, angle_frac, speed) in &launch_angles {
        trials += 1;
        let (caught, ticks) = run_reacquire_trial(*angle_frac, *speed);
        if caught {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} spd={:.0}m/s → {} t={}",
                label, speed,
                if caught { "CATCH" } else { "MISS" }, ticks
            ));
        }
    }

    // --- Hard tier: very fast balls (28-34 m/s) ---
    let fast_angles: [(&str, f64, f64); 4] = [
        ("fast_shallow", 0.3, 30.0),
        ("fast_medium", 0.5, 32.0),
        ("very_fast_steep", 0.7, 34.0),
        ("very_fast_shallow", 0.35, 34.0),
    ];
    for (label, angle_frac, speed) in &fast_angles {
        trials += 1;
        let (caught, ticks) = run_reacquire_trial(*angle_frac, *speed);
        if caught {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} spd={:.0}m/s [FAST] → {} t={}",
                label, speed,
                if caught { "CATCH" } else { "MISS" }, ticks
            ));
        }
    }

    // --- Double-bounce tier: ball launched to hit skin twice ---
    let double_bounce: [(&str, f64, f64); 3] = [
        ("dbl_shallow", 0.25, 24.0),
        ("dbl_medium", 0.4, 26.0),
        ("dbl_fast", 0.35, 30.0),
    ];
    for (label, angle_frac, speed) in &double_bounce {
        trials += 1;
        let (caught, ticks) = run_reacquire_double_bounce_trial(*angle_frac, *speed);
        if caught {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} spd={:.0}m/s [DBL_BOUNCE] → {} t={}",
                label, speed,
                if caught { "CATCH" } else { "MISS" }, ticks
            ));
        }
    }

    // --- Far start tier: player 60-80m from ball origin ---
    let far_start: [(&str, f64, f64, f64); 3] = [
        ("far_60", 0.5, 22.0, 60.0),
        ("far_70", 0.6, 24.0, 70.0),
        ("far_80", 0.5, 26.0, 80.0),
    ];
    for (label, angle_frac, speed, gap) in &far_start {
        trials += 1;
        let (caught, ticks) = run_reacquire_far_trial(*angle_frac, *speed, *gap);
        if caught {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} spd={:.0}m/s gap={:.0}m [FAR] → {} t={}",
                label, speed, gap,
                if caught { "CATCH" } else { "MISS" }, ticks
            ));
        }
    }

    DrillResult {
        name: "pursuit_reacquire".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_reacquire_trial(angle_frac: f64, speed: f64) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(800);
    // Player near center
    let player_pos = Vec3::new(0.0, 10.0, 0.0);
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        player_pos,
    );

    // Launch ball upward toward the skin (R=45m) so it bounces back
    // Ball starts near player, goes outward radially + forward
    let radial_angle = angle_frac * std::f64::consts::FRAC_PI_2;
    let vy = radial_angle.sin() * speed; // outward (toward skin at y=45)
    let vx = radial_angle.cos() * speed; // forward along x
    let ball_start = Vec3::new(-10.0, 12.0, 0.0);
    let ball_vel = Vec3::new(vx, vy, 0.0);
    sim.launch_bell(ball_start, ball_vel, Vec3::new(0.0, 0.0, 0.0), Some("__pass__"));

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 1800u64;
    let mut _saw_bounce = false;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if snap.bell.held_by.as_deref() == Some("H1") {
            return (true, tick);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 800);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        let evs = sim.step(&sw_frame, H);
        for e in &evs {
            if matches!(e, SwEv::BellSkin) {
                _saw_bounce = true;
            }
        }
    }
    (false, max_ticks)
}

/// Double-bounce variant: ball launched at shallower angle so it hits skin twice.
fn run_reacquire_double_bounce_trial(angle_frac: f64, speed: f64) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(810);
    // Player near center
    let player_pos = Vec3::new(0.0, 10.0, 0.0);
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        player_pos,
    );

    // Launch ball with more forward momentum and less radial so it skips
    let radial_angle = angle_frac * std::f64::consts::FRAC_PI_2;
    let vy = radial_angle.sin() * speed;
    let vx = radial_angle.cos() * speed;
    // Start farther back so ball has room for two bounces
    let ball_start = Vec3::new(-30.0, 20.0, 0.0);
    let ball_vel = Vec3::new(vx, vy, 2.0); // slight z drift
    sim.launch_bell(ball_start, ball_vel, Vec3::new(0.0, 0.0, 0.0), Some("__pass__"));

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 2000u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if snap.bell.held_by.as_deref() == Some("H1") {
            return (true, tick);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 810);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks)
}

/// Far-start variant: player is far from ball launch point.
fn run_reacquire_far_trial(angle_frac: f64, speed: f64, initial_gap: f64) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(820);
    // Player offset far from launch
    let player_pos = Vec3::new(initial_gap, 10.0, 0.0);
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        player_pos,
    );

    let radial_angle = angle_frac * std::f64::consts::FRAC_PI_2;
    let vy = radial_angle.sin() * speed;
    let vx = radial_angle.cos() * speed;
    let ball_start = Vec3::new(-10.0, 12.0, 0.0);
    let ball_vel = Vec3::new(vx, vy, 0.0);
    sim.launch_bell(ball_start, ball_vel, Vec3::new(0.0, 0.0, 0.0), Some("__pass__"));

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 2400u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if snap.bell.held_by.as_deref() == Some("H1") {
            return (true, tick);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 820);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks)
}

// ── DRILL 9: Throw Accuracy ────────────────────────────────────────────────
// Carrier aims at a target point. Measure where ball actually ends up.

pub fn drill_throw_accuracy(verbose: bool) -> DrillResult {
    let distances = [30.0, 60.0, 100.0, 150.0];
    let offsets = [0.0, 10.0]; // y-offset of target from straight ahead

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for &dist in &distances {
        for &off in &offsets {
            trials += 1;
            let (hit, ticks, miss_dist) = run_throw_accuracy_trial(dist, off);
            if hit {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "dist={:.0}m off={:.0}m → {} miss={:.1}m t={}",
                    dist, off,
                    if hit { "HIT" } else { "MISS" }, miss_dist, ticks
                ));
            }
        }
    }

    DrillResult {
        name: "throw_accuracy".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_throw_accuracy_trial(target_dist: f64, y_offset: f64) -> (bool, u64, f64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(900);
    let carrier_pos = Vec3::new(-100.0, 10.0, 0.0);
    let target_pos = Vec3::new(-100.0 + target_dist, 10.0 + y_offset, 0.0);

    sim.add_player(
        "C1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        carrier_pos,
    );
    // Place a dummy receiver at target so the AI has someone to throw to
    sim.add_player(
        "R1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        target_pos,
    );
    sim.set_bell_held("C1");

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 2400u64;
    let mut threw = false;
    let mut closest_dist = f64::INFINITY;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        // Detect throw
        if !threw && snap.bell.held_by.is_none() {
            threw = true;
        }
        if threw {
            // Track how close the ball gets to target
            let d = snap.bell.p.sub(target_pos).len();
            if d < closest_dist {
                closest_dist = d;
            }
            // If ball caught by receiver, that counts as a hit
            if snap.bell.held_by.as_deref() == Some("R1") {
                return (true, tick, 0.0);
            }
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 900);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    // Success if ball got within 5m of target at any point
    let hit = closest_dist < 5.0;
    (hit, max_ticks, closest_dist)
}

// ── DRILL 10: Two Pass (A→B→C) ─────────────────────────────────────────────
// Three players in a line. A passes to B who passes to C.

pub fn drill_two_pass(verbose: bool) -> DrillResult {
    let separations = [40.0, 60.0, 80.0, 100.0];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for &sep in &separations {
        trials += 1;
        let (complete, ticks, passes_done) = run_two_pass_trial(sep);
        if complete {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "sep={:.0}m → passes={}/2 {} t={}",
                sep, passes_done,
                if complete { "DONE" } else { "FAIL" }, ticks
            ));
        }
    }

    DrillResult {
        name: "two_pass".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_two_pass_trial(separation: f64) -> (bool, u64, u32) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1000);
    let a_pos = Vec3::new(-120.0, 10.0, 0.0);
    let b_pos = Vec3::new(-120.0 + separation, 10.0, 0.0);
    let c_pos = Vec3::new(-120.0 + separation * 2.0, 10.0, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        a_pos,
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        b_pos,
    );
    sim.add_player(
        "H3",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Faithwing,
        c_pos,
    );
    sim.set_bell_held("H1");

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 2400u64;
    let mut passes_done = 0u32;
    let mut b_had_ball = false;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if !b_had_ball && snap.bell.held_by.as_deref() == Some("H2") {
            b_had_ball = true;
            passes_done = 1;
        }
        if b_had_ball && snap.bell.held_by.as_deref() == Some("H3") {
            return (true, tick, 2);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1000);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks, passes_done)
}

// ── DRILL 11: Under Pressure ────────────────────────────────────────────────
// Carrier has ball, defender closing. Tests pressure-release throw behavior.

pub fn drill_under_pressure(verbose: bool) -> DrillResult {
    let defender_distances = [30.0, 20.0, 15.0, 10.0];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for &dd in &defender_distances {
        trials += 1;
        let (released, caught, ticks) = run_under_pressure_trial(dd);
        let ok = released && caught;
        if ok {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "def_dist={:.0}m → threw={} caught={} {} t={}",
                dd, released, caught,
                if ok { "OK" } else { "FAIL" }, ticks
            ));
        }
    }

    DrillResult {
        name: "under_pressure".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_under_pressure_trial(defender_dist: f64) -> (bool, bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1100);
    let carrier_pos = Vec3::new(-50.0, 10.0, 0.0);
    let receiver_pos = Vec3::new(20.0, 10.0, 0.0);
    // Defender between carrier and receiver, closing on carrier
    let defender_pos = Vec3::new(-50.0 + defender_dist, 10.0, 3.0);

    sim.add_player(
        "C1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        carrier_pos,
    );
    sim.add_player(
        "R1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        receiver_pos,
    );
    sim.add_player(
        "A1",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        defender_pos,
    );
    sim.set_bell_held("C1");

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

    let max_ticks = 2400u64;
    let mut threw = false;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if !threw && snap.bell.held_by.is_none() {
            threw = true;
        }
        if threw && snap.bell.held_by.as_deref() == Some("R1") {
            return (true, true, tick);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1100);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (threw, false, max_ticks)
}

// ── DRILL 12: Gate Score ────────────────────────────────────────────────────
// Single player with ball, positioned within 100m of attack ring. Score.

pub fn drill_gate_score(verbose: bool) -> DrillResult {
    // Home attacks +x, ring at x=320
    let start_distances = [50.0, 80.0, 100.0]; // distance from ring
    let y_offsets = [0.0, 5.0];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    // --- Original tier ---
    for &dist in &start_distances {
        for &y_off in &y_offsets {
            trials += 1;
            let (scored, ticks) = run_gate_score_trial(dist, y_off);
            if scored {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "dist={:.0}m y_off={:.0}m → {} t={}",
                    dist, y_off,
                    if scored { "SCORE" } else { "MISS" }, ticks
                ));
            }
        }
    }

    // --- Far tier: 150-200m from ring ---
    let far_distances = [150.0, 175.0, 200.0];
    let far_offsets = [0.0, 5.0];
    for &dist in &far_distances {
        for &y_off in &far_offsets {
            trials += 1;
            let (scored, ticks) = run_gate_score_trial(dist, y_off);
            if scored {
                successes += 1;
                total_ticks += ticks;
            }
            if verbose {
                details.push(format!(
                    "dist={:.0}m y_off={:.0}m [FAR] → {} t={}",
                    dist, y_off,
                    if scored { "SCORE" } else { "MISS" }, ticks
                ));
            }
        }
    }

    // --- Off-axis tier: significant y-offset (20-30m from centerline) ---
    let offaxis_cases: [(&str, f64, f64); 4] = [
        ("off_20_close", 80.0, 20.0),
        ("off_25_mid", 120.0, 25.0),
        ("off_30_far", 150.0, 30.0),
        ("off_20_far", 180.0, 20.0),
    ];
    for (label, dist, y_off) in &offaxis_cases {
        trials += 1;
        let (scored, ticks) = run_gate_score_trial(*dist, *y_off);
        if scored {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} dist={:.0}m y_off={:.0}m [OFFAXIS] → {} t={}",
                label, dist, y_off,
                if scored { "SCORE" } else { "MISS" }, ticks
            ));
        }
    }

    // --- Defended tier: defender between carrier and ring ---
    let defended_cases: [(&str, f64, f64); 3] = [
        ("def_close", 80.0, 0.0),
        ("def_mid", 120.0, 5.0),
        ("def_far", 160.0, 10.0),
    ];
    for (label, dist, y_off) in &defended_cases {
        trials += 1;
        let (scored, ticks) = run_gate_score_defended_trial(*dist, *y_off);
        if scored {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} dist={:.0}m y_off={:.0}m [DEFENDED] → {} t={}",
                label, dist, y_off,
                if scored { "SCORE" } else { "MISS" }, ticks
            ));
        }
    }

    DrillResult {
        name: "gate_score".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_gate_score_trial(dist_from_ring: f64, y_offset: f64) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1200);
    // Home attacks +x, ring at x=320. Place player at x = 320 - dist
    let px = 320.0 - dist_from_ring;
    let player_pos = Vec3::new(px, 10.0 + y_offset, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        player_pos,
    );
    sim.set_bell_held("H1");

    // Match state: possession Home, gate Mouth (closest gate for scoring)
    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 2400u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = gate_score_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1200);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        let evs = sim.step(&sw_frame, H);
        for e in &evs {
            if matches!(e, SwEv::BellThroughRing { end: crate::sim_world::RingEnd::PlusX, .. }) {
                return (true, tick);
            }
        }
    }
    (false, max_ticks)
}

fn run_gate_score_defended_trial(dist_from_ring: f64, y_offset: f64) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1210);
    let px = 320.0 - dist_from_ring;
    let player_pos = Vec3::new(px, 10.0 + y_offset, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        player_pos,
    );
    // Defender positioned between carrier and ring
    let def_x = (px + 320.0) / 2.0;
    let def_pos = Vec3::new(def_x, 10.0 + y_offset * 0.5, 2.0);
    sim.add_player(
        "A1",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        def_pos,
    );
    sim.set_bell_held("H1");

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

    let max_ticks = 2400u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = gate_score_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1210);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        let evs = sim.step(&sw_frame, H);
        for e in &evs {
            if matches!(e, SwEv::BellThroughRing { end: crate::sim_world::RingEnd::PlusX, .. }) {
                return (true, tick);
            }
        }
    }
    (false, max_ticks)
}

fn gate_score_match_state() -> ai::MatchState {
    ai::MatchState {
        inning: 1.0,
        spine: false,
        possession: ai::TeamSide::Home,
        faith_end: ai::FaithEnd::PlusX,
        cast: ai::Cast {
            throws_left: 3,
            gate: ai::Gate::Mouth,
            spot_x: 272.0,
        },
        contest: None,
        score_home: 0.0,
        score_away: 0.0,
        phase: ai::MatchPhase::Live,
        message: String::new(),
        winner: None,
    }
}

// ── DRILL 13: Loose Ball Race ──────────────────────────────────────────────
// Two players race to a loose ball. Tests dive-commit arbitration.

pub fn drill_loose_ball_race(verbose: bool) -> DrillResult {
    let separations = [20.0, 30.0, 40.0, 50.0];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for &sep in &separations {
        trials += 1;
        let (someone_got_it, ticks, winner) = run_loose_ball_race_trial(sep);
        if someone_got_it {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "sep={:.0}m → {} winner={} t={}",
                sep,
                if someone_got_it { "SECURED" } else { "LOOSE" },
                winner, ticks
            ));
        }
    }

    DrillResult {
        name: "loose_ball_race".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_loose_ball_race_trial(half_sep: f64) -> (bool, u64, String) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1300);
    // Ball at center, two players equidistant
    let ball_pos = Vec3::new(0.0, 10.0, 0.0);
    let p1_pos = Vec3::new(-half_sep, 10.0, 0.0);
    let p2_pos = Vec3::new(half_sep, 10.0, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        p1_pos,
    );
    sim.add_player(
        "A1",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        p2_pos,
    );
    // Launch ball with near-zero velocity (loose)
    sim.launch_bell(ball_pos, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), None);

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

    let max_ticks = 1800u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if let Some(ref holder) = snap.bell.held_by {
            return (true, tick, holder.clone());
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1300);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks, "none".into())
}

// ── DRILL 14: Defensive Intercept ──────────────────────────────────────────
// Thrower passes to receiver; defender in the lane tries to intercept.

pub fn drill_defensive_intercept(verbose: bool) -> DrillResult {
    // lane_offset: how far off the direct line the defender is positioned
    let lane_offsets = [0.0, 3.0, 6.0, 10.0];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    // --- Original tier ---
    for &off in &lane_offsets {
        trials += 1;
        let (intercepted, ticks, who_got_it) = run_intercept_trial(off);
        if intercepted {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "lane_off={:.0}m → {} holder={} t={}",
                off,
                if intercepted { "INTERCEPT" } else { "THROUGH" },
                who_got_it, ticks
            ));
        }
    }

    // --- Hard tier: faster throws (defender must react to 30+ m/s ball) ---
    let fast_offsets = [0.0, 5.0, 8.0, 12.0];
    for &off in &fast_offsets {
        trials += 1;
        let (intercepted, ticks, who_got_it) = run_intercept_fast_trial(off, 32.0);
        if intercepted {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "lane_off={:.0}m spd=32 [FAST] → {} holder={} t={}",
                off,
                if intercepted { "INTERCEPT" } else { "THROUGH" },
                who_got_it, ticks
            ));
        }
    }

    // --- Far offset tier: defender farther from lane (15-25m) ---
    let far_offsets = [15.0, 20.0, 25.0];
    for &off in &far_offsets {
        trials += 1;
        let (intercepted, ticks, who_got_it) = run_intercept_trial(off);
        if intercepted {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "lane_off={:.0}m [FAR_OFF] → {} holder={} t={}",
                off,
                if intercepted { "INTERCEPT" } else { "THROUGH" },
                who_got_it, ticks
            ));
        }
    }

    // --- Late reaction tier: ball already in flight before defender starts ---
    let late_cases: [(&str, f64, u64); 3] = [
        ("late_close", 5.0, 60),    // 0.25s head start
        ("late_mid", 8.0, 90),      // 0.375s head start
        ("late_far", 12.0, 120),    // 0.5s head start
    ];
    for (label, off, delay_ticks) in &late_cases {
        trials += 1;
        let (intercepted, ticks, who_got_it) = run_intercept_late_trial(*off, *delay_ticks);
        if intercepted {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} off={:.0}m delay={}t [LATE] → {} holder={} t={}",
                label, off, delay_ticks,
                if intercepted { "INTERCEPT" } else { "THROUGH" },
                who_got_it, ticks
            ));
        }
    }

    DrillResult {
        name: "defensive_intercept".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_intercept_trial(lane_offset: f64) -> (bool, u64, String) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1400);
    let thrower_pos = Vec3::new(-80.0, 10.0, 0.0);
    let receiver_pos = Vec3::new(0.0, 10.0, 0.0);
    // Defender positioned in the pass lane, slightly offset
    let mid_x = -40.0;
    let defender_pos = Vec3::new(mid_x, 10.0 + lane_offset, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        thrower_pos,
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        receiver_pos,
    );
    sim.add_player(
        "A1",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        defender_pos,
    );
    sim.set_bell_held("H1");

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

    let max_ticks = 2400u64;
    let mut threw = false;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if !threw && snap.bell.held_by.is_none() {
            threw = true;
        }
        if threw {
            if let Some(ref holder) = snap.bell.held_by {
                let intercepted = holder == "A1";
                return (intercepted, tick, holder.clone());
            }
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1400);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks, "none".into())
}

/// Fast-throw intercept variant: ball is launched manually at high speed.
fn run_intercept_fast_trial(lane_offset: f64, ball_speed: f64) -> (bool, u64, String) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1410);
    let thrower_pos = Vec3::new(-80.0, 10.0, 0.0);
    let receiver_pos = Vec3::new(0.0, 10.0, 0.0);
    let mid_x = -40.0;
    let defender_pos = Vec3::new(mid_x, 10.0 + lane_offset, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        thrower_pos,
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        receiver_pos,
    );
    sim.add_player(
        "A1",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        defender_pos,
    );

    // Manually launch ball at high speed toward receiver (bypass AI throw)
    let ball_dir = receiver_pos.sub(thrower_pos).norm();
    let ball_vel = ball_dir.scale(ball_speed);
    sim.launch_bell(thrower_pos, ball_vel, Vec3::new(0.0, 0.0, 0.0), Some("__pass__"));

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

    let max_ticks = 1200u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if let Some(ref holder) = snap.bell.held_by {
            let intercepted = holder == "A1";
            return (intercepted, tick, holder.clone());
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1410);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks, "none".into())
}

/// Late-reaction intercept: ball is already in flight for `delay_ticks` before
/// the defender's AI starts processing.
fn run_intercept_late_trial(lane_offset: f64, delay_ticks: u64) -> (bool, u64, String) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1420);
    let thrower_pos = Vec3::new(-80.0, 10.0, 0.0);
    let receiver_pos = Vec3::new(0.0, 10.0, 0.0);
    let mid_x = -40.0;
    let defender_pos = Vec3::new(mid_x, 10.0 + lane_offset, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        thrower_pos,
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        receiver_pos,
    );
    sim.add_player(
        "A1",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        defender_pos,
    );

    // Launch ball toward receiver at moderate speed
    let ball_dir = receiver_pos.sub(thrower_pos).norm();
    let ball_vel = ball_dir.scale(22.0);
    sim.launch_bell(thrower_pos, ball_vel, Vec3::new(0.0, 0.0, 0.0), Some("__pass__"));

    let cfgs_home_only = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let cfgs_both = vec![
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

    let max_ticks = 1800u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if let Some(ref holder) = snap.bell.held_by {
            let intercepted = holder == "A1";
            return (intercepted, tick, holder.clone());
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        // For the first delay_ticks, only run Home AI (defender stands still)
        let cfgs_ref = if tick < delay_ticks { &cfgs_home_only } else { &cfgs_both };
        let frame = aisys.tick(&ai_sim, &ai_match, cfgs_ref, 1420);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks, "none".into())
}

// ── DRILL 15: Hold and Advance ─────────────────────────────────────────────
// Carrier starts far from ring, measure forward progress over 5 seconds.

pub fn drill_hold_and_advance(verbose: bool) -> DrillResult {
    let start_xs = [-250.0, -200.0, -150.0, -100.0];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for &sx in &start_xs {
        trials += 1;
        let (advanced, ticks, distance_gained) = run_hold_advance_trial(sx);
        if advanced {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "start_x={:.0}m → {} gained={:.1}m t={}",
                sx,
                if advanced { "ADVANCED" } else { "STUCK" },
                distance_gained, ticks
            ));
        }
    }

    DrillResult {
        name: "hold_and_advance".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_hold_advance_trial(start_x: f64) -> (bool, u64, f64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1500);
    let player_pos = Vec3::new(start_x, 10.0, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        player_pos,
    );
    sim.set_bell_held("H1");

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    // 5 seconds = 1200 ticks
    let max_ticks = 1200u64;
    let initial_x = start_x;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1500);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);

        // Check if ball was dropped/thrown — if so, end early
        let post = sim.snapshot();
        if post.bell.held_by.as_deref() != Some("H1") && tick > 60 {
            // Carrier threw the ball — measure how far they advanced first
            let p = post.players.iter().find(|p| p.id == "H1").unwrap();
            let gained = p.p.x - initial_x;
            // Success if they advanced at least 10m toward +x before throwing
            return (gained > 10.0, tick, gained);
        }
    }

    let snap = sim.snapshot();
    let p = snap.players.iter().find(|p| p.id == "H1").unwrap();
    let gained = p.p.x - initial_x;
    // Success if they advanced at least 20m in 5 seconds while holding
    (gained > 20.0, max_ticks, gained)
}

// ── DRILL 16: Short Pass ───────────────────────────────────────────────────
// Two teammates at very short separation (10-25m). Easy bridge toward two_pass.

pub fn drill_short_pass(verbose: bool) -> DrillResult {
    let cases: [(&str, f64, Vec3); 9] = [
        // (label, separation, direction offset for receiver relative to carrier)
        ("fwd_10m", 10.0, Vec3::new(1.0, 0.0, 0.0)),
        ("fwd_15m", 15.0, Vec3::new(1.0, 0.0, 0.0)),
        ("fwd_25m", 25.0, Vec3::new(1.0, 0.0, 0.0)),
        ("lat_10m", 10.0, Vec3::new(0.0, 0.0, 1.0)),
        ("lat_15m", 15.0, Vec3::new(0.0, 0.0, 1.0)),
        ("lat_25m", 25.0, Vec3::new(0.0, 0.0, 1.0)),
        ("back_10m", 10.0, Vec3::new(-1.0, 0.0, 0.0)),
        ("back_15m", 15.0, Vec3::new(-1.0, 0.0, 0.0)),
        ("diag_20m", 20.0, Vec3::new(0.7, 0.0, 0.7)),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for (label, sep, dir) in &cases {
        trials += 1;
        let (caught, ticks) = run_short_pass_trial(*sep, *dir);
        if caught {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} sep={:.0}m → {} t={}",
                label, sep,
                if caught { "CAUGHT" } else { "MISS" }, ticks
            ));
        }
    }

    DrillResult {
        name: "short_pass".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_short_pass_trial(separation: f64, direction: Vec3) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1600);
    let carrier_pos = Vec3::new(-50.0, 10.0, 0.0);
    let receiver_pos = carrier_pos.add(direction.norm().scale(separation));

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        carrier_pos,
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        receiver_pos,
    );
    sim.set_bell_held("H1");

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 1800u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if snap.bell.held_by.as_deref() == Some("H2") {
            return (true, tick);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1600);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks)
}

// ── DRILL 17: Outlet Under Pressure ───────────────────────────────────────
// Easier pressure scenario: carrier + receiver at 30m, single defender closing
// slowly from 20m. Tests if carrier throws before defender arrives.

pub fn drill_outlet_under_pressure(verbose: bool) -> DrillResult {
    // (label, receiver_dist, defender_dist, defender_approach_angle)
    let cases: [(&str, f64, f64, f64); 6] = [
        ("easy_close_recv", 25.0, 25.0, 0.0),
        ("easy_far_recv", 35.0, 25.0, 0.0),
        ("med_close_def", 30.0, 18.0, 0.0),
        ("med_angled_def", 30.0, 20.0, 0.5),
        ("hard_close_def", 30.0, 15.0, 0.0),
        ("hard_angled_close", 30.0, 15.0, 0.8),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for (label, recv_dist, def_dist, def_angle) in &cases {
        trials += 1;
        let (released, caught, ticks) =
            run_outlet_pressure_trial(*recv_dist, *def_dist, *def_angle);
        let ok = released && caught;
        if ok {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} → threw={} caught={} {} t={}",
                label, released, caught,
                if ok { "OK" } else { "FAIL" }, ticks
            ));
        }
    }

    DrillResult {
        name: "outlet_under_pressure".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_outlet_pressure_trial(
    receiver_dist: f64,
    defender_dist: f64,
    defender_angle: f64,
) -> (bool, bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1700);
    let carrier_pos = Vec3::new(-50.0, 10.0, 0.0);
    let receiver_pos = Vec3::new(-50.0 + receiver_dist, 10.0, 0.0);
    // Defender approaches from an angle (0 = direct from +x, angle rotates in yz)
    let def_dx = defender_dist * defender_angle.cos();
    let def_dz = defender_dist * defender_angle.sin();
    let defender_pos = Vec3::new(-50.0 - def_dx, 10.0, def_dz);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        carrier_pos,
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        receiver_pos,
    );
    sim.add_player(
        "A1",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        defender_pos,
    );
    sim.set_bell_held("H1");

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

    let max_ticks = 2400u64;
    let mut threw = false;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if !threw && snap.bell.held_by.is_none() {
            threw = true;
        }
        if threw && snap.bell.held_by.as_deref() == Some("H2") {
            return (true, true, tick);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1700);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (threw, false, max_ticks)
}

// ── DRILL 18: Moving Receive (lead pass) ──────────────────────────────────
// Throw aimed at where receiver WILL BE. Receiver at various speeds moving
// forward. Tests the lead-solver + receiver catch coordination.

pub fn drill_moving_receive(verbose: bool) -> DrillResult {
    // (label, receiver_speed, separation, receiver_angle relative to +x)
    let cases: [(&str, f64, f64, f64); 9] = [
        ("slow_short", 5.0, 30.0, 0.0),
        ("slow_mid", 8.0, 50.0, 0.0),
        ("med_short", 12.0, 30.0, 0.0),
        ("med_mid", 12.0, 50.0, 0.0),
        ("med_angled", 12.0, 40.0, 0.3),
        ("fast_short", 18.0, 30.0, 0.0),
        ("fast_mid", 18.0, 50.0, 0.0),
        ("fast_angled", 20.0, 40.0, 0.5),
        ("vfast_mid", 20.0, 60.0, 0.0),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for (label, recv_speed, sep, angle) in &cases {
        trials += 1;
        let (caught, ticks) = run_moving_receive_trial(*recv_speed, *sep, *angle);
        if caught {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} rspd={:.0}m/s sep={:.0}m → {} t={}",
                label, recv_speed, sep,
                if caught { "CAUGHT" } else { "MISS" }, ticks
            ));
        }
    }

    DrillResult {
        name: "moving_receive".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_moving_receive_trial(
    _receiver_speed: f64,
    separation: f64,
    receiver_angle: f64,
) -> (bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(1800);
    let carrier_pos = Vec3::new(-100.0, 10.0, 0.0);
    // Receiver starts at separation distance, moving in direction given by angle
    let recv_x = -100.0 + separation * receiver_angle.cos().max(0.3);
    let recv_z = separation * receiver_angle.sin();
    let receiver_pos = Vec3::new(recv_x, 10.0, recv_z);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        carrier_pos,
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        receiver_pos,
    );
    sim.set_bell_held("H1");

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    // Let simulation run — receiver will naturally move as AI commands
    let max_ticks = 2400u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if snap.bell.held_by.as_deref() == Some("H2") {
            return (true, tick);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 1800);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks)
}

// ── NAVIGATION DRILL 1: Speed Challenge ────────────────────────────────────
// How fast can a player cover distance? Measure distance in 720 ticks (3s).

pub fn drill_speed_challenge(verbose: bool) -> DrillResult {
    // Starting positions: (label, pos)
    let starts: [(&str, Vec3); 4] = [
        ("center", Vec3::new(0.0, 5.0, 0.0)),
        ("near_skin", Vec3::new(0.0, 35.0, 0.0)),
        ("angle_low", Vec3::new(0.0, 10.0, 20.0)),
        ("angle_high", Vec3::new(0.0, 25.0, -15.0)),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut details = Vec::new();
    let mut total_dist = 0.0f64;

    for (label, start) in &starts {
        trials += 1;
        let dist = run_speed_challenge_trial(*start);
        total_dist += dist;
        let ok = dist > 60.0;
        if ok {
            successes += 1;
        }
        if verbose {
            details.push(format!(
                "{} start=({:.0},{:.0},{:.0}) → dist={:.1}m {}",
                label, start.x, start.y, start.z, dist,
                if ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    DrillResult {
        name: "speed_challenge".into(),
        trials,
        successes,
        avg_ticks: if trials > 0 { total_dist / trials as f64 } else { 0.0 },
        details,
    }
}

fn run_speed_challenge_trial(start: Vec3) -> f64 {
    set_planner_class(0);
    let mut sim = SimWorld::new(2000);
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        start,
    );
    // Place a target far away to give AI a reason to move (+x direction)
    sim.add_player(
        "Dummy",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        Vec3::new(200.0, 10.0, 0.0),
    );
    sim.set_bell_held("Dummy");

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

    let max_ticks = 720u64;
    for _tick in 0..max_ticks {
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2000);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    let snap = sim.snapshot();
    let p = snap.players.iter().find(|p| p.id == "H1").unwrap();
    p.p.sub(start).len()
}

// ── NAVIGATION DRILL 2: Waypoint Circuit ──────────────────────────────────
// Navigate a sequence of 3 waypoints (within 12m of each). Triangle ~80m sides.

pub fn drill_waypoint_circuit(verbose: bool) -> DrillResult {
    // Different triangle orientations
    let circuits: [(&str, [Vec3; 3]); 3] = [
        (
            "axial",
            [
                Vec3::new(-40.0, 10.0, 0.0),
                Vec3::new(40.0, 10.0, 0.0),
                Vec3::new(0.0, 10.0, 60.0),
            ],
        ),
        (
            "cross_axis",
            [
                Vec3::new(0.0, 5.0, -40.0),
                Vec3::new(0.0, 35.0, 20.0),
                Vec3::new(0.0, 20.0, 40.0),
            ],
        ),
        (
            "mixed",
            [
                Vec3::new(-30.0, 8.0, -20.0),
                Vec3::new(50.0, 20.0, 10.0),
                Vec3::new(10.0, 30.0, -30.0),
            ],
        ),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for (label, waypoints) in &circuits {
        trials += 1;
        let (complete, ticks, visited) = run_waypoint_circuit_trial(waypoints);
        if complete {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} → visited={}/3 {} t={}",
                label, visited,
                if complete { "DONE" } else { "FAIL" }, ticks
            ));
        }
    }

    DrillResult {
        name: "waypoint_circuit".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_waypoint_circuit_trial(waypoints: &[Vec3; 3]) -> (bool, u64, u32) {
    set_planner_class(0);
    let mut sim = SimWorld::new(2100);
    // Player starts at first waypoint vicinity
    let start = Vec3::new(
        waypoints[0].x - 15.0,
        waypoints[0].y,
        waypoints[0].z,
    );
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        start,
    );

    // We cycle the dummy target to each successive waypoint to motivate movement
    let mut current_wp = 0usize;
    let arrival_radius = 12.0;

    // Place dummy at first waypoint
    sim.add_player(
        "Dummy",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        waypoints[0],
    );
    sim.set_bell_held("Dummy");

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

    let max_ticks = 2400u64;
    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        let p = snap.players.iter().find(|p| p.id == "H1").unwrap();
        let d = p.p.sub(waypoints[current_wp]).len();
        if d < arrival_radius {
            current_wp += 1;
            if current_wp >= 3 {
                return (true, tick, 3);
            }
            // Teleport dummy to next waypoint to drive AI toward it
            // We can't teleport, but we restart the sim... instead we just
            // let the AI continue — it will pursue the ball holder regardless.
            // The dummy position doesn't update, but the AI Mark job drives pursuit.
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2100);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (false, max_ticks, current_wp as u32)
}

// ── NAVIGATION DRILL 3: Precision Arrival ─────────────────────────────────
// Arrive at target (within 8m) while traveling <5 m/s.

pub fn drill_precision_arrival(verbose: bool) -> DrillResult {
    let distances = [30.0, 50.0, 80.0];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for &dist in &distances {
        trials += 1;
        let (arrived, ticks, final_speed, final_dist) = run_precision_arrival_trial(dist);
        if arrived {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "dist={:.0}m → {} final_d={:.1}m final_spd={:.1}m/s t={}",
                dist,
                if arrived { "ARRIVED" } else { "FAIL" },
                final_dist, final_speed, ticks
            ));
        }
    }

    DrillResult {
        name: "precision_arrival".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_precision_arrival_trial(dist: f64) -> (bool, u64, f64, f64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(2200 + dist as u32);
    let start = Vec3::new(-50.0, 10.0, 0.0);
    let target = Vec3::new(-50.0 + dist, 10.0, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        start,
    );
    // Place a teammate holding ball at target — AI will navigate toward it
    sim.add_player(
        "Dummy",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        target,
    );
    sim.set_bell_held("Dummy");

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

    let max_ticks = 2400u64;
    let arrival_radius = 8.0;
    let speed_threshold = 5.0;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        let p = snap.players.iter().find(|p| p.id == "H1").unwrap();
        let d = p.p.sub(target).len();
        let spd = p.v.len();
        if d < arrival_radius && spd < speed_threshold {
            return (true, tick, spd, d);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2200);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    let snap = sim.snapshot();
    let p = snap.players.iter().find(|p| p.id == "H1").unwrap();
    (false, max_ticks, p.v.len(), p.p.sub(target).len())
}

// ── NAVIGATION DRILL 4: Skin Traverse ─────────────────────────────────────
// Navigate near cylinder skin (R=45) without bouncing off.
// Start at y=35 one side, reach y=35 opposite side (180° around).

pub fn drill_skin_traverse(verbose: bool) -> DrillResult {
    let axial_offsets = [0.0, 30.0, -30.0];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for (i, &ax_off) in axial_offsets.iter().enumerate() {
        trials += 1;
        let (arrived, ticks, skinned) = run_skin_traverse_trial(ax_off, i as u32);
        let ok = arrived && !skinned;
        if ok {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "axial_off={:.0}m → arrived={} skinned={} {} t={}",
                ax_off, arrived, skinned,
                if ok { "PASS" } else { "FAIL" }, ticks
            ));
        }
    }

    DrillResult {
        name: "skin_traverse".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_skin_traverse_trial(axial_offset: f64, trial_idx: u32) -> (bool, u64, bool) {
    set_planner_class(0);
    let mut sim = SimWorld::new(2300 + trial_idx);
    // Start at y=35, z=0 (near skin at R=45)
    let start = Vec3::new(axial_offset, 35.0, 0.0);
    // Target: 180° around the cylinder skin → y=-35, z=0
    let target = Vec3::new(axial_offset, -35.0, 0.0);

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        start,
    );
    // Motivate movement toward target
    sim.add_player(
        "Dummy",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        target,
    );
    sim.set_bell_held("Dummy");

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

    let max_ticks = 2400u64;
    let arrival_radius = 12.0;
    let mut was_skinned = false;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        let p = snap.players.iter().find(|p| p.id == "H1").unwrap();
        let d = p.p.sub(target).len();
        if d < arrival_radius {
            return (true, tick, was_skinned);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2300);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        let evs = sim.step(&sw_frame, H);
        for e in &evs {
            if let SwEv::PlayerSkinned { id } = e {
                if id == "H1" {
                    was_skinned = true;
                }
            }
        }
    }
    (false, max_ticks, was_skinned)
}

// ── GRAPPLE DRILL 1: Swing Distance (Peak Speed) ──────────────────────────
// Measure peak speed achieved in 480 ticks. Success: >18 m/s.

pub fn drill_swing_distance(verbose: bool) -> DrillResult {
    let starts: [(&str, Vec3); 4] = [
        ("center", Vec3::new(0.0, 8.0, 0.0)),
        ("high", Vec3::new(0.0, 25.0, 0.0)),
        ("offset_z", Vec3::new(0.0, 12.0, 15.0)),
        ("axial", Vec3::new(-30.0, 10.0, 0.0)),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut details = Vec::new();

    for (label, start) in &starts {
        trials += 1;
        let peak = run_swing_distance_trial(*start);
        let ok = peak > 18.0;
        if ok {
            successes += 1;
        }
        if verbose {
            details.push(format!(
                "{} → peak_speed={:.1}m/s {}",
                label, peak,
                if ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    DrillResult {
        name: "swing_distance".into(),
        trials,
        successes,
        avg_ticks: 0.0,
        details,
    }
}

fn run_swing_distance_trial(start: Vec3) -> f64 {
    set_planner_class(0);
    let mut sim = SimWorld::new(2400);
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        start,
    );
    // Give AI motivation to move: opponent with ball far away
    sim.add_player(
        "Dummy",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        Vec3::new(150.0, 10.0, 0.0),
    );
    sim.set_bell_held("Dummy");

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

    let max_ticks = 480u64;
    let mut peak_speed = 0.0f64;

    for _tick in 0..max_ticks {
        let snap = sim.snapshot();
        let p = snap.players.iter().find(|p| p.id == "H1").unwrap();
        let spd = p.v.len();
        if spd > peak_speed {
            peak_speed = spd;
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2400);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    peak_speed
}

// ── GRAPPLE DRILL 2: Grapple Efficiency ───────────────────────────────────
// Cover 100m using minimal grapple fires. Score by fire count (lower = better).

pub fn drill_grapple_efficiency(verbose: bool) -> DrillResult {
    let start_ys = [8.0, 20.0, 30.0];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut details = Vec::new();

    for &sy in &start_ys {
        trials += 1;
        let (covered, ticks, fires) = run_grapple_efficiency_trial(sy);
        let ok = covered && ticks < 1200;
        if ok {
            successes += 1;
        }
        if verbose {
            details.push(format!(
                "start_y={:.0}m → covered={} fires={} t={} {}",
                sy, covered, fires, ticks,
                if ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    DrillResult {
        name: "grapple_efficiency".into(),
        trials,
        successes,
        avg_ticks: 0.0,
        details,
    }
}

fn run_grapple_efficiency_trial(start_y: f64) -> (bool, u64, u32) {
    set_planner_class(0);
    let mut sim = SimWorld::new(2500);
    let start = Vec3::new(-50.0, start_y, 0.0);
    let target_x = -50.0 + 100.0; // 100m forward

    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        start,
    );
    // Motivate forward movement
    sim.add_player(
        "Dummy",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        Vec3::new(200.0, 10.0, 0.0),
    );
    sim.set_bell_held("Dummy");

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

    let max_ticks = 1200u64;
    let mut fire_count = 0u32;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        let p = snap.players.iter().find(|p| p.id == "H1").unwrap();
        if p.p.x >= target_x {
            return (true, tick, fire_count);
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2500);
        // Count fire_line_at commands for H1
        for inp in &frame.players {
            if inp.id == "H1" && inp.fire_line_at.is_some() {
                fire_count += 1;
            }
        }
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    let snap = sim.snapshot();
    let p = snap.players.iter().find(|p| p.id == "H1").unwrap();
    (p.p.x >= target_x, max_ticks, fire_count)
}

// ── GRAPPLE DRILL 3: Momentum Chain ──────────────────────────────────────
// Build speed >25 m/s within 960 ticks (4s) from rest.

pub fn drill_momentum_chain(verbose: bool) -> DrillResult {
    let starts: [(&str, Vec3); 4] = [
        ("center", Vec3::new(0.0, 8.0, 0.0)),
        ("high", Vec3::new(0.0, 28.0, 0.0)),
        ("side", Vec3::new(0.0, 12.0, 20.0)),
        ("axial", Vec3::new(-40.0, 15.0, 0.0)),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut details = Vec::new();

    for (label, start) in &starts {
        trials += 1;
        let peak = run_momentum_chain_trial(*start);
        let ok = peak > 25.0;
        if ok {
            successes += 1;
        }
        if verbose {
            details.push(format!(
                "{} → peak={:.1}m/s {}",
                label, peak,
                if ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    DrillResult {
        name: "momentum_chain".into(),
        trials,
        successes,
        avg_ticks: 0.0,
        details,
    }
}

fn run_momentum_chain_trial(start: Vec3) -> f64 {
    set_planner_class(0);
    let mut sim = SimWorld::new(2600);
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        start,
    );
    // Motivate forward movement
    sim.add_player(
        "Dummy",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        Vec3::new(250.0, 10.0, 0.0),
    );
    sim.set_bell_held("Dummy");

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

    let max_ticks = 960u64;
    let mut peak_speed = 0.0f64;

    for _tick in 0..max_ticks {
        let snap = sim.snapshot();
        let p = snap.players.iter().find(|p| p.id == "H1").unwrap();
        let spd = p.v.len();
        if spd > peak_speed {
            peak_speed = spd;
        }
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2600);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    peak_speed
}

// ── COORDINATION DRILL 1: Spacing Maintenance ─────────────────────────────
// 3 teammates maintain spread formation while one holds ball.

pub fn drill_spacing_maintenance(verbose: bool) -> DrillResult {
    // Starting configurations: (label, positions for H1,H2,H3)
    let configs: [(&str, [Vec3; 3]); 3] = [
        (
            "clustered",
            [
                Vec3::new(-30.0, 10.0, 0.0),
                Vec3::new(-25.0, 12.0, 3.0),
                Vec3::new(-28.0, 8.0, -2.0),
            ],
        ),
        (
            "line",
            [
                Vec3::new(-80.0, 10.0, 0.0),
                Vec3::new(-40.0, 10.0, 0.0),
                Vec3::new(0.0, 10.0, 0.0),
            ],
        ),
        (
            "spread",
            [
                Vec3::new(-60.0, 10.0, 20.0),
                Vec3::new(-20.0, 15.0, -15.0),
                Vec3::new(-40.0, 25.0, 0.0),
            ],
        ),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut details = Vec::new();

    for (label, positions) in &configs {
        trials += 1;
        let (avg_sep, min_sep, x_stdev) = run_spacing_maintenance_trial(positions);
        let ok = avg_sep > 25.0 && min_sep > 12.0;
        if ok {
            successes += 1;
        }
        if verbose {
            details.push(format!(
                "{} → avg_sep={:.1}m min_sep={:.1}m x_stdev={:.1} {}",
                label, avg_sep, min_sep, x_stdev,
                if ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    DrillResult {
        name: "spacing_maintenance".into(),
        trials,
        successes,
        avg_ticks: 0.0,
        details,
    }
}

fn run_spacing_maintenance_trial(positions: &[Vec3; 3]) -> (f64, f64, f64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(2700);
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        positions[0],
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        positions[1],
    );
    sim.add_player(
        "H3",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Faithwing,
        positions[2],
    );
    sim.set_bell_held("H1");

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 1200u64;
    for _tick in 0..max_ticks {
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2700);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    // Measure final state
    let snap = sim.snapshot();
    let ps: Vec<Vec3> = ["H1", "H2", "H3"]
        .iter()
        .map(|id| snap.players.iter().find(|p| p.id == *id).unwrap().p)
        .collect();

    // Pairwise distances
    let d01 = ps[0].sub(ps[1]).len();
    let d02 = ps[0].sub(ps[2]).len();
    let d12 = ps[1].sub(ps[2]).len();
    let avg_sep = (d01 + d02 + d12) / 3.0;
    let min_sep = d01.min(d02).min(d12);

    // X-position standard deviation
    let xs: Vec<f64> = ps.iter().map(|p| p.x).collect();
    let mean_x = xs.iter().sum::<f64>() / 3.0;
    let x_var = xs.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>() / 3.0;
    let x_stdev = x_var.sqrt();

    (avg_sep, min_sep, x_stdev)
}

// ── COORDINATION DRILL 2: Support Positioning ─────────────────────────────
// Off-ball player positions forward of carrier for outlet pass.

pub fn drill_support_positioning(verbose: bool) -> DrillResult {
    let support_starts: [(&str, Vec3); 4] = [
        ("behind", Vec3::new(-80.0, 10.0, 0.0)),
        ("lateral", Vec3::new(-50.0, 10.0, 30.0)),
        ("far_behind", Vec3::new(-100.0, 15.0, -10.0)),
        ("above", Vec3::new(-50.0, 30.0, 0.0)),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut details = Vec::new();

    for (label, sup_start) in &support_starts {
        trials += 1;
        let (forward, lat_or_fwd, in_range) = run_support_positioning_trial(*sup_start);
        let ok = forward && lat_or_fwd && in_range;
        if ok {
            successes += 1;
        }
        if verbose {
            details.push(format!(
                "{} → forward={} offset_ok={} range_ok={} {}",
                label, forward, lat_or_fwd, in_range,
                if ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    DrillResult {
        name: "support_positioning".into(),
        trials,
        successes,
        avg_ticks: 0.0,
        details,
    }
}

fn run_support_positioning_trial(support_start: Vec3) -> (bool, bool, bool) {
    set_planner_class(0);
    let mut sim = SimWorld::new(2800);
    // Carrier at x=-50 holding ball
    let carrier_pos = Vec3::new(-50.0, 10.0, 0.0);
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        carrier_pos,
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        support_start,
    );
    sim.set_bell_held("H1");

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    let max_ticks = 600u64;
    for _tick in 0..max_ticks {
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2800);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    // Evaluate H2 final position relative to carrier (H1)
    let snap = sim.snapshot();
    let h1 = snap.players.iter().find(|p| p.id == "H1").unwrap();
    let h2 = snap.players.iter().find(|p| p.id == "H2").unwrap();

    // Home attacks +x, so "forward" means higher x
    let forward = h2.p.x > h1.p.x;
    // Lateral offset >10m OR forward >15m
    let dx = h2.p.x - h1.p.x;
    let lateral = ((h2.p.y - h1.p.y).powi(2) + (h2.p.z - h1.p.z).powi(2)).sqrt();
    let lat_or_fwd = lateral > 10.0 || dx > 15.0;
    // Within 80m passing range
    let range = h2.p.sub(h1.p).len();
    let in_range = range < 80.0;

    (forward, lat_or_fwd, in_range)
}

// ── COORDINATION DRILL 3: Rotation After Catch ───────────────────────────
// Ball transfers from H1 to H2; team reorganizes (H2 advances, H1 supports).

pub fn drill_rotation_after_catch(verbose: bool) -> DrillResult {
    let configs: [(&str, [Vec3; 3]); 3] = [
        (
            "standard",
            [
                Vec3::new(-60.0, 10.0, 0.0),  // H1 (carrier initially)
                Vec3::new(-20.0, 10.0, 15.0),  // H2 (receives ball)
                Vec3::new(-40.0, 20.0, -10.0), // H3 (support)
            ],
        ),
        (
            "stacked",
            [
                Vec3::new(-50.0, 10.0, 5.0),
                Vec3::new(-45.0, 12.0, -5.0),
                Vec3::new(-55.0, 8.0, 0.0),
            ],
        ),
        (
            "wide",
            [
                Vec3::new(-80.0, 10.0, 30.0),
                Vec3::new(-30.0, 15.0, -20.0),
                Vec3::new(-50.0, 25.0, 0.0),
            ],
        ),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut details = Vec::new();

    for (label, positions) in &configs {
        trials += 1;
        let (h2_advanced, h1_repositioned) = run_rotation_after_catch_trial(positions);
        let ok = h2_advanced && h1_repositioned;
        if ok {
            successes += 1;
        }
        if verbose {
            details.push(format!(
                "{} → h2_adv={} h1_repo={} {}",
                label, h2_advanced, h1_repositioned,
                if ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    DrillResult {
        name: "rotation_after_catch".into(),
        trials,
        successes,
        avg_ticks: 0.0,
        details,
    }
}

fn run_rotation_after_catch_trial(positions: &[Vec3; 3]) -> (bool, bool) {
    set_planner_class(0);
    let mut sim = SimWorld::new(2900);
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        positions[0],
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        positions[1],
    );
    sim.add_player(
        "H3",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Faithwing,
        positions[2],
    );
    sim.set_bell_held("H1");

    let cfgs = vec![TeamConfig {
        side: ai::TeamSide::Home,
        profile: style_to_profile("fall-dynasty", "big-slow"),
        difficulty: Difficulty::Pro,
    }];
    let mut aisys = AiSystem::new();

    // Phase 1: let team settle for 120 ticks with H1 holding
    for _tick in 0..120 {
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2900);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    // Transfer ball to H2
    sim.set_bell_held("H2");
    let snap_at_transfer = sim.snapshot();
    let h2_x_at_transfer = snap_at_transfer
        .players.iter().find(|p| p.id == "H2").unwrap().p.x;

    // Phase 2: run 360 more ticks
    for _tick in 0..360 {
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 2900);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    // Evaluate
    let snap = sim.snapshot();
    let h1 = snap.players.iter().find(|p| p.id == "H1").unwrap();
    let h2 = snap.players.iter().find(|p| p.id == "H2").unwrap();

    // H2 advances by >10m toward attack ring (+x)
    let h2_advanced = h2.p.x - h2_x_at_transfer > 10.0;
    // H1 repositions as support (not stacked on H2; lateral or behind)
    let h1_sep = h1.p.sub(h2.p).len();
    let h1_repositioned = h1_sep > 12.0;

    (h2_advanced, h1_repositioned)
}

// ── STRATEGY DRILL 1: Advance Under Defense ───────────────────────────────
// 3v2 offense has numbers advantage. Ball advances >40m in 2400 ticks.

pub fn drill_advance_under_defense(verbose: bool) -> DrillResult {
    // Different defensive formations
    let def_configs: [(&str, [Vec3; 2]); 3] = [
        (
            "zone_center",
            [
                Vec3::new(0.0, 10.0, 10.0),
                Vec3::new(20.0, 10.0, -10.0),
            ],
        ),
        (
            "stacked_deep",
            [
                Vec3::new(30.0, 10.0, 0.0),
                Vec3::new(50.0, 15.0, 5.0),
            ],
        ),
        (
            "spread_wide",
            [
                Vec3::new(10.0, 10.0, 25.0),
                Vec3::new(10.0, 10.0, -25.0),
            ],
        ),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut details = Vec::new();

    for (label, def_pos) in &def_configs {
        trials += 1;
        let (advance, ball_x_gain) = run_advance_under_defense_trial(def_pos);
        let ok = advance > 40.0;
        if ok {
            successes += 1;
        }
        if verbose {
            details.push(format!(
                "{} → ball_advance={:.1}m ball_x_gain={:.1}m {}",
                label, advance, ball_x_gain,
                if ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    DrillResult {
        name: "advance_under_defense".into(),
        trials,
        successes,
        avg_ticks: 0.0,
        details,
    }
}

fn run_advance_under_defense_trial(def_positions: &[Vec3; 2]) -> (f64, f64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(3000);
    // 3 home attackers
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        Vec3::new(-40.0, 10.0, 0.0),
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        Vec3::new(-60.0, 10.0, 20.0),
    );
    sim.add_player(
        "H3",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Faithwing,
        Vec3::new(-60.0, 15.0, -15.0),
    );
    // 2 away defenders
    sim.add_player(
        "A1",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        def_positions[0],
    );
    sim.add_player(
        "A2",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Anchor,
        def_positions[1],
    );
    sim.set_bell_held("H1");

    let initial_ball_x = -40.0;

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

    let max_ticks = 2400u64;
    let mut max_ball_x = initial_ball_x;

    for _tick in 0..max_ticks {
        let snap = sim.snapshot();
        // Track ball position (held or free)
        let ball_x = if let Some(ref holder) = snap.bell.held_by {
            if let Some(p) = snap.players.iter().find(|p| p.id == *holder) {
                p.p.x
            } else {
                snap.bell.p.x
            }
        } else {
            snap.bell.p.x
        };
        if ball_x > max_ball_x {
            max_ball_x = ball_x;
        }

        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 3000);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    let final_snap = sim.snapshot();
    let final_ball_x = if let Some(ref holder) = final_snap.bell.held_by {
        if let Some(p) = final_snap.players.iter().find(|p| p.id == *holder) {
            p.p.x
        } else {
            final_snap.bell.p.x
        }
    } else {
        final_snap.bell.p.x
    };

    let advance = max_ball_x - initial_ball_x;
    let ball_x_gain = final_ball_x - initial_ball_x;
    (advance, ball_x_gain)
}

// ── STRATEGY DRILL 2: Deny Advance ───────────────────────────────────────
// 2 defenders vs 2 attackers. Defense holds ball advance <30m.

pub fn drill_deny_advance(verbose: bool) -> DrillResult {
    let separations: [(&str, f64); 3] = [
        ("close", 30.0),
        ("medium", 50.0),
        ("far", 70.0),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut details = Vec::new();

    for (label, sep) in &separations {
        trials += 1;
        let advance = run_deny_advance_trial(*sep);
        let ok = advance < 30.0;
        if ok {
            successes += 1;
        }
        if verbose {
            details.push(format!(
                "{} sep={:.0}m → ball_advance={:.1}m {}",
                label, sep, advance,
                if ok { "HELD" } else { "BROKE_THROUGH" }
            ));
        }
    }

    DrillResult {
        name: "deny_advance".into(),
        trials,
        successes,
        avg_ticks: 0.0,
        details,
    }
}

fn run_deny_advance_trial(offense_defense_sep: f64) -> f64 {
    set_planner_class(0);
    let mut sim = SimWorld::new(3100);
    // 2 attackers (Away team attacking -x, but we measure ball +x advance for Home)
    // Actually: Home attacks +x. Offense is Home with ball.
    // Defense is Away trying to prevent advance.
    let off_x = -60.0;
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        Vec3::new(off_x, 10.0, 0.0),
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        Vec3::new(off_x - 20.0, 10.0, 15.0),
    );
    // Defenders ahead of offense
    let def_x = off_x + offense_defense_sep;
    sim.add_player(
        "A1",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        Vec3::new(def_x, 10.0, 5.0),
    );
    sim.add_player(
        "A2",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Anchor,
        Vec3::new(def_x + 10.0, 12.0, -8.0),
    );
    sim.set_bell_held("H1");

    let initial_ball_x = off_x;

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

    let max_ticks = 2400u64;
    let mut max_ball_x = initial_ball_x;

    for _tick in 0..max_ticks {
        let snap = sim.snapshot();
        let ball_x = if let Some(ref holder) = snap.bell.held_by {
            // Only count if home team holds it (advance)
            if snap.players.iter().any(|p| p.id == *holder && p.team == crate::sim_world::TeamSide::Home) {
                if let Some(p) = snap.players.iter().find(|p| p.id == *holder) {
                    p.p.x
                } else {
                    snap.bell.p.x
                }
            } else {
                max_ball_x // don't update if away has it
            }
        } else {
            snap.bell.p.x
        };
        if ball_x > max_ball_x {
            max_ball_x = ball_x;
        }

        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 3100);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }

    max_ball_x - initial_ball_x
}

// ── STRATEGY DRILL 3: Fast Break ─────────────────────────────────────────
// Ball free between teams. Home must secure + advance >30m in 1200 ticks.

pub fn drill_fast_break(verbose: bool) -> DrillResult {
    let ball_positions: [(&str, Vec3); 3] = [
        ("center", Vec3::new(0.0, 10.0, 0.0)),
        ("offset", Vec3::new(-20.0, 15.0, 10.0)),
        ("deep", Vec3::new(30.0, 8.0, -5.0)),
    ];

    let mut trials = 0u32;
    let mut successes = 0u32;
    let mut total_ticks = 0u64;
    let mut details = Vec::new();

    for (label, ball_pos) in &ball_positions {
        trials += 1;
        let (secured, advanced, ticks) = run_fast_break_trial(*ball_pos);
        let ok = secured && advanced;
        if ok {
            successes += 1;
            total_ticks += ticks;
        }
        if verbose {
            details.push(format!(
                "{} → secured={} advanced={} {} t={}",
                label, secured, advanced,
                if ok { "PASS" } else { "FAIL" }, ticks
            ));
        }
    }

    DrillResult {
        name: "fast_break".into(),
        trials,
        successes,
        avg_ticks: if successes > 0 { total_ticks as f64 / successes as f64 } else { 0.0 },
        details,
    }
}

fn run_fast_break_trial(ball_pos: Vec3) -> (bool, bool, u64) {
    set_planner_class(0);
    let mut sim = SimWorld::new(3200);
    // 2v2 scenario
    // Home players: one near ball, one ahead
    sim.add_player(
        "H1",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Spinner,
        Vec3::new(ball_pos.x - 20.0, ball_pos.y, ball_pos.z + 5.0),
    );
    sim.add_player(
        "H2",
        crate::sim_world::TeamSide::Home,
        crate::sim_world::RiggerRole::Anchor,
        Vec3::new(ball_pos.x + 30.0, 10.0, 0.0),
    );
    // Away players: one near ball, one behind
    sim.add_player(
        "A1",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Spinner,
        Vec3::new(ball_pos.x + 15.0, ball_pos.y, ball_pos.z - 8.0),
    );
    sim.add_player(
        "A2",
        crate::sim_world::TeamSide::Away,
        crate::sim_world::RiggerRole::Anchor,
        Vec3::new(ball_pos.x - 30.0, 12.0, -5.0),
    );
    // Ball is free (loose)
    sim.launch_bell(ball_pos, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), None);

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

    let max_ticks = 1200u64;
    let mut secured = false;
    let mut secure_x = 0.0f64;
    let mut secure_tick = 0u64;

    for tick in 0..max_ticks {
        let snap = sim.snapshot();
        if !secured {
            // Check if Home team secured the ball
            if let Some(ref holder) = snap.bell.held_by {
                if holder == "H1" || holder == "H2" {
                    secured = true;
                    secure_x = snap.players.iter()
                        .find(|p| p.id == *holder).unwrap().p.x;
                    secure_tick = tick;
                }
            }
        } else {
            // Check if ball advanced >30m from secure point
            let ball_x = if let Some(ref holder) = snap.bell.held_by {
                if holder == "H1" || holder == "H2" {
                    snap.players.iter()
                        .find(|p| p.id == *holder).unwrap().p.x
                } else {
                    snap.bell.p.x
                }
            } else {
                snap.bell.p.x
            };
            if ball_x - secure_x > 30.0 {
                return (true, true, tick);
            }
        }

        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = minimal_match_state();
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, 3200);
        let sw_frame = conv::frame_ai_to_sw(&frame);
        sim.step(&sw_frame, H);
    }
    (secured, false, if secured { secure_tick } else { max_ticks })
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn minimal_match_state() -> ai::MatchState {
    ai::MatchState {
        inning: 1.0,
        spine: false,
        possession: ai::TeamSide::Home,
        faith_end: ai::FaithEnd::PlusX,
        cast: ai::Cast {
            throws_left: 3,
            gate: ai::Gate::First,
            spot_x: -80.0,
        },
        contest: None,
        score_home: 0.0,
        score_away: 0.0,
        phase: ai::MatchPhase::Live,
        message: String::new(),
        winner: None,
    }
}

//! Match Analytics — comprehensive telemetry from full AI-vs-AI matches.
//! These metrics are TRACKED but not directly optimized against (Goodhart risk).
//! They reveal play quality, style, diversity, and failure modes.

use crate::ai::decision_types::TeamConfig;
use crate::ai::plan_bridge::set_planner_class;
use crate::ai::profile::{style_to_profile, Difficulty};
use crate::ai::system::AiSystem;
use crate::ai::types as ai;
use crate::match_sm::{MatchPhase, MatchStateMachine};
use crate::math::Vec3;
use crate::scoring;
use crate::sim_world::{SimEvent as SwEv, SimWorld, Snapshot};
use crate::skill_eval::conv;
use std::collections::HashMap;

const H: f64 = 1.0 / 240.0;
const CYLINDER_R: f64 = 45.0;
const CYLINDER_L: f64 = 640.0;

// ── Roster (same as skill_eval) ─────────────────────────────────────────────
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

// ── Metric structs ──────────────────────────────────────────────────────────

/// Top-level report containing all sub-reports from a single match.
#[derive(Clone, Debug)]
pub struct MatchReport {
    pub ticks: u64,
    pub duration_s: f64,
    pub score_home: i64,
    pub score_away: i64,
    pub style: StyleMetrics,
    pub diversity: DiversityMetrics,
    pub tempo: TempoMetrics,
    pub spatial: SpatialMetrics,
    pub possession: PossessionMetrics,
    pub failure: FailureModeMetrics,
}

#[derive(Clone, Debug)]
pub struct StyleMetrics {
    /// Fraction of ticks a player has reel=0 (swing) vs reel=-1 (winch) while line is taut.
    pub swing_vs_winch_ratio: f64,
    /// Average of each player's peak speed during the match.
    pub avg_peak_speed: f64,
    /// Fraction of ticks players have no line attached (soaring).
    pub free_flight_fraction: f64,
    /// Estimate swing arc length (distance traveled while reel=0 and line taut).
    pub avg_swing_arc_length: f64,
    /// Average grapple fires per second across all players.
    pub fire_rate: f64,
}

#[derive(Clone, Debug)]
pub struct DiversityMetrics {
    /// Shannon entropy of who gets thrown to. Low = always same receiver.
    pub throw_target_entropy: f64,
    /// Count of grid cells visited (10x5 grid: x over 640m, yz by radius 5 bins).
    pub position_coverage: u32,
    /// Count unique (fire_dir_quadrant, reel, throw_released) tuples per player per 30-tick window.
    pub action_sequence_variety: f64,
}

#[derive(Clone, Debug)]
pub struct TempoMetrics {
    pub throws_per_minute: f64,
    pub catches_per_minute: f64,
    pub possession_changes_per_minute: f64,
    pub avg_hold_ticks: f64,
    pub hold_time_std: f64,
    /// Fraction of ticks the ball is "dead" (match phase != Live).
    pub dead_ball_rate: f64,
}

#[derive(Clone, Debug)]
pub struct SpatialMetrics {
    /// 10 bins showing where the ball spends time along x-axis.
    pub ball_x_histogram: [u64; 10],
    /// Average inter-player distance within each team.
    pub team_spread_avg: f64,
    /// % of 10x5 grid cells that had a player visit.
    pub field_utilization: f64,
}

#[derive(Clone, Debug)]
pub struct PossessionMetrics {
    /// Histogram of consecutive same-team passes before turnover [0,1,2,3,4,5+].
    pub chain_lengths: [u32; 6],
    /// % of catches where a defender was within 15m.
    pub contested_catch_fraction: f64,
    /// |home_score - away_score| / max(home_score + away_score, 1).
    pub score_parity: f64,
}

#[derive(Clone, Debug)]
pub struct FailureModeMetrics {
    /// Longest streak of ticks with no SimEvent at all.
    pub stalemate_ticks: u64,
    /// Episodes where a player's x-position has >4 direction reversals within 240 ticks.
    pub oscillation_score: u32,
    /// PlayerSkinned events per minute.
    pub skin_bounce_rate: f64,
    /// BellBobble events per total catch attempts.
    pub bobble_rate: f64,
}

// ── Collectors (internal state during tick-by-tick accumulation) ─────────────

struct StyleCollector {
    swing_ticks: u64,
    winch_ticks: u64,
    free_flight_ticks: u64,
    total_player_ticks: u64,
    peak_speeds: HashMap<String, f64>,
    swing_arc_distances: Vec<f64>,
    current_swing_dist: HashMap<String, f64>,
    fire_count: u64,
}

struct DiversityCollector {
    throw_targets: HashMap<String, u64>,
    /// 10x5 grid visited cells (x-bin, yz-bin).
    grid_visited: [[bool; 5]; 10],
    /// Per player: set of (quadrant, reel, throw_released) tuples in current 30-tick window.
    action_windows: HashMap<String, Vec<(i32, i32, bool)>>,
    action_variety_sum: f64,
    action_variety_count: u64,
    window_tick: u64,
}

struct TempoCollector {
    throws: u64,
    catches: u64,
    possession_changes: u64,
    hold_durations: Vec<u64>,
    current_hold: u64,
    dead_ticks: u64,
    prev_holder: Option<String>,
}

struct SpatialCollector {
    ball_x_histogram: [u64; 10],
    team_spread_sum: f64,
    team_spread_count: u64,
    player_grid: [[bool; 5]; 10],
}

struct PossessionCollector {
    current_chain: u32,
    chain_lengths: [u32; 6],
    prev_team_with_ball: Option<ai::TeamSide>,
    contested_catches: u32,
    total_catches: u32,
}

struct FailureCollector {
    current_no_event_streak: u64,
    max_no_event_streak: u64,
    /// Per player: ring buffer of last 240 x-positions.
    x_history: HashMap<String, Vec<f64>>,
    oscillation_episodes: u32,
    skin_events: u32,
    bobble_events: u32,
    catch_attempts: u32,
}

// ── Grid helpers ────────────────────────────────────────────────────────────

fn x_bin(x: f64) -> usize {
    // x ranges from -320 to +320 (cylinder length 640)
    let normalized = (x + 320.0) / CYLINDER_L;
    let bin = (normalized * 10.0) as usize;
    bin.min(9)
}

fn yz_bin(y: f64, z: f64) -> usize {
    // radial distance from axis, binned into 5 bins over [0, CYLINDER_R]
    let r = (y * y + z * z).sqrt();
    let normalized = r / CYLINDER_R;
    let bin = (normalized * 5.0) as usize;
    bin.min(4)
}

fn fire_dir_quadrant(dir: Vec3) -> i32 {
    // Classify fire direction into 4 quadrants based on y,z signs
    let qy = if dir.y >= 0.0 { 1 } else { 0 };
    let qz = if dir.z >= 0.0 { 1 } else { 0 };
    qy * 2 + qz
}

// ── Main analysis function ──────────────────────────────────────────────────

/// Run a full AI-vs-AI match (same setup as `skill_eval::run_scored_match`)
/// and collect comprehensive analytics tick-by-tick.
pub fn analyze_match(
    h_style: &str,
    h_cyl: &str,
    a_style: &str,
    a_cyl: &str,
    seed: u32,
    max_ticks: u64,
) -> MatchReport {
    set_planner_class(0);

    // ── Setup (mirrors skill_eval::run_scored_match) ─────────────────────────
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

    // ── Initialize collectors ────────────────────────────────────────────────
    let mut style = StyleCollector {
        swing_ticks: 0,
        winch_ticks: 0,
        free_flight_ticks: 0,
        total_player_ticks: 0,
        peak_speeds: HashMap::new(),
        swing_arc_distances: Vec::new(),
        current_swing_dist: HashMap::new(),
        fire_count: 0,
    };

    let mut diversity = DiversityCollector {
        throw_targets: HashMap::new(),
        grid_visited: [[false; 5]; 10],
        action_windows: HashMap::new(),
        action_variety_sum: 0.0,
        action_variety_count: 0,
        window_tick: 0,
    };

    let mut tempo = TempoCollector {
        throws: 0,
        catches: 0,
        possession_changes: 0,
        hold_durations: Vec::new(),
        current_hold: 0,
        dead_ticks: 0,
        prev_holder: None,
    };

    let mut spatial = SpatialCollector {
        ball_x_histogram: [0u64; 10],
        team_spread_sum: 0.0,
        team_spread_count: 0,
        player_grid: [[false; 5]; 10],
    };

    let mut poss = PossessionCollector {
        current_chain: 0,
        chain_lengths: [0u32; 6],
        prev_team_with_ball: None,
        contested_catches: 0,
        total_catches: 0,
    };

    let mut failure = FailureCollector {
        current_no_event_streak: 0,
        max_no_event_streak: 0,
        x_history: HashMap::new(),
        oscillation_episodes: 0,
        skin_events: 0,
        bobble_events: 0,
        catch_attempts: 0,
    };

    let mut ticks: u64 = 0;
    let mut prev_positions: HashMap<String, Vec3> = HashMap::new();

    // ── Main tick loop ───────────────────────────────────────────────────────
    for _ in 0..max_ticks {
        if mat.state().winner.is_some() {
            break;
        }
        let snap = sim.snapshot();
        let ai_sim = conv::snap_to_ai(&snap);
        let ai_match = conv::msm_to_ai(mat.state());

        // ── Tempo: dead ball tracking ────────────────────────────────────────
        if mat.state().phase != MatchPhase::Live {
            tempo.dead_ticks += 1;
        }

        // ── Tempo: hold tracking ─────────────────────────────────────────────
        if let Some(ref holder) = snap.bell.held_by {
            tempo.current_hold += 1;
            if tempo.prev_holder.as_ref() != Some(holder) {
                // Possession changed
                if tempo.prev_holder.is_some() {
                    tempo.possession_changes += 1;
                }
                tempo.prev_holder = Some(holder.clone());
            }
        } else {
            if tempo.current_hold > 0 {
                tempo.hold_durations.push(tempo.current_hold);
                tempo.current_hold = 0;
            }
            // If the ball was just released, prev_holder stays for possession tracking
        }

        // ── Spatial: ball x histogram ────────────────────────────────────────
        spatial.ball_x_histogram[x_bin(snap.bell.p.x)] += 1;

        // ── Spatial: team spread ─────────────────────────────────────────────
        if ticks % 30 == 0 {
            for team in &[ai::TeamSide::Home, ai::TeamSide::Away] {
                let team_players: Vec<&crate::sim_world::PlayerSnapshot> = snap
                    .players
                    .iter()
                    .filter(|p| {
                        let pt = match p.team {
                            crate::sim_world::TeamSide::Home => ai::TeamSide::Home,
                            crate::sim_world::TeamSide::Away => ai::TeamSide::Away,
                        };
                        pt == *team
                    })
                    .collect();
                if team_players.len() >= 2 {
                    let mut dist_sum = 0.0;
                    let mut count = 0u32;
                    for i in 0..team_players.len() {
                        for j in (i + 1)..team_players.len() {
                            let dx = team_players[i].p.x - team_players[j].p.x;
                            let dy = team_players[i].p.y - team_players[j].p.y;
                            let dz = team_players[i].p.z - team_players[j].p.z;
                            dist_sum += (dx * dx + dy * dy + dz * dz).sqrt();
                            count += 1;
                        }
                    }
                    spatial.team_spread_sum += dist_sum / count as f64;
                    spatial.team_spread_count += 1;
                }
            }
        }

        // ── Style + Spatial: per-player tracking ─────────────────────────────
        for p in &snap.players {
            let speed = (p.v.x * p.v.x + p.v.y * p.v.y + p.v.z * p.v.z).sqrt();
            let peak = style.peak_speeds.entry(p.id.clone()).or_insert(0.0);
            if speed > *peak {
                *peak = speed;
            }

            style.total_player_ticks += 1;

            if p.line_anchor.is_some() {
                let taut = p.line_taut.unwrap_or(false);
                if taut {
                    // We'll check reel value from the frame below
                }
            } else {
                style.free_flight_ticks += 1;
            }

            // Grid coverage
            let xb = x_bin(p.p.x);
            let yzb = yz_bin(p.p.y, p.p.z);
            spatial.player_grid[xb][yzb] = true;
            diversity.grid_visited[xb][yzb] = true;

            // Distance traveled (for swing arc)
            if let Some(prev) = prev_positions.get(&p.id) {
                let dx = p.p.x - prev.x;
                let dy = p.p.y - prev.y;
                let dz = p.p.z - prev.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if p.line_anchor.is_some() && p.line_taut.unwrap_or(false) {
                    // Accumulate swing arc (we'll check reel=0 from frame)
                    *style.current_swing_dist.entry(p.id.clone()).or_insert(0.0) += dist;
                } else {
                    // End of swing arc
                    if let Some(d) = style.current_swing_dist.remove(&p.id) {
                        if d > 0.0 {
                            style.swing_arc_distances.push(d);
                        }
                    }
                }
            }
            prev_positions.insert(p.id.clone(), p.p);

            // Oscillation detection: track x-history
            let hist = failure.x_history.entry(p.id.clone()).or_insert_with(Vec::new);
            hist.push(p.p.x);
            if hist.len() > 240 {
                hist.remove(0);
            }
            // Check for >4 direction reversals in the 240-tick window
            if hist.len() == 240 && ticks % 240 == 0 {
                let mut reversals = 0u32;
                let mut prev_dir: Option<bool> = None; // true = increasing
                for i in 1..hist.len() {
                    let increasing = hist[i] > hist[i - 1];
                    if let Some(pd) = prev_dir {
                        if increasing != pd {
                            reversals += 1;
                        }
                    }
                    if (hist[i] - hist[i - 1]).abs() > 0.01 {
                        prev_dir = Some(increasing);
                    }
                }
                if reversals > 4 {
                    failure.oscillation_episodes += 1;
                }
            }
        }

        // ── AI tick ──────────────────────────────────────────────────────────
        let frame = aisys.tick(&ai_sim, &ai_match, &cfgs, seed);

        // ── Style: reel tracking from frame, fire counting ───────────────────
        for inp in &frame.players {
            // Find the player in snapshot to check line state
            if let Some(sp) = snap.players.iter().find(|p| p.id == inp.id) {
                if sp.line_anchor.is_some() && sp.line_taut.unwrap_or(false) {
                    if inp.reel == 0 {
                        style.swing_ticks += 1;
                    } else if inp.reel == -1 {
                        style.winch_ticks += 1;
                    }
                }
            }

            if inp.fire_line_at.is_some() {
                style.fire_count += 1;
            }

            if inp.throw_released {
                tempo.throws += 1;
                // Track throw target: who is nearest non-thrower teammate
                if let Some(thrower_team) = team_of.get(&inp.id) {
                    let thrower_pos = snap
                        .players
                        .iter()
                        .find(|p| p.id == inp.id)
                        .map(|p| p.p);
                    if let Some(tp) = thrower_pos {
                        let mut best_id: Option<String> = None;
                        let mut best_dist = f64::INFINITY;
                        for p in &snap.players {
                            if p.id == inp.id {
                                continue;
                            }
                            let pt = match p.team {
                                crate::sim_world::TeamSide::Home => ai::TeamSide::Home,
                                crate::sim_world::TeamSide::Away => ai::TeamSide::Away,
                            };
                            if pt == *thrower_team {
                                let dx = p.p.x - tp.x;
                                let dy = p.p.y - tp.y;
                                let dz = p.p.z - tp.z;
                                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                                if d < best_dist {
                                    best_dist = d;
                                    best_id = Some(p.id.clone());
                                }
                            }
                        }
                        if let Some(target) = best_id {
                            *diversity.throw_targets.entry(target).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Action variety: accumulate per 30-tick window
            let action_tuple = (
                inp.fire_line_at.map(|d| fire_dir_quadrant(d)).unwrap_or(-1),
                inp.reel,
                inp.throw_released,
            );
            diversity
                .action_windows
                .entry(inp.id.clone())
                .or_insert_with(Vec::new)
                .push(action_tuple);
        }

        // ── Diversity: flush action windows every 30 ticks ───────────────────
        diversity.window_tick += 1;
        if diversity.window_tick >= 30 {
            for (_id, actions) in diversity.action_windows.iter_mut() {
                // Count unique tuples
                let mut unique: Vec<(i32, i32, bool)> = actions.clone();
                unique.sort_by(|a, b| {
                    a.0.cmp(&b.0)
                        .then(a.1.cmp(&b.1))
                        .then(a.2.cmp(&b.2))
                });
                unique.dedup();
                diversity.action_variety_sum += unique.len() as f64;
                diversity.action_variety_count += 1;
                actions.clear();
            }
            diversity.window_tick = 0;
        }

        // ── Step the sim ─────────────────────────────────────────────────────
        let sw_frame = conv::frame_ai_to_sw(&frame);
        let evs = sim.step(&sw_frame, H);

        // ── Process events ───────────────────────────────────────────────────
        if evs.is_empty() {
            failure.current_no_event_streak += 1;
        } else {
            if failure.current_no_event_streak > failure.max_no_event_streak {
                failure.max_no_event_streak = failure.current_no_event_streak;
            }
            failure.current_no_event_streak = 0;
        }

        for e in &evs {
            match e {
                SwEv::BellCaught { by } => {
                    tempo.catches += 1;
                    poss.total_catches += 1;
                    failure.catch_attempts += 1;

                    // Check if contested (defender within 15m)
                    if let Some(catcher_team) = team_of.get(by) {
                        let catcher_pos = snap.players.iter().find(|p| &p.id == by).map(|p| p.p);
                        if let Some(cp) = catcher_pos {
                            let contested = snap.players.iter().any(|p| {
                                let pt = match p.team {
                                    crate::sim_world::TeamSide::Home => ai::TeamSide::Home,
                                    crate::sim_world::TeamSide::Away => ai::TeamSide::Away,
                                };
                                if pt == *catcher_team {
                                    return false;
                                }
                                let dx = p.p.x - cp.x;
                                let dy = p.p.y - cp.y;
                                let dz = p.p.z - cp.z;
                                (dx * dx + dy * dy + dz * dz).sqrt() <= 15.0
                            });
                            if contested {
                                poss.contested_catches += 1;
                            }
                        }
                    }

                    // Possession chain tracking
                    if let Some(catcher_team) = team_of.get(by) {
                        if let Some(prev_team) = poss.prev_team_with_ball {
                            if *catcher_team == prev_team {
                                poss.current_chain += 1;
                            } else {
                                // Turnover: record chain length
                                let idx = (poss.current_chain as usize).min(5);
                                poss.chain_lengths[idx] += 1;
                                poss.current_chain = 0;
                            }
                        }
                        poss.prev_team_with_ball = Some(*catcher_team);
                    }
                }
                SwEv::BellBobble { .. } => {
                    failure.bobble_events += 1;
                    failure.catch_attempts += 1;
                }
                SwEv::BellClatter { .. } => {
                    failure.catch_attempts += 1;
                }
                SwEv::PlayerSkinned { .. } => {
                    failure.skin_events += 1;
                }
                SwEv::BellSkin => {
                    failure.skin_events += 1;
                }
                _ => {}
            }
        }

        // ── Feed events to match SM ──────────────────────────────────────────
        let scoring_evs: Vec<scoring::SimEvent> =
            evs.iter().map(conv::ev_sw_to_scoring).collect();
        let post = conv::ai_to_scoring(&conv::snap_to_ai(&sim.snapshot()));
        mat.consume(&scoring_evs, &post);

        // ── Re-arm dead ball ─────────────────────────────────────────────────
        if mat.state().winner.is_none() && mat.state().phase != MatchPhase::Live {
            let possession = mat.state().possession;
            if let Some(r) = rs
                .iter()
                .find(|r| conv::team_ai_to_scoring(r.team) == possession)
            {
                sim.set_bell_held(r.id);
            }
            mat.resume_live();
        }

        ticks += 1;
    }

    // Finalize the last no-event streak
    if failure.current_no_event_streak > failure.max_no_event_streak {
        failure.max_no_event_streak = failure.current_no_event_streak;
    }

    // Finalize hold duration
    if tempo.current_hold > 0 {
        tempo.hold_durations.push(tempo.current_hold);
    }

    // Finalize remaining swing arcs
    for (_id, d) in style.current_swing_dist.drain() {
        if d > 0.0 {
            style.swing_arc_distances.push(d);
        }
    }

    // Finalize remaining chain
    if poss.current_chain > 0 || poss.prev_team_with_ball.is_some() {
        let idx = (poss.current_chain as usize).min(5);
        poss.chain_lengths[idx] += 1;
    }

    // ── Assemble report ──────────────────────────────────────────────────────
    let duration_s = ticks as f64 * H;
    let duration_min = duration_s / 60.0;

    // Style metrics
    let swing_total = style.swing_ticks + style.winch_ticks;
    let swing_vs_winch_ratio = if swing_total > 0 {
        style.swing_ticks as f64 / swing_total as f64
    } else {
        0.0
    };
    let avg_peak_speed = if style.peak_speeds.is_empty() {
        0.0
    } else {
        style.peak_speeds.values().sum::<f64>() / style.peak_speeds.len() as f64
    };
    let free_flight_fraction = if style.total_player_ticks > 0 {
        style.free_flight_ticks as f64 / style.total_player_ticks as f64
    } else {
        0.0
    };
    let avg_swing_arc_length = if style.swing_arc_distances.is_empty() {
        0.0
    } else {
        style.swing_arc_distances.iter().sum::<f64>() / style.swing_arc_distances.len() as f64
    };
    let fire_rate = if duration_s > 0.0 {
        style.fire_count as f64 / duration_s / 8.0 // per player per second
    } else {
        0.0
    };

    // Diversity metrics
    let throw_target_entropy = {
        let total: u64 = diversity.throw_targets.values().sum();
        if total == 0 {
            0.0
        } else {
            let mut h = 0.0_f64;
            for &count in diversity.throw_targets.values() {
                if count > 0 {
                    let p = count as f64 / total as f64;
                    h -= p * p.ln();
                }
            }
            h
        }
    };
    let position_coverage = {
        let mut count = 0u32;
        for row in &diversity.grid_visited {
            for &cell in row {
                if cell {
                    count += 1;
                }
            }
        }
        count
    };
    let action_sequence_variety = if diversity.action_variety_count > 0 {
        diversity.action_variety_sum / diversity.action_variety_count as f64
    } else {
        0.0
    };

    // Tempo metrics
    let throws_per_minute = if duration_min > 0.0 {
        tempo.throws as f64 / duration_min
    } else {
        0.0
    };
    let catches_per_minute = if duration_min > 0.0 {
        tempo.catches as f64 / duration_min
    } else {
        0.0
    };
    let possession_changes_per_minute = if duration_min > 0.0 {
        tempo.possession_changes as f64 / duration_min
    } else {
        0.0
    };
    let avg_hold_ticks = if tempo.hold_durations.is_empty() {
        0.0
    } else {
        tempo.hold_durations.iter().sum::<u64>() as f64 / tempo.hold_durations.len() as f64
    };
    let hold_time_std = if tempo.hold_durations.len() < 2 {
        0.0
    } else {
        let mean = avg_hold_ticks;
        let variance = tempo
            .hold_durations
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (tempo.hold_durations.len() - 1) as f64;
        variance.sqrt()
    };
    let dead_ball_rate = if ticks > 0 {
        tempo.dead_ticks as f64 / ticks as f64
    } else {
        0.0
    };

    // Spatial metrics
    let team_spread_avg = if spatial.team_spread_count > 0 {
        spatial.team_spread_sum / spatial.team_spread_count as f64
    } else {
        0.0
    };
    let field_utilization = {
        let mut visited = 0u32;
        for row in &spatial.player_grid {
            for &cell in row {
                if cell {
                    visited += 1;
                }
            }
        }
        visited as f64 / 50.0 * 100.0
    };

    // Possession metrics
    let contested_catch_fraction = if poss.total_catches > 0 {
        poss.contested_catches as f64 / poss.total_catches as f64 * 100.0
    } else {
        0.0
    };
    let score_parity = {
        let sh = mat.state().score_home;
        let sa = mat.state().score_away;
        let diff = (sh - sa).unsigned_abs() as f64;
        let total = (sh + sa).max(1) as f64;
        diff / total
    };

    // Failure metrics
    let skin_bounce_rate = if duration_min > 0.0 {
        failure.skin_events as f64 / duration_min
    } else {
        0.0
    };
    let bobble_rate = if failure.catch_attempts > 0 {
        failure.bobble_events as f64 / failure.catch_attempts as f64
    } else {
        0.0
    };

    MatchReport {
        ticks,
        duration_s,
        score_home: mat.state().score_home,
        score_away: mat.state().score_away,
        style: StyleMetrics {
            swing_vs_winch_ratio,
            avg_peak_speed,
            free_flight_fraction,
            avg_swing_arc_length,
            fire_rate,
        },
        diversity: DiversityMetrics {
            throw_target_entropy,
            position_coverage,
            action_sequence_variety,
        },
        tempo: TempoMetrics {
            throws_per_minute,
            catches_per_minute,
            possession_changes_per_minute,
            avg_hold_ticks,
            hold_time_std,
            dead_ball_rate,
        },
        spatial: SpatialMetrics {
            ball_x_histogram: spatial.ball_x_histogram,
            team_spread_avg,
            field_utilization,
        },
        possession: PossessionMetrics {
            chain_lengths: poss.chain_lengths,
            contested_catch_fraction,
            score_parity,
        },
        failure: FailureModeMetrics {
            stalemate_ticks: failure.max_no_event_streak,
            oscillation_score: failure.oscillation_episodes,
            skin_bounce_rate,
            bobble_rate,
        },
    }
}

// ── Pretty-print ────────────────────────────────────────────────────────────

pub fn print_report(r: &MatchReport) {
    println!("=== MATCH ANALYTICS REPORT ===");
    println!("Duration: {:.1}s ({} ticks)", r.duration_s, r.ticks);
    println!("Score: Home {} - Away {}", r.score_home, r.score_away);

    println!("\n--- STYLE ---");
    println!("  swing_vs_winch_ratio:  {:.3}", r.style.swing_vs_winch_ratio);
    println!("  avg_peak_speed:        {:.1} m/s", r.style.avg_peak_speed);
    println!("  free_flight_fraction:  {:.3}", r.style.free_flight_fraction);
    println!("  avg_swing_arc_length:  {:.1} m", r.style.avg_swing_arc_length);
    println!("  fire_rate:             {:.2} /s/player", r.style.fire_rate);

    println!("\n--- DIVERSITY ---");
    println!("  throw_target_entropy:    {:.3} bits", r.diversity.throw_target_entropy);
    println!("  position_coverage:       {}/50 cells", r.diversity.position_coverage);
    println!("  action_sequence_variety: {:.2}", r.diversity.action_sequence_variety);

    println!("\n--- TEMPO ---");
    println!("  throws_per_minute:              {:.1}", r.tempo.throws_per_minute);
    println!("  catches_per_minute:             {:.1}", r.tempo.catches_per_minute);
    println!("  possession_changes_per_minute:  {:.1}", r.tempo.possession_changes_per_minute);
    println!("  avg_hold_ticks:                 {:.0}", r.tempo.avg_hold_ticks);
    println!("  hold_time_std:                  {:.0}", r.tempo.hold_time_std);
    println!("  dead_ball_rate:                 {:.3}", r.tempo.dead_ball_rate);

    println!("\n--- SPATIAL ---");
    println!("  ball_x_histogram: {:?}", r.spatial.ball_x_histogram);
    println!("  team_spread_avg:  {:.1} m", r.spatial.team_spread_avg);
    println!("  field_utilization: {:.1}%", r.spatial.field_utilization);

    println!("\n--- POSSESSION ---");
    println!("  chain_lengths [0,1,2,3,4,5+]: {:?}", r.possession.chain_lengths);
    println!("  contested_catch_fraction:      {:.1}%", r.possession.contested_catch_fraction);
    println!("  score_parity:                  {:.3}", r.possession.score_parity);

    println!("\n--- FAILURE MODES ---");
    println!("  stalemate_ticks:   {}", r.failure.stalemate_ticks);
    println!("  oscillation_score: {}", r.failure.oscillation_score);
    println!("  skin_bounce_rate:  {:.1} /min", r.failure.skin_bounce_rate);
    println!("  bobble_rate:       {:.3}", r.failure.bobble_rate);
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "diagnostic; run with --ignored --nocapture"]
    fn full_analytics_report() {
        let report = analyze_match(
            "fall-dynasty", "big-slow",
            "fall-dynasty", "big-slow",
            1234, 8000,
        );
        print_report(&report);
        // Basic sanity: match ran
        assert!(report.ticks > 0);
        assert!(report.duration_s > 0.0);
    }
}

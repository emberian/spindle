//! 1:1 faithful Rust port of src/ai/Director.ts (the FROZEN SPEC).
//!
//! Director — per-team strategic AI layer, runs at ~2 Hz (every 30 sim
//! ticks). It produces a *committed* plan that the per-player role policies
//! execute UNCHANGED until the next Director tick.
//!
//! The shared output types (`DirectorState`, `PlayerAssignment`,
//! `CastPosture`, `Job`) live in `crate::ai::decision_types` — this module
//! imports and constructs them, never redefines them.
//!
//! Mapping notes (TS → Rust):
//!   * TS `number` → `f64`; `T | null` → `Option<T>`; the `job` string-union
//!     → the substrate `Job` enum; the posture string-union → `CastPosture`.
//!   * `Record<string, PlayerAssignment>` → `HashMap<String, PlayerAssignment>`.
//!   * `rng: () => number` → `rng: &mut AiRng`; every `rng()` → `rng.next()`,
//!     preserving the EXACT draw order/count (Director draws exactly once:
//!     `styleNoise = rng() * 2 - 1`, after all assignment logic).
//!   * `REG.R` (Director.ts) → `crate::tuning::R`; `GATE_X` →
//!     `crate::tuning::GATE_X`.
//!   * `Math.sign` → `f64::signum` (matches for the non-zero ring x's here;
//!     `aRingX`/`faithX` are always `±GATE_X`, never 0 or NaN, so
//!     `signum` is behaviour-equivalent to JS `Math.sign`).
//!
//! Director.ts line spans ported:
//!   * 101-113  `axisRadius` / `dist3` helpers
//!   * 119-126  `evFaith` → `ev_faith`
//!   * 132-142  `evFree`  → `ev_free`
//!   * 151-174  `choosePosture` → `choose_posture`
//!   * 180-192  `pickLoopSetter` → `pick_loop_setter`
//!   * 198-223  `shouldContest` → `should_contest`
//!   * 230-302  `assignMarks` → `assign_marks`
//!   * 308-385  `assignReceivers` → `assign_receivers`
//!   * 392-556  `runDirector` → `run_director`
//!
//! ScoreEV symbols used: `crate::ai::score_ev::{p_fall, p_rise, p_loop}`
//! (the TS `import { pFall, pRise, pLoop } from './decide/ScoreEV'`).

use super::decision_types::{
    ActivePlay, CastPosture, DirectorState, Job, PlayAssignment, PlayKind, PlayRole,
    PlayerAssignment, DIRECTOR_TICK_INTERVAL,
};
use super::efe;
use super::orientation::{
    attack_ring_x, attack_sign, defend_ring_x, forward_progress, gate_world_x,
};
use super::profile::TeamProfile;
use super::rng::AiRng;
use super::score_ev::{p_fall, p_loop, p_rise};
use super::types::{FaithEnd, MatchState, PlayerSim, RiggerRole, SimState, TeamSide};
use crate::math::Vec3;
use crate::tuning::{GATE_X, R as REG_R};
use std::collections::HashMap;

// ── small geometry helpers (Director.ts:101-113) ─────────────────────────────

/// `axisRadius(p) = sqrt(p.y*p.y + p.z*p.z)` (Director.ts:101-103).
fn axis_radius(p: &Vec3) -> f64 {
    (p.y * p.y + p.z * p.z).sqrt()
}

/// `dist3(a, b)` (Director.ts:105-113).
fn dist3(a: &Vec3, b: &Vec3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ── EV calculators (public surface) ──────────────────────────────────────────

/// Compute the EV of throwing toward the Faith (spinward) ring.
/// `E[Faith] = p_fall · 2` (Director.ts:119-126).
pub fn ev_faith(bell_pos: &Vec3, faith_x: f64) -> f64 {
    let dist = (bell_pos.x - faith_x).abs();
    let radius = axis_radius(bell_pos);
    p_fall(dist, radius) * 2.0
}

/// Compute the EV of throwing toward the Free (antispinward) ring.
/// `E[Free] = p_rise · 5 + p_loop · 7` (Director.ts:132-142).
pub fn ev_free(bell_pos: &Vec3, free_x: f64, loop_propensity: f64) -> f64 {
    let dist = (bell_pos.x - free_x).abs();
    let radius = axis_radius(bell_pos);
    let pr = p_rise(dist, radius);
    let pl = p_loop(radius, dist, loop_propensity);
    pr * 5.0 + pl * 7.0
}

// ── posture (Director.ts:151-174) ────────────────────────────────────────────

/// Choose cast posture based on score differential, inning, possession,
/// spine. Branch order is identical to the TS (first match wins).
fn choose_posture(
    m: &MatchState,
    profile: &TeamProfile,
    team_side: TeamSide,
    ev_faith_val: f64,
    ev_free_val: f64,
) -> CastPosture {
    let my_score = if team_side == TeamSide::Home {
        m.score_home
    } else {
        m.score_away
    };
    let opp_score = if team_side == TeamSide::Home {
        m.score_away
    } else {
        m.score_home
    };
    let score_diff = my_score - opp_score;
    // `Math.max(1, 9 - inning + (spine ? 0 : 1))`.
    let innings_left = (9.0 - m.inning + if m.spine { 0.0 } else { 1.0 }).max(1.0);

    // Spine (overtime) or significantly behind → chase.
    if score_diff < -6.0 || (m.spine && score_diff <= 0.0) {
        return CastPosture::Chase;
    }
    // Behind late → chase even on a smaller deficit.
    if score_diff < 0.0 && innings_left <= 3.0 {
        return CastPosture::Chase;
    }
    // Comfortably ahead with few innings left → grind out the clock.
    if score_diff > 4.0 && innings_left <= 3.0 {
        return CastPosture::Grind;
    }
    // Ahead, mid-game → mild grind for high-discipline teams.
    if score_diff > 2.0 && profile.variance < 0.35 {
        return CastPosture::Grind;
    }
    // High-variance teams chase when the Free EV is genuinely there.
    if profile.variance > 0.65 && ev_free_val > ev_faith_val * 1.2 {
        return CastPosture::Chase;
    }
    CastPosture::Tempo
}

// ── loop-setter pick (Director.ts:180-192) ───────────────────────────────────

/// Pick which Spinner should set the loop play this cast. Prefers Spinners
/// already near the axis (low axis-radius). Iterates `players` in order,
/// strict `<` so the FIRST minimal-radius spinner wins (TS-identical).
fn pick_loop_setter(players: &[PlayerSim], team_side: TeamSide) -> Option<String> {
    let mut best: Option<&PlayerSim> = None;
    let mut best_radius = f64::INFINITY;
    for p in players {
        if p.team != team_side || p.role != RiggerRole::Spinner {
            continue;
        }
        let r = axis_radius(&p.p);
        if r < best_radius {
            best_radius = r;
            best = Some(p);
        }
    }
    best.map(|p| p.id.clone())
}

// ── contest decision (Director.ts:198-223) ───────────────────────────────────

/// Decide whether to commit to a contest. Pure heuristic.
fn should_contest(
    m: &MatchState,
    profile: &TeamProfile,
    team_side: TeamSide,
    posture: CastPosture,
    contest_radius: f64,
) -> bool {
    let mut score = profile.contest_aggression;

    let radius_frac = contest_radius / REG_R;
    if radius_frac > 0.7 {
        score -= 0.1; // deep is risky even for the brave
    } else {
        score += 0.05; // high (near axis) contests are reversible
    }

    if posture == CastPosture::Chase {
        score += 0.15;
    }
    if posture == CastPosture::Grind {
        score -= 0.1;
    }

    // Whose contest is it? If we're behind, "commit" means hold/fight.
    if m.contest.is_some() {
        let my_score = if team_side == TeamSide::Home {
            m.score_home
        } else {
            m.score_away
        };
        let opp_score = if team_side == TeamSide::Home {
            m.score_away
        } else {
            m.score_home
        };
        if my_score - opp_score < 0.0 {
            score += 0.08; // behind → fight for it
        }
    }

    score > 0.5
}

// ── defensive marks (Director.ts:230-302) ────────────────────────────────────

/// Assign defensive marks: each of our players man-marks the nearest
/// unmarked opponent (greedy). The opponent holding the bell gets priority.
fn assign_marks(
    my_players: &[PlayerSim],
    opponents: &[PlayerSim],
    bell_holder_id: Option<&str>,
    assignments: &mut HashMap<String, PlayerAssignment>,
    aggression: f64,
) {
    if opponents.is_empty() {
        return;
    }
    let mut used_opp: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Sort opponents by threat: bell holder first. The TS comparator returns
    // -1 if `a` is the holder, 1 if `b` is the holder, else 0. With a stable
    // sort this floats the (single) holder ahead while preserving the
    // original order of everyone else — replicated here with a stable sort.
    let mut ranked: Vec<&PlayerSim> = opponents.iter().collect();
    ranked.sort_by(|a, b| {
        let a_holder = bell_holder_id == Some(a.id.as_str());
        let b_holder = bell_holder_id == Some(b.id.as_str());
        if a_holder {
            std::cmp::Ordering::Less
        } else if b_holder {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });

    // DYNAMISM: defenders HUNT. Carrier-marker presses onto the carrier; the
    // next two get close shadows; the rest a tighter goal-side screen.
    // `defenders` is the mutable pool; `splice(bestI,1)` → swap-free remove.
    let mut defenders: Vec<&PlayerSim> = my_players.iter().collect();
    let mut rank: i32 = 0;
    for opp in &ranked {
        if defenders.is_empty() {
            break;
        }
        if used_opp.contains(&opp.id) {
            continue;
        }
        let mut best_i: i32 = -1;
        let mut best_d = f64::INFINITY;
        for (i, def) in defenders.iter().enumerate() {
            let d = dist3(&def.p, &opp.p);
            if d < best_d - 1e-6 {
                best_d = d;
                best_i = i as i32;
            }
        }
        if best_i < 0 {
            break;
        }
        // Array `.splice(bestI, 1)[0]` removes & returns that element,
        // shifting the rest left (order-preserving) — `Vec::remove`.
        let def = defenders.remove(best_i as usize);
        let is_carrier = bell_holder_id == Some(opp.id.as_str());
        let pressure: f64 = if is_carrier {
            (0.85 + aggression * 0.15).min(1.0)
        } else if rank <= 2 {
            (0.28 + aggression * 0.30).min(0.55)
        } else {
            (0.12 + aggression * 0.23).min(0.35)
        };
        assignments.insert(
            def.id.clone(),
            PlayerAssignment {
                job: Job::Mark,
                mark_id: Some(opp.id.clone()),
                depth_slot: 0.0,
                radius_slot: 0.0,
                pressure,
            },
        );
        used_opp.insert(opp.id.clone());
        rank += 1;
    }

    // Leftover defenders roam a help zone at varied radii.
    let mut zi: i32 = 0;
    for def in &defenders {
        assignments.insert(
            def.id.clone(),
            PlayerAssignment {
                job: Job::Zone,
                mark_id: None,
                depth_slot: 0.0,
                radius_slot: 0.3 + (zi % 3) as f64 * 0.28,
                pressure: 0.0,
            },
        );
        zi += 1;
    }
}

// ── offensive receiver slots (Director.ts:308-385) ───────────────────────────

/// Assign offensive receiver slots: spread receivers across distinct axial
/// depths and axis-radii so the carrier always has well-separated outlets.
fn assign_receivers(
    receivers: &[PlayerSim],
    loop_setter_id: Option<&str>,
    assignments: &mut HashMap<String, PlayerAssignment>,
) {
    // Stable ordering by id (ids unique, so the TS `a.id < b.id ? -1 : 1`
    // total order is exactly an ascending sort by id).
    let mut ordered: Vec<&PlayerSim> = receivers.iter().collect();
    ordered.sort_by(|a, b| a.id.cmp(&b.id));
    let n = ordered.len();
    let non_loop_count = ordered
        .iter()
        .filter(|p| Some(p.id.as_str()) != loop_setter_id)
        .count();
    let mut non_loop_index: usize = 0;
    for p in ordered.iter().take(n) {
        if Some(p.id.as_str()) == loop_setter_id {
            // Loop-setter sits high & near the carrier's depth.
            assignments.insert(
                p.id.clone(),
                PlayerAssignment {
                    job: Job::Receive,
                    mark_id: None,
                    depth_slot: 0.4,
                    radius_slot: 0.15,
                    pressure: 0.0,
                },
            );
            continue;
        }
        let k = non_loop_index;
        non_loop_index += 1;
        if k == 0 {
            // The mandatory safe outlet: shallow, mid-radius.
            assignments.insert(
                p.id.clone(),
                PlayerAssignment {
                    job: Job::Receive,
                    mark_id: None,
                    depth_slot: 0.18,
                    radius_slot: 0.35,
                    pressure: 0.0,
                },
            );
            continue;
        }
        // Spread the rest from a mid outlet to a DEEP scoring threat.
        let frac = if non_loop_count > 2 {
            (k - 1) as f64 / (non_loop_count - 2) as f64
        } else {
            0.6
        };
        let depth_slot = 0.45 + frac * 0.54; // 0.45 (mid) .. 0.99 (ring)
        let radius_slot: f64 = match p.role {
            RiggerRole::Freewing => 0.12,  // high / near-axis
            RiggerRole::Faithwing => 0.38, // spinward, in corridor
            RiggerRole::Anchor => 0.48,    // outer band, still reachable
            RiggerRole::Spinner => 0.28,
            // TS `default:` — the only remaining role is `reach`.
            _ => 0.22 + 0.26 * frac,
        };
        assignments.insert(
            p.id.clone(),
            PlayerAssignment {
                job: Job::Receive,
                mark_id: None,
                depth_slot,
                radius_slot,
                pressure: 0.0,
            },
        );
    }
}

// ── PLAY SELECTION ─────────────────────────────────────────────────────────────
// Coordinated multi-player play patterns. This layer runs AFTER role/job
// assignment and adds interlocking nav targets ON TOP — it does not override
// the Director's Job assignments.

/// Maximum tube radius for clamped positions (rho < R * 0.85).
const TUBE_CLAMP: f64 = REG_R * 0.85;

/// Duration (in ticks) a play remains valid before re-evaluation.
const PLAY_DURATION_TICKS: f64 = 90.0; // ~3 Director windows at 2Hz (30 ticks each)

/// Clamp a Vec3 so its radial component (y,z) stays inside the tube.
fn clamp_inside_tube(p: Vec3) -> Vec3 {
    let rho = (p.y * p.y + p.z * p.z).sqrt();
    if rho <= TUBE_CLAMP {
        p
    } else {
        let scale = TUBE_CLAMP / rho;
        Vec3::new(p.x, p.y * scale, p.z * scale)
    }
}

/// Select a coordinated play pattern based on game state. Pure + deterministic.
fn select_play(
    state: &SimState,
    _m: &MatchState,
    team_side: TeamSide,
    _profile: &TeamProfile,
    director_out: &DirectorState,
) -> ActivePlay {
    let a_sign = director_out.attack_sign;
    let a_ring_x = director_out.attack_ring_x;
    let d_ring_x = director_out.defend_ring_x;
    let tick = state.tick;

    // Gather our team's players (excluding P1 human) and opponents.
    let my_players: Vec<&PlayerSim> = state
        .players
        .iter()
        .filter(|p| p.team == team_side && p.id != "P1")
        .collect();
    let opponents: Vec<&PlayerSim> = state
        .players
        .iter()
        .filter(|p| p.team != team_side)
        .collect();

    // ── Phase determination ────────────────────────────────────────────────
    let bell_pos = state.bell.p;
    let _held_by_us = director_out.has_possession && !director_out.bell_loose;
    let bell_loose = director_out.bell_loose;
    let opponent_has_ball = !director_out.has_possession && !bell_loose;

    // ── LOOSE BALL ─────────────────────────────────────────────────────────
    if bell_loose {
        // Mark the play kind so rigger_ai knows a loose-ball play is active.
        // Leave concrete assignments minimal — the existing recover/shadow
        // system handles the actual chasing.
        return ActivePlay {
            kind: PlayKind::LooseBallRecovery,
            started_tick: tick,
            assignments: HashMap::new(),
        };
    }

    // ── DEFENSE (opponent has possession) ──────────────────────────────────
    if opponent_has_ball {
        return select_recovery_formation(
            state, &my_players, bell_pos, d_ring_x, a_sign, tick,
        );
    }

    // ── OFFENSE (we hold the bell) ─────────────────────────────────────────
    // Find the carrier.
    let carrier: Option<&PlayerSim> = director_out
        .carrier_id
        .as_ref()
        .and_then(|cid| state.players.iter().find(|p| &p.id == cid));

    let carrier = match carrier {
        Some(c) => c,
        None => {
            // Fallback: no identifiable carrier, use SpreadAdvance.
            return select_spread_advance(
                &my_players, bell_pos, a_sign, a_ring_x, tick,
            );
        }
    };

    let carrier_pos = carrier.p;

    // Distance from carrier to attack ring.
    let dist_to_ring = (carrier_pos.x - a_ring_x).abs();

    // Nearest defender distance to carrier.
    let nearest_def_dist = opponents
        .iter()
        .map(|o| dist3(&carrier_pos, &o.p))
        .fold(f64::INFINITY, f64::min);

    // ── Condition evaluation (priority order) ──────────────────────────────
    if nearest_def_dist < 14.0 {
        // Carrier under immediate pressure → PressureRelease.
        return select_pressure_release(
            carrier, &my_players, a_sign, tick,
        );
    }

    if dist_to_ring < 120.0 {
        // Near the attack gate with a reasonably clear lane.
        // "Reasonably clear" = no defender within 10m of the direct axis
        // path from carrier to ring.
        let lane_clear = opponents.iter().all(|o| {
            // Project opponent onto the carrier→ring axis line and check
            // perpendicular distance.
            let dx = a_ring_x - carrier_pos.x;
            if dx.abs() < 1.0 {
                return true; // already at ring
            }
            let t = ((o.p.x - carrier_pos.x) / dx).clamp(0.0, 1.0);
            let proj = Vec3::new(
                carrier_pos.x + t * dx,
                carrier_pos.y,
                carrier_pos.z,
            );
            dist3(&o.p, &proj) > 10.0
        });
        if lane_clear {
            return select_gate_run(
                carrier, &my_players, &opponents, a_sign, a_ring_x, tick,
            );
        }
    }

    // Default offensive play: OutletChain.
    select_outlet_chain(carrier, &my_players, &opponents, a_sign, a_ring_x, tick)
}

/// GateRun: carrier advances to within 50-70m of gate, receivers stage
/// near and past the gate plane.
fn select_gate_run(
    carrier: &PlayerSim,
    my_players: &[&PlayerSim],
    opponents: &[&PlayerSim],
    a_sign: f64,
    a_ring_x: f64,
    tick: f64,
) -> ActivePlay {
    let mut play_assignments: HashMap<String, PlayAssignment> = HashMap::new();

    // Carrier target: advance to 50-70m from the gate, stay near axis.
    let carrier_target_x = a_ring_x - a_sign * 60.0; // 60m from ring
    let carrier_target = clamp_inside_tube(Vec3::new(carrier_target_x, 0.0, 0.0));
    play_assignments.insert(
        carrier.id.clone(),
        PlayAssignment {
            play_role: PlayRole::Carrier,
            target: carrier_target,
        },
    );

    // Find teammates (non-carrier).
    let teammates: Vec<&&PlayerSim> = my_players
        .iter()
        .filter(|p| p.id != carrier.id)
        .collect();

    // Primary receiver: best-positioned teammate near the gate (furthest
    // forward with best separation from defenders).
    let primary = pick_best_receiver(&teammates, opponents, a_sign);
    if let Some(prim) = primary {
        // Just past gate plane, on-axis, offset slightly in Coriolis direction.
        // Coriolis-favored = positive y (spinward drift in our coordinate frame).
        let prim_target = clamp_inside_tube(Vec3::new(
            a_ring_x - a_sign * 15.0, // just before gate
            5.0,                        // slight Coriolis offset
            0.0,
        ));
        play_assignments.insert(
            prim.id.clone(),
            PlayAssignment {
                play_role: PlayRole::PrimaryReceiver,
                target: prim_target,
            },
        );

        // Secondary: hold 25m ahead of carrier at medium radius — safety valve.
        let secondary = teammates
            .iter()
            .filter(|p| p.id != prim.id)
            .min_by(|a, b| {
                dist3(&a.p, &carrier.p)
                    .partial_cmp(&dist3(&b.p, &carrier.p))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(sec) = secondary {
            let sec_target = clamp_inside_tube(Vec3::new(
                carrier.p.x + a_sign * 25.0,
                0.0,
                15.0, // medium radius offset
            ));
            play_assignments.insert(
                sec.id.clone(),
                PlayAssignment {
                    play_role: PlayRole::SecondaryReceiver,
                    target: sec_target,
                },
            );
        }
    }

    // Remaining teammates: Screen or DeepOption.
    let assigned_ids: Vec<String> = play_assignments.keys().cloned().collect();
    for tm in &teammates {
        if assigned_ids.contains(&tm.id) {
            continue;
        }
        let role_target = clamp_inside_tube(Vec3::new(
            carrier.p.x + a_sign * 10.0,
            tm.p.y,
            tm.p.z,
        ));
        play_assignments.insert(
            tm.id.clone(),
            PlayAssignment {
                play_role: PlayRole::Screen,
                target: role_target,
            },
        );
    }

    ActivePlay {
        kind: PlayKind::GateRun,
        started_tick: tick,
        assignments: play_assignments,
    }
}

/// OutletChain: carrier advances, receivers spread at 60% and 20m distances.
fn select_outlet_chain(
    carrier: &PlayerSim,
    my_players: &[&PlayerSim],
    opponents: &[&PlayerSim],
    a_sign: f64,
    a_ring_x: f64,
    tick: f64,
) -> ActivePlay {
    let mut play_assignments: HashMap<String, PlayAssignment> = HashMap::new();

    // Carrier: advance at current rate (target slightly ahead).
    let carrier_target = clamp_inside_tube(Vec3::new(
        carrier.p.x + a_sign * 20.0,
        carrier.p.y,
        carrier.p.z,
    ));
    play_assignments.insert(
        carrier.id.clone(),
        PlayAssignment {
            play_role: PlayRole::Carrier,
            target: carrier_target,
        },
    );

    let teammates: Vec<&&PlayerSim> = my_players
        .iter()
        .filter(|p| p.id != carrier.id)
        .collect();

    // Primary: a catchable pass distance ahead (40-80m), capped so the
    // receiver stays within throw range rather than sprinting to half-field.
    let dist_to_ring = (carrier.p.x - a_ring_x).abs();
    let primary = pick_best_receiver(&teammates, opponents, a_sign);
    if let Some(prim) = primary {
        let lead = (dist_to_ring * 0.3).min(80.0).max(40.0);
        let prim_x = carrier.p.x + a_sign * lead;
        let prim_target = clamp_inside_tube(Vec3::new(prim_x, 0.0, 20.0));
        play_assignments.insert(
            prim.id.clone(),
            PlayAssignment {
                play_role: PlayRole::PrimaryReceiver,
                target: prim_target,
            },
        );

        // Secondary: 20m ahead of carrier, radius 8-12m (short dump).
        let secondary = teammates
            .iter()
            .filter(|p| p.id != prim.id)
            .min_by(|a, b| {
                dist3(&a.p, &carrier.p)
                    .partial_cmp(&dist3(&b.p, &carrier.p))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(sec) = secondary {
            let sec_target = clamp_inside_tube(Vec3::new(
                carrier.p.x + a_sign * 20.0,
                0.0,
                10.0,
            ));
            play_assignments.insert(
                sec.id.clone(),
                PlayAssignment {
                    play_role: PlayRole::SecondaryReceiver,
                    target: sec_target,
                },
            );
        }
    }

    // Remaining teammates: DeepOption.
    let assigned_ids: Vec<String> = play_assignments.keys().cloned().collect();
    for tm in &teammates {
        if assigned_ids.contains(&tm.id) {
            continue;
        }
        let deep_target = clamp_inside_tube(Vec3::new(
            carrier.p.x + a_sign * 40.0,
            tm.p.y,
            tm.p.z,
        ));
        play_assignments.insert(
            tm.id.clone(),
            PlayAssignment {
                play_role: PlayRole::DeepOption,
                target: deep_target,
            },
        );
    }

    ActivePlay {
        kind: PlayKind::OutletChain,
        started_tick: tick,
        assignments: play_assignments,
    }
}

/// PressureRelease: carrier holds, nearest cuts toward, rest holds.
fn select_pressure_release(
    carrier: &PlayerSim,
    my_players: &[&PlayerSim],
    _a_sign: f64,
    tick: f64,
) -> ActivePlay {
    let mut play_assignments: HashMap<String, PlayAssignment> = HashMap::new();

    // Carrier: hold current position.
    let carrier_target = clamp_inside_tube(carrier.p);
    play_assignments.insert(
        carrier.id.clone(),
        PlayAssignment {
            play_role: PlayRole::Carrier,
            target: carrier_target,
        },
    );

    let teammates: Vec<&&PlayerSim> = my_players
        .iter()
        .filter(|p| p.id != carrier.id)
        .collect();

    // Primary (nearest teammate): cut TOWARD carrier (halve the gap).
    let primary = teammates
        .iter()
        .min_by(|a, b| {
            dist3(&a.p, &carrier.p)
                .partial_cmp(&dist3(&b.p, &carrier.p))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    if let Some(prim) = primary {
        // Halve the gap between primary and carrier.
        let midpoint = Vec3::new(
            (prim.p.x + carrier.p.x) * 0.5,
            (prim.p.y + carrier.p.y) * 0.5,
            (prim.p.z + carrier.p.z) * 0.5,
        );
        let prim_target = clamp_inside_tube(midpoint);
        play_assignments.insert(
            prim.id.clone(),
            PlayAssignment {
                play_role: PlayRole::PrimaryReceiver,
                target: prim_target,
            },
        );

        // Secondary: stay put — the quick dump resolves in one throw.
        let secondary = teammates
            .iter()
            .filter(|p| p.id != prim.id)
            .min_by(|a, b| {
                dist3(&a.p, &carrier.p)
                    .partial_cmp(&dist3(&b.p, &carrier.p))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(sec) = secondary {
            let sec_target = clamp_inside_tube(sec.p);
            play_assignments.insert(
                sec.id.clone(),
                PlayAssignment {
                    play_role: PlayRole::SecondaryReceiver,
                    target: sec_target,
                },
            );
        }
    }

    // Remaining teammates: Screen (hold position).
    let assigned_ids: Vec<String> = play_assignments.keys().cloned().collect();
    for tm in &teammates {
        if assigned_ids.contains(&tm.id) {
            continue;
        }
        let screen_target = clamp_inside_tube(tm.p);
        play_assignments.insert(
            tm.id.clone(),
            PlayAssignment {
                play_role: PlayRole::Screen,
                target: screen_target,
            },
        );
    }

    ActivePlay {
        kind: PlayKind::PressureRelease,
        started_tick: tick,
        assignments: play_assignments,
    }
}

/// RecoveryFormation: station players between bell and our ring at staggered
/// depths.
fn select_recovery_formation(
    _state: &SimState,
    my_players: &[&PlayerSim],
    bell_pos: Vec3,
    d_ring_x: f64,
    a_sign: f64,
    tick: f64,
) -> ActivePlay {
    let mut play_assignments: HashMap<String, PlayAssignment> = HashMap::new();

    // Sort players by distance to the bell (nearest first).
    let mut sorted: Vec<&PlayerSim> = my_players.to_vec();
    sorted.sort_by(|a, b| {
        dist3(&a.p, &bell_pos)
            .partial_cmp(&dist3(&b.p, &bell_pos))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // The axis between bell and our ring.
    let bell_to_ring_x = d_ring_x - bell_pos.x;

    for (i, player) in sorted.iter().enumerate() {
        let (role, target) = match i {
            0 => {
                // Station 1 (nearest to bell): between bell and our ring, 30m back.
                let t_x = bell_pos.x - a_sign * 30.0;
                (PlayRole::PrimaryReceiver, clamp_inside_tube(Vec3::new(t_x, 0.0, 0.0)))
            }
            1 => {
                // Station 2 (midfield): halfway between bell and our ring.
                let t_x = bell_pos.x + bell_to_ring_x * 0.5;
                (PlayRole::SecondaryReceiver, clamp_inside_tube(Vec3::new(t_x, 0.0, 10.0)))
            }
            2 => {
                // Station 3 (deep): near our ring — last back.
                let t_x = d_ring_x - a_sign * 40.0; // 40m in front of our ring
                (PlayRole::DeepOption, clamp_inside_tube(Vec3::new(t_x, 0.0, -10.0)))
            }
            _ => {
                // Extra players: screen at varied positions.
                let t_x = bell_pos.x + bell_to_ring_x * (0.3 + 0.15 * i as f64);
                (PlayRole::Screen, clamp_inside_tube(Vec3::new(t_x, player.p.y, player.p.z)))
            }
        };
        play_assignments.insert(
            player.id.clone(),
            PlayAssignment {
                play_role: role,
                target,
            },
        );
    }

    ActivePlay {
        kind: PlayKind::RecoveryFormation,
        started_tick: tick,
        assignments: play_assignments,
    }
}

/// SpreadAdvance fallback: spread players forward.
fn select_spread_advance(
    my_players: &[&PlayerSim],
    bell_pos: Vec3,
    a_sign: f64,
    _a_ring_x: f64,
    tick: f64,
) -> ActivePlay {
    let mut play_assignments: HashMap<String, PlayAssignment> = HashMap::new();

    for (i, player) in my_players.iter().enumerate() {
        let offset = 20.0 + 15.0 * i as f64;
        let lateral = ((i as f64) * 12.0) - 18.0; // spread across y
        let target = clamp_inside_tube(Vec3::new(
            bell_pos.x + a_sign * offset,
            lateral,
            0.0,
        ));
        let role = match i {
            0 => PlayRole::Carrier,
            1 => PlayRole::PrimaryReceiver,
            2 => PlayRole::SecondaryReceiver,
            _ => PlayRole::DeepOption,
        };
        play_assignments.insert(
            player.id.clone(),
            PlayAssignment {
                play_role: role,
                target,
            },
        );
    }

    ActivePlay {
        kind: PlayKind::SpreadAdvance,
        started_tick: tick,
        assignments: play_assignments,
    }
}

/// Pick the best receiver from teammates: the one with the most forward
/// progress and best separation from defenders. Deterministic (id tie-break).
fn pick_best_receiver<'a>(
    teammates: &[&&'a PlayerSim],
    opponents: &[&PlayerSim],
    _a_sign: f64,
) -> Option<&'a PlayerSim> {
    let mut best: Option<(f64, &'a PlayerSim)> = None;
    for tm in teammates {
        let fwd = forward_progress(tm.team, tm.p.x);
        // Separation: nearest opponent distance (bounded).
        let mut nearest_opp = f64::INFINITY;
        for o in opponents {
            let d = dist3(&tm.p, &o.p);
            if d < nearest_opp {
                nearest_opp = d;
            }
        }
        if !nearest_opp.is_finite() {
            nearest_opp = 60.0;
        }
        // Score: forward progress + bounded separation bonus.
        let score = fwd + nearest_opp.min(40.0) * 0.5;
        let take = match &best {
            None => true,
            Some((bs, bp)) => {
                score > *bs + 1e-9
                    || ((score - *bs).abs() <= 1e-9 && tm.id < bp.id)
            }
        };
        if take {
            best = Some((score, *tm));
        }
    }
    best.map(|(_, p)| p)
}

// ── the Director itself (Director.ts:392-556) ────────────────────────────────

/// Run the Director for one team. Call this at ~2 Hz; cache the result.
/// The returned plan is meant to be held CONSTANT by every per-player
/// policy until the next Director tick.
pub fn run_director(
    state: &SimState,
    m: &MatchState,
    profile: &TeamProfile,
    team_side: TeamSide,
    rng: &mut AiRng,
) -> DirectorState {
    // ORIENTATION: physical attack direction is FIXED by team, NOT faithEnd.
    let a_sign = attack_sign(team_side);
    let a_ring_x = attack_ring_x(team_side);
    let d_ring_x = defend_ring_x(team_side);
    // Is the ring we physically attack the Faith (2-pt) or Free (5-pt) ring?
    let faith_x = match m.faith_end {
        FaithEnd::PlusX => GATE_X,
        FaithEnd::MinusX => -GATE_X,
    };
    let free_x = -faith_x;
    let our_ring_is_faith = a_ring_x.signum() == faith_x.signum();

    // evFaith/evFree keep their LEGACY meaning (abs distance to a fixed ring).
    let ef = ev_faith(&state.bell.p, faith_x);
    let efr = ev_free(&state.bell.p, free_x, profile.loop_propensity);

    let posture = choose_posture(m, profile, team_side, ef, efr);

    // The ring THIS team physically attacks is the Free (5-pt) ring.
    let attacking_free = !our_ring_is_faith;

    let use_loop = profile.loop_propensity > 0.25 && posture != CastPosture::Grind;
    let loop_setter_id = if use_loop {
        pick_loop_setter(&state.players, team_side)
    } else {
        None
    };

    // Possession / loose-bell determination.
    // P1 is always the human-controlled player in live play (the AI system
    // never emits inputs for it). In full-AI matches (skill_eval) there is
    // no player named "P1" so this filter is a harmless no-op.
    let my_players: Vec<&PlayerSim> = state
        .players
        .iter()
        .filter(|p| p.team == team_side && p.id != "P1")
        .collect();
    let opponents: Vec<&PlayerSim> = state
        .players
        .iter()
        .filter(|p| p.team != team_side)
        .collect();
    let all_my_players: Vec<&PlayerSim> = state
        .players
        .iter()
        .filter(|p| p.team == team_side)
        .collect();
    let holder: Option<&PlayerSim> = match &state.bell.held_by {
        Some(hb) => state.players.iter().find(|p| &p.id == hb),
        None => None,
    };
    let bell_loose = holder.is_none();
    let held_by_us = holder.map(|h| h.team == team_side).unwrap_or(false);
    // In-flight from us counts as possession.
    let thrown_by_us = bell_loose
        && state.bell.thrown_by.is_some()
        && state
            .players
            .iter()
            .find(|p| Some(&p.id) == state.bell.thrown_by.as_ref())
            .map(|p| p.team == team_side)
            .unwrap_or(false);
    let has_possession = held_by_us || thrown_by_us || m.possession == team_side;

    // Contest decision.
    let mut contest_commit = false;
    if let Some(contest) = &m.contest {
        let contester = state.players.iter().find(|p| p.id == contest.contester);
        let contest_radius = match contester {
            Some(c) => axis_radius(&c.p),
            None => REG_R * 0.5,
        };
        contest_commit = should_contest(m, profile, team_side, posture, contest_radius);
    }

    // ── Assignments (committed for the Director window) ──────────────────────
    let mut assignments: HashMap<String, PlayerAssignment> = HashMap::new();
    let mut carrier_id: Option<String> = None;
    let mut recover_id: Option<String> = None;

    if bell_loose {
        // ── LOOSE-BELL COORDINATION (was: 3 nearest → all Recover = the
        // gawk swarm). Now a co-designed unit assigned by predicted
        // TIME-TO-INTERCEPT of the canon bell band, with distinct roles:
        //
        //   PRIMARY  (1)            min t_intercept → Job::Recover,
        //                           recover_id = this id. The sole
        //                           committed diver (is_dive_committer
        //                           keeps it exactly one).
        //   SHADOW   (1, if ≥3)     next-best t_intercept → Job::Recover
        //                           but flagged SHADOW: it does NOT dive
        //                           the live bell — it stations at the
        //                           predicted REBOUND LOCUS (efe::
        //                           rebound_locus) so a primary bobble
        //                           drops the ball onto it. It can become
        //                           the committer next window iff a
        //                           deflection genuinely wins it the race
        //                           (recover_id recompute) — never a
        //                           second simultaneous diver.
        //   OUTLET   (1, if ≥4)     best down-field free body in OPEN
        //                           space → Job::Receive, staged so a
        //                           clean catch has an immediate pass
        //                           target (catch → outlet → advance).
        //   REST                    unchanged: assign_receivers (if we
        //                           will possess) / assign_marks (deny
        //                           the opponent's best counter).
        //
        // Pure geometry + predicted t_intercept + id tie-breaks. No rng.
        let band = efe::BellBand::predict(
            state.bell.p, state.bell.v, state.omega, REG_R,
        );

        // (t_intercept, &player) for every available teammate, sorted by
        // race-winner first, id tie-break (HashMap/iteration-order-free).
        let mut raced: Vec<(f64, &PlayerSim)> = my_players
            .iter()
            .map(|p| (band.time_to_intercept(p.p, p.v), *p))
            .collect();
        raced.sort_by(|a, b| {
            if (a.0 - b.0).abs() > 1e-9 {
                a.0.partial_cmp(&b.0).unwrap()
            } else {
                a.1.id.cmp(&b.1.id)
            }
        });
        let avail = raced.len();

        // PRIMARY: the race winner.
        let primary_id: Option<String> =
            raced.first().map(|(_, p)| p.id.clone());
        recover_id = primary_id.clone();
        // SHADOW: the runner-up, but only if we can spare a 3rd body for
        // the REST (a 2-body team must not strip its whole field).
        let shadow_id: Option<String> = if avail >= 3 {
            raced.get(1).map(|(_, p)| p.id.clone())
        } else {
            None
        };

        // The recover unit (primary + maybe shadow) — assigned FIRST so
        // the REST passes below operate on the remaining bodies, then
        // RE-ASSERTED last so receiver/mark passes never clobber a role.
        let mut role_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        if let Some(pid) = &primary_id {
            role_ids.insert(pid.clone());
            assignments
                .insert(pid.clone(), PlayerAssignment::recover_primary());
        }
        if let Some(sid) = &shadow_id {
            role_ids.insert(sid.clone());
            assignments
                .insert(sid.clone(), PlayerAssignment::recover_shadow());
        }

        // OUTLET: only if we have a 4th free body (primary + shadow + at
        // least one REST defender/receiver still leaves a spare). Pick the
        // teammate furthest DOWN-FIELD (max forward_progress toward our
        // attack ring) among the non-role bodies that is also in the most
        // OPEN space (fewest nearby opponents) so a clean catch has a
        // real, uncontested pass target. Deterministic score, id
        // tie-break.
        let outlet_id: Option<String> = if avail >= 4 {
            let mut best: Option<(f64, &PlayerSim)> = None;
            for p in my_players.iter() {
                if role_ids.contains(&p.id) {
                    continue;
                }
                let fwd = forward_progress(team_side, p.p.x);
                // Openness: nearest opponent distance (bounded), bigger =
                // more open. Pure geometry over the stable opponents Vec.
                let mut nearest_opp = f64::INFINITY;
                for o in opponents.iter() {
                    let d = dist3(&p.p, &o.p);
                    if d < nearest_opp {
                        nearest_opp = d;
                    }
                }
                if !nearest_opp.is_finite() {
                    nearest_opp = 60.0;
                }
                // Down-field, in open space: forward progress plus a
                // bounded openness credit (a far opponent ⇒ safe outlet).
                let score = fwd + nearest_opp.min(30.0);
                let take = match &best {
                    None => true,
                    Some((bs, bp)) => {
                        score > *bs + 1e-9
                            || ((score - *bs).abs() <= 1e-9
                                && p.id < bp.id)
                    }
                };
                if take {
                    best = Some((score, *p));
                }
            }
            best.map(|(_, p)| p.id.clone())
        } else {
            None
        };
        if let Some(oid) = &outlet_id {
            role_ids.insert(oid.clone());
            // Job::Receive so wants_catch + the receiver/gate-route slot
            // machinery already light this body up as a pass target.
            assignments
                .insert(oid.clone(), PlayerAssignment::recover_outlet());
        }

        // REST: unchanged behavior — receivers if we'll possess, else
        // man-mark to deny the opponent's best counter-intercept.
        let rest: Vec<PlayerSim> = my_players
            .iter()
            .filter(|p| !role_ids.contains(&p.id))
            .map(|p| (*p).clone())
            .collect();
        if has_possession || thrown_by_us {
            assign_receivers(&rest, loop_setter_id.as_deref(), &mut assignments);
        } else {
            assign_marks(
                &rest,
                &opponents.iter().map(|p| (*p).clone()).collect::<Vec<_>>(),
                None,
                &mut assignments,
                profile.aggression,
            );
        }

        // Re-assert the coordinated roles LAST so the receiver/mark passes
        // never overwrite primary/shadow/outlet.
        if let Some(pid) = &primary_id {
            assignments
                .insert(pid.clone(), PlayerAssignment::recover_primary());
        }
        if let Some(sid) = &shadow_id {
            assignments
                .insert(sid.clone(), PlayerAssignment::recover_shadow());
        }
        if let Some(oid) = &outlet_id {
            assignments
                .insert(oid.clone(), PlayerAssignment::recover_outlet());
        }
    } else if held_by_us {
        // OFFENSE: the holder is the carrier; the rest are receivers.
        let cid = holder.unwrap().id.clone();
        carrier_id = Some(cid.clone());
        let receivers: Vec<PlayerSim> = my_players
            .iter()
            .filter(|p| p.id != cid)
            .map(|p| (*p).clone())
            .collect();
        assign_receivers(&receivers, loop_setter_id.as_deref(), &mut assignments);
        // carrierId may not be in myPlayers (e.g. P1) — slots still hold.
        if my_players.iter().any(|p| p.id == cid) {
            assignments.insert(
                cid.clone(),
                PlayerAssignment {
                    job: Job::Carry,
                    mark_id: None,
                    depth_slot: 1.0,
                    radius_slot: 0.4,
                    pressure: 0.0,
                },
            );
        }
    } else {
        // DEFENSE: the opponent holds it — man-mark, prioritising carrier.
        assign_marks(
            &my_players.iter().map(|p| (*p).clone()).collect::<Vec<_>>(),
            &opponents.iter().map(|p| (*p).clone()).collect::<Vec<_>>(),
            holder.map(|h| h.id.as_str()),
            &mut assignments,
            profile.aggression,
        );
    }

    // Stable style noise for the window (the Director's ONLY rng draw).
    let style_noise = rng.next() * 2.0 - 1.0;

    // Safety net: any of our players without an assignment gets a default.
    for p in &all_my_players {
        if p.id == "P1" {
            continue;
        }
        if !assignments.contains_key(&p.id) {
            let a = if has_possession {
                PlayerAssignment {
                    job: Job::Support,
                    mark_id: None,
                    depth_slot: 0.4,
                    radius_slot: 0.45,
                    pressure: 0.0,
                }
            } else {
                PlayerAssignment {
                    job: Job::Zone,
                    mark_id: None,
                    depth_slot: 0.0,
                    radius_slot: 0.5,
                    pressure: 0.0,
                }
            };
            assignments.insert(p.id.clone(), a);
        }
    }

    // KEEPER NEVER ABANDONS GOAL. Force the Reach off mark/zone onto a
    // pass-through `receive` job so the in-lane keeper policy runs.
    for p in &my_players {
        if p.role != RiggerRole::Reach {
            continue;
        }
        if let Some(a) = assignments.get(&p.id) {
            if a.job == Job::Mark || a.job == Job::Zone {
                assignments.insert(
                    p.id.clone(),
                    PlayerAssignment {
                        job: Job::Receive,
                        mark_id: None,
                        depth_slot: 0.0,
                        radius_slot: 0.0,
                        pressure: 0.0,
                    },
                );
            }
        }
    }

    // OFFENSE REBUILD — gate progression target. Stage the gate receiver a
    // little PAST the next cast-gate line (in attack dir) so a completed
    // carry→pass spends the throw with the bell past the gate ⇒ the match
    // clears it. Pick the deepest receiver as the gate runner (the existing
    // deep-threat slot); fall back to any receiver.
    let gate_lead = 22.0;
    let gate_stage_x = gate_world_x(team_side, m.cast.gate) + a_sign * gate_lead;
    // Deepest Receiver runs the gate route; deterministic id tie-break
    // (HashMap iteration order is non-deterministic — must not fold over it).
    let gate_receiver_id = {
        let mut chosen: Option<String> = None;
        let mut chosen_depth = f64::NEG_INFINITY;
        for (id, a) in assignments.iter() {
            if a.job != Job::Receive
                || Some(id.as_str()) == loop_setter_id.as_deref()
            {
                continue;
            }
            if a.depth_slot > chosen_depth + 1e-9
                || ((a.depth_slot - chosen_depth).abs() <= 1e-9
                    && chosen.as_deref().map(|c| id.as_str() < c).unwrap_or(true))
            {
                chosen_depth = a.depth_slot;
                chosen = Some(id.clone());
            }
        }
        chosen
    };

    let mut director_out = DirectorState {
        attacking_free,
        attack_sign: a_sign,
        attack_ring_x: a_ring_x,
        defend_ring_x: d_ring_x,
        posture,
        loop_setter_id,
        contest_commit,
        ev_faith: ef,
        ev_free: efr,
        update_in: DIRECTOR_TICK_INTERVAL,
        has_possession,
        bell_loose,
        carrier_id,
        recover_id,
        assignments,
        style_noise,
        gate_stage_x,
        gate_receiver_id,
        active_play: None,
    };

    // ── PLAY SELECTION ─────────────────────────────────────────────────
    // After role/job assignment, select a coordinated play pattern that
    // gives the assigned players interlocking targets. This layer does NOT
    // override Job assignments — it adds coordinated nav targets ON TOP.
    let active_play = select_play(state, m, team_side, profile, &director_out);
    director_out.active_play = Some(active_play);

    director_out
}

// ── RL-observation Director hints (a DERIVED READ for the policy obs) ───────
//
// The shared-parameter RL policy needs to SEE its Director assignment +
// the real gate-plane geometry of the controlled side (the audit found
// the obs had none, so coordination could not be learned). This helper
// runs the SAME pure `run_director` over the public state with a FIXED
// baseline profile + a FIXED seeded `AiRng` (so it is identical in the
// native gym AND the browser `policy_wasm` — train==inference) and
// projects out exactly the wasm-safe codes the policy obs carries. It is
// a DERIVED READ: it never mutates sim state, never enters
// `hash_snapshot`, and does not drive any rigger (the policy replaces the
// controlled riggers' inputs entirely — this is only a feature hint).
// Determinism: pure fn of (state, match, side); the `AiRng` seed is the
// constant 0 with the current tick, exactly the deterministic idiom
// `AiSystem` uses, no wall-clock.
pub struct ObsDirectorHints {
    /// Per-rigger-id: (Job discriminant 0..=5, mark/target id).
    pub assignments: Vec<(String, u8, Option<String>)>,
    /// Controlled side's downrange sign (+1 Home / −1 Away).
    pub attack_sign: f64,
    /// Signed world-x of the CURRENT cast-gate plane for the side.
    pub gate_plane_x: f64,
}

fn job_code(j: Job) -> u8 {
    match j {
        Job::Carry => 0,
        Job::Recover => 1,
        Job::Receive => 2,
        Job::Mark => 3,
        Job::Support => 4,
        Job::Zone => 5,
    }
}

/// Compute the policy-observation Director hints for `side`. Pure +
/// wasm-safe (no rng beyond a fixed-seed `AiRng`, no clock, no HashMap
/// fold leaking iteration order — the assignment list is emitted in the
/// ORDERED `state.players` order, then by id, a total deterministic key).
pub fn obs_director_hints(
    state: &SimState,
    m: &MatchState,
    side: TeamSide,
) -> ObsDirectorHints {
    let profile = TeamProfile::baseline();
    let tick_u = state.tick as u32;
    let mut rng = AiRng::make(0, tick_u, 0);
    let ds = run_director(state, m, &profile, side, &mut rng);

    // Emit per-rigger assignments in the ORDERED player vector order,
    // restricted to `side` (no HashMap iteration fold).
    let mut assignments: Vec<(String, u8, Option<String>)> = Vec::new();
    for p in state.players.iter() {
        if p.team != side {
            continue;
        }
        if let Some(asg) = ds.assignments.get(&p.id) {
            assignments.push((
                p.id.clone(),
                job_code(asg.job),
                asg.mark_id.clone(),
            ));
        }
    }
    ObsDirectorHints {
        assignments,
        attack_sign: attack_sign(side),
        gate_plane_x: gate_world_x(side, m.cast.gate),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{
        BellState, Cast, ContestDirection, Gate, MatchContest, MatchPhase,
    };
    use crate::math::Quat;

    fn vec(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    fn player(id: &str, team: TeamSide, role: RiggerRole, p: Vec3) -> PlayerSim {
        PlayerSim {
            id: id.to_string(),
            team,
            role,
            p,
            v: vec(0.0, 0.0, 0.0),
            q: Quat::ident(),
            line: None,
            dv_budget: 0.0,
            contact_ref: None,
            grounded: false,
        }
    }

    fn bell(p: Vec3, held_by: Option<&str>, thrown_by: Option<&str>) -> BellState {
        BellState {
            p,
            v: vec(0.0, 0.0, 0.0),
            q: Quat::ident(),
            w: vec(0.0, 0.0, 0.0),
            chime: 0.0,
            held_by: held_by.map(|s| s.to_string()),
            thrown_by: thrown_by.map(|s| s.to_string()),
            touched_since_throw: false,
            release_pos: vec(0.0, 0.0, 0.0),
            release_tick: 0.0,
            pass_chain: vec![],
        }
    }

    fn match_state() -> MatchState {
        MatchState {
            inning: 3.0,
            spine: false,
            possession: TeamSide::Home,
            faith_end: FaithEnd::PlusX,
            cast: Cast {
                throws_left: 3,
                gate: Gate::First,
                spot_x: 0.0,
            },
            contest: None,
            score_home: 4.0,
            score_away: 2.0,
            phase: MatchPhase::Live,
            message: String::new(),
            winner: None,
        }
    }

    /// A roster with a full Home unit + an Away unit, bell loose mid-field.
    fn sim_state_loose() -> SimState {
        let players = vec![
            player("H1", TeamSide::Home, RiggerRole::Anchor, vec(10.0, 5.0, 0.0)),
            player("H2", TeamSide::Home, RiggerRole::Spinner, vec(-30.0, 2.0, 4.0)),
            player("H3", TeamSide::Home, RiggerRole::Faithwing, vec(50.0, 8.0, -3.0)),
            player("H4", TeamSide::Home, RiggerRole::Freewing, vec(-5.0, 1.0, 9.0)),
            player("H5", TeamSide::Home, RiggerRole::Reach, vec(300.0, 0.0, 0.0)),
            player("A1", TeamSide::Away, RiggerRole::Anchor, vec(20.0, 4.0, 1.0)),
            player("A2", TeamSide::Away, RiggerRole::Spinner, vec(-40.0, 3.0, 2.0)),
            player("A3", TeamSide::Away, RiggerRole::Freewing, vec(60.0, 6.0, -2.0)),
            player("A4", TeamSide::Away, RiggerRole::Reach, vec(-300.0, 0.0, 0.0)),
        ];
        SimState {
            tick: 240.0,
            omega: 0.32,
            bell: bell(vec(0.0, 3.0, 0.0), None, None),
            players,
        }
    }

    fn sim_state_held_by(holder: &str) -> SimState {
        let mut s = sim_state_loose();
        s.bell = bell(vec(0.0, 3.0, 0.0), Some(holder), None);
        s
    }

    #[test]
    fn determinism_field_by_field() {
        let s = sim_state_loose();
        let m = match_state();
        let prof = TeamProfile::baseline();
        let mut r1 = AiRng::make(424242, 240, 0);
        let mut r2 = AiRng::make(424242, 240, 0);
        let a = run_director(&s, &m, &prof, TeamSide::Home, &mut r1);
        let b = run_director(&s, &m, &prof, TeamSide::Home, &mut r2);

        assert_eq!(a.attacking_free, b.attacking_free);
        assert_eq!(a.attack_sign.to_bits(), b.attack_sign.to_bits());
        assert_eq!(a.attack_ring_x.to_bits(), b.attack_ring_x.to_bits());
        assert_eq!(a.defend_ring_x.to_bits(), b.defend_ring_x.to_bits());
        assert_eq!(a.posture, b.posture);
        assert_eq!(a.loop_setter_id, b.loop_setter_id);
        assert_eq!(a.contest_commit, b.contest_commit);
        assert_eq!(a.ev_faith.to_bits(), b.ev_faith.to_bits());
        assert_eq!(a.ev_free.to_bits(), b.ev_free.to_bits());
        assert_eq!(a.update_in.to_bits(), b.update_in.to_bits());
        assert_eq!(a.has_possession, b.has_possession);
        assert_eq!(a.bell_loose, b.bell_loose);
        assert_eq!(a.carrier_id, b.carrier_id);
        assert_eq!(a.recover_id, b.recover_id);
        assert_eq!(a.style_noise.to_bits(), b.style_noise.to_bits());
        assert_eq!(a.assignments, b.assignments);
    }

    fn assert_structural(d: &DirectorState, team_ids: &[&str]) {
        // Every (non-P1) team player has an assignment.
        for id in team_ids {
            assert!(
                d.assignments.contains_key(*id),
                "missing assignment for {id}: {:?}",
                d.assignments.keys().collect::<Vec<_>>()
            );
        }
        // No stray assignments outside the team.
        for k in d.assignments.keys() {
            assert!(team_ids.contains(&k.as_str()), "stray assignment {k}");
        }
        // Slots in [0,1]; pressure in [0,1]. EXCEPTION: the documented
        // loose-bell SHADOW role is encoded by the reserved out-of-range
        // sentinel `SHADOW_DEPTH_FLAG` on a `Job::Recover` assignment
        // (role via existing job + reserved slot, no new Job variant) —
        // that is a deliberate distinct encoding, not slot corruption,
        // so the invariant explicitly recognizes it.
        for (id, a) in &d.assignments {
            assert!(
                a.is_shadow() || (0.0..=1.0).contains(&a.depth_slot),
                "{id} depth_slot {} out of [0,1]",
                a.depth_slot
            );
            assert!(
                (0.0..=1.0).contains(&a.radius_slot),
                "{id} radius_slot {} out of [0,1]",
                a.radius_slot
            );
            assert!(
                (0.0..=1.0).contains(&a.pressure),
                "{id} pressure {} out of [0,1]",
                a.pressure
            );
            if a.job == Job::Mark {
                assert!(a.mark_id.is_some(), "{id} mark job without mark_id");
            }
        }
        // style_noise in [-1,1]; EVs finite.
        assert!(
            (-1.0..=1.0).contains(&d.style_noise),
            "style_noise {} out of [-1,1]",
            d.style_noise
        );
        assert!(d.ev_faith.is_finite(), "ev_faith not finite");
        assert!(d.ev_free.is_finite(), "ev_free not finite");
        assert!(d.update_in == DIRECTOR_TICK_INTERVAL);
        assert!(d.attack_sign == 1.0 || d.attack_sign == -1.0);
    }

    #[test]
    fn structural_loose_bell_home_possession() {
        let s = sim_state_loose();
        let m = match_state();
        let prof = TeamProfile::baseline();
        let mut r = AiRng::make(7, 240, 0);
        let d = run_director(&s, &m, &prof, TeamSide::Home, &mut r);
        // Home has possession (match.possession == Home), bell loose →
        // recover pack designated, recoverId set, no carrier.
        assert!(d.bell_loose);
        assert!(d.has_possession);
        assert!(d.recover_id.is_some());
        assert!(d.carrier_id.is_none());
        let pack = d
            .assignments
            .values()
            .filter(|a| a.job == Job::Recover)
            .count();
        assert!(pack >= 1 && pack <= 3, "pack size {pack}");
        assert_structural(&d, &["H1", "H2", "H3", "H4", "H5"]);
    }

    #[test]
    fn structural_offense_carrier() {
        let s = sim_state_held_by("H1");
        let m = match_state();
        let prof = TeamProfile::baseline();
        let mut r = AiRng::make(99, 240, 0);
        let d = run_director(&s, &m, &prof, TeamSide::Home, &mut r);
        assert!(!d.bell_loose);
        assert!(d.has_possession);
        assert_eq!(d.carrier_id.as_deref(), Some("H1"));
        assert_eq!(d.assignments["H1"].job, Job::Carry);
        assert_structural(&d, &["H1", "H2", "H3", "H4", "H5"]);
    }

    #[test]
    fn structural_defense_marks() {
        // Away holds the bell → Home defends with man-marks.
        let s = sim_state_held_by("A1");
        let m = match_state();
        let prof = TeamProfile::baseline();
        let mut r = AiRng::make(1234, 240, 0);
        let d = run_director(&s, &m, &prof, TeamSide::Home, &mut r);
        assert!(!d.bell_loose);
        assert!(d.carrier_id.is_none());
        // The Reach is forced off mark/zone onto receive (keeper rule).
        assert_eq!(d.assignments["H5"].job, Job::Receive);
        assert_structural(&d, &["H1", "H2", "H3", "H4", "H5"]);
    }

    #[test]
    fn contest_radius_fallback_when_contester_absent() {
        let mut s = sim_state_loose();
        s.bell = bell(vec(0.0, 3.0, 0.0), Some("A1"), None);
        let mut m = match_state();
        m.contest = Some(MatchContest {
            thrower: "H1".to_string(),
            contester: "GHOST".to_string(), // not in players → REG_R*0.5
            count: 1,
            radius: 5.0,
            direction: ContestDirection::Fair,
        });
        let prof = TeamProfile::baseline();
        let mut r = AiRng::make(5, 240, 0);
        let d = run_director(&s, &m, &prof, TeamSide::Home, &mut r);
        // Just exercises the contest branch deterministically.
        let mut r2 = AiRng::make(5, 240, 0);
        let d2 = run_director(&s, &m, &prof, TeamSide::Home, &mut r2);
        assert_eq!(d.contest_commit, d2.contest_commit);
        assert_structural(&d, &["H1", "H2", "H3", "H4", "H5"]);
    }
}

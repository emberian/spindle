//! 1:1 faithful Rust port of src/ai/RiggerAI.ts (the FROZEN SPEC) —
//! the per-player decision integrator.
//!
//! COMMITMENT MODEL (RiggerAI.ts:6-25): discrete decisions (which receiver
//! to throw to, whether to throw, which nav target to swing toward) are
//! *committed* into a per-player cache and only re-decided when EITHER the
//! Director ticked OR a hard trigger fired. Between decisions the player
//! smoothly EXECUTES the plan. Hysteresis (SWITCH_MARGIN) prevents target
//! dithering between near-equal receivers. Asymmetric difficulty wired via
//! DifficultyScaling (the DIFFICULTY_PROFILES table lives in the frozen
//! decision_types substrate).
//!
//! Shared types (PlayerCommit, PlayerCommitCache, DirectorState,
//! DifficultyScaling, difficulty_scaling, PlayerAssignment, Job,
//! CastPosture) come from `crate::ai::decision_types` — NOT redefined here.
//! `RoleStyle` is `crate::ai::types::RoleStyle`. `Difficulty` is
//! `crate::ai::profile::Difficulty`.

use super::decision_types::{
    difficulty_scaling, CastPosture, DifficultyScaling, DirectorState, Job, PlayerAssignment,
    PlayerCommit, PlayerCommitCache,
};
use super::gate_solve::solve_gate_throw;
use super::lead_predict::solve_lead_velocity;
use super::orientation::{attack_ring_x, attack_sign, defend_ring_x, forward_progress};
use super::plan_bridge::{plan_grapple, GrapplePlan, Sticky};
use super::profile::{Difficulty, TeamProfile};
use super::rng::AiRng;
use super::roles::anchor::anchor_policy;
use super::roles::faithwing::faithwing_policy;
use super::roles::freewing::freewing_policy;
use super::roles::reach::reach_policy;
use super::roles::spinner::spinner_policy;
use super::throw_score::{score_throw, ThrowCandidate};
use super::types::{Gate, MatchState, PlayerInput, PlayerSim, RiggerRole, RoleStyle, SimState};
use crate::math::Vec3;
use crate::trajectory::{rk4_step, PointState};

// REG mirror (RiggerAI.ts imports REG from sim/RegConstants).
const REG_R: f64 = crate::tuning::R; // 45
const REG_L: f64 = crate::tuning::L; // 640
const REG_GATE_RADIUS: f64 = crate::tuning::GATE_RADIUS; // 8
const REG_OMEGA: f64 = crate::tuning::OMEGA; // 0.32

// Planner swoop/rrt knobs — in Rust these are constants (planner.rs is the
// canonical source; planner = 0 MPC ⇒ rrt_active() == false).
const SWOOP_MIN_V: f64 = crate::planner::SWOOP_MIN_V; // 8.0
const SWOOP_ALIGN: f64 = crate::planner::SWOOP_ALIGN; // 0.6
/// TS `rrtActive()` — false on the canonical planner=0 MPC path.
fn rrt_active() -> bool {
    false
}
/// TS `rrtReplanTicks()` = Math.max(1, round(tune('rrtReplan',30))) = 30.
fn rrt_replan_ticks() -> f64 {
    30.0_f64.round().max(1.0)
}

// ── helpers (vec ops expressed on crate::math::Vec3) ──────────────────────────
#[inline]
fn vsub(a: Vec3, b: Vec3) -> Vec3 {
    a.sub(b)
}
#[inline]
fn vadd(a: Vec3, b: Vec3) -> Vec3 {
    a.add(b)
}
#[inline]
fn vscale(a: Vec3, s: f64) -> Vec3 {
    a.scale(s)
}
#[inline]
fn vlen(a: Vec3) -> f64 {
    a.len()
}
#[inline]
fn vdot(a: Vec3, b: Vec3) -> f64 {
    a.dot(b)
}
#[inline]
fn vnorm(a: Vec3) -> Vec3 {
    a.norm()
}
#[inline]
fn v3z() -> Vec3 {
    Vec3::new(0.0, 0.0, 0.0)
}

// Predictor skin bounce — EXACT mirror of sim Collision.skinBounce
// (RiggerAI.ts:50-70). RESTITUTION 0.55, no rng.
const PRED_SKIN_RESTITUTION: f64 = 0.55;
fn predict_skin_bounce(st: &mut PointState) {
    let rho = st.p.y.hypot(st.p.z);
    if rho < REG_R {
        return;
    }
    let ny = st.p.y / rho;
    let nz = st.p.z / rho;
    let vn = st.v.y * ny + st.v.z * nz;
    if vn > 0.0 {
        st.v.y -= (1.0 + PRED_SKIN_RESTITUTION) * vn * ny;
        st.v.z -= (1.0 + PRED_SKIN_RESTITUTION) * vn * nz;
    }
    let s = (REG_R - 1e-3) / rho;
    st.p.y *= s;
    st.p.z *= s;
}

// ── C3: athletic micro-control constants (RiggerAI.ts:72-85) ─────────────────
const SETTLE_RADIUS: f64 = 6.0;
const MICRO_DV_MAX: f64 = 2.0;
const PUSHOFF_ALIGN_COS: f64 = 0.35;

fn axis_radius(p: Vec3) -> f64 {
    (p.y * p.y + p.z * p.z).sqrt()
}

/// estimateOpenness (RiggerAI.ts:192-219).
fn estimate_openness(thrower: &PlayerSim, receiver: &PlayerSim, opponents: &[&PlayerSim]) -> f64 {
    let mut worst = 1.0_f64;
    let tr = vsub(receiver.p, thrower.p);
    let tr_len = vlen(tr);
    let tr_dir = if tr_len > 1e-6 {
        vscale(tr, 1.0 / tr_len)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };

    for opp in opponents {
        let d_recv = vlen(vsub(receiver.p, opp.p));
        let prox_open = (d_recv / 12.0).min(1.0);

        let to = vsub(opp.p, thrower.p);
        let proj = vdot(to, tr_dir).min(tr_len).max(0.0);
        let lane_closest = vadd(thrower.p, vscale(tr_dir, proj));
        let lane_dist = vlen(vsub(opp.p, lane_closest));
        let lane_open = (lane_dist / 9.0).min(1.0);

        let opp_open = prox_open.min(0.4 + 0.6 * lane_open);
        if opp_open < worst {
            worst = opp_open;
        }
    }
    worst
}

/// perturbDirection (RiggerAI.ts:225-234). Draws rng exactly twice when
/// sigma ≥ 0.001 (the gaussian via Box-Muller), zero draws otherwise.
fn perturb_direction(dir: Vec3, sigma: f64, rng: &mut AiRng) -> Vec3 {
    if sigma < 0.001 {
        return dir;
    }
    let u1 = rng.next().max(1e-10);
    let u2 = rng.next();
    let mag = (-2.0 * u1.ln()).sqrt() * sigma;
    let dx = mag * (2.0 * std::f64::consts::PI * u2).cos();
    let dy = mag * (2.0 * std::f64::consts::PI * u2).sin();
    vnorm(Vec3::new(dir.x + dx, dir.y + dy, dir.z))
}

/// hardTrigger (RiggerAI.ts:237-249).
fn hard_trigger(
    commit: &PlayerCommit,
    state: &SimState,
    m: &MatchState,
    player: &PlayerSim,
) -> bool {
    let held_by = state.bell.held_by.as_deref();
    let have_bell = held_by == Some(player.id.as_str());
    let contest_on = m.contest.is_some();
    if commit.saw_held_by.as_deref() != held_by {
        return true;
    }
    if commit.saw_had_bell != have_bell {
        return true;
    }
    if commit.saw_contest != contest_on {
        return true;
    }
    if have_bell && commit.hold_ticks > 0.0 && (commit.hold_ticks as i64) % 80 == 0 {
        return true;
    }
    false
}

/// computePlayerInput (RiggerAI.ts:255-474). Mutates `cache` in place.
#[allow(clippy::too_many_arguments)]
pub fn compute_player_input(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    profile: &TeamProfile,
    director: &DirectorState,
    difficulty: Difficulty,
    rng: &mut AiRng,
    cache: &mut PlayerCommitCache,
    director_refreshed: bool,
    director_interval: f64,
) -> PlayerInput {
    let scaling = difficulty_scaling(difficulty);
    let tick = state.tick;

    let have_bell_now = state.bell.held_by.as_deref() == Some(player.id.as_str());
    if let Some(commit) = cache.value.as_mut() {
        commit.hold_ticks = if have_bell_now {
            commit.hold_ticks + 1.0
        } else {
            0.0
        };
    }
    let first_ever = cache.value.is_none();
    let triggered = match cache.value.as_ref() {
        Some(commit) => hard_trigger(commit, state, m, player),
        None => true,
    };

    // Re-decide when: first time, Director refreshed, or a hard trigger fired.
    let re_decide = cache.value.is_none() || director_refreshed || triggered;

    if re_decide {
        if cache.value.is_none() {
            cache.value = Some(PlayerCommit {
                throw_target_id: None,
                throw_go: false,
                throw_dir: None,
                throw_spin: 0.0,
                throw_charge: 0.0,
                nav_target: None,
                catch_offset: v3z(),
                aim_dither: v3z(),
                saw_held_by: state.bell.held_by.clone(),
                saw_contest: m.contest.is_some(),
                saw_had_bell: have_bell_now,
                react_gate_until_tick: 0.0,
                last_anchor_pos: None,
                last_anchor_reel: -1,
                style_angle: 0.0,
                style_radius: 0.0,
                saw_role: None,
                saw_job: None,
                hold_ticks: 0.0,
                rrt_plan: None,
                rrt_plan_tick: -1.0,
                rrt_plan_target: None,
            });
        }

        // Reaction latency: arm the gate ONCE per distinct change, count down,
        // never re-roll, never compound (RiggerAI.ts:313-329).
        {
            let commit = cache.value.as_mut().unwrap();
            if !first_ever && triggered && tick >= commit.react_gate_until_tick {
                let lag_windows = scaling.reaction_delay * 2.0;
                let lag_ticks =
                    (lag_windows * director_interval * (0.5 + rng.next())).round();
                commit.react_gate_until_tick = tick + lag_ticks;
                // Consume the trigger NOW (snapshot the situation the moment
                // we ARM the gate) — see RiggerAI.ts:317-328.
                commit.saw_held_by = state.bell.held_by.clone();
                commit.saw_contest = m.contest.is_some();
                commit.saw_had_bell = have_bell_now;
            }
        }

        let gated = {
            let commit = cache.value.as_ref().unwrap();
            tick < commit.react_gate_until_tick
        };

        if !gated {
            // Refresh the situation snapshot now that we are acting on it.
            {
                let commit = cache.value.as_mut().unwrap();
                commit.saw_held_by = state.bell.held_by.clone();
                commit.saw_contest = m.contest.is_some();
                commit.saw_had_bell = have_bell_now;
            }

            // director.assignments[player.id] ?? default.
            let assignment: PlayerAssignment = director
                .assignments
                .get(&player.id)
                .cloned()
                .unwrap_or(PlayerAssignment {
                    job: Job::Support,
                    mark_id: None,
                    depth_slot: 0.4,
                    radius_slot: 0.45,
                    pressure: 0.0,
                });

            // C2 — commit role jitter ONCE; re-draw only on a GENUINE
            // role/job change (RiggerAI.ts:355-363). String snapshots mirror
            // the TS sawRole/sawJob (player.role / assignment.job names).
            let role_name = role_str(player.role);
            let job_name = job_str(assignment.job);
            {
                let commit = cache.value.as_mut().unwrap();
                if commit.saw_role.as_deref() != Some(role_name)
                    || commit.saw_job.as_deref() != Some(job_name)
                {
                    commit.style_angle = rng.next();
                    commit.style_radius = rng.next();
                    commit.saw_role = Some(role_name.to_string());
                    commit.saw_job = Some(job_name.to_string());
                }
            }

            if have_bell_now {
                decide_throw(
                    player, state, m, profile, director, &scaling, cache, rng,
                );
            } else {
                let commit = cache.value.as_mut().unwrap();
                commit.throw_go = false;
                commit.throw_target_id = None;
                commit.throw_dir = None;
            }

            let (style_angle, style_radius) = {
                let commit = cache.value.as_ref().unwrap();
                (commit.style_angle, commit.style_radius)
            };
            let nav_target = decide_nav_target(
                player,
                state,
                m,
                profile,
                director,
                &assignment,
                &scaling,
                rng,
                Some(RoleStyle {
                    angle: style_angle,
                    radius: style_radius,
                }),
            );
            cache.value.as_mut().unwrap().nav_target = Some(nav_target);

            // Skill-scaled receiver mistiming (RiggerAI.ts:389-408).
            {
                let commit = cache.value.as_mut().unwrap();
                if assignment.job == Job::Receive || assignment.job == Job::Support {
                    if rng.next() < scaling.drop_chance * 6.0 {
                        let mm = 4.0 + scaling.throw_variance * 30.0;
                        commit.catch_offset = Vec3::new(
                            (rng.next() * 2.0 - 1.0) * mm,
                            (rng.next() * 2.0 - 1.0) * mm,
                            (rng.next() * 2.0 - 1.0) * mm,
                        );
                    } else {
                        let mm = scaling.throw_variance * 6.0;
                        commit.catch_offset = Vec3::new(
                            (rng.next() * 2.0 - 1.0) * mm,
                            (rng.next() * 2.0 - 1.0) * mm,
                            (rng.next() * 2.0 - 1.0) * mm,
                        );
                    }
                } else {
                    commit.catch_offset = v3z();
                }

                // Per-window seeded aim micro-dither (RiggerAI.ts:414-419).
                let ds = 0.004 + scaling.throw_variance * 0.012;
                commit.aim_dither = Vec3::new(
                    (rng.next() * 2.0 - 1.0) * ds,
                    (rng.next() * 2.0 - 1.0) * ds,
                    (rng.next() * 2.0 - 1.0) * ds,
                );
            }
        }

        cache.decided_tick = tick;
    }

    // commit = cache.value! (RiggerAI.ts:426)
    // ── EXECUTE the committed plan (smooth, every tick) ───────────────────────

    {
        let commit = cache.value.as_ref().unwrap();
        if have_bell_now {
            if let (true, Some(dir)) = (commit.throw_go, commit.throw_dir) {
                let charged = commit.throw_charge;
                return PlayerInput {
                    id: player.id.clone(),
                    aim: dir,
                    fire_line_at: None,
                    reel: 0,
                    release: false,
                    pushoff: false,
                    throw_charge: charged,
                    throw_released: true,
                    throw_spin: commit.throw_spin,
                    thrumbler: v3z(),
                };
            }
        }
    }

    // Otherwise execute committed navigation toward the cached target.
    let (target, aim_dither) = {
        let commit = cache.value.as_ref().unwrap();
        let target = match commit.nav_target {
            Some(nt) => vadd(nt, commit.catch_offset),
            None => player.p,
        };
        (target, commit.aim_dither)
    };

    let partial = {
        let commit = cache.value.as_mut().unwrap();
        navigate_to(player, state, target, commit)
    };
    let nav_aim = vnorm(vadd(
        partial.aim.unwrap_or(Vec3::new(1.0, 0.0, 0.0)),
        aim_dither,
    ));

    let pushoff = should_pushoff(player, target);
    let thrumbler = settle_thrumbler(player, target);

    PlayerInput {
        id: player.id.clone(),
        aim: nav_aim,
        fire_line_at: partial.fire_line_at.unwrap_or(None),
        reel: partial.reel.unwrap_or(0),
        release: partial.release.unwrap_or(false),
        pushoff: partial.pushoff.unwrap_or(false) || pushoff,
        throw_charge: 0.0,
        throw_released: false,
        throw_spin: 0.0,
        thrumbler,
    }
}

// ── role/job name snapshots (TS uses the string-union values) ─────────────────
fn role_str(r: RiggerRole) -> &'static str {
    match r {
        RiggerRole::Anchor => "anchor",
        RiggerRole::Spinner => "spinner",
        RiggerRole::Faithwing => "faithwing",
        RiggerRole::Freewing => "freewing",
        RiggerRole::Reach => "reach",
    }
}
fn job_str(j: Job) -> &'static str {
    match j {
        Job::Carry => "carry",
        Job::Recover => "recover",
        Job::Receive => "receive",
        Job::Mark => "mark",
        Job::Support => "support",
        Job::Zone => "zone",
    }
}

/// Partial<PlayerInput> — the planToInput / navigateTo return shape.
#[derive(Clone, Debug, Default)]
struct PartialInput {
    aim: Option<Vec3>,
    fire_line_at: Option<Option<Vec3>>,
    reel: Option<i32>,
    release: Option<bool>,
    pushoff: Option<bool>,
}

/// planToInput (GrapplePlanner.ts:916-942) — ported here per spec. A small
/// pure fn building PlayerInput fields; caller merges aim/throw* around it.
fn plan_to_input(plan: Option<&GrapplePlan>, aim: Vec3) -> PartialInput {
    match plan {
        None => PartialInput {
            aim: Some(aim),
            fire_line_at: Some(None),
            reel: Some(0),
            release: Some(false),
            pushoff: Some(false),
        },
        Some(p) => PartialInput {
            aim: Some(aim),
            fire_line_at: Some(Some(p.anchor_pos)),
            reel: Some(p.reel),
            release: Some(false),
            pushoff: Some(false),
        },
    }
}

/// decideNavTarget (RiggerAI.ts:489-630).
#[allow(clippy::too_many_arguments)]
fn decide_nav_target(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    profile: &TeamProfile,
    director: &DirectorState,
    assignment: &PlayerAssignment,
    _scaling: &DifficultyScaling,
    rng: &mut AiRng,
    style: Option<RoleStyle>,
) -> Vec3 {
    // Loose / in-flight bell pursuit (recover).
    if assignment.job == Job::Recover {
        let b = &state.bell;
        let dx = b.p.x - player.p.x;
        let dy = b.p.y - player.p.y;
        let dz = b.p.z - player.p.z;
        let gap = (dx * dx + dy * dy + dz * dz).sqrt();
        let recover_close_v = 20.0;
        let t_lead = (gap / recover_close_v).max(0.15).min(2.0);
        let mut st = PointState { p: b.p, v: b.v };
        let h: f64 = 1.0 / 60.0;
        let mut t_acc = 0.0_f64;
        while t_acc < t_lead {
            let step = h.min(t_lead - t_acc);
            st = rk4_step(st, REG_OMEGA, step);
            predict_skin_bounce(&mut st);
            t_acc += step;
        }
        let bs = st.v.x.hypot(st.v.y).hypot(st.v.z);
        if bs > 1e-3 {
            let tuck = 6.0;
            return Vec3::new(
                st.p.x - (st.v.x / bs) * tuck,
                st.p.y - (st.v.y / bs) * tuck,
                st.p.z - (st.v.z / bs) * tuck,
            );
        }
        return Vec3::new(st.p.x, st.p.y, st.p.z);
    }

    // CARRIER: actively gain ground toward OUR attacking ring
    // (keyed off ACTUAL possession too).
    if assignment.job == Job::Carry
        || state.bell.held_by.as_deref() == Some(player.id.as_str())
    {
        let sgn = attack_sign(player.team);
        let ring_x = attack_ring_x(player.team);
        let dist_to_ring = (ring_x - player.p.x).abs();
        let step = 110.0_f64.min(dist_to_ring);
        let ahead_x = player.p.x + sgn * step;
        let r = axis_radius(player.p);
        let target_r = r.min(6.0);
        let yz_len = if r > 1e-6 { r } else { 1.0 };
        return Vec3::new(
            ahead_x,
            (player.p.y / yz_len) * target_r,
            (player.p.z / yz_len) * target_r,
        );
    }

    // Man-marking.
    if assignment.job == Job::Mark {
        if let Some(mark_id) = &assignment.mark_id {
            if let Some(mark) = state.players.iter().find(|p| &p.id == mark_id) {
                let t_ring = Vec3::new(defend_ring_x(player.team), 0.0, 0.0);
                let to_ring = vnorm(vsub(t_ring, mark.p));
                let standoff = 8.0 - assignment.pressure * 5.4;
                let base = vadd(mark.p, vscale(to_ring, standoff));
                if assignment.pressure > 0.7 {
                    let to_bell = vsub(state.bell.p, base);
                    let bl = vlen(to_bell);
                    if bl > 1e-6 {
                        let lean = bl.min(2.5);
                        return vadd(base, vscale(to_bell, lean / bl));
                    }
                }
                return base;
            }
        }
    }

    // SUPPORT (offense, off-ball): cut into a distinct lane.
    if assignment.job == Job::Support {
        let sgn = director.attack_sign;
        let style_radius = style.map(|s| s.radius).unwrap_or(0.5);
        let style_angle = style.map(|s| s.angle).unwrap_or(0.5);
        let cut_x = state.bell.p.x + sgn * (45.0 + style_radius * 55.0);
        let ang = std::f64::consts::PI * (0.12 + style_angle * 1.4);
        let r = 14.0 + style_radius * 30.0;
        return Vec3::new(cut_x, r * ang.cos(), r * ang.sin());
    }

    // ZONE (defense, off-ball / help).
    if assignment.job == Job::Zone {
        let dx = defend_ring_x(player.team);
        let sgn = attack_sign(player.team);
        let style_radius = style.map(|s| s.radius).unwrap_or(0.5);
        let style_angle = style.map(|s| s.angle).unwrap_or(0.0);
        let guard_x = dx + sgn * (22.0 + style_radius * 30.0);
        let ang = std::f64::consts::PI
            * (0.15 + assignment.radius_slot * 1.3 + style_angle * 0.2);
        let r = 16.0 + assignment.radius_slot * 34.0;
        return Vec3::new(guard_x, r * ang.cos(), r * ang.sin());
    }

    // Otherwise defer to the role policy.
    role_target(player, state, m, profile, director, assignment, rng, style)
}

/// roleTarget (RiggerAI.ts:633-700).
#[allow(clippy::too_many_arguments)]
fn role_target(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    profile: &TeamProfile,
    director: &DirectorState,
    assignment: &PlayerAssignment,
    rng: &mut AiRng,
    style: Option<RoleStyle>,
) -> Vec3 {
    let mut base: Vec3 = match player.role {
        RiggerRole::Anchor => {
            anchor_policy(player, state, m, profile, rng, style).target_pos
        }
        RiggerRole::Spinner => {
            let is_loop_setter =
                director.loop_setter_id.as_deref() == Some(player.id.as_str());
            spinner_policy(player, state, m, profile, is_loop_setter, rng, style)
                .target_pos
        }
        RiggerRole::Faithwing => {
            faithwing_policy(player, state, m, profile, rng, style).target_pos
        }
        RiggerRole::Freewing => {
            freewing_policy(player, state, m, profile, rng, style).target_pos
        }
        RiggerRole::Reach => {
            reach_policy(player, state, m, profile, rng, style).target_pos
        }
    };

    // Offensive receiver: override radius & axial depth from the Director slot.
    if (assignment.job == Job::Receive || assignment.job == Job::Support)
        && player.role != RiggerRole::Reach
    {
        let carrier = director
            .carrier_id
            .as_ref()
            .and_then(|cid| state.players.iter().find(|p| &p.id == cid));
        let target_ring_x = director.attack_ring_x;
        let from_x = carrier.map(|c| c.p.x).unwrap_or(state.bell.p.x);
        let min_lead = director.attack_sign * 14.0;
        let mut depth_x = from_x + (target_ring_x - from_x) * assignment.depth_slot;
        if director.attack_sign > 0.0 {
            depth_x = depth_x.max(from_x + min_lead);
        } else {
            depth_x = depth_x.min(from_x + min_lead);
        }
        let r_const = 45.0;
        let r = r_const * (0.12 + assignment.radius_slot * 0.7);
        let ang = std::f64::consts::PI
            * (0.15 + assignment.radius_slot * 0.7 + director.style_noise * 0.12);
        base = Vec3::new(depth_x, r * ang.cos(), r * ang.sin());
    }
    base
}

/// shouldPushoff (RiggerAI.ts:715-732). Pure geometry, no rng.
fn should_pushoff(player: &PlayerSim, target: Vec3) -> bool {
    let in_contact = player.contact_ref.is_some() || player.grounded;
    if !in_contact {
        return false;
    }
    let r = (player.p.y * player.p.y + player.p.z * player.p.z).sqrt();
    if r < 1e-6 {
        return false;
    }
    let radial_out = Vec3::new(0.0, player.p.y / r, player.p.z / r);
    let push_dir = if player.grounded {
        vscale(radial_out, -1.0)
    } else {
        radial_out
    };
    let to_target = vsub(target, player.p);
    let len = vlen(to_target);
    if len < 1e-6 {
        return false;
    }
    let align = vdot(vscale(to_target, 1.0 / len), push_dir);
    align > PUSHOFF_ALIGN_COS
}

/// settleThrumbler (RiggerAI.ts:742-762). Pure geometry, no rng.
fn settle_thrumbler(player: &PlayerSim, target: Vec3) -> Vec3 {
    let to_target = vsub(target, player.p);
    let dist = vlen(to_target);
    if dist > SETTLE_RADIUS || dist < 1e-6 {
        return v3z();
    }
    let approach = vscale(to_target, 1.0 / dist);
    let approach_mag = MICRO_DV_MAX.min(dist * 0.5);
    let speed = vlen(player.v);
    let brake = if speed > 1e-6 {
        vscale(player.v, -MICRO_DV_MAX.min(speed) / speed)
    } else {
        v3z()
    };
    let closeness = 1.0 - dist / SETTLE_RADIUS;
    let dv = vadd(
        vscale(approach, approach_mag * (1.0 - 0.5 * closeness)),
        vscale(brake, MICRO_DV_MAX * 0.6 * closeness),
    );
    let mm = vlen(dv);
    if mm > MICRO_DV_MAX {
        vscale(dv, MICRO_DV_MAX / mm)
    } else {
        dv
    }
}

/// navigateTo (RiggerAI.ts:769-833). C1 sticky anchor + RRT cache branch
/// (the non-default path; rrt_active() == false on canonical MPC).
fn navigate_to(
    player: &PlayerSim,
    state: &SimState,
    target: Vec3,
    commit: &mut PlayerCommit,
) -> PartialInput {
    let sticky = commit.last_anchor_pos.map(|pos| Sticky {
        pos,
        reel: commit.last_anchor_reel,
    });

    let plan: Option<GrapplePlan> = if rrt_active() {
        let age = state.tick - commit.rrt_plan_tick;
        let tgt_jump = match commit.rrt_plan_target {
            Some(t) => ((target.x - t.x).powi(2)
                + (target.y - t.y).powi(2)
                + (target.z - t.z).powi(2))
            .sqrt(),
            None => f64::INFINITY,
        };
        if commit.rrt_plan.is_some()
            && age >= 0.0
            && age < rrt_replan_ticks()
            && tgt_jump < 25.0
        {
            commit.rrt_plan
        } else {
            let p = plan_grapple(player, target, state, true, sticky);
            commit.rrt_plan = p;
            commit.rrt_plan_tick = state.tick;
            commit.rrt_plan_target = Some(Vec3::new(target.x, target.y, target.z));
            p
        }
    } else {
        plan_grapple(player, target, state, true, sticky)
    };

    if let Some(p) = plan {
        commit.last_anchor_pos = Some(p.anchor_pos);
        commit.last_anchor_reel = p.reel;
    } else {
        commit.last_anchor_pos = None;
    }

    // SWOOP RELEASE.
    if let Some(line) = &player.line {
        if line.taut {
            if let Some(p) = plan {
                if p.reel == 0 {
                    let sp = vlen(player.v);
                    let to_t = vsub(target, player.p);
                    let dl = vlen(to_t);
                    if sp > SWOOP_MIN_V
                        && dl > 1e-6
                        && vdot(player.v, to_t) / (sp * dl) > SWOOP_ALIGN
                    {
                        commit.last_anchor_pos = None;
                        let soar_aim = vnorm(player.v);
                        let mut pi = plan_to_input(None, soar_aim);
                        pi.release = Some(true);
                        return pi;
                    }
                }
            }
        }
    }

    let aim = match plan {
        Some(p) => vnorm(vsub(p.anchor_pos, player.p)),
        None => Vec3::new(1.0, 0.0, 0.0),
    };
    plan_to_input(plan.as_ref(), aim)
}

/// decideThrow (RiggerAI.ts:841-1066). Mutates the commit in place.
#[allow(clippy::too_many_arguments)]
fn decide_throw(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    profile: &TeamProfile,
    director: &DirectorState,
    scaling: &DifficultyScaling,
    cache: &mut PlayerCommitCache,
    rng: &mut AiRng,
) {
    let teammates: Vec<&PlayerSim> = state
        .players
        .iter()
        .filter(|p| p.team == player.team && p.id != player.id)
        .collect();
    let opponents: Vec<&PlayerSim> =
        state.players.iter().filter(|p| p.team != player.team).collect();

    let ring_x = attack_ring_x(player.team);
    let dist_to_ring = (ring_x - player.p.x).abs();

    let posture_mul = match director.posture {
        CastPosture::Chase => 1.12,
        CastPosture::Grind => 0.92,
        CastPosture::Tempo => 1.0,
    };
    let throw_speed = (18.0 + profile.aggression * 12.0) * posture_mul;

    // ── PRIORITY 1: a REAL scoring throw through OUR ring. ────────────────────
    if dist_to_ring > 6.0 {
        let gate = solve_gate_throw(
            player.p,
            player.team,
            state.omega,
            throw_speed,
            profile.loop_propensity,
            player.v,
        );
        // match.cast is always present in Rust (non-Option) → TS ternaries
        // collapse to the present-branch.
        let throws_left = m.cast.throws_left as f64;
        let at_mouth = m.cast.gate == Gate::Mouth;
        let threads_well = match &gate {
            Some(g) => g.arrive_rho <= REG_GATE_RADIUS * 0.6,
            None => false,
        };
        let in_range = dist_to_ring < REG_L * 0.45;
        let stall_hard = 540.0;
        let hold_ticks = cache.value.as_ref().unwrap().hold_ticks;
        let must_shoot = throws_left <= 1.0 || hold_ticks > stall_hard;
        if let Some(g) = &gate {
            if (at_mouth && threads_well && in_range) || must_shoot {
                let dir = perturb_direction(
                    vnorm(g.throw_vec),
                    scaling.throw_variance * 0.04,
                    rng,
                );
                let throw_spin = if director.attacking_free {
                    -(0.3 + profile.loop_propensity * 0.7)
                } else {
                    0.2 + (1.0 - profile.free_end_bias) * 0.3
                };
                let throw_min = 9.0;
                let throw_max = 34.0;
                let charge = ((g.release_speed - throw_min) / (throw_max - throw_min))
                    .max(0.0)
                    .min(1.0);
                let commit = cache.value.as_mut().unwrap();
                commit.throw_go = true;
                commit.throw_target_id = None;
                commit.throw_dir = Some(dir);
                commit.throw_spin = throw_spin.clamp(-1.0, 1.0);
                commit.throw_charge = charge;
                return;
            }
        }
    }

    // ── PRIORITY 2: a ground-gaining pass to an OPEN, ADVANCING receiver. ─────
    let mut best_score = f64::NEG_INFINITY;
    let mut best_tm: Option<&PlayerSim> = None;
    let mut best_v0: Option<Vec3> = None;
    let mut held_score = f64::NEG_INFINITY;

    let committed_target = cache.value.as_ref().unwrap().throw_target_id.clone();

    for tm in &teammates {
        if rng.next() < scaling.miss_open_chance {
            continue;
        }

        let lead =
            solve_lead_velocity(player.p, throw_speed, tm.p, tm.v, state.omega);
        let lead = match lead {
            Some(l) => l,
            None => continue,
        };

        let gain = forward_progress(player.team, lead.intercept.x)
            - forward_progress(player.team, player.p.x);

        let openness = estimate_openness(player, tm, &opponents);

        let candidate = ThrowCandidate {
            receiver: (*tm).clone(),
            aim_pos: lead.intercept,
            flight_time: lead.flight_time,
            openness,
            receiver_radius: axis_radius(tm.p),
        };

        let noise = (rng.next() - 0.5) * (1.0 - scaling.read_quality) * 0.3;
        let result = score_throw(player, candidate, m, *profile, noise);

        let gain_norm = (gain / 90.0).max(-1.0).min(1.5);
        let rcv_r = axis_radius(tm.p);
        let radius_penalty = ((rcv_r - 16.0) / REG_R).max(0.0) * 0.6;
        let score = result.score + gain_norm * 0.35 - radius_penalty;

        if committed_target.as_deref() == Some(tm.id.as_str()) {
            held_score = score;
        }

        if score > best_score {
            best_score = score;
            best_tm = Some(*tm);
            best_v0 = Some(lead.v0);
        }
    }

    // Hysteresis: keep the committed target unless clearly beaten.
    let switch_margin = 0.12;
    if let Some(ct) = &committed_target {
        if held_score > f64::NEG_INFINITY {
            if let Some(bt) = best_tm {
                if bt.id != *ct && best_score < held_score + switch_margin {
                    if let Some(held) =
                        teammates.iter().find(|p| p.id == *ct).copied()
                    {
                        if let Some(lead) = solve_lead_velocity(
                            player.p,
                            throw_speed,
                            held.p,
                            held.v,
                            state.omega,
                        ) {
                            best_tm = Some(held);
                            best_v0 = Some(lead.v0);
                            best_score = held_score;
                        }
                    }
                }
            }
        }
    }

    // Decide WHETHER to pass at all.
    let mut best_gain = f64::NEG_INFINITY;
    if let Some(bt) = best_tm {
        best_gain = forward_progress(player.team, bt.p.x)
            - forward_progress(player.team, player.p.x);
    }

    let opp = nearest_opponent_dist(player, &opponents);
    let pressured = opp < 14.0;
    let stall = cache.value.as_ref().unwrap().hold_ticks;
    let stall_hard2 = 540.0;
    let ease = ((stall - 240.0) / 300.0).max(0.0).min(1.0);
    let pass_gains_ground = best_gain > 8.0 - ease * 24.0;
    let safe_outlet = best_score > 0.15 - ease * 0.25 && best_gain > -25.0 - ease * 30.0;
    let force_release = stall > stall_hard2 && best_tm.is_some() && best_v0.is_some();
    let acceptable = best_tm.is_some()
        && best_v0.is_some()
        && (force_release
            || (best_score > -0.02 - ease * 0.35
                && (pass_gains_ground || (pressured && safe_outlet))));

    if !acceptable {
        let commit = cache.value.as_mut().unwrap();
        commit.throw_go = false;
        commit.throw_target_id = None;
        commit.throw_dir = None;
        return;
    }

    let noisy_dir =
        perturb_direction(vnorm(best_v0.unwrap()), scaling.throw_variance, rng);
    let throw_spin = if director.attacking_free {
        -(0.3 + profile.loop_propensity * 0.7)
    } else {
        0.2 + (1.0 - profile.free_end_bias) * 0.3
    };
    let bt_id = best_tm.unwrap().id.clone();
    let charge = (0.7 + rng.next() * 0.3).min(1.0);

    let commit = cache.value.as_mut().unwrap();
    commit.throw_go = true;
    commit.throw_target_id = Some(bt_id);
    commit.throw_dir = Some(noisy_dir);
    commit.throw_spin = throw_spin.clamp(-1.0, 1.0);
    commit.throw_charge = charge;
}

/// nearestOpponentDist (RiggerAI.ts:1069-1076).
fn nearest_opponent_dist(player: &PlayerSim, opponents: &[&PlayerSim]) -> f64 {
    let mut best = f64::INFINITY;
    for o in opponents {
        let d = vlen(vsub(o.p, player.p));
        if d < best {
            best = d;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::profile::style_to_profile;
    use crate::ai::types::{BellState, Cast, MatchPhase, TeamSide};
    use crate::math::Quat;
    use std::collections::HashMap;

    fn vz() -> Vec3 {
        Vec3::new(0.0, 0.0, 0.0)
    }

    fn mk_player(id: &str, team: TeamSide, role: RiggerRole, p: Vec3) -> PlayerSim {
        PlayerSim {
            id: id.to_string(),
            team,
            role,
            p,
            v: Vec3::new(1.0, 0.5, -0.5),
            q: Quat::ident(),
            line: None,
            dv_budget: 5.0,
            contact_ref: None,
            grounded: false,
        }
    }

    fn mk_state(players: Vec<PlayerSim>, held: Option<&str>) -> SimState {
        SimState {
            tick: 100.0,
            omega: crate::tuning::OMEGA,
            bell: BellState {
                p: Vec3::new(10.0, 3.0, 2.0),
                v: Vec3::new(4.0, 1.0, -2.0),
                q: Quat::ident(),
                w: vz(),
                chime: 1.0,
                held_by: held.map(|s| s.to_string()),
                thrown_by: None,
                touched_since_throw: false,
                release_pos: vz(),
                release_tick: 0.0,
                pass_chain: vec![],
            },
            players,
        }
    }

    fn mk_match() -> MatchState {
        MatchState {
            inning: 1.0,
            spine: false,
            possession: TeamSide::Home,
            faith_end: crate::ai::types::FaithEnd::PlusX,
            cast: Cast {
                throws_left: 3,
                gate: Gate::First,
                spot_x: 0.0,
            },
            contest: None,
            score_home: 0.0,
            score_away: 0.0,
            phase: MatchPhase::Live,
            message: String::new(),
            winner: None,
        }
    }

    fn mk_director() -> DirectorState {
        let mut assignments = HashMap::new();
        assignments.insert(
            "H1".to_string(),
            PlayerAssignment {
                job: Job::Carry,
                mark_id: None,
                depth_slot: 0.5,
                radius_slot: 0.4,
                pressure: 0.0,
            },
        );
        assignments.insert(
            "H2".to_string(),
            PlayerAssignment {
                job: Job::Receive,
                mark_id: None,
                depth_slot: 0.6,
                radius_slot: 0.5,
                pressure: 0.0,
            },
        );
        DirectorState {
            attacking_free: true,
            attack_sign: 1.0,
            attack_ring_x: crate::tuning::GATE_X,
            defend_ring_x: -crate::tuning::GATE_X,
            posture: CastPosture::Tempo,
            loop_setter_id: None,
            contest_commit: false,
            ev_faith: 0.0,
            ev_free: 0.0,
            update_in: 0.0,
            has_possession: true,
            bell_loose: false,
            carrier_id: Some("H1".to_string()),
            recover_id: None,
            assignments,
            style_noise: 0.3,
        }
    }

    fn scenario() -> (Vec<PlayerSim>, MatchState, DirectorState, TeamProfile) {
        let players = vec![
            mk_player("H1", TeamSide::Home, RiggerRole::Spinner, Vec3::new(-50.0, 4.0, 1.0)),
            mk_player("H2", TeamSide::Home, RiggerRole::Faithwing, Vec3::new(20.0, 6.0, -3.0)),
            mk_player("A1", TeamSide::Away, RiggerRole::Anchor, Vec3::new(0.0, 5.0, 5.0)),
        ];
        (
            players,
            mk_match(),
            mk_director(),
            style_to_profile("balanced", "medium"),
        )
    }

    fn run_once(held: Option<&str>) -> (PlayerInput, PlayerCommit) {
        let (players, m, dir, profile) = scenario();
        let state = mk_state(players, held);
        let player = state.players[0].clone();
        let mut rng = AiRng::make(424242, 100, 0);
        let mut cache = PlayerCommitCache::default();
        let out = compute_player_input(
            &player, &state, &m, &profile, &dir, Difficulty::Pro, &mut rng,
            &mut cache, true, 30.0,
        );
        (out, cache.value.unwrap())
    }

    #[test]
    fn deterministic_same_inputs_same_output_and_commit() {
        let (o1, c1) = run_once(Some("H1"));
        let (o2, c2) = run_once(Some("H1"));
        assert_eq!(o1.id, o2.id);
        assert_eq!(o1.aim.x.to_bits(), o2.aim.x.to_bits());
        assert_eq!(o1.aim.y.to_bits(), o2.aim.y.to_bits());
        assert_eq!(o1.aim.z.to_bits(), o2.aim.z.to_bits());
        assert_eq!(o1.reel, o2.reel);
        assert_eq!(o1.throw_charge.to_bits(), o2.throw_charge.to_bits());
        assert_eq!(o1.throw_released, o2.throw_released);
        assert_eq!(o1.throw_spin.to_bits(), o2.throw_spin.to_bits());
        assert_eq!(c1.throw_target_id, c2.throw_target_id);
        assert_eq!(c1.throw_go, c2.throw_go);
        assert_eq!(c1.throw_charge.to_bits(), c2.throw_charge.to_bits());
        assert_eq!(
            c1.nav_target.map(|v| (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())),
            c2.nav_target.map(|v| (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())),
        );
    }

    #[test]
    fn output_invariants_true_by_construction() {
        for held in [Some("H1"), None] {
            let (o, _c) = run_once(held);
            assert!(o.aim.x.is_finite() && o.aim.y.is_finite() && o.aim.z.is_finite());
            assert!(o.reel == -1 || o.reel == 0 || o.reel == 1);
            assert!((0.0..=1.0).contains(&o.throw_charge));
            assert!((-1.0..=1.0).contains(&o.throw_spin));
            assert!(
                o.thrumbler.x.is_finite()
                    && o.thrumbler.y.is_finite()
                    && o.thrumbler.z.is_finite()
            );
        }
    }

    #[test]
    fn carrier_executes_or_navigates() {
        // Holding the bell → either a committed throw or a nav frame; both
        // finite aim. Cache commit must persist. hold_ticks follows the
        // FAITHFUL TS semantics (RiggerAI.ts:270-272): the increment runs
        // only when a commit ALREADY exists, so the first-ever tick (fresh
        // cache) leaves hold_ticks == 0 (fresh commit, RiggerAI.ts:301);
        // it reaches 1 on the SECOND tick while still holding.
        let (players, m, dir, profile) = scenario();
        let state = mk_state(players, Some("H1"));
        let player = state.players[0].clone();
        let mut cache = PlayerCommitCache::default();

        let mut rng1 = AiRng::make(424242, 100, 0);
        let o = compute_player_input(
            &player, &state, &m, &profile, &dir, Difficulty::Pro, &mut rng1,
            &mut cache, true, 30.0,
        );
        assert!(o.aim.x.is_finite() && o.aim.y.is_finite() && o.aim.z.is_finite());
        let c1 = cache.value.clone().expect("commit persists");
        assert_eq!(c1.hold_ticks, 0.0, "first-ever tick: increment skipped");

        let mut rng2 = AiRng::make(424242, 101, 0);
        compute_player_input(
            &player, &state, &m, &profile, &dir, Difficulty::Pro, &mut rng2,
            &mut cache, false, 30.0,
        );
        let c2 = cache.value.clone().expect("commit persists");
        assert_eq!(c2.hold_ticks, 1.0, "second tick still holding → 1");
    }
}

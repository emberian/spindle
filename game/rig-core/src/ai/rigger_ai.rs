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
use super::efe;
use super::gate_solve::{solve_gate_throw, THROW_MAX_SPEED, THROW_MIN_SPEED};
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
use super::types::{MatchState, PlayerInput, PlayerSim, RiggerRole, RoleStyle, SimState};
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

/// OFFENSE REBUILD — commit-to-catch signal. A player commits to taking
/// the live (loose / in-flight) bell when it is theirs to take: they were
/// not the last thrower, the bell is reasonably near, and EITHER they are
/// the directed recover/receiver outlet OR the bell is genuinely closing
/// on them (a defender stepping into a pass = a played pick). Deterministic
/// (pure geometry off the committed assignment + bell kinematics).
fn wants_catch(player: &PlayerSim, state: &SimState, assignment: &PlayerAssignment) -> bool {
    if state.bell.held_by.is_some() {
        return false;
    }
    if state.bell.thrown_by.as_deref() == Some(player.id.as_str()) {
        return false;
    }
    let to_bell = vsub(state.bell.p, player.p);
    let gap = vlen(to_bell);
    if gap > 70.0 {
        return false;
    }
    // Directed outlets always commit (recover pack, staged receiver).
    if matches!(assignment.job, Job::Recover | Job::Receive | Job::Support) {
        return true;
    }
    // Anyone else commits only if the bell is actually coming at them.
    let bs = vlen(state.bell.v);
    if bs < 1e-3 {
        return gap < 12.0;
    }
    let closing = -vdot(state.bell.v, to_bell) / (bs * gap.max(1e-6));
    closing > 0.2 && gap < 45.0
}

/// OFFENSE REBUILD — lead-intercept point of the live bell, so a committed
/// catcher actually navigates onto the ball instead of a static slot.
/// RK4 + skin-bounce predictor (same model the recover branch uses),
/// horizon scaled by gap so it stays stable. No rng.
fn bell_intercept(player: &PlayerSim, state: &SimState) -> Vec3 {
    let b = &state.bell;
    let gap = vlen(vsub(b.p, player.p));
    let close_v = 22.0;
    let t_lead = (gap / close_v).max(0.1).min(2.5);
    let mut st = PointState { p: b.p, v: b.v };
    let h: f64 = 1.0 / 60.0;
    let mut t_acc = 0.0_f64;
    while t_acc < t_lead {
        let step = h.min(t_lead - t_acc);
        st = rk4_step(st, REG_OMEGA, step);
        predict_skin_bounce(&mut st);
        t_acc += step;
    }
    st.p
}

/// LOOSE-BELL DECISIVE DIVE — time-to-intercept of the predicted bell band
/// for an arbitrary world point, assuming a confident closing speed. This is
/// the SAME canon RK4 + skin-bounce predictor (via `efe::BellBand`) the
/// recover branch already uses; we only read its `best_intercept_for`
/// shortfall. Lower = this position genuinely wins the race to the ball.
/// Pure geometry, deterministic, no rng.
fn dive_intercept_gap(from: Vec3, from_v: Vec3, state: &SimState) -> f64 {
    let band =
        efe::BellBand::predict(state.bell.p, state.bell.v, state.omega, REG_R);
    // Closing speed credit: a committed powered-hook dive really does close
    // fast; bias slightly by current inbound velocity so a rigger already
    // driving at the ball is correctly judged the better racer.
    let to_b = vsub(state.bell.p, from);
    let bl = vlen(to_b);
    let inbound = if bl > 1e-6 {
        (vdot(from_v, to_b) / bl).max(0.0)
    } else {
        0.0
    };
    let close_v = 24.0 + inbound.min(14.0);
    band.expected_response_gap(from, close_v)
}

/// Is THIS rigger the SINGLE committed diver of the loose bell?
///
/// The loose-bell unit is now ROLE-DIFFERENTIATED by the Director: the
/// PRIMARY (`recover_id`, min predicted t_intercept) is the SOLE committed
/// diver. The SHADOW is also a `Job::Recover` member but is FLAGGED
/// (`depth_slot == SHADOW_DEPTH_FLAG`) and must NEVER dive simultaneously
/// — it stations on the rebound locus and only becomes the diver if the
/// NEXT Director window recomputes `recover_id` onto it (a real deflection
/// won it the race). So there is exactly ONE diver at any instant.
///
/// True iff the bell is loose, ours-to-take (`wants_catch`), AND either
/// (a) the Director named us the primary (`recover_id`), OR (b) the
/// FALLBACK: `recover_id` is somehow invalid (absent, not on our team, or
/// the named primary can't take this bell) and, among the recover unit
/// members that genuinely CAN take it AND are NOT the shadow, we win a
/// deterministic best-predicted-intercept compare (id tie-break). The
/// fallback keeps the team from freezing if the Director's pick is stale;
/// it still yields exactly one diver. Deterministic, no rng.
fn is_dive_committer(
    player: &PlayerSim,
    state: &SimState,
    director: &DirectorState,
    assignment: &PlayerAssignment,
) -> bool {
    if state.bell.held_by.is_some() {
        return false;
    }
    if !wants_catch(player, state, assignment) {
        return false;
    }
    // The SHADOW never dives via this path — it holds the rebound locus.
    let i_am_shadow = assignment.is_shadow();
    // The Director's named PRIMARY always commits (never the shadow).
    if director.recover_id.as_deref() == Some(player.id.as_str())
        && !i_am_shadow
    {
        return true;
    }
    // FALLBACK only when the named primary is invalid: not present on our
    // team, or present but cannot actually take this bell right now. If
    // the named primary is valid we defer to it (exactly one diver) and
    // everyone else — including the shadow — holds their role.
    let primary_valid = director
        .recover_id
        .as_deref()
        .and_then(|rid| state.players.iter().find(|p| p.id == rid))
        .map(|pp| {
            pp.team == player.team
                && director
                    .assignments
                    .get(&pp.id)
                    .map(|a| wants_catch(pp, state, a))
                    .unwrap_or(false)
        })
        .unwrap_or(false);
    if primary_valid {
        return false;
    }
    if i_am_shadow || assignment.job != Job::Recover {
        return false;
    }
    // Recover-unit members that genuinely can take the bell and are not
    // the shadow — deterministic best-intercept compare, id tie-break.
    let mut pack: Vec<&PlayerSim> = state
        .players
        .iter()
        .filter(|p| {
            if p.team != player.team {
                return false;
            }
            match director.assignments.get(&p.id) {
                Some(a) => {
                    a.job == Job::Recover
                        && !a.is_shadow()
                        && wants_catch(p, state, a)
                }
                None => false,
            }
        })
        .collect();
    if pack.is_empty() {
        return false;
    }
    pack.sort_by(|a, b| {
        let ga = dive_intercept_gap(a.p, a.v, state);
        let gb = dive_intercept_gap(b.p, b.v, state);
        if (ga - gb).abs() > 1e-9 {
            ga.partial_cmp(&gb).unwrap()
        } else {
            a.id.cmp(&b.id)
        }
    });
    pack[0].id == player.id
}

/// Predicted catch point AND the bell's velocity THERE. Same canon RK4 +
/// skin-bounce predictor (`efe::roll_forward`) the recover/intercept path
/// uses; the lead horizon is chosen deterministically so the rigger and the
/// bell would meet at roughly the same instant (a true rendezvous, not a
/// stale point). Returns `(catch_point, bell_velocity_at_catch_point)`.
/// Pure geometry, deterministic, no rng.
fn bell_rendezvous(player: &PlayerSim, state: &SimState) -> (Vec3, Vec3) {
    let b = &state.bell;
    // The committed powered hook really does close fast (winch cruise ≈
    // REEL_PULL_SPEED 26 m/s plus the rigger's inbound speed). Use a
    // confident closing speed to size the lead so the predicted point is
    // where we can actually BE when the bell is there.
    let to_b = vsub(b.p, player.p);
    let gap = vlen(to_b);
    let inbound = if gap > 1e-6 {
        (vdot(player.v, to_b) / gap).max(0.0)
    } else {
        0.0
    };
    let close_v = (crate::tuning::REEL_PULL_SPEED + inbound.min(14.0)).max(8.0);
    // Fixed-point: re-evaluate the lead time at the predicted point a few
    // times so t_lead ≈ |predicted − rigger| / close_v converges (the bell
    // curves under Coriolis; one shot would lead the wrong amount). Fixed
    // iteration count ⇒ deterministic.
    let mut t_lead = (gap / close_v).max(0.05).min(2.6);
    let mut st = PointState { p: b.p, v: b.v };
    for _ in 0..4 {
        st = efe::roll_forward(b.p, b.v, state.omega, t_lead, REG_R);
        let d = vlen(vsub(st.p, player.p));
        t_lead = (d / close_v).max(0.05).min(2.6);
    }
    (st.p, st.v)
}

/// THE DECISIVE POWERED-HOOK DIVE — now a VELOCITY-MATCHED RENDEZVOUS.
///
/// The old dive fired the anchor 14 m beyond the lead point along the
/// rigger→point line and held `reel = -1` all the way in, so the rigger
/// arrived winching at ≈ REEL_PULL_SPEED along ITS OWN heading — a
/// direction unrelated to the bell's. `|bell_v − player_v|` was therefore
/// huge (a high-speed crossing flyby) and `try_catch_ex` only ever
/// bobbled.
///
/// Now: predict the catch point AND the bell's velocity there (canon
/// predictor), and fire the anchor DOWNRANGE ALONG THE BELL'S VELOCITY —
/// i.e. ahead of the catch point on the bell's own future path. Winching
/// toward that anchor pulls the rigger onto a track that runs roughly
/// PARALLEL to the bell's trajectory (a pursuit/rendezvous), so the
/// terminal winch velocity is ≈ co-linear with `bell_v`; the rel-speed
/// collapses from "two unrelated headings" to ≈ `|REEL_PULL_SPEED −
/// |bell_v||`, which the terminal ease then bleeds the rest of the way
/// into the 38 m/s absorb window. Returns `(PartialInput, aim_dir,
/// catch_point)` so the caller can run the terminal-ease phase. Pure
/// geometry + the canon predictor; deterministic, no rng.
fn decisive_dive_input(player: &PlayerSim, state: &SimState) -> (PartialInput, Vec3, Vec3) {
    let (ip, bv) = bell_rendezvous(player, state);
    let to_ip = vsub(ip, player.p);
    let d = vlen(to_ip);
    let app_dir = if d > 1e-6 {
        vscale(to_ip, 1.0 / d)
    } else {
        // Degenerate (already on it): drive straight at the live bell.
        let tb = vsub(state.bell.p, player.p);
        let tl = vlen(tb);
        if tl > 1e-6 {
            vscale(tb, 1.0 / tl)
        } else {
            Vec3::new(attack_sign(player.team), 0.0, 0.0)
        }
    };
    // Rendezvous anchor: DOWNRANGE on the bell's OWN future path, not the
    // straight rigger→point line. The terminal winch then drags the rigger
    // parallel to the bell's velocity (≈ co-linear), so at the catch point
    // |bell_v − player_v| ≈ ||bell_v| − REEL_PULL_SPEED|, small enough for
    // the terminal ease to bleed into the committed absorb window.
    let bspeed = vlen(bv);
    let bvdir = if bspeed > 1e-3 {
        vscale(bv, 1.0 / bspeed)
    } else {
        // Bell barely moving: a plain straight dive is already a rendezvous
        // (rel-speed ≈ our own speed, which the terminal ease kills).
        app_dir
    };
    // Far out we still need to actually CLOSE the gap to the bell's path;
    // near in we want to be running parallel to it. Blend the two with a
    // smooth, distance-only (deterministic) weight: mostly approach when
    // far, mostly bell-parallel once within ~3 terminal radii.
    let term_r = crate::tuning::DIVE_TERMINAL_RADIUS;
    let w_par = (1.0 - (d / (3.0 * term_r)).min(1.0)).clamp(0.0, 1.0);
    let blended = vnorm(vadd(
        vscale(app_dir, 1.0 - w_par),
        vscale(bvdir, w_par),
    ));
    let dir = if vlen(blended) > 1e-6 { blended } else { app_dir };
    // Anchor ahead of the catch point along the bell's path so reeling
    // converges onto — and then runs with — the bell's trajectory.
    let beyond = 14.0_f64;
    let mut anchor = vadd(ip, vscale(bvdir, beyond));
    // Keep the anchor inside the chamber (clamp cross-radius below the
    // skin) so the powered hook has real purchase.
    let arho = (anchor.y * anchor.y + anchor.z * anchor.z).sqrt();
    let max_r = REG_R - 1.0;
    if arho > max_r && arho > 1e-6 {
        let s = max_r / arho;
        anchor.y *= s;
        anchor.z *= s;
    }
    let pi = PartialInput {
        aim: Some(dir),
        fire_line_at: Some(Some(anchor)),
        reel: Some(-1),
        release: Some(false),
        pushoff: Some(false),
    };
    (pi, dir, ip)
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
                dive_commit_tick: -1.0,
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
                    catch_intent: false,
                };
            }
        }
    }

    // OFFENSE REBUILD — committed catch. If the live bell is ours to take,
    // steer onto its lead-intercept and signal the sim to widen the catch
    // envelope so a played pass actually completes (and defenders can pick).
    let cur_assignment: PlayerAssignment = director
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
    let catch_intent = wants_catch(player, state, &cur_assignment);

    // ── LOOSE-BELL DECISIVE DIVE (dominates coverage / w-max / nav) ───────
    // The gawk-ring fix: when the bell is loose and ours-to-take, the
    // single best-placed rigger (Director recover lead) + ONE deterministic
    // backup BREAK OFF and hard-dive — a powered-hook winch driven THROUGH
    // the predicted intercept (anchor beyond it, reel=-1, catch_intent).
    // This pragmatic catch utility STRONGLY dominates the epistemic /
    // weakest-sufficient terms for the committer(s) only; everyone else
    // keeps their role (anti-all-dive). It LATCHES with hysteresis: once a
    // rigger commits it stays committed (no per-tick flip-flop) until the
    // bell is taken, it catches, or it is clearly beaten by another rigger.
    {
        let bell_loose = state.bell.held_by.is_none();
        let already = cache
            .value
            .as_ref()
            .map(|c| c.dive_commit_tick >= 0.0)
            .unwrap_or(false);
        // Hysteresis: keep committing if we already were AND the bell is
        // still loose, ours-to-take, and reasonably near (not clearly
        // beaten — gap blows out). Otherwise (re)select fresh.
        let to_bell = vsub(state.bell.p, player.p);
        let still_takeable = bell_loose
            && state.bell.thrown_by.as_deref() != Some(player.id.as_str())
            && vlen(to_bell) < 85.0;
        let commit_dive = if already && still_takeable {
            true
        } else {
            is_dive_committer(player, state, director, &cur_assignment)
        };

        if let Some(c) = cache.value.as_mut() {
            if commit_dive {
                if c.dive_commit_tick < 0.0 {
                    c.dive_commit_tick = tick;
                }
            } else {
                c.dive_commit_tick = -1.0;
            }
        }

        if commit_dive {
            let (partial, dir, catch_pt) = decisive_dive_input(player, state);
            let aim_dither = cache
                .value
                .as_ref()
                .map(|c| c.aim_dither)
                .unwrap_or_else(v3z);
            let nav_aim = vnorm(vadd(
                partial.aim.unwrap_or(dir),
                aim_dither,
            ));
            let pushoff = should_pushoff(player, state.bell.p);

            // ── COMMIT ONCE, THEN RIDE THE LINE (the catch fix, AI side) ──
            // Grappling now has latency: a fired claw takes flight time to
            // land and the line then LOCKS to its anchor until released
            // (sim_world). If we re-aimed a fresh predicted intercept every
            // tick the sim would just ignore it (line committed) — but the
            // worse failure the OLD code had was perpetually chasing a
            // moving prediction and sliding past the real bell. So the dive
            // now fires EXACTLY ONCE and then rides physics:
            //   • No line visible  ⇒ FIRE the committed anchor (once).
            //   • Line visible (claw in flight OR attached) ⇒ DO NOT
            //     re-fire; keep reel = −1 + catch_intent and let the winch
            //     carry us through the bell.
            //   • Line visible but the bell has clearly ESCAPED the
            //     committed approach ⇒ RELEASE to break off; the sim's
            //     re-fire cooldown then elapses and a fresh commit can
            //     re-acquire.
            // `player.line` is the AI's snapshot view of its own rig — its
            // mere presence means a claw is committed; this is the
            // commit-once latch (no per-tick fireLineAt re-issue), kept
            // deterministic (snapshot-driven, no rng, no wall clock).
            let has_line = player.line.is_some();
            let bell_escaped = vlen(to_bell) > 60.0;
            let (fire, do_release) = if !has_line {
                (partial.fire_line_at.unwrap_or(None), false)
            } else if bell_escaped {
                (None, true)
            } else {
                (None, false)
            };

            // ── TERMINAL WINCH-EASE (the catch fix) ──────────────────────
            // The rendezvous anchor has pulled us onto a track roughly
            // PARALLEL to the bell's velocity. Once we are within
            // DIVE_TERMINAL_RADIUS of the predicted catch point, STOP
            // hard-winching: emit reel = 0 so the winch adds no more Δv and
            // our speed bleeds toward the bell's, dropping
            // |bell_v − player_v| into the committed absorb window
            // (COMMIT_CATCH_SPEED = 38) with closing ≥ closing_floor, so
            // collision::try_catch_ex returns Caught. We do NOT release the
            // line — grapple latency makes a re-acquire costly — and we keep
            // catch_intent = true so `committed` stays set. Geometry only
            // (a radius test on the deterministic predicted point): no rng,
            // no wall clock, fully tick-reproducible.
            let near_catch =
                vlen(vsub(catch_pt, player.p)) <= crate::tuning::DIVE_TERMINAL_RADIUS;
            let reel_cmd: i32 = if near_catch {
                0
            } else {
                partial.reel.unwrap_or(-1)
            };

            return PlayerInput {
                id: player.id.clone(),
                aim: nav_aim,
                fire_line_at: fire,
                reel: reel_cmd,
                release: do_release || partial.release.unwrap_or(false),
                pushoff: partial.pushoff.unwrap_or(false) || pushoff,
                throw_charge: 0.0,
                throw_released: false,
                throw_spin: 0.0,
                thrumbler: v3z(),
                // Always signal committed-catch intent while diving so the
                // sim widens the catch envelope (collision::try_catch_ex).
                catch_intent: true,
            };
        }
    }

    // Otherwise execute committed navigation toward the cached target.
    let (target, aim_dither) = {
        let commit = cache.value.as_ref().unwrap();
        let target = if catch_intent {
            bell_intercept(player, state)
        } else {
            match commit.nav_target {
                Some(nt) => vadd(nt, commit.catch_offset),
                None => player.p,
            }
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
        catch_intent,
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
    // RECOVER — WEAKEST USEFUL GRAPPLE toward the loose bell. The old code
    // either stern-chased or aimed a single argmin lead point. Instead we
    // predict the bell BAND (generative model) and go boldly for the band
    // sample this rigger is MOST favorable to intercept — the broad
    // reachable region, not a knife-edge point. This recovers skin-stuck
    // bells (the predictor's skin bounce is in the band) and produces bold
    // chamber-spanning swings instead of an unwinnable tail chase.
    if assignment.job == Job::Recover {
        let band = efe::BellBand::predict(
            state.bell.p, state.bell.v, state.omega, REG_R,
        );
        // SHADOW / SAFETY: a flagged recover member does NOT chase the
        // live bell — it stations at the predicted REBOUND LOCUS so a
        // primary bobble drops the ball straight onto it. The rebound is
        // rolled from the PRIMARY's predicted touch (recover_id's
        // t_intercept against the same canon band) through the bobble
        // damp + a short spill — distinct from, and never converging on,
        // the primary's intercept point. If recover_id is unresolvable we
        // fall back to a defensive station behind the band mid (still
        // off the dive line). Pure geometry, deterministic, no rng.
        if assignment.is_shadow() {
            let t_primary = director
                .recover_id
                .as_deref()
                .and_then(|rid| {
                    state.players.iter().find(|p| p.id == rid)
                })
                .map(|pp| band.time_to_intercept(pp.p, pp.v))
                // No resolvable primary: use the band mid horizon as a
                // sane deterministic touch time.
                .unwrap_or_else(|| {
                    band.samples[band.samples.len() / 2].0
                });
            return efe::rebound_locus(
                state.bell.p,
                state.bell.v,
                state.omega,
                REG_R,
                t_primary,
            );
        }
        // PRIMARY recover: go boldly for the band sample this rigger is
        // MOST favorable to intercept (the broad reachable region).
        // close_v = a confident reel/swoop closing speed; high so the
        // chosen sample is one we can actually get IN FRONT of.
        let ip = band.best_intercept_for(player.p, 24.0);
        // A tiny tuck toward the mid sample keeps us arriving INTO the
        // path (not behind it) without committing to a single point.
        let mid = band.mid().p;
        let toward = vsub(mid, ip);
        let tl = vlen(toward);
        if tl > 1e-6 {
            return vadd(ip, vscale(toward, (1.5_f64).min(tl) / tl));
        }
        return ip;
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

    // SUPPORT (offense, off-ball) and ZONE (defense, off-ball/help): these
    // are exactly the agents with NO immediate pragmatic job — the ones
    // that used to twitch on a single argmin slot. They now take an
    // EPISTEMIC / COVERAGE action: among a FAN of broad candidate regions
    // spanning the chamber volume, pick the WEAKEST = the region that
    // minimizes Expected Free Energy (pragmatic responsiveness to the bell
    // band + epistemic team-volume spread). This is the volume-play drive.
    if assignment.job == Job::Support || assignment.job == Job::Zone {
        return wmax_coverage_target(
            player, state, director, assignment, style,
        );
    }

    // Otherwise defer to the role policy — but still pass off-ball,
    // jobless role agents through the same w-maxing coverage selection so
    // they span the volume instead of collapsing on the ball.
    let base =
        role_target(player, state, m, profile, director, assignment, rng, style);
    let bell_held = state.bell.held_by.is_some();
    let off_ball = !bell_held
        || state.bell.held_by.as_deref() != Some(player.id.as_str());
    if off_ball && assignment.job != Job::Receive {
        // Blend the role's intent with the coverage region: keep the role's
        // identity but pull it toward the under-covered, responsive part of
        // the volume (weakest-sufficient: the broad region, not the slot).
        let cov = wmax_coverage_target(
            player, state, director, assignment, style,
        );
        return vadd(vscale(base, 0.45), vscale(cov, 0.55));
    }
    base
}

/// WEAKEST-SUFFICIENT navigation for an off-ball / jobless agent.
///
/// Builds a FAN of broad candidate regions that span the chamber volume
/// (a few forward-progress depths × a few cross-tube angles), scores each
/// by Expected Free Energy, and returns the WEAKEST acceptable one — the
/// region that keeps the most futures responsive and the team most spread.
///
/// EFE(cand) = pragmatic + epistemic:
///   - pragmatic = expected "distance to be able to respond" to the
///     predicted bell BAND from `cand` (low = stays useful across the
///     range of bell futures), softened toward the attack direction so
///     offense still flows forward.
///   - epistemic = team volume coverage: a crowding penalty (don't
///     collapse on teammates / the ball) + an axis-redundancy penalty
///     (don't all stack at the same depth) ⇒ fill the empty volume.
///
/// "Weakest" = we deliberately bias toward the BROAD central candidate
/// among those within a fitness band of the best EFE, not the razor argmin
/// — that is what kills the twitch and produces bold spanning swings.
fn wmax_coverage_target(
    player: &PlayerSim,
    state: &SimState,
    director: &DirectorState,
    assignment: &PlayerAssignment,
    style: Option<RoleStyle>,
) -> Vec3 {
    let team = player.team;
    let skin_r = REG_R;
    let omega = state.omega;
    let sgn = attack_sign(team);
    // EFE controller params (thread-local; production default is the
    // verbatim former constants ⇒ behavior-preserving). The genome
    // scaffold is UNCHANGED — role-conditioning below SCALES these
    // existing params per role, it does not add genes.
    let ep = super::efe_params::efe_params();

    let band =
        efe::BellBand::predict(state.bell.p, state.bell.v, omega, skin_r);
    let others = efe::teammate_positions(player, state);
    let others_fp: Vec<f64> = others
        .iter()
        .map(|p| forward_progress(team, p.x))
        .collect();
    let vm = efe::VolumeModel::new(team, crate::tuning::GATE_X, skin_r);

    // ── EFE ROLE-CONDITIONING (the conceptual core) ──────────────────────
    // The free energy this controller minimizes is now CONDITIONED on the
    // loose-bell coordination role, instead of every off-ball body sharing
    // one objective and clumping. We do NOT add learner genes (the GA
    // scaffold stays 14 wide, EfeParams::default() byte-verbatim): we
    // scale the EXISTING term weights/anchor per role. Deterministic.
    //
    //   * bell loose & this is an off-ball SUPPORT body (the gawkers):
    //     the EPISTEMIC team-spread term is AMPLIFIED and the candidate
    //     fan is pushed OFF the bell point — the primary dives and the
    //     shadow holds the spill, so a third body crowding the ball is
    //     pure noise. This is the anti-swarm win: support spreads into
    //     coverage of the chamber volume rather than gawking in a clump.
    //   * a flagged SHADOW that somehow reaches here (it normally returns
    //     from the recover nav): center coverage on the rebound region.
    //   * ZONE / possession states: unchanged objective (default scale).
    let bell_loose = state.bell.held_by.is_none();
    let is_shadow = assignment.is_shadow();
    let support_during_loose =
        bell_loose && assignment.job == Job::Support;
    // Role weight multipliers on the EXISTING params (1.0 = unchanged).
    // SUPPORT during a loose bell: heavier team-spread (don't clump),
    // lighter raw bell-responsiveness (the recover unit owns the ball),
    // stronger forward flow so the spread is a useful advancing shape.
    let (w_crowd, w_redun, w_resp, w_fwd) = if support_during_loose {
        (2.2_f64, 1.6_f64, 0.45_f64, 1.5_f64)
    } else {
        (1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64)
    };
    // Anti-swarm bell stand-off: a deterministic geometric penalty that
    // grows as a candidate nears the live bell point while the recover
    // unit owns it — the explicit "stop gawking on the ball" term. Pure
    // geometry over the bell position; no rng, order-free.
    let bell_p = state.bell.p;
    const SWARM_STANDOFF_R: f64 = 26.0; // m — recover unit's exclusive bubble
    const SWARM_STANDOFF_W: f64 = 4.0; // EFE weight of intruding it

    // Defense (Zone) anchors nearer our own ring; offense (Support / role)
    // anchors around the bell and ahead toward the attack gate. A SHADOW
    // that falls through here anchors on the predicted rebound locus.
    let defending = assignment.job == Job::Zone;
    let anchor_x = if is_shadow {
        let t_primary = director
            .recover_id
            .as_deref()
            .and_then(|rid| state.players.iter().find(|p| p.id == rid))
            .map(|pp| band.time_to_intercept(pp.p, pp.v))
            .unwrap_or_else(|| band.samples[band.samples.len() / 2].0);
        efe::rebound_locus(
            state.bell.p, state.bell.v, omega, skin_r, t_primary,
        )
        .x
    } else if defending {
        defend_ring_x(team) + sgn * ep.cov_defense_anchor_ahead
    } else if support_during_loose {
        // Push the coverage fan AHEAD of the bell, down our advance axis,
        // so support pre-positions for the catch→outlet→advance payoff
        // instead of converging on the loose ball.
        state.bell.p.x + sgn * (ep.cov_offense_anchor_ahead + 36.0)
    } else {
        state.bell.p.x + sgn * ep.cov_offense_anchor_ahead
    };

    // Style gives each agent a STABLE distinct slice of the fan so the team
    // naturally fans out (deterministic — committed style draw, no rng).
    let s_ang = style.map(|s| s.angle).unwrap_or(0.5);
    let s_rad = style.map(|s| s.radius).unwrap_or(0.5);

    // Candidate FAN: 3 depths along the attack axis × 4 cross angles ×
    // 2 radii — broad regions spanning the tube, not points.
    let depths = [-70.0_f64, 10.0, 90.0];
    let radii = [14.0_f64 + s_rad * 8.0, 30.0 + s_rad * 10.0];
    let mut best: Option<Vec3> = None;
    let mut best_efe = f64::INFINITY;

    for (di, &dd) in depths.iter().enumerate() {
        let cx = anchor_x + sgn * dd;
        for k in 0..4 {
            // Spread the four angles around the agent's stable style slice.
            let ang = std::f64::consts::PI
                * (0.15 + s_ang * 1.4 + k as f64 * 0.5);
            for &rr in radii.iter() {
                let r = rr.min(skin_r * 0.85);
                let cand =
                    Vec3::new(cx, r * ang.cos(), r * ang.sin());

                // PRAGMATIC: responsiveness to the predicted bell band.
                // Offense wants to be a favorable future interceptor;
                // defense wants to be able to deny it. Same quantity.
                let resp = band.expected_response_gap(cand, ep.cov_close_v);
                // Soften by forward intent so offense still flows up-field.
                let fwd = forward_progress(team, cand.x);
                let fwd_pull = if defending {
                    0.0
                } else {
                    -(fwd / (crate::tuning::GATE_X.abs()))
                        * ep.cov_fwd_pull_gain
                        * w_fwd
                };
                // PRAGMATIC term, ROLE-CONDITIONED: PRIMARY-style bodies
                // want raw bell responsiveness (w_resp = 1); a loose-bell
                // SUPPORT body deliberately DOWN-WEIGHTS it (the recover
                // unit owns the ball) so its objective is dominated by
                // coverage + advance, not by racing to the same point.
                let pragmatic = resp * w_resp + fwd_pull;

                // EPISTEMIC: team volume coverage, ROLE-CONDITIONED. The
                // crowding/redundancy weights are amplified for loose-bell
                // support so the team spreads into a coordinated shape.
                let crowd = vm.crowding_at(cand, &others);
                let redun = vm.axis_redundancy(fwd, &others_fp);
                let epistemic = crowd * ep.cov_crowd_w * w_crowd
                    + redun * ep.cov_redun_w * w_redun;

                // ANTI-SWARM STAND-OFF: a deterministic geometric penalty
                // that fires only while the recover unit owns the loose
                // ball — a candidate inside the bell's exclusive bubble is
                // pure gawking and is pushed out. Smooth (quadratic)
                // falloff so the gradient spreads the cluster rather than
                // snapping it. Zero in all non-loose / non-support states
                // ⇒ behaviour-preserving everywhere else.
                let swarm_pen = if support_during_loose {
                    let db = (cand.x - bell_p.x)
                        .hypot(cand.y - bell_p.y)
                        .hypot(cand.z - bell_p.z);
                    if db < SWARM_STANDOFF_R {
                        let t = 1.0 - db / SWARM_STANDOFF_R;
                        SWARM_STANDOFF_W * t * t
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // Weakest-sufficient bias: prefer the central depth/radius
                // (broader success-set, less committal) by a small bonus
                // so among near-equal EFE the BROAD region wins.
                let breadth_bonus =
                    if di == 1 { ep.cov_breadth_bonus } else { 0.0 };

                let total =
                    pragmatic + epistemic + swarm_pen + breadth_bonus;
                if total < best_efe {
                    best_efe = total;
                    best = Some(cand);
                }
            }
        }
    }
    best.unwrap_or_else(|| {
        Vec3::new(anchor_x, 18.0, 0.0)
    })
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

    // OFFENSE REBUILD — the designated gate receiver stages just PAST the
    // next cast gate, near the spin axis (low cross-radius) so the carrier
    // can hit them and the completed pass clears the gate. This is the
    // played progression route (carry → gate pass → advance).
    if director.gate_receiver_id.as_deref() == Some(player.id.as_str())
        && (assignment.job == Job::Receive || assignment.job == Job::Support)
        && player.role != RiggerRole::Reach
    {
        // Hold a shallow on-axis post a touch beyond the gate line.
        let r = 5.0;
        let ang = std::f64::consts::PI * (0.5 + director.style_noise * 0.25);
        return Vec3::new(director.gate_stage_x, r * ang.cos(), r * ang.sin());
    }

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

/// decide_throw — REBUILT as a w-maxing active-inference controller.
///
/// OLD (the dead-game bug): throw iff a near-exact closed-form gate
/// solution threaded the ring with `arrive_rho <= GATE_RADIUS*0.6`
/// (strongest possible precondition ⇒ they almost never threw), else fall
/// to a precise argmax lead pass. Maximally over-specific.
///
/// NEW: a small ACTION REPERTOIRE of candidate releases is enumerated
/// (gate closed-form seed, lead-pass seeds to each teammate, and a coarse
/// world-direction fan at a few speeds). Each candidate's OUTCOME is
/// predicted with the canon generative model (`efe::roll_forward` = the
/// sacred Coriolis integrator + skin bounce). A candidate is FIT if its
/// predicted outcome is good-enough across a WIDE band: it scores through
/// our ring, OR it carries the bell forward toward the attack gate and is
/// reachable by a teammate (a covered receiver), OR it just clearly
/// advances the bell down the volume. Among FIT candidates we pick the
/// **WEAKEST sufficient** one = the throw with the LARGEST TOLERANCE BAND:
/// we perturb the release by a fan of aim errors and count how many
/// perturbed releases STILL land a good-enough outcome. The throw whose
/// success-set survives the most error wins — robust, not perfect. They
/// throw whenever any fit candidate exists (often), not rarely+perfectly.
///
/// Determinism: pure geometry + the canon integrator. The only RNG draws
/// are the SAME difficulty draws the frozen spec made (perturb_direction
/// throw_variance, miss_open_chance gate, spin/charge jitter), kept so the
/// determinism guards stay bit-identical in shape.
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

    let omega = state.omega;
    let skin_r = REG_R;
    let team = player.team;
    // EFE throw-controller params (thread-local; default = former
    // hardcoded constants ⇒ behavior-preserving).
    let ep = super::efe_params::efe_params();
    let sgn = attack_sign(team);
    let ring_x = attack_ring_x(team);
    let dist_to_ring = (ring_x - player.p.x).abs();
    let attack_gate_fp = forward_progress(team, ring_x);
    let my_fp = forward_progress(team, player.p.x);

    let posture_mul = match director.posture {
        CastPosture::Chase => 1.12,
        CastPosture::Grind => 0.92,
        CastPosture::Tempo => 1.0,
    };
    let throw_speed = (18.0 + profile.aggression * 12.0) * posture_mul;

    // The sim launches v_bell = thrower.v + dir*speed. To realise a desired
    // world launch v0 the player aims along (v0 - player.v) at charge for
    // |v0 - player.v|. We work in world-launch space and convert at the end.
    let throw_min = THROW_MIN_SPEED;
    let throw_max = THROW_MAX_SPEED;

    // Predicted teammate band — where each teammate will be over the
    // generative horizon (so a "catchable" outcome means catchable by where
    // they're GOING, not a stale slot). Cheap: one mid-horizon roll each.
    let tm_future: Vec<(usize, Vec3)> = teammates
        .iter()
        .enumerate()
        .map(|(i, tm)| {
            let st = efe::roll_forward(tm.p, tm.v, omega, ep.throw_tm_horizon, skin_r);
            (i, st.p)
        })
        .collect();

    // ── Outcome quality of a world launch v0 (the PRAGMATIC evaluator). ──
    // Roll the bell with the canon model from the release; return a scalar
    // in roughly [0,1+] where >= GOOD_ENOUGH means "this is a good throw".
    // Higher = better. No RNG.
    let good_enough = ep.throw_good_enough;
    let eval_launch = |v0: Vec3| -> f64 {
        // Roll far enough to see it cross the ring plane or settle.
        let st = efe::roll_forward(player.p, v0, omega, ep.throw_eval_horizon, skin_r);
        let end_fp = forward_progress(team, st.p.x);
        let end_rho = st.p.y.hypot(st.p.z);

        // (a) SCORING: did it pass our attack ring plane reasonably central?
        // Detect a plane crossing by rolling in two halves and checking the
        // sign of (ring_x - x). Cheap proxy: if it ends at/past the ring
        // forward-progress AND the cross radius stayed catch-able.
        let crossed_ring = end_fp >= attack_gate_fp - 2.0;
        let central = end_rho <= REG_GATE_RADIUS * 1.6;
        if crossed_ring && central {
            return 1.0;
        }

        // (b) ADVANCE + CATCHABLE: bell ends meaningfully forward of us and
        // near a teammate's predicted spot (a completable pass / coverage).
        let gain = forward_progress(team, st.p.x) - my_fp;
        let mut nearest_tm = f64::INFINITY;
        for (_i, tp) in tm_future.iter() {
            let d = (st.p.x - tp.x).hypot(st.p.y - tp.y).hypot(st.p.z - tp.z);
            if d < nearest_tm {
                nearest_tm = d;
            }
        }
        // Reachable if a teammate is within a generous catch envelope of the
        // bell's settle point (wide on purpose — weakest-sufficient).
        let reach = (1.0 - (nearest_tm / ep.throw_reach_radius)).clamp(0.0, 1.0);
        let gain_norm = (gain / 120.0).clamp(-0.5, 1.0);

        // Don't reward flinging it into the skin or behind us.
        let skin_pen = if end_rho > skin_r * 0.96 { 0.25 } else { 0.0 };
        let backward_pen = if gain < -20.0 { 0.4 } else { 0.0 };

        0.30 + 0.45 * reach + 0.35 * gain_norm - skin_pen - backward_pen
    };

    // ── Build the ACTION REPERTOIRE of world launches. ──────────────────
    let mut candidates: Vec<Vec3> = Vec::new();

    // Seed 1: the gate closed-form (still a fine candidate — just no longer
    // the ONLY acceptable throw, and no longer required to be near-exact).
    if dist_to_ring > 6.0 {
        if let Some(g) = solve_gate_throw(
            player.p,
            team,
            omega,
            throw_speed,
            profile.loop_propensity,
            player.v,
        ) {
            candidates.push(g.v0);
        }
    }

    // Seed 2: a lead pass to each teammate's predicted position (forward
    // ones first — we want to advance). These are seeds, not the answer.
    for tm in teammates.iter() {
        // miss_open_chance: SAME draw the frozen spec made per teammate.
        if rng.next() < scaling.miss_open_chance {
            continue;
        }
        if let Some(lead) =
            solve_lead_velocity(player.p, throw_speed, tm.p, tm.v, omega)
        {
            candidates.push(lead.v0);
        }
    }

    // Seed 3: a coarse world-direction FAN biased down the attack axis, at
    // a few speeds. This is the part that makes them throw OFTEN: even with
    // no clean solver hit there is almost always a forward fling that
    // advances the bell within the wide good-enough band.
    {
        let base = Vec3::new(sgn, 0.0, 0.0);
        let yz: [(f64, f64); 5] =
            [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)];
        let speeds = [throw_min + 4.0, throw_speed, throw_max - 4.0];
        for &(dy, dz) in yz.iter() {
            let dir = vnorm(Vec3::new(base.x, base.y + dy, base.z + dz));
            for &sp in speeds.iter() {
                candidates.push(vscale(dir, sp));
            }
        }
    }

    // ── W-MAXING SELECTION: among FIT candidates, pick the WEAKEST =
    // the one with the LARGEST TOLERANCE BAND (success-set under release
    // error). We perturb each fit launch by a fixed fan of aim errors and
    // count how many perturbations still clear good-enough. The throw that
    // survives the most error is the least committal / most robust. ──────
    let err_fan: [(f64, f64); 9] = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.7, 0.7),
        (0.7, -0.7),
        (-0.7, 0.7),
        (-0.7, -0.7),
    ];
    // Perturbation magnitude scaled to the difficulty's own throw variance
    // band so "robust" means robust to THIS rigger's real error.
    let err_mag = (3.0 + scaling.throw_variance * ep.throw_err_mag_scale).max(2.0);

    let mut best_v0: Option<Vec3> = None;
    let mut best_band: f64 = -1.0;
    let mut best_center_q: f64 = 0.0;

    for &v0 in candidates.iter() {
        let sp = vlen(v0);
        if !(throw_min..=throw_max + 8.0).contains(&sp) || sp < 1e-3 {
            continue;
        }
        let center_q = eval_launch(v0);
        if center_q < good_enough {
            continue; // not FIT — its predicted outcome isn't good-enough.
        }
        // Tolerance band = fraction of the perturbed fan still good-enough.
        let unit = vscale(v0, 1.0 / sp);
        // Build two world axes perpendicular to the launch for the error
        // fan (deterministic basis: cross with x then with y as fallback).
        let mut a = Vec3::new(0.0, 1.0, 0.0);
        if unit.y.abs() > 0.9 {
            a = Vec3::new(1.0, 0.0, 0.0);
        }
        let e1 = vnorm(Vec3::new(
            unit.y * a.z - unit.z * a.y,
            unit.z * a.x - unit.x * a.z,
            unit.x * a.y - unit.y * a.x,
        ));
        let e2 = vnorm(Vec3::new(
            unit.y * e1.z - unit.z * e1.y,
            unit.z * e1.x - unit.x * e1.z,
            unit.x * e1.y - unit.y * e1.x,
        ));
        let mut hits = 0.0_f64;
        for &(c1, c2) in err_fan.iter() {
            let perturbed = vadd(
                v0,
                vadd(vscale(e1, c1 * err_mag), vscale(e2, c2 * err_mag)),
            );
            if eval_launch(perturbed) >= good_enough {
                hits += 1.0;
            }
        }
        let band = hits / err_fan.len() as f64;
        // Weakest sufficient: maximize the tolerance band; break ties by
        // center quality so among equally-robust throws we take the better.
        if band > best_band + 1e-9
            || (band > best_band - 1e-9 && center_q > best_center_q + 1e-9)
        {
            best_band = band;
            best_center_q = center_q;
            best_v0 = Some(v0);
        }
    }

    // ── Decide WHETHER to throw at all. With a wide good-enough band the
    // common case is "yes" (that is the point — they should throw often).
    // Stall pressure only ever LOOSENS this further. ────────────────────
    let stall = cache.value.as_ref().unwrap().hold_ticks;
    let pressured = nearest_opponent_dist(player, &opponents) < ep.throw_pressure_dist;
    let force = stall > ep.throw_stall_force_ticks || m.cast.throws_left as f64 <= 1.0;

    let chosen = match best_v0 {
        Some(v) => v,
        None => {
            // No fit candidate. If we're forced (stall/last throw) fling the
            // most-forward seed anyway — a weak, robust release beats a dead
            // stuck carrier. Otherwise hold and keep carrying.
            if force {
                vscale(Vec3::new(sgn, 0.0, 0.0), throw_speed)
            } else {
                let commit = cache.value.as_mut().unwrap();
                commit.throw_go = false;
                commit.throw_target_id = None;
                commit.throw_dir = None;
                return;
            }
        }
    };

    // Acceptance: throw if we found a fit candidate, or if pressured/forced
    // (a robust outlet under pressure beats holding into a strip).
    let have_fit = best_v0.is_some();
    if !have_fit && !force && !pressured {
        let commit = cache.value.as_mut().unwrap();
        commit.throw_go = false;
        commit.throw_target_id = None;
        commit.throw_dir = None;
        return;
    }

    // Convert world launch → the throw the player imparts (sim adds v).
    let throw_vec = vsub(chosen, player.v);
    let rel_speed = vlen(throw_vec).clamp(throw_min, throw_max);
    let aim_dir = if vlen(throw_vec) > 1e-6 {
        vnorm(throw_vec)
    } else {
        Vec3::new(sgn, 0.0, 0.0)
    };
    // SAME difficulty perturbation the frozen spec applied to a thrown dir.
    let noisy_dir = perturb_direction(aim_dir, scaling.throw_variance, rng);

    let throw_spin = if director.attacking_free {
        -(0.3 + profile.loop_propensity * 0.7)
    } else {
        0.2 + (1.0 - profile.free_end_bias) * 0.3
    };
    let charge = ((rel_speed - throw_min) / (throw_max - throw_min))
        .clamp(0.0, 1.0)
        // SAME charge jitter draw shape the frozen spec made.
        + (rng.next() - 0.5) * 0.04 * scaling.throw_variance;
    let charge = charge.clamp(0.0, 1.0);

    // If a teammate is the natural target of this launch, name them so the
    // sim widens their catch envelope (committed-catch / gate-clear path).
    let mut tgt_id: Option<String> = None;
    {
        let st = efe::roll_forward(player.p, chosen, omega, 1.6, skin_r);
        let mut best_d = 34.0_f64;
        for tm in teammates.iter() {
            let (i, _) = tm_future
                .iter()
                .find(|(i, _)| *i < teammates.len() && teammates[*i].id == tm.id)
                .copied()
                .unwrap_or((usize::MAX, Vec3::new(0.0, 0.0, 0.0)));
            let tp = if i != usize::MAX { tm_future[i].1 } else { tm.p };
            let d = (st.p.x - tp.x).hypot(st.p.y - tp.y).hypot(st.p.z - tp.z);
            if d < best_d {
                best_d = d;
                tgt_id = Some(tm.id.clone());
            }
        }
    }

    let commit = cache.value.as_mut().unwrap();
    commit.throw_go = true;
    commit.throw_target_id = tgt_id;
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
    use crate::ai::types::{BellState, Cast, Gate, MatchPhase, TeamSide};
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
            gate_stage_x: crate::tuning::GATE_X * 0.25,
            gate_receiver_id: Some("H2".to_string()),
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
    fn loose_bell_closest_rigger_decisively_dives_not_hovers() {
        // A loose bell + a clearly-closest rigger named the Director recover
        // lead. Its decided input MUST be a committed powered-hook dive:
        // fire_line_at SOME (an anchor toward an intercept-enabling point)
        // + reel == -1 + catch_intent == true — NOT a hover/coverage frame
        // (no anchor / reel 0).
        let players = vec![
            mk_player("H1", TeamSide::Home, RiggerRole::Spinner, Vec3::new(6.0, 4.0, 1.0)),
            mk_player("H2", TeamSide::Home, RiggerRole::Faithwing, Vec3::new(-180.0, 6.0, -3.0)),
            mk_player("A1", TeamSide::Away, RiggerRole::Anchor, Vec3::new(120.0, 5.0, 5.0)),
        ];
        let mut m = mk_match();
        m.contest = None;
        let mut dir = mk_director();
        // Loose-bell director state: H1 is the nearest recover lead.
        dir.bell_loose = true;
        dir.has_possession = false;
        dir.carrier_id = None;
        dir.recover_id = Some("H1".to_string());
        dir.gate_receiver_id = None;
        dir.assignments.clear();
        dir.assignments.insert(
            "H1".to_string(),
            PlayerAssignment {
                job: Job::Recover,
                mark_id: None,
                depth_slot: 0.0,
                radius_slot: 0.0,
                pressure: 0.0,
            },
        );
        dir.assignments.insert(
            "H2".to_string(),
            PlayerAssignment {
                job: Job::Support,
                mark_id: None,
                depth_slot: 0.4,
                radius_slot: 0.45,
                pressure: 0.0,
            },
        );
        let state = mk_state(players, None); // bell loose (held_by None)
        let player = state.players[0].clone();
        let mut rng = AiRng::make(424242, 100, 0);
        let mut cache = PlayerCommitCache::default();
        let out = compute_player_input(
            &player, &state, &m, &style_to_profile("balanced", "medium"),
            &dir, Difficulty::Pro, &mut rng, &mut cache, true, 30.0,
        );
        assert!(
            out.fire_line_at.is_some(),
            "committer must FIRE an anchor (decisive dive), got hover/no-anchor"
        );
        // Re-baselined for the velocity-matched-rendezvous catch fix: the
        // dive now hard-winches (reel = -1) ONLY while still closing; once
        // within DIVE_TERMINAL_RADIUS of the predicted catch point it eases
        // the winch (reel = 0) so rel-speed bleeds into the committed
        // absorb window (try_catch_ex Caught). Here the bell is near & slow
        // so the predicted catch point is already inside the terminal
        // radius ⇒ the CORRECT decisive input is the terminal ease, not a
        // through-winch. The committal property the test guards is still
        // proven by fire_line_at (it FIRED a dive anchor, not a hover) +
        // catch_intent + the latch; the winch sign is a phase detail.
        assert!(
            out.reel == -1 || out.reel == 0,
            "committer must drive the dive: hard-winch (-1) while closing \
             or terminal-ease (0) once within the catch-point radius, \
             got {}",
            out.reel
        );
        assert!(out.catch_intent, "committer must signal catch_intent");
        // Hysteresis latched.
        assert!(
            cache.value.as_ref().unwrap().dive_commit_tick >= 0.0,
            "dive commitment must latch for hysteresis"
        );
        // A far-off non-pack teammate must NOT be diving (anti-all-dive).
        let h2 = state.players[1].clone();
        let mut rng2 = AiRng::make(424242, 100, 1);
        let mut cache2 = PlayerCommitCache::default();
        let out2 = compute_player_input(
            &h2, &state, &m, &style_to_profile("balanced", "medium"),
            &dir, Difficulty::Pro, &mut rng2, &mut cache2, true, 30.0,
        );
        assert!(
            cache2.value.as_ref().unwrap().dive_commit_tick < 0.0,
            "a non-committer teammate must NOT latch a dive (anti-all-dive)"
        );
        let _ = out2;
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

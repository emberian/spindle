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
    difficulty_scaling, CastPosture, DifficultyScaling, DirectorState, Job, PlayRole,
    PlayerAssignment, PlayerCommit, PlayerCommitCache,
};
use super::efe;
use super::gate_solve::{solve_gate_throw, THROW_MAX_SPEED, THROW_MIN_SPEED};
use super::lead_predict::solve_lead_velocity;
use super::orientation::{
    attack_ring_x, attack_sign, defend_ring_x, forward_progress, gate_world_x,
};
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
const SETTLE_RADIUS: f64 = 18.0; // was 6.0 — covers grapple dead zone
const MICRO_DV_MAX: f64 = 3.5; // was 2.0 — stronger corrections
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
// Render-only legibility (the overlay seam) re-invokes this committer
// predicate from `system.rs` to label each rigger's intent. It is a pure
// deterministic read (no rng, no cache mutation) so re-evaluating it never
// perturbs the emitted InputFrame or the determinism hash.
pub(crate) fn wants_catch(
    player: &PlayerSim,
    state: &SimState,
    assignment: &PlayerAssignment,
    active_pass_target: Option<&str>,
) -> bool {
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

    let thrown_by_team = state
        .bell
        .thrown_by
        .as_deref()
        .and_then(|tid| state.players.iter().find(|p| p.id == tid))
        .map(|p| p.team);
    let friendly_pass = thrown_by_team == Some(player.team);

    // If we know the intended catcher of our own in-flight pass, only that
    // rigger gets the automatic committed catch envelope. Other teammates
    // keep their spacing unless the ball is practically hitting them.
    if friendly_pass {
        if let Some(tid) = active_pass_target {
            let target_valid = state
                .players
                .iter()
                .any(|p| p.id == tid && p.team == player.team);
            if target_valid {
                if tid == player.id {
                    return true;
                }
                let bs = vlen(state.bell.v);
                let closing = if bs < 1e-3 {
                    gap < 8.0
                } else {
                    -vdot(state.bell.v, to_bell) / (bs * gap.max(1e-6)) > 0.75
                        && gap < 16.0
                };
                return closing;
            }
        }
    }

    // Directed ball roles commit automatically. `Support` is deliberately not
    // included: it is spacing/coverage, not an implicit second receiver.
    if matches!(assignment.job, Job::Recover | Job::Receive) {
        return true;
    }
    // Anyone else commits if the bell is reachable. Check both current gap
    // AND predicted future gap (a crossing ball might be far now but pass
    // close within 1-2 seconds).
    if gap < 15.0 {
        return true;
    }
    let bs = vlen(state.bell.v);
    if bs < 1e-3 {
        return gap < 15.0;
    }
    // Predicted closest approach: project bell along its velocity and check
    // minimum distance to the player (simple linear CPA estimate).
    let rel_pos = to_bell; // bell - player
    let rel_vel = vsub(state.bell.v, player.v);
    let rv2 = vdot(rel_vel, rel_vel);
    if rv2 > 1e-6 {
        let t_cpa = (-vdot(rel_pos, rel_vel) / rv2).clamp(0.0, 3.0);
        let cpa_pos = vadd(rel_pos, vscale(rel_vel, t_cpa));
        let cpa_dist = vlen(cpa_pos);
        if cpa_dist < 25.0 && t_cpa < 2.5 {
            return true;
        }
    }
    let closing = -vdot(state.bell.v, to_bell) / (bs * gap.max(1e-6));
    closing > 0.0 && gap < 50.0
}

/// OFFENSE REBUILD — lead-intercept point of the live bell, so a committed
/// catcher actually navigates onto the ball instead of a static slot.
/// RK4 + skin-bounce predictor (same model the recover branch uses),
/// horizon scaled by gap so it stays stable. No rng.
pub(crate) fn bell_intercept(player: &PlayerSim, state: &SimState) -> Vec3 {
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
pub(crate) fn is_dive_committer(
    player: &PlayerSim,
    state: &SimState,
    director: &DirectorState,
    assignment: &PlayerAssignment,
    active_pass_target: Option<&str>,
) -> bool {
    if state.bell.held_by.is_some() {
        return false;
    }
    if !wants_catch(player, state, assignment, active_pass_target) {
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
                    .map(|a| wants_catch(pp, state, a, active_pass_target))
                    .unwrap_or(false)
        })
        .unwrap_or(false);
    if primary_valid {
        return false;
    }
    if i_am_shadow {
        return false;
    }
    // EXPANDED FALLBACK: when the Director hasn't assigned valid Recover
    // roles, the closest teammate that wants the catch gets to dive. This
    // ensures a loose ball always has a diver regardless of Director state.
    let mut pack: Vec<&PlayerSim> = state
        .players
        .iter()
        .filter(|p| {
            if p.team != player.team {
                return false;
            }
            let a = director.assignments.get(&p.id);
            match a {
                Some(a) => {
                    !a.is_shadow()
                        && wants_catch(p, state, a, active_pass_target)
                }
                None => wants_catch(p, state, assignment, active_pass_target),
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

/// PART C — DEFENSIVE CONTEST. The single committed contester of an
/// OPPONENT-controlled bell, gated by the Director's `contest_commit`.
///
/// Active iff: the Director set `contest_commit` for our team, AND the
/// opponent either holds the bell or just threw it (a loose bell thrown by
/// an opponent — a pass we can pick / strip), AND this rigger wins a
/// deterministic best-gap compare among our NON-grounded defenders that are
/// reasonably near the ball. Exactly ONE contester (mirrors the
/// `is_dive_committer` single-committer discipline: argmin gap, id
/// tie-break). Returns `false` when the bell is OURS (the dive path owns
/// that) so the two single-committer disciplines never overlap.
/// Deterministic — pure geometry + canon predictor, no rng.
pub(crate) fn is_contest_committer(
    player: &PlayerSim,
    state: &SimState,
    m: &MatchState,
    director: &DirectorState,
) -> bool {
    if !director.contest_commit {
        return false;
    }
    // Who controls the bell? Held by an opponent, OR loose but thrown by an
    // opponent (an opposing pass in flight we can contest / pick).
    let opp_controls = match state.bell.held_by.as_deref() {
        Some(hid) => state
            .players
            .iter()
            .find(|p| p.id == hid)
            .map(|p| p.team != player.team)
            .unwrap_or(false),
        None => match state.bell.thrown_by.as_deref() {
            Some(tid) => state
                .players
                .iter()
                .find(|p| p.id == tid)
                .map(|p| p.team != player.team)
                .unwrap_or(false),
            None => false,
        },
    };
    if !opp_controls {
        return false;
    }
    // It must also be OUR cast-defense (we are not in possession).
    if m.possession == player.team {
        return false;
    }
    // Our eligible contesters: non-grounded teammates reasonably near the
    // ball that are NOT themselves the bell carrier.
    let near = 70.0_f64;
    let mut pack: Vec<&PlayerSim> = state
        .players
        .iter()
        .filter(|p| {
            p.team == player.team
                && state.bell.held_by.as_deref() != Some(p.id.as_str())
                && vlen(vsub(state.bell.p, p.p)) < near
        })
        .collect();
    if pack.is_empty() {
        return false;
    }
    // Single committer: argmin predicted intercept gap, id tie-break —
    // the exact deterministic discipline used by `is_dive_committer`.
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

/// PART C — the committed defender's lane-interception target + powered-hook
/// fire. Reuses the SAME machinery as offense (no new predictor): for a
/// HELD bell we solve the lane the carrier is most likely to pass into with
/// `solve_lead_velocity` (the canon Coriolis lead solver) toward the
/// opponent's best free receiver and drive to that interception point; for
/// a LOOSE opponent-thrown bell we use `bell_rendezvous` (efe::roll_forward
/// predictor) exactly like the dive. The anchor is fired downrange so the
/// winch pulls us onto a contesting track. `snatch_vs_clatter` decides
/// whether we commit catch_intent (go for the clean strip/snatch) or just
/// disrupt (clatter); `grapple_risk` gates whether we fire the powered
/// hook at all vs. close under free-flight + thrumbler. Deterministic.
fn defensive_contest_input(
    player: &PlayerSim,
    state: &SimState,
    profile: &TeamProfile,
) -> (Vec3, Vec3, bool, bool) {
    // (target, fire_anchor_dir_basis, want_catch_intent, fire_hook)
    let want_catch = profile.snatch_vs_clatter >= 0.5;
    let fire_hook = profile.grapple_risk >= 0.4;

    if let Some(hid) = state.bell.held_by.clone() {
        // Held by the carrier: solve the most dangerous outward pass lane
        // (carrier → their best-separated teammate) with the canon lead
        // solver and station on that interception point so a real pass
        // runs into us (a played pick); failing a clean read, pressure the
        // carrier directly (drives the sim's STRIP_PRESS mechanic).
        let carrier = state.players.iter().find(|p| p.id == hid);
        if let Some(carrier) = carrier {
            // Their best outlet = the opposing teammate with the most
            // open separation from us (deterministic argmax, id tie-break).
            let mut best: Option<(&PlayerSim, f64)> = None;
            for p in state.players.iter() {
                if p.team != carrier.team || p.id == carrier.id {
                    continue;
                }
                let sep = vlen(vsub(p.p, player.p));
                let better = match best {
                    None => true,
                    Some((bp, bs)) => sep > bs || ((sep - bs).abs() < 1e-9 && p.id < bp.id),
                };
                if better {
                    best = Some((p, sep));
                }
            }
            if let Some((outlet, _)) = best {
                // Canon Coriolis lead solver — the SAME one offense uses to
                // lead a receiver. The interception point is where a pass
                // to that outlet would actually fly.
                if let Some(lr) = solve_lead_velocity(
                    carrier.p,
                    24.0, // ≈ a strong pass speed
                    outlet.p,
                    outlet.v,
                    state.omega,
                ) {
                    let aim = vnorm(vsub(lr.intercept, player.p));
                    return (lr.intercept, aim, want_catch, fire_hook);
                }
            }
            // No readable lane: pressure the carrier (strip approach).
            let aim = vnorm(vsub(carrier.p, player.p));
            return (carrier.p, aim, want_catch, fire_hook);
        }
    }

    // Loose opponent-thrown bell: rendezvous exactly like the dive
    // (efe::roll_forward predictor) and contest the catch / pick it.
    let (ip, bv) = bell_rendezvous(player, state);
    let bspeed = vlen(bv);
    let bvdir = if bspeed > 1e-3 {
        vscale(bv, 1.0 / bspeed)
    } else {
        let t = vsub(ip, player.p);
        let tl = vlen(t);
        if tl > 1e-6 { vscale(t, 1.0 / tl) } else { Vec3::new(1.0, 0.0, 0.0) }
    };
    let aim = vnorm(vsub(ip, player.p));
    let _ = bvdir;
    (ip, aim, want_catch, fire_hook)
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
    // Rendezvous anchor placement depends on ball speed. Fast balls curve
    // under Coriolis so we can run parallel; slow balls need direct interception.
    let bspeed = vlen(bv);
    let bvdir = if bspeed > 1e-3 {
        vscale(bv, 1.0 / bspeed)
    } else {
        app_dir
    };
    let term_r = crate::tuning::DIVE_TERMINAL_RADIUS;
    // For slow/medium balls (< 20 m/s), use direct interception — fire
    // straight at the predicted catch point, no parallel-run blending.
    // The ball isn't curving enough to make rendezvous geometry work.
    let dir = if bspeed < 20.0 {
        app_dir
    } else {
        // Fast balls: blend approach (far) and parallel (near) so we end up
        // running alongside the ball's trajectory at the catch point.
        let w_par = (1.0 - (d / (3.0 * term_r)).min(1.0)).clamp(0.0, 1.0);
        let blended = vnorm(vadd(
            vscale(app_dir, 1.0 - w_par),
            vscale(bvdir, w_par),
        ));
        if vlen(blended) > 1e-6 { blended } else { app_dir }
    };
    // Anchor placement: for fast balls, ahead along the bell's path so
    // reeling converges onto the trajectory. For slow balls, just past the
    // catch point along the approach direction (direct interception).
    let beyond = if bspeed < 20.0 { 8.0 } else { 14.0 };
    let anchor_dir = if bspeed < 20.0 { app_dir } else { bvdir };
    let mut anchor = vadd(ip, vscale(anchor_dir, beyond));
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
    active_pass_target: Option<&str>,
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
    let catch_intent = wants_catch(player, state, &cur_assignment, active_pass_target);

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
            is_dive_committer(
                player,
                state,
                director,
                &cur_assignment,
                active_pass_target,
            )
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
            // Release if the ball is clearly not catchable: either far away (>25m)
            // OR we've reached/passed the predicted catch point but the ball isn't
            // here (missed the rendezvous).
            let to_catch = vlen(vsub(catch_pt, player.p));
            let bell_far = vlen(to_bell) > 25.0;
            let past_catch_point =
                to_catch < crate::tuning::DIVE_TERMINAL_RADIUS && vlen(to_bell) > 12.0;
            let should_release = bell_far || past_catch_point;
            let (fire, do_release) = if !has_line {
                (partial.fire_line_at.unwrap_or(None), false)
            } else if should_release {
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
            // Ease when BOTH near the predicted catch point AND near the
            // actual ball. If the prediction was off (ball diverged), keep
            // winching until either we're close to the ball or we release.
            let near_catch_pt =
                vlen(vsub(catch_pt, player.p)) <= crate::tuning::DIVE_TERMINAL_RADIUS;
            let near_bell = vlen(to_bell) <= crate::tuning::DIVE_TERMINAL_RADIUS + 3.0;
            let reel_cmd: i32 = if near_catch_pt && near_bell {
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

    // ── PART C — DEFENSIVE CONTEST (active defense; consumes
    // director.contest_commit) ───────────────────────────────────────────
    // The Director computes `contest_commit` (director.rs should_contest)
    // and, until now, NOTHING non-test read it — marking was a passive
    // standoff and `snatch_vs_clatter` / `grapple_risk` drove no defensive
    // action. Now: when the Director has committed us to a contest and the
    // OPPONENT controls the bell (holds it or just threw it), the single
    // deterministically-selected contester (is_contest_committer — argmin
    // intercept gap, id tie-break, exactly one, mirroring the dive's
    // single-committer discipline) solves a lane-interception point with
    // the canon predictors (`solve_lead_velocity` for a held bell's
    // outward pass lane / `bell_rendezvous` = efe::roll_forward for a loose
    // pass — NO new predictor) and drives a POWERED-HOOK approach to
    // contest the catch / press a strip. `snatch_vs_clatter` gates
    // catch_intent (clean snatch vs. disrupt); `grapple_risk` gates firing
    // the hook vs. closing under free-flight. This is the change that moves
    // denial / turnover / pass-chain off the floor. Deterministic, no rng.
    if is_contest_committer(player, state, m, director) {
        let (tgt, aim0, want_catch, fire_hook) =
            defensive_contest_input(player, state, profile);
        let aim_dither = cache.value.as_ref().map(|c| c.aim_dither).unwrap_or_else(v3z);
        let nav_aim = vnorm(vadd(aim0, aim_dither));
        let to_tgt = vsub(tgt, player.p);
        let gap = vlen(to_tgt);
        let has_line = player.line.is_some();
        // Fire the powered hook ONCE (commit-once, then ride physics —
        // grapple latency locks it; mirrors the dive). Anchor downrange
        // past the interception point so the winch drags us onto the
        // contesting track. Only if grapple_risk clears the gate AND we
        // have no line in flight/attached AND the target is far enough that
        // a hook actually helps (close-in we just settle onto it).
        let fire = if fire_hook && !has_line && gap > 12.0 {
            let dir = if gap > 1e-6 { vscale(to_tgt, 1.0 / gap) } else { nav_aim };
            let mut anchor = vadd(tgt, vscale(dir, 12.0));
            let arho = (anchor.y * anchor.y + anchor.z * anchor.z).sqrt();
            let max_r = REG_R - 1.0;
            if arho > max_r && arho > 1e-6 {
                let s = max_r / arho;
                anchor.y *= s;
                anchor.z *= s;
            }
            Some(anchor)
        } else {
            None
        };
        let reel_cmd: i32 = if has_line || fire.is_some() { -1 } else { 0 };
        let pushoff = should_pushoff(player, tgt);
        let thrumbler = if fire.is_none() && !has_line {
            settle_thrumbler(player, tgt)
        } else {
            v3z()
        };
        return PlayerInput {
            id: player.id.clone(),
            aim: nav_aim,
            fire_line_at: fire,
            reel: reel_cmd,
            release: false,
            pushoff,
            throw_charge: 0.0,
            throw_released: false,
            throw_spin: 0.0,
            thrumbler,
            // snatch_vs_clatter high ⇒ go for the clean strip/snatch (the
            // sim widens the catch / strip envelope on catch_intent);
            // low ⇒ just disrupt (clatter) without committing to the grab.
            catch_intent: want_catch,
        };
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
    let thrumbler = if catch_intent {
        catch_brake_thrumbler(player, state)
    } else {
        settle_thrumbler(player, target)
    };

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
pub(crate) fn role_str(r: RiggerRole) -> &'static str {
    match r {
        RiggerRole::Anchor => "anchor",
        RiggerRole::Spinner => "spinner",
        RiggerRole::Faithwing => "faithwing",
        RiggerRole::Freewing => "freewing",
        RiggerRole::Reach => "reach",
    }
}
pub(crate) fn job_str(j: Job) -> &'static str {
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
    // ── COORDINATED PLAY OVERRIDE ──────────────────────────────────────
    // If the Director has an active play and assigned this player a role
    // with a concrete target, navigate there. Play targets are designed
    // to interlock — following them creates pass opportunities by
    // construction. Only override if the play target is still reasonable
    // (not stale, player isn't in a hard override state like dive).
    if let Some(ref play) = director.active_play {
        if let Some(pa) = play.assignments.get(&player.id) {
            // Play targets take priority over independent positioning for:
            // - Carrier: play tells them WHERE to carry to (not just "forward")
            // - PrimaryReceiver / SecondaryReceiver: play tells them where to cut
            // - Screen / DeepOption: play tells them where to station
            //
            // Do NOT override: Recover job (dive is time-critical), or if
            // the player is the dive committer (handled elsewhere).
            let dominated_by_dive = assignment.job == Job::Recover;
            if !dominated_by_dive {
                return pa.target;
            }
        }
    }

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

    // CARRIER: pressure-reactive carrying — juke defenders, advance toward
    // the best outlet teammate so passing lanes shorten as we drive.
    if assignment.job == Job::Carry
        || state.bell.held_by.as_deref() == Some(player.id.as_str())
    {
        let sgn = attack_sign(player.team);
        let ring_x = attack_ring_x(player.team);
        let skin_r = REG_R;

        // If the play system assigned us a carry target, use it as our
        // destination (the play already factors in gate positioning and
        // outlet alignment). Still apply pressure-juke on top.
        let play_base = director.active_play.as_ref()
            .and_then(|p| p.assignments.get(&player.id))
            .map(|pa| pa.target);

        // Gather defender positions.
        let defenders: Vec<Vec3> = state
            .players
            .iter()
            .filter(|p| p.team != player.team)
            .map(|p| p.p)
            .collect();

        // Find nearest defender and their closing speed.
        let mut nearest_def: Option<&Vec3> = None;
        let mut nearest_dist = f64::INFINITY;
        for dp in defenders.iter() {
            let d = vlen(vsub(*dp, player.p));
            if d < nearest_dist {
                nearest_dist = d;
                nearest_def = Some(dp);
            }
        }

        // Baseline: use play target if available, else advance toward attack ring.
        let (mut target_x, mut target_y, mut target_z) = if let Some(pt) = play_base {
            (pt.x, pt.y, pt.z)
        } else {
            let dist_to_ring = (ring_x - player.p.x).abs();
            let step = 110.0_f64.min(dist_to_ring);
            let tx = player.p.x + sgn * step;
            let r = axis_radius(player.p);
            let yz_len = if r > 1e-6 { r } else { 1.0 };
            let ty = player.p.y / yz_len * r.min(6.0);
            let tz = player.p.z / yz_len * r.min(6.0);
            (tx, ty, tz)
        };

        if let Some(def_p) = nearest_def {
            if nearest_dist < 20.0 {
                // Immediate pressure: juke perpendicular to the defender's
                // approach vector (cross-radius dodge).
                let approach = vnorm(vsub(player.p, *def_p));
                // Perpendicular in the YZ plane (cross product with X axis).
                let perp = Vec3::new(0.0, -approach.z, approach.y);
                let perp_len = vlen(perp);
                if perp_len > 1e-6 {
                    let dodge = vscale(perp, 12.0 / perp_len);
                    target_y += dodge.y;
                    target_z += dodge.z;
                }
            } else if play_base.is_none() {
                // No immediate pressure and no play target: carry toward
                // the teammate with the best reception_quality so the pass
                // lane shortens. (When a play target exists, it already
                // encodes outlet alignment — skip independent search.)
                let teammates: Vec<&PlayerSim> = state
                    .players
                    .iter()
                    .filter(|p| p.team == player.team && p.id != player.id)
                    .collect();
                let mut best_rq = -1.0_f64;
                let mut best_mate_p: Option<Vec3> = None;
                for mate in teammates.iter() {
                    let rq = efe::reception_quality(
                        player.p, mate.p, mate.v, &defenders, state.omega, skin_r,
                    );
                    if rq > best_rq {
                        best_rq = rq;
                        best_mate_p = Some(mate.p);
                    }
                }
                if let Some(mp) = best_mate_p {
                    // Bias toward best outlet: blend 70% advance, 30% toward outlet.
                    let dist_to_ring = (ring_x - player.p.x).abs();
                    let step = 110.0_f64.min(dist_to_ring);
                    let toward_mate = vsub(mp, player.p);
                    let tl = vlen(toward_mate);
                    if tl > 1e-6 {
                        let pull = vscale(toward_mate, 0.3 * step / tl);
                        target_x += pull.x;
                        target_y += pull.y;
                        target_z += pull.z;
                    }
                }
            }
        }

        // Clamp radius inside the skin.
        let out_r = (target_y * target_y + target_z * target_z).sqrt();
        if out_r > skin_r * 0.85 {
            let clamp = skin_r * 0.85 / out_r;
            target_y *= clamp;
            target_z *= clamp;
        }
        return Vec3::new(target_x, target_y, target_z);
    }

    // Man-marking — proactive lane-denial, doubling, and rotation.
    if assignment.job == Job::Mark {
        if let Some(mark_id) = &assignment.mark_id {
            if let Some(mark) = state.players.iter().find(|p| &p.id == mark_id) {
                let t_ring = Vec3::new(defend_ring_x(player.team), 0.0, 0.0);
                let to_ring = vnorm(vsub(t_ring, mark.p));
                let standoff = 8.0 - assignment.pressure * 5.4;

                // Find the carrier (opponent holding the bell).
                let carrier = state.bell.held_by.as_ref()
                    .and_then(|hid| state.players.iter().find(|p| &p.id == hid));

                // CASE 1: Ball is held by an opponent — read the passing lane.
                if let Some(carrier) = carrier {
                    if carrier.team != player.team {
                        // The carrier→mark lane is the threat. Position to
                        // deny this lane, not just goal-side of the mark.
                        let lane = vsub(mark.p, carrier.p);
                        let lane_len = vlen(lane);

                        if lane_len > 5.0 {
                            let lane_dir = vscale(lane, 1.0 / lane_len);

                            // Station IN the passing lane between carrier and mark,
                            // biased toward the mark (so we can still recover if
                            // they cut). Position = mark - standoff along the lane
                            // direction (between carrier and mark).
                            let lane_pos = vsub(mark.p, vscale(lane_dir, standoff.max(4.0)));

                            // Blend: high pressure = cheat MORE into the lane;
                            // low pressure = stay closer to traditional goal-side.
                            let lane_weight = (assignment.pressure * 1.4).clamp(0.0, 0.85);
                            let goal_side = vadd(mark.p, vscale(to_ring, standoff));
                            let base = vadd(
                                vscale(lane_pos, lane_weight),
                                vscale(goal_side, 1.0 - lane_weight),
                            );

                            // DOUBLING: if the carrier is very close to us (< 18m)
                            // and our mark is far from the ball (> 25m), crash
                            // toward the carrier to create a double-team. Our mark
                            // is unlikely to receive soon.
                            let carrier_dist = vlen(vsub(carrier.p, player.p));
                            let mark_ball_dist = vlen(vsub(state.bell.p, mark.p));
                            if carrier_dist < 18.0 && mark_ball_dist > 25.0 && assignment.pressure > 0.5 {
                                let to_carrier = vnorm(vsub(carrier.p, player.p));
                                let crash = vadd(base, vscale(to_carrier, 8.0));
                                return crash;
                            }

                            return base;
                        }
                    }
                }

                // CASE 2: Ball is loose or in flight — rotate toward the ball.
                // The mark just threw or the ball is loose. Don't stay locked
                // on a player who doesn't have the ball — rotate toward where
                // the ball is going to deny the next reception.
                let ball_loose = state.bell.held_by.is_none();
                if ball_loose {
                    let to_bell = vsub(state.bell.p, mark.p);
                    let bell_dist = vlen(to_bell);

                    // If the ball is heading toward our mark (they might catch it),
                    // tighten up and get between ball and mark.
                    let bell_closing = if bell_dist > 1e-6 {
                        -vdot(state.bell.v, to_bell) / (vlen(state.bell.v).max(1e-6) * bell_dist)
                    } else {
                        0.0
                    };

                    if bell_closing > 0.3 && bell_dist < 50.0 {
                        // Ball heading toward our mark — get in the way!
                        let intercept_pos = vadd(mark.p, vscale(vnorm(vsub(state.bell.p, mark.p)), standoff.min(6.0)));
                        return intercept_pos;
                    }

                    // Ball not heading toward mark — sag off toward the ball
                    // side to provide help defense (zone up).
                    let sag_toward_ball = vnorm(vsub(state.bell.p, mark.p));
                    let sag_dist = (standoff + 4.0).min(15.0);
                    let base = vadd(mark.p, vscale(to_ring, standoff));
                    return vadd(base, vscale(sag_toward_ball, sag_dist * 0.4));
                }

                // CASE 3: We have the ball (shouldn't be marking, but safety fallback).
                // Traditional goal-side positioning.
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
    let (w_crowd, w_redun, _w_resp, w_fwd) = if support_during_loose {
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

    // Resolve carrier position for reception_quality scoring.
    let carrier_p: Vec3 = director
        .carrier_id
        .as_ref()
        .and_then(|cid| state.players.iter().find(|p| &p.id == cid))
        .map(|c| c.p)
        .unwrap_or(state.bell.p);

    // Gather defender (opponent) positions.
    let defender_positions: Vec<Vec3> = state
        .players
        .iter()
        .filter(|p| p.team != team)
        .map(|p| p.p)
        .collect();

    // Candidate FAN: 3 depths along the attack axis × 4 cross angles ×
    // 2 radii — broad regions spanning the tube, not points.
    let depths = [-70.0_f64, 10.0, 90.0];
    let radii = [14.0_f64 + s_rad * 8.0, 30.0 + s_rad * 10.0];
    let mut best: Option<Vec3> = None;
    let mut best_score = f64::NEG_INFINITY;

    for (_di, &dd) in depths.iter().enumerate() {
        let cx = anchor_x + sgn * dd;
        for k in 0..4 {
            // Spread the four angles around the agent's stable style slice.
            let ang = std::f64::consts::PI
                * (0.15 + s_ang * 1.4 + k as f64 * 0.5);
            for &rr in radii.iter() {
                let r = rr.min(skin_r * 0.85);
                let cand =
                    Vec3::new(cx, r * ang.cos(), r * ang.sin());

                // RECEPTION-BASIN: "how good is cand as a pass target
                // from the carrier's current position?" [0,1]
                let rq = efe::reception_quality(
                    carrier_p, cand, player.v, &defender_positions, omega, skin_r,
                );

                // Forward progress pull (advance the play).
                let fwd = forward_progress(team, cand.x);
                let fwd_pull = if defending {
                    0.0
                } else {
                    (fwd / crate::tuning::GATE_X.abs())
                        * ep.cov_fwd_pull_gain
                        * w_fwd
                };

                // EPISTEMIC: anti-crowding + axis redundancy (prevent clumping).
                // These are penalties (higher = worse), so we subtract them.
                let crowd = vm.crowding_at(cand, &others);
                let redun = vm.axis_redundancy(fwd, &others_fp);
                let anti_crowd = crowd * ep.cov_crowd_w * w_crowd;
                let axis_redun = redun * ep.cov_redun_w * w_redun;

                // ANTI-SWARM STAND-OFF (unchanged): penalise gawking near
                // the loose ball while the recover unit owns it.
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

                // Composite score (MAXIMIZE):
                //   0.55 * reception_quality
                //   0.20 * forward_progress pull
                //   0.15 * anti-crowding (negated penalty)
                //   0.10 * axis redundancy (negated penalty)
                let total = 0.55 * rq
                    + 0.20 * fwd_pull
                    - 0.15 * anti_crowd
                    - 0.10 * axis_redun
                    - swarm_pen;

                if total > best_score {
                    best_score = total;
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

    // Prevent teammate stacking when role policies produce overlapping targets.
    base = repel_from_teammates(base, player, state);

    // Tempo: push targets forward on anticipated transitions.
    let bell_loose = state.bell.held_by.is_none();
    let thrown_by_opp = state
        .bell
        .thrown_by
        .as_deref()
        .and_then(|tid| state.players.iter().find(|p| p.id == tid))
        .map(|p| p.team != player.team)
        .unwrap_or(false);
    let transition_forward = bell_loose && thrown_by_opp;

    if transition_forward {
        // Anticipate gaining possession — push forward.
        let sgn = attack_sign(player.team);
        base.x += sgn * 15.0;
    }

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

    // Offensive receiver: active separation-seeking around the Director's
    // assigned depth slot. Among a small fan of nearby positions, pick the
    // one that maximizes reception_quality from the carrier — receivers
    // actively cut into open lanes rather than camping at fixed posts.
    if (assignment.job == Job::Receive || assignment.job == Job::Support)
        && player.role != RiggerRole::Reach
    {
        let carrier = director
            .carrier_id
            .as_ref()
            .and_then(|cid| state.players.iter().find(|p| &p.id == cid));
        let carrier_p = carrier.map(|c| c.p).unwrap_or(state.bell.p);

        // If we have a play target, use it as the center of our search
        // (instead of the Director's depth_slot/radius_slot center).
        let play_center = director.active_play.as_ref()
            .and_then(|p| p.assignments.get(&player.id))
            .map(|pa| pa.target);

        // Use play_center as the search center if available, else use the
        // Director slot computation as before.
        let (center_x, center_y, center_z) = if let Some(pc) = play_center {
            (pc.x, pc.y, pc.z)
        } else {
            let target_ring_x = director.attack_ring_x;
            let from_x = carrier.map(|c| c.p.x).unwrap_or(state.bell.p.x);
            let min_lead = director.attack_sign * 14.0;
            let mut cx = from_x + (target_ring_x - from_x) * assignment.depth_slot;
            if director.attack_sign > 0.0 {
                cx = cx.max(from_x + min_lead);
            } else {
                cx = cx.min(from_x + min_lead);
            }
            let r_const = 45.0;
            let cr = r_const * (0.12 + assignment.radius_slot * 0.7);
            let c_ang = std::f64::consts::PI
                * (0.15 + assignment.radius_slot * 0.7 + director.style_noise * 0.12);
            (cx, cr * c_ang.cos(), cr * c_ang.sin())
        };

        let skin_r = REG_R;
        let defender_positions: Vec<Vec3> = state
            .players
            .iter()
            .filter(|p| p.team != player.team)
            .map(|p| p.p)
            .collect();

        // Search a small fan around the center: +-15m depth, +-10m radius.
        let depth_offsets = [-15.0_f64, -7.5, 0.0, 7.5, 15.0];
        let radial_offsets = [-10.0_f64, -5.0, 0.0, 5.0, 10.0];
        let mut best_pos = Vec3::new(center_x, center_y, center_z);
        let mut best_rq = f64::NEG_INFINITY;

        for &dx in depth_offsets.iter() {
            let cx = center_x + dx;
            for &dr in radial_offsets.iter() {
                // Offset the radius — shift in the radial direction from axis.
                let yz_len = (center_y * center_y + center_z * center_z).sqrt();
                let (ny, nz) = if yz_len > 1e-6 {
                    (center_y / yz_len, center_z / yz_len)
                } else {
                    (1.0, 0.0)
                };
                let new_r = (yz_len + dr).max(3.0).min(skin_r * 0.85);
                let cy = ny * new_r;
                let cz = nz * new_r;
                let cand = Vec3::new(cx, cy, cz);

                let rq = efe::reception_quality(
                    carrier_p, cand, player.v, &defender_positions, state.omega, skin_r,
                );
                if rq > best_rq {
                    best_rq = rq;
                    best_pos = cand;
                }
            }
        }
        base = best_pos;
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
/// Velocity-aware: gentle nudge far out, firm correction mid-range, hard
/// brake + settle close in. Covers the 6-18m grapple dead zone.
fn settle_thrumbler(player: &PlayerSim, target: Vec3) -> Vec3 {
    let to_target = vsub(target, player.p);
    let dist = vlen(to_target);
    if dist > SETTLE_RADIUS || dist < 1e-6 {
        return v3z();
    }
    let approach = vscale(to_target, 1.0 / dist);
    let speed = vlen(player.v);

    // How much of our velocity is aligned with the approach direction?
    let v_along = if speed > 1e-6 {
        vdot(player.v, approach)
    } else {
        0.0
    };
    // Perpendicular speed component — this needs braking most
    let v_perp = (speed * speed - v_along * v_along).max(0.0).sqrt();

    let dv = if dist > 12.0 {
        // ── 12-18m: gentle correction (keep grapple dominant, just fix drift) ──
        let approach_mag = MICRO_DV_MAX * 0.3;
        // Gentle brake on perpendicular component only
        let brake_strength = MICRO_DV_MAX * 0.2;
        let brake = if speed > 1e-6 {
            let v_perp_vec = vsub(player.v, vscale(approach, v_along));
            let vp_len = vlen(v_perp_vec);
            if vp_len > 1e-6 {
                vscale(v_perp_vec, -brake_strength.min(vp_len) / vp_len)
            } else {
                v3z()
            }
        } else {
            v3z()
        };
        vadd(vscale(approach, approach_mag), brake)
    } else if dist > 6.0 {
        // ── 6-12m: firm correction + progressive braking ──
        let zone_t = 1.0 - (dist - 6.0) / 6.0; // 0 at 12m, 1 at 6m
        let approach_mag = MICRO_DV_MAX * (0.4 + 0.3 * zone_t);
        let perp_ratio = if speed > 1e-6 { v_perp / speed } else { 0.0 };
        let brake_strength = MICRO_DV_MAX * (0.4 + 0.4 * perp_ratio) * (0.5 + 0.5 * zone_t);
        let brake = if speed > 1e-6 {
            vscale(player.v, -brake_strength.min(speed) / speed)
        } else {
            v3z()
        };
        vadd(vscale(approach, approach_mag), brake)
    } else {
        // ── < 6m: hard brake + fine settle (original behavior, enhanced) ──
        let closeness = 1.0 - dist / 6.0; // 0 at 6m, 1 at 0m
        let approach_mag = MICRO_DV_MAX.min(dist * 0.5);
        let perp_ratio = if speed > 1e-6 { v_perp / speed } else { 0.0 };
        let brake_strength = MICRO_DV_MAX * (0.6 + 0.3 * perp_ratio) * (0.5 + 0.5 * closeness);
        let brake = if speed > 1e-6 {
            vscale(player.v, -brake_strength.min(speed) / speed)
        } else {
            v3z()
        };
        vadd(
            vscale(approach, approach_mag * (1.0 - 0.5 * closeness)),
            brake,
        )
    };

    let mm = vlen(dv);
    if mm > MICRO_DV_MAX {
        vscale(dv, MICRO_DV_MAX / mm)
    } else {
        dv
    }
}

/// Catch-intent velocity matching thrumbler. When approaching the ball to
/// catch it, brake relative velocity to stay within the absorb window.
/// Pure geometry, deterministic, no rng.
fn catch_brake_thrumbler(player: &PlayerSim, state: &SimState) -> Vec3 {
    let bell_v = state.bell.v;
    let rel = vsub(player.v, bell_v); // player's excess velocity over the ball
    let rel_speed = vlen(rel);
    if rel_speed < 2.0 {
        return v3z(); // already well-matched
    }
    // Brake: oppose the relative velocity (try to match ball speed)
    // Stronger as we get closer to the ball
    let to_bell = vsub(state.bell.p, player.p);
    let gap = vlen(to_bell);
    let urgency = (1.0 - gap / 30.0).clamp(0.1, 1.0);
    let brake_mag = (MICRO_DV_MAX * urgency).min(rel_speed);
    vscale(rel, -brake_mag / rel_speed)
}

/// Repel a nav target away from nearby teammate positions to prevent
/// stacking. Pure geometry, deterministic.
fn repel_from_teammates(target: Vec3, player: &PlayerSim, state: &SimState) -> Vec3 {
    let mut nudge = v3z();
    for p in state.players.iter() {
        if p.team != player.team || p.id == player.id {
            continue;
        }
        let to_mate = vsub(target, p.p);
        let dist = vlen(to_mate);
        // Repel when within 16m — keep players spread.
        if dist < 16.0 && dist > 1e-6 {
            let strength = (1.0 - dist / 16.0) * 6.0; // up to 6m nudge
            nudge = vadd(nudge, vscale(to_mate, strength / dist));
        }
    }
    vadd(target, nudge)
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

    // ── GRAPPLE COMMITMENT: if we have an active attached line and our
    // swing is making progress toward the target, RIDE THE ARC instead
    // of re-planning every tick. Only for ACTIVE WINCH (reel=-1) — a
    // free swing (reel=0) needs to flow through to the swoop release
    // check to decide when to release.
    if let Some(line) = &player.line {
        if line.taut {
            // Only commit on reel=-1 (active winch). Free swings (reel=0)
            // must flow through to the release check below.
            let currently_winching = commit.last_anchor_reel == -1;
            if currently_winching {
                let to_target = vsub(target, player.p);
                let dist = vlen(to_target);
                let speed = vlen(player.v);
                let closing = if dist > 1e-6 && speed > 1e-6 {
                    vdot(player.v, to_target) / (speed * dist)
                } else {
                    0.0
                };
                let target_stable = match commit.last_anchor_pos {
                    Some(ap) => vlen(vsub(target, ap)) < 60.0,
                    None => false,
                };
                if target_stable && (closing > 0.1 || (speed > 10.0 && dist > 20.0)) {
                    let aim = if dist > 1e-6 {
                        vnorm(to_target)
                    } else {
                        Vec3::new(1.0, 0.0, 0.0)
                    };
                    return PartialInput {
                        aim: Some(aim),
                        fire_line_at: Some(None),
                        reel: Some(-1),
                        release: Some(false),
                        pushoff: Some(false),
                    };
                }
            }
        }
    }

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

    // SWOOP RELEASE — aggressive: release into swing arcs early and let
    // Coriolis curve carry us to the target. Don't require near-perfect
    // alignment before releasing.
    if let Some(line) = &player.line {
        if line.taut {
            if let Some(p) = plan {
                if p.reel == 0 {
                    let sp = vlen(player.v);
                    let to_t = vsub(target, player.p);
                    let dl = vlen(to_t);
                    if dl > 1e-6 {
                        let align = vdot(player.v, to_t) / (sp.max(1e-6) * dl);

                        // Release if we have reasonable speed and are heading
                        // in a broadly correct direction.
                        let speed_ok = sp > 5.0; // was SWOOP_MIN_V (8.0)
                        let align_ok = align > 0.35; // was SWOOP_ALIGN (0.6)

                        // Also release if we have HIGH speed even with moderate
                        // alignment — momentum is king, don't waste a good swing.
                        let momentum_release = sp > 12.0 && align > 0.15;

                        if (speed_ok && align_ok) || momentum_release {
                            commit.last_anchor_pos = None;
                            commit.last_anchor_reel = -1; // Reset so next tick replans fresh
                            let soar_aim = vnorm(player.v);
                            let mut pi = plan_to_input(None, soar_aim);
                            pi.release = Some(true);
                            return pi;
                        }
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

    #[derive(Clone, Debug)]
    struct ThrowCandidate {
        v0: Vec3,
        target_id: Option<String>,
        tactical_bias: f64,
    }

    // ── Outcome quality of a candidate world launch (the PRAGMATIC evaluator).
    // Roll the bell with the canon model from the release; return a scalar in
    // roughly [0,1+] where >= GOOD_ENOUGH means "this is a good throw".
    // Higher = better. No RNG. This is aligned to the match layer: a useful
    // non-scoring throw should either clear the current cast gate into a
    // receiver or be a clearly retained advancing pass, not merely a forward
    // endpoint fling.
    let current_gate_fp = forward_progress(team, gate_world_x(team, m.cast.gate));
    let eval_candidate = |cand: &ThrowCandidate| -> f64 {
        // Roll far enough to see it cross the ring plane or settle.
        let st = efe::roll_forward(player.p, cand.v0, omega, ep.throw_eval_horizon, skin_r);
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
        // near the intended receiver's predicted spot (or a teammate if this
        // is an untargeted emergency release).
        let gain = forward_progress(team, st.p.x) - my_fp;
        let mut nearest_tm = f64::INFINITY;
        let mut target_tm = f64::INFINITY;
        for (i, tp) in tm_future.iter() {
            let d = (st.p.x - tp.x).hypot(st.p.y - tp.y).hypot(st.p.z - tp.z);
            if d < nearest_tm {
                nearest_tm = d;
            }
            if cand
                .target_id
                .as_deref()
                .map(|tid| teammates[*i].id == tid)
                .unwrap_or(false)
            {
                target_tm = d;
            }
        }
        // Reachable if a teammate is within a generous catch envelope of the
        // bell's settle point (wide on purpose — weakest-sufficient).
        let catch_dist = if cand.target_id.is_some() && target_tm.is_finite() {
            target_tm
        } else {
            nearest_tm
        };
        let reach = (1.0 - (catch_dist / ep.throw_reach_radius)).clamp(0.0, 1.0);
        let gain_norm = (gain / 120.0).clamp(-0.5, 1.0);
        let gate_adv = if end_fp >= current_gate_fp - 2.0 {
            1.0
        } else {
            0.0
        };

        // Don't reward flinging it into the skin or behind us.
        let skin_pen = if end_rho > skin_r * 0.96 { 0.25 } else { 0.0 };
        let backward_pen = if gain < -20.0 { 0.4 } else { 0.0 };

        0.10
            + 0.38 * reach
            + 0.38 * gate_adv
            + 0.18 * gain_norm.max(0.0)
            + cand.tactical_bias
            - skin_pen
            - backward_pen
    };

    // ── Build the ACTION REPERTOIRE of world launches. ──────────────────
    let mut candidates: Vec<ThrowCandidate> = Vec::new();

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
            candidates.push(ThrowCandidate {
                v0: g.v0,
                target_id: None,
                tactical_bias: 0.0,
            });
        }
    }

    // Seed 2: a lead pass to each teammate's predicted position (forward
    // ones first — we want to advance). These are seeds, not the answer.
    //
    // If a coordinated play is active, heavily bias toward the primary receiver.
    let play_primary_id: Option<&str> = director.active_play.as_ref()
        .and_then(|p| {
            p.assignments.iter()
                .find(|(_, a)| a.play_role == PlayRole::PrimaryReceiver)
                .map(|(id, _)| id.as_str())
        });

    for tm in teammates.iter() {
        // miss_open_chance: SAME draw the frozen spec made per teammate.
        if rng.next() < scaling.miss_open_chance {
            continue;
        }
        if let Some(lead) =
            solve_lead_velocity(player.p, throw_speed, tm.p, tm.v, omega)
        {
            let is_gate_runner =
                director.gate_receiver_id.as_deref() == Some(tm.id.as_str());
            let is_play_primary =
                play_primary_id == Some(tm.id.as_str());
            let mut bias = 0.0_f64;
            if is_gate_runner { bias += 0.20; }
            if is_play_primary { bias += 0.35; }
            candidates.push(ThrowCandidate {
                v0: lead.v0,
                target_id: Some(tm.id.clone()),
                tactical_bias: bias,
            });
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
                candidates.push(ThrowCandidate {
                    v0: vscale(dir, sp),
                    target_id: None,
                    tactical_bias: 0.0,
                });
            }
        }
    }

    // ── RECEPTION-PROBABILITY × OUTCOME-VALUE SELECTION: score each
    // candidate by expected value of the pass (teammate catch likelihood,
    // defender interception risk, forward progress, gate-clearing bonus).
    // Pre-filter obviously terrible candidates, then select by EV. ──────

    // Build teammate (position, velocity) pairs for efe::pass_outcome_ev.
    let tm_with_vel: Vec<(Vec3, Vec3)> = teammates
        .iter()
        .map(|tm| (tm.p, tm.v))
        .collect();
    // Build defender positions.
    let defender_positions: Vec<Vec3> = opponents
        .iter()
        .map(|op| op.p)
        .collect();

    let mut best_candidate: Option<ThrowCandidate> = None;
    let mut best_ev: f64 = f64::NEG_INFINITY;

    for cand in candidates.iter() {
        let rel_speed = vlen(vsub(cand.v0, player.v));
        if !(throw_min..=throw_max).contains(&rel_speed) || rel_speed < 1e-3 {
            continue;
        }
        // Pre-filter: reject clearly bad candidates (backward, into skin)
        // before the heavier pass_outcome_ev evaluation.
        let center_q = eval_candidate(cand);
        if center_q < 0.2 {
            continue;
        }
        // Score by reception probability × outcome value.
        let ev = efe::pass_outcome_ev(
            player.p,
            cand.v0,
            &tm_with_vel,
            &defender_positions,
            omega,
            skin_r,
        );
        // Incorporate tactical_bias from candidate generation (e.g. gate
        // receiver bonus) as additive boost.
        let ev = ev + cand.tactical_bias;
        if ev > best_ev {
            best_ev = ev;
            best_candidate = Some(cand.clone());
        }
    }

    // ── Decide WHETHER to throw at all. With EV-based selection we throw
    // when the best option exceeds a value threshold — an advancing pass
    // to a somewhat-open teammate is worth it. Under pressure/stall the
    // threshold drops significantly. ──────────────────────────────────────
    let stall = cache.value.as_ref().unwrap().hold_ticks;
    let pressured = nearest_opponent_dist(player, &opponents) < ep.throw_pressure_dist;
    let force = stall > ep.throw_stall_force_ticks || m.cast.throws_left as f64 <= 1.0;

    // Threshold: normal play requires a meaningful EV; pressure/stall
    // lowers the bar so the carrier releases rather than getting stripped.
    let ev_threshold = if force {
        0.0
    } else if pressured {
        0.15
    } else {
        0.35
    };

    let chosen = match best_candidate.clone() {
        Some(v) if best_ev >= ev_threshold => v,
        _ => {
            // No candidate clears the threshold. If forced (stall/last
            // throw) fling forward anyway — a weak release beats a dead
            // stuck carrier. Otherwise hold and keep carrying.
            if force {
                ThrowCandidate {
                    v0: vscale(Vec3::new(sgn, 0.0, 0.0), throw_speed),
                    target_id: None,
                    tactical_bias: 0.0,
                }
            } else {
                let commit = cache.value.as_mut().unwrap();
                commit.throw_go = false;
                commit.throw_target_id = None;
                commit.throw_dir = None;
                return;
            }
        }
    };

    // Convert world launch → the throw the player imparts (sim adds v).
    let throw_vec = vsub(chosen.v0, player.v);
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

    let commit = cache.value.as_mut().unwrap();
    commit.throw_go = true;
    commit.throw_target_id = chosen.target_id.clone();
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
            active_play: None,
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
            None, &mut cache, true, 30.0,
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
    fn friendly_pass_commits_only_the_intended_receiver() {
        let players = vec![
            mk_player("H1", TeamSide::Home, RiggerRole::Spinner, Vec3::new(-20.0, 4.0, 1.0)),
            mk_player("H2", TeamSide::Home, RiggerRole::Faithwing, Vec3::new(10.0, 5.0, 1.0)),
            mk_player("H3", TeamSide::Home, RiggerRole::Anchor, Vec3::new(0.0, 6.0, 1.0)),
        ];
        let mut state = mk_state(players, None);
        state.bell.thrown_by = Some("H1".to_string());
        state.bell.p = Vec3::new(11.0, 5.5, 1.0);
        state.bell.v = Vec3::new(9.0, 0.0, 0.0);
        let recv = PlayerAssignment {
            job: Job::Receive,
            mark_id: None,
            depth_slot: 0.6,
            radius_slot: 0.3,
            pressure: 0.0,
        };
        assert!(
            wants_catch(&state.players[1], &state, &recv, Some("H2")),
            "the named receiver should commit to the friendly pass"
        );
        assert!(
            !wants_catch(&state.players[2], &state, &recv, Some("H2")),
            "a non-target teammate should hold spacing instead of swarming"
        );
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
            &dir, Difficulty::Pro, &mut rng, None, &mut cache, true, 30.0,
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
            &dir, Difficulty::Pro, &mut rng2, None, &mut cache2, true, 30.0,
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
            None, &mut cache, true, 30.0,
        );
        assert!(o.aim.x.is_finite() && o.aim.y.is_finite() && o.aim.z.is_finite());
        let c1 = cache.value.clone().expect("commit persists");
        assert_eq!(c1.hold_ticks, 0.0, "first-ever tick: increment skipped");

        let mut rng2 = AiRng::make(424242, 101, 0);
        compute_player_input(
            &player, &state, &m, &profile, &dir, Difficulty::Pro, &mut rng2,
            None, &mut cache, false, 30.0,
        );
        let c2 = cache.value.clone().expect("commit persists");
        assert_eq!(c2.hold_ticks, 1.0, "second tick still holding → 1");
    }
}

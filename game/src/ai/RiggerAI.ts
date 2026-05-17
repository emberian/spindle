// RiggerAI — per-player AI dispatcher.
//
// Given the current SimState + MatchState + TeamProfile + Director plan +
// difficulty + rng, produce a PlayerInput for one AI-controlled player.
//
// COMMITMENT MODEL (the core competence win):
//   Discrete decisions — which receiver to throw to, whether to throw at all,
//   which navigation target to swing toward — are *committed* into a
//   per-player cache and only re-decided when EITHER
//     (a) the Director ticked (≈2 Hz), OR
//     (b) a hard trigger fires (bell possession changed, a contest opened,
//         we just gained/lost the bell, the committed target became invalid).
//   Between decisions the player smoothly EXECUTES the plan (keeps swinging
//   toward the same anchor, keeps charging the same throw). This stops the
//   240 Hz intent-flicker that made players look spastic and incompetent.
//
//   Hysteresis: a freshly switched throw target must beat the held target by
//   a margin before we switch again, preventing dithering between two
//   near-equal receivers.
//
// Asymmetric difficulty (now actually wired):
//   rookie  — slow committed reactions, wide throw cone, poor reads,
//             sloppy lead/aim (drop-prone receiving via mistimed positioning)
//   pro     — balanced
//   legend  — tight execution, sharp reads, crisp leads

import type { Vec3 } from '../sim/vec';
import { vnorm, vsub, vadd, vscale, vlen, vdot, v3 } from '../sim/vec';
import type { PlayerSim, SimState, MatchState, PlayerInput } from '../sim/types';
import type { TeamProfile } from '../league/teams';
import type { DirectorState, PlayerAssignment } from './Director';
import { planToInput, planGrapple } from './nav/GrapplePlanner';
import { anchorPolicy } from './roles/Anchor';
import { spinnerPolicy } from './roles/Spinner';
import { faithwingPolicy } from './roles/Faithwing';
import { freewingPolicy } from './roles/Freewing';
import { reachPolicy } from './roles/Reach';
import { solveLeadVelocity } from './decide/LeadPredict';
import { scoreThrow } from './decide/ThrowScore';
import type { ThrowCandidate } from './decide/ThrowScore';
import { solveGateThrow } from './decide/GateSolve';
import { attackSign, attackRingX, defendRingX, forwardProgress } from './Orientation';

export type Difficulty = 'rookie' | 'pro' | 'legend';

export interface DifficultyScaling {
  /** Mean committed reaction latency in Director-windows before acting on a
   *  new situation. Modeled as a *commitment* lag, not per-tick coin flips. */
  reactionDelay: number;
  /** Gaussian sigma on throw velocity direction (radians) */
  throwVariance: number;
  /** Scale on read quality [0,1] — lower = picks worse options */
  readQuality: number;
  /** Probability a receiver mistimes/misjudges its catch position this
   *  decision → modeled as a positioning/lead offset, NOT a fake sim event. */
  dropChance: number;
  /** Probability of not seeing an open receiver (per re-decision) */
  missOpenChance: number;
}

const DIFFICULTY_PROFILES: Record<Difficulty, DifficultyScaling> = {
  rookie: {
    reactionDelay: 0.35,
    throwVariance: 0.18,
    readQuality: 0.45,
    dropChance: 0.08,
    missOpenChance: 0.2,
  },
  pro: {
    reactionDelay: 0.12,
    throwVariance: 0.06,
    readQuality: 0.8,
    dropChance: 0.02,
    missOpenChance: 0.06,
  },
  legend: {
    reactionDelay: 0.04,
    throwVariance: 0.02,
    readQuality: 0.97,
    dropChance: 0.005,
    missOpenChance: 0.01,
  },
};

// ── Commitment cache types (owned/instantiated by index.ts) ────────────────────

export interface PlayerCommit {
  /** Committed throw target receiver id (when we hold the bell), or null. */
  throwTargetId: string | null;
  /** Committed decision: should we be executing a throw? */
  throwGo: boolean;
  /** Committed throw direction + spin (stable while executing the throw). */
  throwDir: Vec3 | null;
  throwSpin: number;
  throwCharge: number;
  /** Cached navigation target (world point) we are committed to swinging to. */
  navTarget: Vec3 | null;
  /** Per-receiver positioning/timing offset (skill-scaled catch realism). */
  catchOffset: Vec3;
  /** Per-window seeded aim micro-dither (sub-degree); keeps seed live. */
  aimDither: Vec3;
  /** Snapshot of the situation we decided against, for hard-trigger detection. */
  sawHeldBy: string | null;
  sawContest: boolean;
  sawHadBell: boolean;
  /** Reaction gate: ticks remaining before we ACT on a freshly-seen change. */
  reactGateUntilTick: number;
}

export interface PlayerCommitCache {
  value: PlayerCommit | null;
  decidedTick: number;
}

function axisRadius(p: Vec3): number {
  return Math.sqrt(p.y * p.y + p.z * p.z);
}

/**
 * Estimate how open a receiver is, accounting for the defender that most
 * threatens the PASSING LANE (not just nearest body): a defender close to the
 * receiver AND between thrower and receiver is far more dangerous.
 * Returns [0,1]: 1 = wide open, 0 = blanketed/covered.
 */
function estimateOpenness(
  thrower: PlayerSim,
  receiver: PlayerSim,
  opponents: PlayerSim[],
): number {
  let worst = 1;
  const tr = vsub(receiver.p, thrower.p);
  const trLen = vlen(tr);
  const trDir = trLen > 1e-6 ? vscale(tr, 1 / trLen) : v3(1, 0, 0);

  for (const opp of opponents) {
    // Proximity to the receiver (catch-contest).
    const dRecv = vlen(vsub(receiver.p, opp.p));
    const proxOpen = Math.min(1, dRecv / 12);

    // Distance from the defender to the throwing lane segment (interception).
    const to = vsub(opp.p, thrower.p);
    const proj = Math.max(0, Math.min(trLen, vdot(to, trDir)));
    const laneClosest = vadd(thrower.p, vscale(trDir, proj));
    const laneDist = vlen(vsub(opp.p, laneClosest));
    // Only count lane risk for defenders actually in front of the receiver-ish.
    const laneOpen = Math.min(1, laneDist / 9);

    const oppOpen = Math.min(proxOpen, 0.4 + 0.6 * laneOpen);
    if (oppOpen < worst) worst = oppOpen;
  }
  return worst;
}

/**
 * Apply throw variance: perturb a direction vector by `sigma` radians of
 * isotropic noise. Uses rng() (injected — no Math.random).
 */
function perturbDirection(dir: Vec3, sigma: number, rng: () => number): Vec3 {
  if (sigma < 0.001) return dir;
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  const mag = Math.sqrt(-2 * Math.log(u1)) * sigma;
  const dx = mag * Math.cos(2 * Math.PI * u2);
  const dy = mag * Math.sin(2 * Math.PI * u2);
  const perturbed: Vec3 = { x: dir.x + dx, y: dir.y + dy, z: dir.z };
  return vnorm(perturbed);
}

/** Did the macro-situation change enough to force an immediate re-decision? */
function hardTrigger(commit: PlayerCommit, state: SimState, match: MatchState, player: PlayerSim): boolean {
  const heldBy = state.bell.heldBy;
  const haveBell = heldBy === player.id;
  const contestOn = match.contest !== null;
  if (commit.sawHeldBy !== heldBy) return true; // possession changed hands
  if (commit.sawHadBell !== haveBell) return true; // we gained/lost the bell
  if (commit.sawContest !== contestOn) return true; // a contest opened/closed
  return false;
}

/**
 * Build a complete PlayerInput for one AI player this tick.
 * Stateful via `commit` (the index.ts cache); deterministic via injected rng.
 */
export function computePlayerInput(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  director: DirectorState,
  difficulty: Difficulty,
  rng: () => number,
  cache: PlayerCommitCache,
  directorRefreshed: boolean,
  directorInterval: number,
): PlayerInput {
  const scaling = DIFFICULTY_PROFILES[difficulty];
  const tick = state.tick;

  let commit = cache.value;
  const firstEver = !commit;
  const triggered = commit ? hardTrigger(commit, state, match, player) : true;

  // Re-decide when: first time, Director refreshed, or a hard trigger fired.
  const reDecide = !commit || directorRefreshed || triggered;

  if (reDecide) {
    if (!commit) {
      commit = {
        throwTargetId: null,
        throwGo: false,
        throwDir: null,
        throwSpin: 0,
        throwCharge: 0,
        navTarget: null,
        catchOffset: v3(),
        aimDither: v3(),
        sawHeldBy: state.bell.heldBy,
        sawContest: match.contest !== null,
        sawHadBell: state.bell.heldBy === player.id,
        reactGateUntilTick: 0,
      };
    }

    // Reaction latency: on a *hard* change, a less-skilled player needs a
    // beat before its committed plan updates (commitment lag, not flicker).
    // The gate is ARMED ONCE per distinct change and counted down; it never
    // re-rolls per tick and never compounds (a still-pending gate from the
    // same unhandled change is left alone so the player can't freeze).
    if (!firstEver && triggered && tick >= commit.reactGateUntilTick) {
      const lagWindows = scaling.reactionDelay * 2; // up to ~2 director windows
      const lagTicks = Math.round(lagWindows * directorInterval * (0.5 + rng()));
      commit.reactGateUntilTick = tick + lagTicks;
      // CRITICAL: consume the trigger by snapshotting the situation NOW, the
      // moment we ARM the gate — NOT only when we later act. Otherwise the
      // snapshot stays stale for the whole gated window, hardTrigger() keeps
      // returning true every tick, and the gate re-arms forever the instant
      // it expires → the player is perpetually gated and NEVER decides. (This
      // is the home/away asymmetry: H1 holds the bell from frame 0 so it hits
      // the firstEver bypass and skips gate-arming; A1 only gains the bell
      // mid-match, arms the gate, and then starved its own throw/carry
      // decision forever — hence "away stalls, never throws".)
      commit.sawHeldBy = state.bell.heldBy;
      commit.sawContest = match.contest !== null;
      commit.sawHadBell = state.bell.heldBy === player.id;
    }

    const gated = tick < commit.reactGateUntilTick;

    if (!gated) {
      // Refresh the situation snapshot now that we are acting on it (also
      // covers the firstEver / Director-refresh paths that don't arm a gate).
      commit.sawHeldBy = state.bell.heldBy;
      commit.sawContest = match.contest !== null;
      commit.sawHadBell = state.bell.heldBy === player.id;

      const assignment = director.assignments[player.id] ?? {
        job: 'support' as const,
        markId: null,
        depthSlot: 0.4,
        radiusSlot: 0.45,
      };

      if (state.bell.heldBy === player.id) {
        decideThrow(player, state, match, profile, director, scaling, commit, rng);
      } else {
        commit.throwGo = false;
        commit.throwTargetId = null;
        commit.throwDir = null;
      }

      // Commit a navigation target for the window (executed smoothly below).
      commit.navTarget = decideNavTarget(
        player,
        state,
        match,
        profile,
        director,
        assignment,
        scaling,
        rng,
      );

      // Skill-scaled receiver mistiming: a stable per-window positioning
      // offset (looks like a slightly mistimed/misjudged catch run, never a
      // fake sim event — the SIM owns physics). Bigger for rookies.
      if (assignment.job === 'receive' || assignment.job === 'support') {
        if (rng() < scaling.dropChance * 6) {
          const m = 4 + scaling.throwVariance * 30;
          commit.catchOffset = {
            x: (rng() * 2 - 1) * m,
            y: (rng() * 2 - 1) * m,
            z: (rng() * 2 - 1) * m,
          };
        } else {
          // Mild jitter scaled by skill — keeps receivers from being perfect.
          const m = scaling.throwVariance * 6;
          commit.catchOffset = {
            x: (rng() * 2 - 1) * m,
            y: (rng() * 2 - 1) * m,
            z: (rng() * 2 - 1) * m,
          };
        }
      } else {
        commit.catchOffset = v3();
      }

      // Per-window seeded aim micro-dither: sub-degree, committed (stable
      // within the window so it never reintroduces jitter), but ensures the
      // seed always reaches the output even on a purely geometric frame.
      // Symmetric → competence & team-style divergence unaffected.
      const ds = 0.004 + scaling.throwVariance * 0.012;
      commit.aimDither = {
        x: (rng() * 2 - 1) * ds,
        y: (rng() * 2 - 1) * ds,
        z: (rng() * 2 - 1) * ds,
      };
    }

    cache.value = commit;
    cache.decidedTick = tick;
  }

  commit = cache.value!;

  // ── EXECUTE the committed plan (smooth, every tick) ───────────────────────

  if (state.bell.heldBy === player.id && commit.throwGo && commit.throwDir) {
    // Execute the committed throw — keep charging & release; stable direction.
    const charged = commit.throwCharge;
    return {
      id: player.id,
      aim: commit.throwDir,
      fireLineAt: null,
      reel: 0,
      release: false,
      pushoff: false,
      throwCharge: charged,
      throwReleased: true,
      throwSpin: commit.throwSpin,
      thrumbler: v3(),
    };
  }

  // Otherwise execute committed navigation toward the cached target.
  const target = commit.navTarget
    ? vadd(commit.navTarget, commit.catchOffset)
    : player.p;
  const partial = navigateTo(player, state, target);
  const navAim = vnorm(vadd(partial.aim ?? v3(1, 0, 0), commit.aimDither));

  return {
    id: player.id,
    aim: navAim,
    fireLineAt: partial.fireLineAt ?? null,
    reel: partial.reel ?? 0,
    release: partial.release ?? false,
    pushoff: partial.pushoff ?? false,
    throwCharge: 0,
    throwReleased: false,
    throwSpin: 0,
    thrumbler: partial.thrumbler ?? v3(),
  };
}

/**
 * Decide a committed navigation target. Roles still own the *shape* of where
 * they want to be; the Director assignment overrides job-level intent so the
 * team coordinates (carrier lane, receiver depth slots, marks, recovery).
 */
function decideNavTarget(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  director: DirectorState,
  assignment: PlayerAssignment,
  _scaling: DifficultyScaling,
  rng: () => number,
): Vec3 {
  // ROBUSTNESS: loose-bell recoverer always pursues the bell directly (lead
  // the bell slightly so the swing arrives where it's going).
  if (assignment.job === 'recover') {
    const b = state.bell;
    return {
      x: b.p.x + b.v.x * 0.6,
      y: b.p.y + b.v.y * 0.6,
      z: b.p.z + b.v.z * 0.6,
    };
  }

  // CARRIER: actively gain ground toward OUR attacking ring. Drive a target a
  // good chunk down-axis from the carrier (capped at the ring), pulled toward
  // the axis so we end up in a scoring lane rather than skinned at the rim.
  //
  // SYMMETRY FIX (the away-stall bug): the de-facto carrier each cast is
  // whoever the harness grips the bell to (the SPINNER H1/A1), which is NOT
  // always the Director's `carry` assignee — the Director runs at 2 Hz and
  // may lag a re-grip, and a spinner's own role policy only nudges ~40 m and
  // sits at mid-radius, so the real holder would dawdle / heave instead of
  // driving to the ring. Key the carrier-advance off ACTUAL possession
  // (state.bell.heldBy === player.id) so home AND away drive identically.
  if (assignment.job === 'carry' || state.bell.heldBy === player.id) {
    const sgn = attackSign(player.team);
    const ringX = attackRingX(player.team);
    const distToRing = Math.abs(ringX - player.p.x);
    // Advance hard down-axis toward OUR ring (never past it).
    const step = Math.min(110, distToRing);
    const aheadX = player.p.x + sgn * step;
    const r = axisRadius(player.p);
    // Aggressively ease toward the axis so a real (rho≤8) gate solution
    // exists ASAP — the closed-form gate throw needs us near-axial.
    const targetR = Math.min(r, 6);
    const yzLen = r > 1e-6 ? r : 1;
    return {
      x: aheadX,
      y: (player.p.y / yzLen) * targetR,
      z: (player.p.z / yzLen) * targetR,
    };
  }

  // Man-marking: sit between the marked opponent and OUR defended ring
  // (orientation-correct: home defends -X, away defends +X).
  if (assignment.job === 'mark' && assignment.markId) {
    const mark = state.players.find(p => p.id === assignment.markId);
    if (mark) {
      const tRing = { x: defendRingX(player.team), y: 0, z: 0 };
      const toRing = vnorm(vsub(tRing, mark.p));
      // 6 m goal-side of the mark (between mark and the ring we defend).
      return vadd(mark.p, vscale(toRing, 6));
    }
  }

  // Otherwise defer to the role policy for positional craft, but feed it the
  // Director assignment so receivers spread to distinct depth/radius slots.
  return roleTarget(player, state, match, profile, director, assignment, rng);
}

/** Role-shaped target, modulated by the Director's depth/radius slot. */
function roleTarget(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  director: DirectorState,
  assignment: PlayerAssignment,
  rng: () => number,
): Vec3 {
  // Use the role POLICY (the role's intended destination), not the grapple
  // anchor. This keeps each role's positional craft + profile-driven style
  // (preserving the style-divergence test) while the Director slot can refine
  // offensive receiver spacing below.
  let base: Vec3;
  switch (player.role) {
    case 'anchor':
      base = { ...anchorPolicy(player, state, match, profile, rng).targetPos };
      break;
    case 'spinner': {
      const isLoopSetter = director.loopSetterId === player.id;
      base = {
        ...spinnerPolicy(player, state, match, profile, isLoopSetter, rng).targetPos,
      };
      break;
    }
    case 'faithwing':
      base = { ...faithwingPolicy(player, state, match, profile, rng).targetPos };
      break;
    case 'freewing':
      base = { ...freewingPolicy(player, state, match, profile, rng).targetPos };
      break;
    case 'reach':
      base = { ...reachPolicy(player, state, match, profile, rng).targetPos };
      break;
    default:
      base = { ...player.p };
  }

  // If we're an offensive receiver, override radius & axial depth from the
  // Director slot so receivers fan out AHEAD of the carrier toward OUR
  // attacking ring (orientation-correct: uses director.attackRingX, never
  // faithEnd). depthSlot interpolates carrier → ring, so receivers always
  // lead into space the carrier is driving toward.
  if (assignment.job === 'receive' || assignment.job === 'support') {
    const carrier = director.carrierId
      ? state.players.find(p => p.id === director.carrierId)
      : null;
    const targetRingX = director.attackRingX;
    const fromX = carrier ? carrier.p.x : state.bell.p.x;
    // Always at least a little ahead of the carrier toward the ring.
    const minLead = director.attackSign * 14;
    let depthX = fromX + (targetRingX - fromX) * assignment.depthSlot;
    if (director.attackSign > 0) depthX = Math.max(depthX, fromX + minLead);
    else depthX = Math.min(depthX, fromX + minLead);
    // Place at the slotted axis-radius; keep an angular spread from styleNoise.
    const R = 45;
    const r = R * (0.12 + assignment.radiusSlot * 0.7);
    const ang =
      Math.PI *
      (0.15 +
        assignment.radiusSlot * 0.7 +
        director.styleNoise * 0.12);
    base = { x: depthX, y: r * Math.cos(ang), z: r * Math.sin(ang) };
  }
  return base;
}

/** Run the grappler toward a committed target (executed every tick). */
function navigateTo(
  player: PlayerSim,
  state: SimState,
  target: Vec3,
): Partial<PlayerInput> {
  // planToInput is the frozen bridge; planGrapple is invoked by role nav but
  // here we need a direct plan to the committed point.
  // Re-use the planner via a tiny shim role-agnostic call.
  const plan = planGrapple(player, target, state);
  const aim = plan ? vnorm(vsub(plan.anchorPos, player.p)) : v3(1, 0, 0);
  return planToInput(plan, aim);
}

/**
 * Decide (and commit) whether/where to throw when holding the bell.
 * Opponent-aware: scores receivers by lane openness and EV, leads into space,
 * and applies difficulty-scaled read noise. Hysteresis prevents target
 * dithering between near-equal options.
 */
function decideThrow(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  director: DirectorState,
  scaling: DifficultyScaling,
  commit: PlayerCommit,
  rng: () => number,
): void {
  const teammates = state.players.filter(
    p => p.team === player.team && p.id !== player.id,
  );
  const opponents = state.players.filter(p => p.team !== player.team);

  const ringX = attackRingX(player.team);
  const distToRing = Math.abs(ringX - player.p.x);

  // Posture shapes throw speed/aggression.
  const postureMul =
    director.posture === 'chase' ? 1.12 : director.posture === 'grind' ? 0.92 : 1;
  const throwSpeed = (18 + profile.aggression * 12) * postureMul;

  // ── PRIORITY 1: a REAL scoring throw through OUR ring. ────────────────────
  // The Coriolis gate solver returns a CLOSED-FORM launch that threads
  // rho ≈ 0 at x = ringX (or null if no skin-safe, dead-ball-safe, throwable
  // solution exists from here). It already enforces flight-time and
  // release-speed limits, so we attempt it from anywhere on the field rather
  // than gating on an arbitrary distance — the solver itself is the gate.
  if (distToRing > 6) {
    const gate = solveGateThrow(
      player.p,
      player.team,
      state.omega,
      throwSpeed,
      profile.loopPropensity,
      player.v, // sim ADDS thrower velocity — solver cancels it
    );
    if (gate) {
      // Aim along the REQUIRED throw vector (sim adds player.v back to it);
      // charge so the sim's release speed equals |throwVec| exactly. This
      // makes the realised free-bell velocity match the solved v0, so the
      // rho≈0 thread holds in the live sim.
      //
      // A ring shot is a PRECISION throw: the gate is only 8 m across at the
      // end of a multi-second Coriolis arc, so even a fraction of a degree of
      // aim noise sails it wide. Use a heavily-attenuated variance (kept
      // nonzero so the seed still reaches output and skill still separates)
      // instead of the full pass-grade cone.
      const dir = perturbDirection(
        vnorm(gate.throwVec),
        scaling.throwVariance * 0.04,
        rng,
      );
      // Spin: loop/curl ride needs spin; a flat Fall/Rise wants modest spin.
      const throwSpin = director.attackingFree
        ? -(0.3 + profile.loopPropensity * 0.7)
        : 0.2 + (1 - profile.freeEndBias) * 0.3;
      commit.throwGo = true;
      commit.throwTargetId = null; // direct ring shot, no receiver
      commit.throwDir = dir;
      commit.throwSpin = Math.max(-1, Math.min(1, throwSpin));
      // Sim: speed = THROW_MIN + charge·(THROW_MAX − THROW_MIN). Invert it so
      // the release speed equals the solver's required releaseSpeed exactly.
      const THROW_MIN = 9.0;
      const THROW_MAX = 34.0;
      commit.throwCharge = Math.max(
        0,
        Math.min(
          1,
          (gate.releaseSpeed - THROW_MIN) / (THROW_MAX - THROW_MIN),
        ),
      );
      return;
    }
  }

  // ── PRIORITY 2: a ground-gaining pass to an OPEN, ADVANCING receiver. ─────
  // We only pass if it meaningfully advances the bell toward OUR ring (or is
  // the mandatory safe outlet). Never a hopeless heave with no receiver.
  let bestScore = -Infinity;
  let bestTm: PlayerSim | null = null;
  let bestV0: Vec3 | null = null;
  let heldScore = -Infinity;

  for (const tm of teammates) {
    if (rng() < scaling.missOpenChance) continue;

    const lead = solveLeadVelocity(player.p, throwSpeed, tm.p, tm.v, state.omega);
    if (!lead) continue;

    // Forward gain toward OUR ring at the intercept (orientation-correct).
    const gain =
      forwardProgress(player.team, lead.intercept.x) -
      forwardProgress(player.team, player.p.x);

    const openness = estimateOpenness(player, tm, opponents);

    const candidate: ThrowCandidate = {
      receiver: tm,
      aimPos: lead.intercept,
      flightTime: lead.flightTime,
      openness,
      receiverRadius: axisRadius(tm.p),
    };

    const noise = (rng() - 0.5) * (1 - scaling.readQuality) * 0.3;
    const result = scoreThrow(player, candidate, match, profile, noise);

    // Reward forward progress strongly so the team always works down-axis;
    // a backward pass is only acceptable as a pressure-release safety valve.
    const gainNorm = Math.max(-1, Math.min(1.5, gain / 90));
    const score = result.score + gainNorm * 0.35;

    if (tm.id === commit.throwTargetId) heldScore = score;

    if (score > bestScore) {
      bestScore = score;
      bestTm = tm;
      bestV0 = lead.v0;
    }
  }

  // Hysteresis: keep the committed target unless clearly beaten.
  const SWITCH_MARGIN = 0.12;
  if (
    commit.throwTargetId &&
    heldScore > -Infinity &&
    bestTm &&
    bestTm.id !== commit.throwTargetId &&
    bestScore < heldScore + SWITCH_MARGIN
  ) {
    const held = teammates.find(p => p.id === commit.throwTargetId);
    if (held) {
      const lead = solveLeadVelocity(player.p, throwSpeed, held.p, held.v, state.omega);
      if (lead) {
        bestTm = held;
        bestV0 = lead.v0;
        bestScore = heldScore;
      }
    }
  }

  // Decide WHETHER to pass at all. Compute the best forward gain available.
  let bestGain = -Infinity;
  if (bestTm) {
    bestGain =
      forwardProgress(player.team, bestTm.p.x) -
      forwardProgress(player.team, player.p.x);
  }

  // HOLD & keep advancing (never deadlock, never heave) UNLESS the pass is
  // both reasonably safe AND gains ground (or is a safe near-lateral outlet
  // when we're pressured). The carrier's own nav advances the bell otherwise.
  const opp = nearestOpponentDist(player, opponents);
  const pressured = opp < 14;
  const passGainsGround = bestGain > 18;
  const safeOutlet = bestScore > 0.2 && bestGain > -25;
  const acceptable =
    bestTm != null &&
    bestV0 != null &&
    bestScore > -0.02 &&
    (passGainsGround || (pressured && safeOutlet));

  if (!acceptable) {
    commit.throwGo = false;
    commit.throwTargetId = null;
    commit.throwDir = null;
    return;
  }

  const noisyDir = perturbDirection(vnorm(bestV0!), scaling.throwVariance, rng);
  const throwSpin = director.attackingFree
    ? -(0.3 + profile.loopPropensity * 0.7)
    : 0.2 + (1 - profile.freeEndBias) * 0.3;

  commit.throwGo = true;
  commit.throwTargetId = bestTm!.id;
  commit.throwDir = noisyDir;
  commit.throwSpin = Math.max(-1, Math.min(1, throwSpin));
  commit.throwCharge = Math.min(1, 0.7 + rng() * 0.3);
}

/** Distance from a player to the nearest opponent. */
function nearestOpponentDist(player: PlayerSim, opponents: PlayerSim[]): number {
  let best = Infinity;
  for (const o of opponents) {
    const d = vlen(vsub(o.p, player.p));
    if (d < best) best = d;
  }
  return best;
}

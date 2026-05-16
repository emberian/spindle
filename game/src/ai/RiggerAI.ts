// RiggerAI — per-player AI dispatcher.
// Given the current SimState + MatchState + TeamProfile + difficulty + rng,
// produce a PlayerInput for one AI-controlled player.
//
// Asymmetric difficulty:
//   rookie  — delayed reaction, wide throw variance, poor read quality, drop/miss-open
//   pro     — balanced
//   legend  — tight execution, high EV choices, sharp reads

import type { Vec3 } from '../sim/vec';
import { vnorm, vsub, vlen, v3 } from '../sim/vec';
import type { PlayerSim, SimState, MatchState, PlayerInput } from '../sim/types';
import type { TeamProfile } from '../league/teams';
import type { DirectorState } from './Director';
import { planToInput } from './nav/GrapplePlanner';
import { anchorNavigate } from './roles/Anchor';
import { spinnerNavigate } from './roles/Spinner';
import { faithwingNavigate } from './roles/Faithwing';
import { freewingNavigate } from './roles/Freewing';
import { reachNavigate } from './roles/Reach';
import { solveLeadVelocity } from './decide/LeadPredict';
import { scoreThrow } from './decide/ThrowScore';
import type { ThrowCandidate } from './decide/ThrowScore';

export type Difficulty = 'rookie' | 'pro' | 'legend';

export interface DifficultyScaling {
  /** Fraction of ticks the player "reacts" — skip decision this tick [0,1] */
  reactionDelay: number;
  /** Gaussian sigma on throw velocity direction (radians) */
  throwVariance: number;
  /** Scale on ScoreEV quality [0,1] — lower = picks worse options */
  readQuality: number;
  /** Probability of dropping a catchable bell */
  dropChance: number;
  /** Probability of not seeing an open receiver */
  missOpenChance: number;
}

const DIFFICULTY_PROFILES: Record<Difficulty, DifficultyScaling> = {
  rookie: {
    reactionDelay: 0.35,
    throwVariance: 0.18,
    readQuality: 0.45,
    dropChance: 0.08,
    missOpenChance: 0.20,
  },
  pro: {
    reactionDelay: 0.12,
    throwVariance: 0.06,
    readQuality: 0.80,
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

function axisRadius(p: Vec3): number {
  return Math.sqrt(p.y * p.y + p.z * p.z);
}

/** Estimate how open a receiver is: 1 = no defenders near, 0 = fully covered. */
function estimateOpenness(
  receiver: PlayerSim,
  opponents: PlayerSim[],
): number {
  let minDist = Infinity;
  for (const opp of opponents) {
    const d = vlen(vsub(receiver.p, opp.p));
    if (d < minDist) minDist = d;
  }
  return Math.min(1, minDist / 12);
}

/**
 * Apply throw variance: perturb a direction vector by `sigma` radians of
 * isotropic noise. Uses rng() (injected — no Math.random).
 */
function perturbDirection(dir: Vec3, sigma: number, rng: () => number): Vec3 {
  if (sigma < 0.001) return dir;
  // Box-Muller for two independent gaussians.
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  const mag = Math.sqrt(-2 * Math.log(u1)) * sigma;
  const dx = mag * Math.cos(2 * Math.PI * u2);
  const dy = mag * Math.sin(2 * Math.PI * u2);
  // Build a perturbed direction.
  const perturbed: Vec3 = { x: dir.x + dx, y: dir.y + dy, z: dir.z };
  return vnorm(perturbed);
}

/**
 * Build a complete PlayerInput for one AI player this tick.
 * Returns null if the player should skip this tick (reaction delay).
 */
export function computePlayerInput(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  director: DirectorState,
  difficulty: Difficulty,
  rng: () => number,
): PlayerInput {
  const scaling = DIFFICULTY_PROFILES[difficulty];

  // Reaction delay: skip decision this tick and coast.
  if (rng() < scaling.reactionDelay) {
    return coastInput(player);
  }

  // Determine role and navigate.
  const nav = navigateForRole(player, state, match, profile, director, rng);

  // If the player holds the bell, decide whether to throw.
  if (state.bell.heldBy === player.id) {
    return computeThrowInput(player, state, match, profile, director, scaling, nav.aim, rng);
  }

  return {
    id: player.id,
    aim: nav.aim,
    fireLineAt: nav.fireLineAt ?? null,
    reel: nav.reel ?? 0,
    release: nav.release ?? false,
    pushoff: nav.pushoff ?? false,
    throwCharge: 0,
    throwReleased: false,
    throwSpin: 0,
    thrumbler: nav.thrumbler ?? v3(),
  };
}

/** Return a "do nothing" input — the player coasts on its current grapple. */
function coastInput(player: PlayerSim): PlayerInput {
  return {
    id: player.id,
    aim: v3(1, 0, 0),
    fireLineAt: null,
    reel: player.line ? (player.line.taut ? -1 : 0) : 0,
    release: false,
    pushoff: false,
    throwCharge: 0,
    throwReleased: false,
    throwSpin: 0,
    thrumbler: v3(),
  };
}

/**
 * Compute grapple navigation for the player's role.
 * Returns a partial PlayerInput with grapple fields populated.
 */
function navigateForRole(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  director: DirectorState,
  rng: () => number,
): PlayerInput {
  let plan = null;

  switch (player.role) {
    case 'anchor':
      plan = anchorNavigate(player, state, match, profile, rng);
      break;
    case 'spinner': {
      const isLoopSetter = director.loopSetterId === player.id;
      plan = spinnerNavigate(player, state, match, profile, isLoopSetter, rng);
      break;
    }
    case 'faithwing':
      plan = faithwingNavigate(player, state, match, profile, rng);
      break;
    case 'freewing':
      plan = freewingNavigate(player, state, match, profile, rng);
      break;
    case 'reach':
      plan = reachNavigate(player, state, match, profile, rng);
      break;
  }

  const aim = plan ? vnorm(vsub(plan.anchorPos, player.p)) : v3(1, 0, 0);
  const partial = planToInput(plan, aim);

  return {
    id: player.id,
    aim: partial.aim ?? v3(1, 0, 0),
    fireLineAt: partial.fireLineAt ?? null,
    reel: partial.reel ?? 0,
    release: partial.release ?? false,
    pushoff: partial.pushoff ?? false,
    throwCharge: partial.throwCharge ?? 0,
    throwReleased: partial.throwReleased ?? false,
    throwSpin: partial.throwSpin ?? 0,
    thrumbler: partial.thrumbler ?? v3(),
  };
}

/**
 * Compute throw input when the player holds the bell.
 * Scores all teammates as throw candidates and picks the best one.
 */
function computeThrowInput(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  director: DirectorState,
  scaling: DifficultyScaling,
  _defaultAim: Vec3,
  rng: () => number,
): PlayerInput {
  const teammates = state.players.filter(
    p => p.team === player.team && p.id !== player.id,
  );
  const opponents = state.players.filter(p => p.team !== player.team);

  // Throw speed: influenced by profile aggression.
  const throwSpeed = 18 + profile.aggression * 12;

  let bestScore = -Infinity;
  let bestCandidate: ThrowCandidate | null = null;
  let bestV0: Vec3 | null = null;

  for (const tm of teammates) {
    // Miss-open: difficulty-scaled chance to not see a receiver.
    if (rng() < scaling.missOpenChance) continue;

    const openness = estimateOpenness(tm, opponents);
    // Solve Coriolis lead.
    const lead = solveLeadVelocity(
      player.p,
      throwSpeed,
      tm.p,
      tm.v,
      state.omega,
    );
    if (!lead) continue;

    const candidate: ThrowCandidate = {
      receiver: tm,
      aimPos: lead.intercept,
      flightTime: lead.flightTime,
      openness,
      receiverRadius: axisRadius(tm.p),
    };

    // Add read-quality noise to score.
    const noise = (rng() - 0.5) * (1 - scaling.readQuality) * 0.3;
    const result = scoreThrow(player, candidate, match, profile, noise);

    if (result.score > bestScore) {
      bestScore = result.score;
      bestCandidate = candidate;
      bestV0 = lead.v0;
    }
  }

  // No good throw option: hold and reposition.
  if (!bestCandidate || !bestV0 || bestScore < -0.1) {
    return coastInput(player);
  }

  // Apply throw variance.
  const noisyDir = perturbDirection(vnorm(bestV0), scaling.throwVariance, rng);
  const throwSpin = director.attackingFree
    ? -(0.3 + profile.loopPropensity * 0.7)  // antispinward spin for cross/loop
    : (0.2 + (1 - profile.freeEndBias) * 0.3); // spinward for Faith throws

  return {
    id: player.id,
    aim: noisyDir,
    fireLineAt: null,
    reel: 0,
    release: false,
    pushoff: false,
    throwCharge: 0.7 + rng() * 0.3,
    throwReleased: true,
    throwSpin,
    thrumbler: v3(),
  };
}

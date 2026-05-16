// Director — per-team strategic AI layer, runs at ~2 Hz (every 30 sim-ticks).
//
// Responsibilities:
//   1. Faith vs Free end EV decision: E[Faith]=p·2, E[Free]=p·5+pLoop·7
//   2. Cast posture: grind / tempo / chase
//   3. Pick the loop-setter (which Spinner runs the loop play)
//   4. Contest commit / decline (whether to contest a bell in flight)
//
// The Director produces a DirectorState consumed by role policies each tick.
// Parameterised by TeamProfile — different franchise philosophies emerge.

import type { TeamProfile } from '../league/teams';
import type { SimState, MatchState, PlayerSim } from '../sim/types';
import { REG, GATE_X } from '../sim/RegConstants';
import { pFall, pRise, pLoop } from './decide/ScoreEV';

// ── Director output ────────────────────────────────────────────────────────────

export type CastPosture = 'grind' | 'tempo' | 'chase';

export interface DirectorState {
  /** Which end the team is currently attacking toward. */
  attackingFree: boolean;
  /** Current cast posture. */
  posture: CastPosture;
  /** PlayerId of the designated loop-setter (a Spinner), or null. */
  loopSetterId: string | null;
  /** Should the team commit to contesting the current bell in flight? */
  contestCommit: boolean;
  /**
   * E[Faith] and E[Free] values from current position — used by role policies
   * for fine-grained scoring decisions.
   */
  evFaith: number;
  evFree: number;
  /** How many ticks until the next Director update. */
  updateIn: number;
}

// Director runs at ~2 Hz: once every 30 sim ticks.
export const DIRECTOR_TICK_INTERVAL = 30;

function axisRadius(p: { y: number; z: number }): number {
  return Math.sqrt(p.y * p.y + p.z * p.z);
}

/**
 * Compute the EV of throwing toward the Faith (spinward) ring.
 * E[Faith] = p_fall · 2
 */
export function evFaith(
  bellPos: { x: number; y: number; z: number },
  faithX: number,
): number {
  const dist = Math.abs(bellPos.x - faithX);
  const radius = axisRadius(bellPos);
  return pFall(dist, radius) * 2;
}

/**
 * Compute the EV of throwing toward the Free (antispinward) ring.
 * E[Free] = p_rise · 5 + p_loop · 7
 */
export function evFree(
  bellPos: { x: number; y: number; z: number },
  freeX: number,
  loopPropensity: number,
): number {
  const dist = Math.abs(bellPos.x - freeX);
  const radius = axisRadius(bellPos);
  const pr = pRise(dist, radius);
  const pl = pLoop(radius, dist, loopPropensity);
  return pr * 5 + pl * 7;
}

/**
 * Choose cast posture based on score differential, inning, and possession.
 *
 * grind:  conservative, minimize turnovers (ahead, or late in match)
 * tempo:  balanced, normal execution
 * chase:  aggressive, take risks (behind, need big scores)
 */
function choosePosture(
  match: MatchState,
  profile: TeamProfile,
  evFaithVal: number,
  evFreeVal: number,
): CastPosture {
  const scoreDiff = match.scoreHome - match.scoreAway;
  const inningsLeft = Math.max(1, 9 - match.inning + (match.spine ? 0 : 1));
  // Chase: significantly behind or in spine (overtime).
  if (scoreDiff < -6 || match.spine) {
    return 'chase';
  }
  // Grind: comfortably ahead with few innings left.
  if (scoreDiff > 4 && inningsLeft <= 3) {
    return 'grind';
  }
  // High-variance teams always prefer chase-style if the EV is there.
  if (profile.variance > 0.65 && evFreeVal > evFaithVal * 1.2) {
    return 'chase';
  }
  return 'tempo';
}

/**
 * Pick which Spinner should set the loop play this cast.
 * Prefers Spinners already near the axis (high).
 */
function pickLoopSetter(players: PlayerSim[], teamSide: PlayerSim['team']): string | null {
  const spinners = players.filter(p => p.team === teamSide && p.role === 'spinner');
  if (spinners.length === 0) return null;

  // Closest to axis = best loop-setter.
  let best = spinners[0];
  let bestRadius = axisRadius(best.p);
  for (const s of spinners.slice(1)) {
    const r = axisRadius(s.p);
    if (r < bestRadius) {
      bestRadius = r;
      best = s;
    }
  }
  return best.id;
}

/**
 * Decide whether to commit to a contest (one-on-one duel).
 *
 * Factors:
 *   - contestAggression profile
 *   - Contest position (deep = violent/final, high = chess/reversible)
 *   - Current posture
 */
function shouldContest(
  _match: MatchState,
  profile: TeamProfile,
  posture: CastPosture,
  contestRadius: number,
): boolean {
  // Base: contestAggression from profile.
  let score = profile.contestAggression;

  // Deep contests (high radius) are violent and final — commit only if aggressive.
  const radiusFrac = contestRadius / REG.R;
  if (radiusFrac > 0.7) {
    score -= 0.1; // Deep is risky even for aggressive teams.
  } else {
    score += 0.05; // High (near axis) contests are reversible — commit more.
  }

  // Chasing: must contest to regain possession.
  if (posture === 'chase') score += 0.15;
  if (posture === 'grind') score -= 0.1;

  return score > 0.5;
}

/**
 * Run the Director for one team.
 * Call this at ~2 Hz; cache the result in DirectorState.updateIn.
 */
export function runDirector(
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  teamSide: PlayerSim['team'],
  _rng: () => number,
): DirectorState {
  const faithX = match.faithEnd === '+x' ? GATE_X : -GATE_X;
  const freeX  = match.faithEnd === '+x' ? -GATE_X : GATE_X;

  const ef = evFaith(state.bell.p, faithX);
  const efr = evFree(state.bell.p, freeX, profile.loopPropensity);

  // Attack Free if its EV is better, weighted by the team's freeEndBias.
  const biasedFree = efr * (0.6 + profile.freeEndBias * 0.8);
  const biasedFaith = ef * (0.6 + (1 - profile.freeEndBias) * 0.8);
  const attackingFree = biasedFree > biasedFaith;

  const posture = choosePosture(match, profile, ef, efr);

  // Loop-setter: only designate one if posture isn't grind and loop propensity is high.
  const useLoop = profile.loopPropensity > 0.25 && posture !== 'grind';
  const loopSetterId = useLoop ? pickLoopSetter(state.players, teamSide) : null;

  // Contest decision.
  let contestCommit = false;
  if (match.contest !== null) {
    const contester = state.players.find(p => p.id === match.contest!.contester);
    const contestRadius = contester ? axisRadius(contester.p) : REG.R * 0.5;
    contestCommit = shouldContest(match, profile, posture, contestRadius);
  }

  return {
    attackingFree,
    posture,
    loopSetterId,
    contestCommit,
    evFaith: ef,
    evFree: efr,
    updateIn: DIRECTOR_TICK_INTERVAL,
  };
}

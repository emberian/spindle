// Per-target throw scoring for the AI. Given a potential receiver and the
// current game state, compute a scalar "throw score" used by role policies to
// pick the best recipient. Integrates LeadPredict (Coriolis trajectory) and
// ScoreEV (end-zone option value) with TeamProfile biases.
//
// No RNG here — the call-site injects noise for difficulty variance.

import type { Vec3 } from '../../sim/vec';
import type { PlayerSim, MatchState } from '../../sim/types';
import type { TeamProfile } from '../../league/teams';
import { computeScoreEV } from './ScoreEV';
import { REG, GATE_X } from '../../sim/RegConstants';

export interface ThrowCandidate {
  receiver: PlayerSim;
  /** World position the throw is aimed at (lead intercept) */
  aimPos: Vec3;
  /** Estimated flight time in seconds */
  flightTime: number;
  /** How open the receiver is [0, 1] — 1 = totally uncontested */
  openness: number;
  /** Axis-radius of receiver position */
  receiverRadius: number;
}

export interface ThrowScoreResult {
  candidate: ThrowCandidate;
  /** Final scalar score — higher = better throw */
  score: number;
  /** Preferred scoring intent if the receiver is at or near the goal mouth */
  scoringIntent: 'fall' | 'rise' | 'loop' | 'curl' | 'advance';
}

/**
 * Score a single throw candidate.
 *
 * Scoring factors:
 *   - openness (primary — don't throw into coverage)
 *   - yardage gain along the axis toward the target ring
 *   - ScoreEV for receivers near the goal mouth
 *   - radius: consistent with profile (Fall teams want deep, Rise teams want high)
 *   - estimated flight time (shorter = safer)
 */
export function scoreThrow(
  thrower: PlayerSim,
  candidate: ThrowCandidate,
  match: MatchState,
  profile: TeamProfile,
  addNoise: number = 0,
): ThrowScoreResult {
  const { aimPos, openness, flightTime, receiverRadius } = candidate;

  // Determine which ring is the attack target.
  const attackFaith = match.possession === 'home'
    ? match.faithEnd === '+x'
    : match.faithEnd === '-x';

  const faithX = match.faithEnd === '+x' ? GATE_X : -GATE_X;
  const freeX  = match.faithEnd === '+x' ? -GATE_X : GATE_X;

  const distToFaithRing = Math.abs(aimPos.x - faithX);
  const distToFreeRing  = Math.abs(aimPos.x - freeX);

  // Axial progress toward the team's preferred end.
  const throwerX = thrower.p.x;
  const axialGain = attackFaith
    ? (faithX > 0 ? aimPos.x - throwerX : throwerX - aimPos.x)
    : (freeX  > 0 ? aimPos.x - throwerX : throwerX - aimPos.x);

  const normalizedGain = Math.max(-0.5, Math.min(1.5, axialGain / (REG.L * 0.25)));

  // Openness is the dominant term.
  const opennessScore = openness * 0.45;

  // Axial gain toward goal.
  const gainScore = normalizedGain * (0.2 + profile.aggression * 0.15);

  // Flight-time penalty (short = safe, long = risky).
  const timePenalty = Math.min(0.2, flightTime * 0.03);

  // Radius alignment: Fall teams want deep (high radius), Rise teams want high (low radius).
  const receiverRadiusFrac = receiverRadius / REG.R;
  const radiusScore = attackFaith
    ? receiverRadiusFrac * 0.1 * (1 - profile.freeEndBias)  // fall: deep bonus
    : (1 - receiverRadiusFrac) * 0.1 * profile.freeEndBias;  // rise: axis bonus

  // Near-goal EV bonus.
  let scoringIntent: ThrowScoreResult['scoringIntent'] = 'advance';
  let evBonus = 0;
  const isNearGoal = Math.min(distToFaithRing, distToFreeRing) < REG.L * 0.12;
  if (isNearGoal) {
    const evCtx = {
      distToFaithRing,
      distToFreeRing,
      radiusFromAxis: receiverRadius,
      match,
      profile,
    };
    const ev = computeScoreEV(evCtx);
    const bestEv = Math.max(ev.evFall, ev.evRise, ev.evLoop, ev.evCurl);
    evBonus = Math.min(0.3, bestEv * 0.06);
    scoringIntent = ev.preferred;
  }

  const rawScore =
    opennessScore + gainScore + radiusScore + evBonus - timePenalty + addNoise;

  return { candidate, score: rawScore, scoringIntent };
}

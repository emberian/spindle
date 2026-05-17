// Expected-value calculations for Fall / Rise / Loop / Curl scoring attempts.
// Parameterised by a TeamProfile so different franchise philosophies produce
// different shot selections. No RNG — call-site injects variance if desired.

import type { TeamProfile } from '../../league/teams';
import type { MatchState } from '../../sim/types';
import { REG } from '../../sim/RegConstants';

// ── Scoring values (canon §6 rules.md) ───────────────────────────────────────
export const SCORE_FALL = 2;
export const SCORE_RISE = 5;
export const SCORE_LOOP = 7;
export const SCORE_GROUND = 1; // awarded to defense

// ── Completion-probability models ─────────────────────────────────────────────
// These are rough sigmoid-shaped models keyed on distance to goal ring and
// current axis-radius. Outer values calibrated to match the "Rise rarely
// scores in a season" and "Fall is the bread" canon.

/**
 * Estimated probability of a clean Fall (spinward Faith-end score).
 * Easier than Rise; higher from depth (deep = heavy, committed but true).
 * p_fall = base * distFactor * radiusFactor
 */
export function pFall(
  distToFaithRing: number,
  radiusFromAxis: number,
): number {
  // Base: 80% if right at the gate, falling off with range.
  const distFactor = Math.max(0, 1 - distToFaithRing / (REG.L * 0.55));
  // Deep play (high radius) strengthens a Fall — "deep is faithful".
  const radiusFactor = 0.65 + 0.35 * Math.min(1, radiusFromAxis / REG.R);
  return Math.min(1, 0.82 * distFactor * radiusFactor);
}

/**
 * Estimated probability of a clean Rise (antispinward Free-end score).
 * Harder — rises against the world's curve. Most players never score one.
 */
export function pRise(
  distToFreeRing: number,
  radiusFromAxis: number,
): number {
  // Base: only ~35% even close to the ring.
  const distFactor = Math.max(0, 1 - distToFreeRing / (REG.L * 0.4));
  // Being closer to the axis helps on the Free end — high is free.
  const radiusFactor = 0.75 - 0.25 * Math.min(1, radiusFromAxis / REG.R);
  return Math.min(1, 0.35 * distFactor * radiusFactor);
}

/**
 * Estimated probability of a Loop completing (7-pt score).
 * Extremely difficult; requires very specific positions and spin window.
 */
export function pLoop(
  radiusFromAxis: number,
  distToRing: number,
  loopPropensity: number,
): number {
  // Only achievable near-axis (high) with the right spin. Window & base
  // raised (user call: hunt Loops more — the marquee 7 should actually
  // show in a watched match, not "never in a season"). Team loopPropensity
  // still scales it, so franchise style divergence is preserved.
  const radiusFactor = Math.max(0, 1 - radiusFromAxis / (REG.R * 0.9));
  const distFactor = Math.max(0, 1 - distToRing / (REG.L * 0.5));
  return Math.min(0.7, 0.20 * radiusFactor * distFactor * (0.4 + loopPropensity));
}

/**
 * Curl: a partial-loop that curves back from the Free end.
 * Worth 5 (a Rise) if it makes it through; modeled as a lower-prob Rise.
 */
export function pCurl(
  radiusFromAxis: number,
  distToFreeRing: number,
  loopPropensity: number,
): number {
  const base = pRise(distToFreeRing, radiusFromAxis);
  return base * (0.5 + 0.5 * loopPropensity);
}

// ── EV calculators ─────────────────────────────────────────────────────────────

export interface ScoreEVContext {
  distToFaithRing: number;
  distToFreeRing: number;
  radiusFromAxis: number;
  match: MatchState;
  profile: TeamProfile;
}

export interface ScoreEVResult {
  evFall: number;
  evRise: number;
  evLoop: number;
  evCurl: number;
  preferred: 'fall' | 'rise' | 'loop' | 'curl';
}

/**
 * Compute expected-value for each scoring option at the current field position.
 *
 * E[Fall]  = p_fall · 2
 * E[Rise]  = p_rise · 5
 * E[Loop]  = p_loop · 7
 * E[Curl]  = p_curl · 5  (same ring value as Rise)
 *
 * Team profile biases are applied additively after the physics-based EV so
 * that a rise-chaos team still shoots loops even when the raw math says no.
 */
export function computeScoreEV(ctx: ScoreEVContext): ScoreEVResult {
  const { distToFaithRing, distToFreeRing, radiusFromAxis, profile } = ctx;

  const pf = pFall(distToFaithRing, radiusFromAxis);
  const pr = pRise(distToFreeRing, radiusFromAxis);
  const pl = pLoop(radiusFromAxis, Math.min(distToFaithRing, distToFreeRing), profile.loopPropensity);
  const pc = pCurl(radiusFromAxis, distToFreeRing, profile.loopPropensity);

  // Raw EVs.
  const rawFall = pf * SCORE_FALL;
  const rawRise = pr * SCORE_RISE;
  const rawLoop = pl * SCORE_LOOP;
  const rawCurl = pc * SCORE_RISE;

  // Bias adjustments from team profile (scale, not add, to preserve relative ordering).
  const fallBias  = 1 + (1 - profile.freeEndBias) * 0.4;   // Fall teams inflate Faith EV
  const freeBias  = 1 + profile.freeEndBias * 0.6;          // Rise teams inflate Free EV
  const loopBias  = 1 + profile.loopPropensity * 0.8;       // Loop specialists inflate Loop EV

  const evFall = rawFall * fallBias;
  const evRise = rawRise * freeBias;
  const evLoop = rawLoop * loopBias;
  const evCurl = rawCurl * freeBias * (0.6 + profile.loopPropensity * 0.4);

  // Preferred option: highest EV.
  let preferred: ScoreEVResult['preferred'] = 'fall';
  let best = evFall;
  if (evRise > best) { best = evRise; preferred = 'rise'; }
  if (evLoop > best) { best = evLoop; preferred = 'loop'; }
  if (evCurl > best) { preferred = 'curl'; }

  return { evFall, evRise, evLoop, evCurl, preferred };
}

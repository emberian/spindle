// Contest.ts — pure one-on-one duel resolution for RIG.
//
// FROZEN contract: codes only against sim/types.ts.
//
// A contest ("a one") resolves over THREE exchanges (the count).
// Canon (rules.md §5 / sport.md §3.4):
//   - Thrower wins the count → complete; cast continues.
//   - Contester wins the count → turnover at that radius/direction.
//   - Depth (radius from axis) governs decisiveness:
//       deep (large radius) = fast, violent, final → low variance, radius-heavy outcome
//       high (small radius) = slow, chess-like, reversible → higher variance
//
// Determinism: rngNext is injected by the caller; the caller owns the seed.
// This function is pure: given the same inputs it produces the same output.

import type { MatchState } from '../sim/types';

export interface ContestState {
  thrower: string;
  contester: string;
  count: 0 | 1 | 2 | 3;
  radius: number;
  direction: 'fair' | 'cross';
}

export interface ContestResult {
  winner: 'thrower' | 'contester';
  /** How many exchanges were needed (1–3). */
  exchanges: number;
}

/**
 * Resolve a full RIG contest (up to 3 exchanges).
 *
 * @param contest — the contest state at the moment of freeze
 * @param _matchSnap — match snapshot (unused currently; provided for future gradient hooks)
 * @param rngNext — injected deterministic RNG draw: returns a number in [0, 1)
 *
 * Depth model:
 *   At the axis (radius ≈ 0) the contest is purely a coin flip each exchange.
 *   At skin (radius ≈ R ≈ 45 m) the gradient gives a heavy advantage
 *   to the thrower (the aggressive party) — deeper commitment = more decisive.
 *   We encode this as a thrower-win probability per exchange that rises with radius:
 *     p_thrower = 0.5 + 0.25 * clamp(radius / R, 0, 1)
 *   (range 0.5 at axis → 0.75 at skin)
 *   "Cross" direction (antispinward) slightly favours the contester (the defender
 *   riding the world's hand against the thrower's curve):
 *     p_thrower -= 0.05 if direction === 'cross'
 *   Final p_thrower is clamped to [0.25, 0.85].
 *
 * Best-of-3 logic: first to 2 exchange-wins takes the contest.
 */
export function resolveContest(
  contest: ContestState,
  _matchSnap: MatchState,
  rngNext: () => number,
): ContestResult {
  const R = 45; // regulation skin radius (from RegConstants, inlined for purity)

  let p = 0.5 + 0.25 * Math.min(1, Math.max(0, contest.radius / R));
  if (contest.direction === 'cross') p -= 0.05;
  p = Math.min(0.85, Math.max(0.25, p));

  let throwerWins = 0;
  let contesterWins = 0;
  let exchanges = 0;

  while (throwerWins < 2 && contesterWins < 2) {
    exchanges++;
    if (rngNext() < p) {
      throwerWins++;
    } else {
      contesterWins++;
    }
  }

  return {
    winner: throwerWins >= 2 ? 'thrower' : 'contester',
    exchanges,
  };
}

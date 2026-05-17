// Orientation — the single source of truth for "which way is forward".
//
// THE BUG THIS FIXES: the rest of the AI historically derived a team's
// attacking ring from `match.faithEnd`. That is WRONG. Per the frozen match
// rules (src/match/MatchStateMachine.ts → attackDir):
//
//     home ALWAYS attacks the +X ring (x = +GATE_X)
//     away ALWAYS attacks the -X ring (x = -GATE_X)
//
// `faithEnd` only labels which physical ring scores as "Faith" (2 pts) vs
// "Free" (5 pts); it does NOT determine the direction a team drives the bell.
//
// Every nav target, receiver depth slot, throw lead, and Director posture must
// be expressed relative to THIS, never to faithEnd. This module is that anchor.

import type { MatchState, TeamSide, SimState } from '../sim/types';
import { GATE_X } from '../sim/RegConstants';

/** Signed attack direction for a team: +1 → drives toward +X ring, -1 → -X. */
export function attackSign(team: TeamSide): 1 | -1 {
  return team === 'home' ? 1 : -1;
}

/** World-x of the ring `team` is attacking (scoring at). */
export function attackRingX(team: TeamSide): number {
  return attackSign(team) * GATE_X;
}

/** World-x of the ring `team` is defending (their own ring). */
export function defendRingX(team: TeamSide): number {
  return -attackSign(team) * GATE_X;
}

/**
 * Is the ring `team` attacks the Faith (spinward, 2-pt) ring?
 * faithEnd '+x' means the +X ring is Faith. home attacks +X, away attacks -X.
 */
export function attackingFaithRing(team: TeamSide, match: MatchState): boolean {
  const ringX = attackRingX(team);
  const faithX = match.faithEnd === '+x' ? GATE_X : -GATE_X;
  return Math.sign(ringX) === Math.sign(faithX);
}

/**
 * Which team currently "owns the attack" — i.e. whose attacking direction the
 * offensive shape should be built around. Prefers the actual bell holder, then
 * the last thrower (in-flight from us), then match.possession.
 */
export function attackingTeam(state: SimState, match: MatchState): TeamSide {
  const heldBy = state.bell.heldBy;
  if (heldBy) {
    const h = state.players.find(p => p.id === heldBy);
    if (h) return h.team;
  }
  const thrownBy = state.bell.thrownBy;
  if (thrownBy && !state.bell.heldBy) {
    const t = state.players.find(p => p.id === thrownBy);
    if (t) return t.team;
  }
  return match.possession;
}

/** Forward (toward `team`'s attacking ring) progress of an x coordinate. */
export function forwardProgress(team: TeamSide, x: number): number {
  return attackSign(team) * x;
}

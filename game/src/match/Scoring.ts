// Scoring.ts — pure scoring attribution for RIG bell_through_ring events.
//
// FROZEN contract: codes only against sim/types.ts.
//
// Canonical point table (rules.md §6 / sport.md §3.5):
//   Fall  = 2  — bell cleanly through the Faith (spinward) ring, touched OK
//   Rise  = 5  — bell through the Free (antispinward) ring
//   Loop  = 7  — untouched closed Coriolis arc (loopTier === 'loop'); ends inning
//   Curl  = base+2, capped below Loop:
//            Faith-curl = 4 (Fall base 2 + 2), Free-curl = 6 (Rise base 5-1=4? → cap at 6)
//   Ground = 1 — awarded to the DEFENSE when attacker is player_skinned
//
// Faith end identification:
//   faithEnd '+x' → ring at +x is Faith (spinward), ring at -x is Free.
//   faithEnd '-x' → ring at -x is Faith, ring at +x is Free.
//
// Pass-chain-safe attribution:
//   The attacking team is the team of bell.thrownBy (last thrower).
//   If bell.passChain has entries, the scorer is the last person in passChain
//   whose team is the attacking team — credit follows the chain correctly.
//   For Ground, the defending team is the team that did NOT last touch.

import type { SimEvent, SimState, MatchState, TeamSide } from '../sim/types';

export type ScoreKind = 'fall' | 'rise' | 'loop' | 'curl' | 'ground';

export interface ScoreResult {
  team: TeamSide;
  kind: ScoreKind;
  points: number;
  endsInning: boolean;
}

/** Is this ring-end the Faith end? */
function isFaithEnd(end: '+x' | '-x', faithEnd: '+x' | '-x'): boolean {
  return end === faithEnd;
}

/**
 * Determine the attacking team from the bell state.
 * The attacker is the team of the last known thrower/holder.
 * Returns null if attribution is impossible.
 */
function attackingTeam(sim: SimState): TeamSide | null {
  // Prefer passChain last entry
  const chain = sim.bell.passChain;
  if (chain.length > 0) {
    const lastId = chain[chain.length - 1];
    const pl = sim.players.find((p) => p.id === lastId);
    if (pl) return pl.team;
  }
  // Fallback: thrownBy
  if (sim.bell.thrownBy) {
    const pl = sim.players.find((p) => p.id === sim.bell.thrownBy);
    if (pl) return pl.team;
  }
  // Fallback: heldBy
  if (sim.bell.heldBy) {
    const pl = sim.players.find((p) => p.id === sim.bell.heldBy);
    if (pl) return pl.team;
  }
  return null;
}

/**
 * Map a bell_through_ring event to a scoring result.
 *
 * Returns null if attribution fails (no known thrower).
 */
export function scoreFor(
  ev: Extract<SimEvent, { type: 'bell_through_ring' }>,
  match: MatchState,
  sim: SimState,
): ScoreResult | null {
  const attacker = attackingTeam(sim);
  if (attacker === null) return null;

  const faith = isFaithEnd(ev.end, match.faithEnd);
  const untouched = !ev.touched;

  // Loop: untouched closed Coriolis arc — tier from sim's loopTier field
  if (ev.loopTier === 'loop' && untouched) {
    return { team: attacker, kind: 'loop', points: 7, endsInning: true };
  }

  // Curl: partial Coriolis arc, untouched
  if (ev.loopTier === 'curl' && untouched) {
    // Faith-curl: base 2 + 2 = 4; Free-curl: base 5 + 2 = 7 but capped to 6 (< Loop 7)
    const base = faith ? 2 : 5;
    const points = Math.min(base + 2, 6);
    return { team: attacker, kind: 'curl', points, endsInning: false };
  }

  // Fall (Faith ring, any touch state for ordinary score)
  if (faith) {
    return { team: attacker, kind: 'fall', points: 2, endsInning: false };
  }

  // Rise (Free ring)
  return { team: attacker, kind: 'rise', points: 5, endsInning: false };
}

/**
 * Map a player_skinned event to a Ground score for the defense.
 *
 * Ground = 1 pt to the team that is NOT the attacker.
 * The attacker is the team currently holding / last touching the bell.
 * Returns null if attribution fails.
 */
export function groundFor(
  ev: Extract<SimEvent, { type: 'player_skinned' }>,
  sim: SimState,
): { team: TeamSide; kind: 'ground'; points: 1; endsInning: false } | null {
  // The skinned player's team is the team that was scored on (they get -defensive credit)
  // Ground is awarded to the DEFENSE (team opposite the attacker).
  const attacker = attackingTeam(sim);
  if (attacker === null) return null;

  // Check that the skinned player is on the attacking team
  const skinned = sim.players.find((p) => p.id === ev.id);
  if (!skinned) return null;
  if (skinned.team !== attacker) return null; // defender skinned → no ground

  const defense: TeamSide = attacker === 'home' ? 'away' : 'home';
  return { team: defense, kind: 'ground', points: 1, endsInning: false };
}

// Re-export for convenience
export type { TeamSide };

// Reach role policy — goal-ring keeper, mobile.
// The Reach defends the scoring arc and is the one player allowed to clip the
// goal ring itself and pivot on it. Reaches are the sport's lunatics.
// Decision tick: 5 Hz (every 12 sim-ticks at 60 Hz) — they hold long.
//
// Responsibilities:
//   - Cover the ring mouth on defense.
//   - On offense: position beyond the ring to contest incoming passes.
//   - ONLY role allowed to fire their line at the goal ring.

import type { Vec3 } from '../../sim/vec';
import type { PlayerSim, SimState, MatchState } from '../../sim/types';
import type { TeamProfile } from '../../league/teams';
import { REG, GATE_X } from '../../sim/RegConstants';
import { planGrapple, type GrapplePlan } from '../nav/GrapplePlanner';

// Reach patrols at moderate radius, close to the goal ring.
const REACH_PATROL_RADIUS = REG.R * 0.48;
const REACH_ATTACK_RADIUS = REG.R * 0.35;

export interface ReachIntent {
  targetPos: Vec3;
  /** Whether to fire line at the ring itself (only the Reach may do this). */
  clipRing: boolean;
  intent: 'defend-mouth' | 'pivot-ring' | 'contest-approach' | 'clear-zone';
}

export function reachPolicy(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  _profile: TeamProfile,
  rng: () => number,
): ReachIntent {
  const bell = state.bell;
  const possession = match.possession === player.team;
  const faithX = match.faithEnd === '+x' ? GATE_X : -GATE_X;
  const freeX  = match.faithEnd === '+x' ? -GATE_X : GATE_X;

  if (!possession) {
    // Defense: guard the mouth of the ring we're protecting.
    // Which end is the Reach guarding? They defend the team's own ring.
    // The team attacks toward the opponent's ring, so we defend our own.
    const defendX = faithX; // simplified — defending the Faith end

    const bellDistToOurRing = Math.abs(bell.p.x - defendX);
    const isHot = bellDistToOurRing < REG.L * 0.18;

    if (isHot) {
      // Clip the ring — pivot on it to extend reach.
      const angle = Math.atan2(bell.p.z, bell.p.y);
      return {
        targetPos: {
          x: defendX,
          y: REG.gateRadius * Math.cos(angle),
          z: REG.gateRadius * Math.sin(angle),
        },
        clipRing: true,
        intent: 'pivot-ring',
      };
    }

    // Normal defense: float in front of the ring.
    const guardX = defendX + (faithX > 0 ? -18 : 18);
    const angle = Math.PI * (0.3 + rng() * 0.2);
    return {
      targetPos: {
        x: guardX,
        y: REACH_PATROL_RADIUS * Math.cos(angle),
        z: REACH_PATROL_RADIUS * Math.sin(angle),
      },
      clipRing: false,
      intent: 'defend-mouth',
    };
  } else {
    // Offense: contest passes approaching the opponent's ring.
    const attackX = freeX + (freeX > 0 ? -20 : 20);
    const angle = Math.PI * (0.2 + rng() * 0.2);
    return {
      targetPos: {
        x: attackX,
        y: REACH_ATTACK_RADIUS * Math.cos(angle),
        z: REACH_ATTACK_RADIUS * Math.sin(angle),
      },
      clipRing: false,
      intent: 'contest-approach',
    };
  }
}

export function reachNavigate(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  rng: () => number,
): GrapplePlan | null {
  const { targetPos, clipRing } = reachPolicy(player, state, match, profile, rng);

  if (clipRing) {
    // Anchor directly to the goal ring — planGrapple handles this via the target.
    return planGrapple(player, targetPos, state);
  }

  return planGrapple(player, targetPos, state);
}

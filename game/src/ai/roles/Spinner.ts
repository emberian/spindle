// Spinner role policy — midfield engine, trades altitude for tempo.
// Spinners live in the slope, run the lines, and set loop plays.
// Decision tick: 10 Hz.
//
// Responsibilities:
//   - Maintain midfield position, bridging Anchor and Wings.
//   - Set loop-setter position when loop play is called.
//   - Pressure bell carriers on defense.
//   - High grapple activity — Spinners swing constantly.

import type { Vec3 } from '../../sim/vec';
import type { PlayerSim, SimState, MatchState } from '../../sim/types';
import type { TeamProfile } from '../../league/teams';
import { REG, GATE_X } from '../../sim/RegConstants';
import { planGrapple, type GrapplePlan } from '../nav/GrapplePlanner';

// Spinner operates in the mid-radius band.
const SPINNER_TARGET_RADIUS = REG.R * 0.42;
const SPINNER_HIGH_RADIUS = REG.R * 0.20;

export interface SpinnerIntent {
  targetPos: Vec3;
  intent: 'midfield' | 'loop-setter' | 'pressure' | 'swing-line';
  isLoopSetter: boolean;
}

/**
 * Spinner decision policy.
 * Spinners alternate between midfield holding and aggressive line-setting.
 */
export function spinnerPolicy(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  isLoopSetter: boolean,
  rng: () => number,
): SpinnerIntent {
  const bell = state.bell;
  const possession = match.possession === player.team;
  const faithX = match.faithEnd === '+x' ? GATE_X : -GATE_X;
  const freeX = match.faithEnd === '+x' ? -GATE_X : GATE_X;

  if (possession) {
    if (isLoopSetter && profile.loopPropensity > 0.3) {
      // Loop-setter: position near axis, ready to launch the loop.
      // Optimal position: mid-field, high (close to axis), with angle.
      const loopX = (faithX + freeX) / 2 + (rng() - 0.5) * 60;
      const angle = Math.PI * (0.1 + rng() * 0.15);
      const targetY = SPINNER_HIGH_RADIUS * Math.cos(angle);
      const targetZ = SPINNER_HIGH_RADIUS * Math.sin(angle);
      return {
        targetPos: { x: loopX, y: targetY, z: targetZ },
        intent: 'loop-setter',
        isLoopSetter: true,
      };
    }

    // Normal offense: midfield bridge position.
    const midX = player.p.x + (faithX > 0 ? 40 : -40) * profile.aggression;
    const angle = Math.PI * (0.3 + rng() * 0.25);
    const r = SPINNER_TARGET_RADIUS * (0.8 + rng() * 0.4);
    return {
      targetPos: { x: midX, y: r * Math.cos(angle), z: r * Math.sin(angle) },
      intent: 'midfield',
      isLoopSetter: false,
    };
  } else {
    // Defense: pressure the ball carrier from midfield.
    const bellHolder = state.players.find(p => p.id === bell.heldBy);
    if (bellHolder && bellHolder.team !== player.team) {
      // Pressure from the spinning side.
      const pressureX = bellHolder.p.x + (faithX > 0 ? 20 : -20);
      return {
        targetPos: { x: pressureX, y: bellHolder.p.y * 0.7, z: bellHolder.p.z * 0.7 },
        intent: 'pressure',
        isLoopSetter: false,
      };
    }

    // Hold midfield position on defense.
    const defX = (faithX + freeX) / 2;
    return {
      targetPos: { x: defX, y: SPINNER_TARGET_RADIUS * 0.7, z: 0 },
      intent: 'midfield',
      isLoopSetter: false,
    };
  }
}

export function spinnerNavigate(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  isLoopSetter: boolean,
  rng: () => number,
): GrapplePlan | null {
  const { targetPos } = spinnerPolicy(player, state, match, profile, isLoopSetter, rng);
  return planGrapple(player, targetPos, state);
}

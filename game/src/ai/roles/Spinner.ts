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
import { REG } from '../../sim/RegConstants';
import { planGrapple, type GrapplePlan } from '../nav/GrapplePlanner';
import { attackSign, defendRingX } from '../Orientation';

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
  // ORIENTATION-CORRECT: advance toward the ring THIS team attacks.
  const sgn = attackSign(player.team); // +1 home, -1 away
  const ourDefX = defendRingX(player.team);

  if (possession) {
    if (isLoopSetter && profile.loopPropensity > 0.3) {
      // Loop-setter: position near axis, ready to launch the loop.
      // Optimal position: mid-field, high (close to axis), with angle.
      const loopX = 0 + (rng() - 0.5) * 60;
      const angle = Math.PI * (0.1 + rng() * 0.15);
      const targetY = SPINNER_HIGH_RADIUS * Math.cos(angle);
      const targetZ = SPINNER_HIGH_RADIUS * Math.sin(angle);
      return {
        targetPos: { x: loopX, y: targetY, z: targetZ },
        intent: 'loop-setter',
        isLoopSetter: true,
      };
    }

    // Normal offense: bridge AHEAD of self toward OUR attacking ring.
    const midX = player.p.x + sgn * 40 * (0.6 + profile.aggression);
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
      // Pressure goal-side: get between the carrier and OUR defended ring.
      const pressureX = bellHolder.p.x + (ourDefX > 0 ? 20 : -20);
      return {
        targetPos: { x: pressureX, y: bellHolder.p.y * 0.7, z: bellHolder.p.z * 0.7 },
        intent: 'pressure',
        isLoopSetter: false,
      };
    }

    // Hold midfield position on defense.
    return {
      targetPos: { x: 0, y: SPINNER_TARGET_RADIUS * 0.7, z: 0 },
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

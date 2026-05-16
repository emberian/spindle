// Faithwing role policy — the power wing, spinward side.
// Devastating in the Faith direction; the heavy reliable scorer.
// Decision tick: 10 Hz.
//
// Responsibilities:
//   - Position to receive on the Faith (spinward) side.
//   - Drive toward the Faith ring when bell approaches.
//   - On defense: shadow the opposing Freewing (the most dangerous scorer).

import type { Vec3 } from '../../sim/vec';
import type { PlayerSim, SimState, MatchState } from '../../sim/types';
import type { TeamProfile } from '../../league/teams';
import { REG, GATE_X } from '../../sim/RegConstants';
import { planGrapple, type GrapplePlan } from '../nav/GrapplePlanner';

const FAITHWING_RADIUS = REG.R * 0.55;

export interface FaithwingIntent {
  targetPos: Vec3;
  intent: 'faith-attack' | 'receive' | 'shadow-freewing';
}

export function faithwingPolicy(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  rng: () => number,
): FaithwingIntent {
  const bell = state.bell;
  const possession = match.possession === player.team;
  const faithX = match.faithEnd === '+x' ? GATE_X : -GATE_X;

  if (possession) {
    // Check if the bell is close to the Faith ring.
    const bellDistToFaith = Math.abs(bell.p.x - faithX);
    if (bellDistToFaith < REG.L * 0.3) {
      // Attack mode: sprint into position to receive a scoring pass.
      const attackX = faithX + (faithX > 0 ? -15 : 15);
      const angle = Math.PI * 0.35 * (rng() * 0.4 + 0.8);
      return {
        targetPos: {
          x: attackX,
          y: FAITHWING_RADIUS * Math.cos(angle),
          z: FAITHWING_RADIUS * Math.sin(angle),
        },
        intent: 'faith-attack',
      };
    }

    // Midfield receiving position, biased toward the Faith side.
    const midX = bell.p.x + (faithX > 0 ? 50 * profile.aggression : -50 * profile.aggression);
    const angle = Math.PI * (0.25 + rng() * 0.2);
    return {
      targetPos: { x: midX, y: FAITHWING_RADIUS * Math.cos(angle), z: FAITHWING_RADIUS * Math.sin(angle) },
      intent: 'receive',
    };
  } else {
    // Defense: shadow the opposing Freewing (they're the dangerous one).
    const opponentFreewing = state.players.find(
      p => p.team !== player.team && p.role === 'freewing',
    );
    if (opponentFreewing) {
      // Position between the Freewing and the Faith ring.
      const shadowX = (opponentFreewing.p.x + faithX) / 2;
      return {
        targetPos: {
          x: shadowX,
          y: opponentFreewing.p.y * 0.8,
          z: opponentFreewing.p.z * 0.8,
        },
        intent: 'shadow-freewing',
      };
    }

    // Fall back to guarding the Faith end.
    const guardX = faithX + (faithX > 0 ? -40 : 40);
    return {
      targetPos: { x: guardX, y: FAITHWING_RADIUS * 0.7, z: 0 },
      intent: 'shadow-freewing',
    };
  }
}

export function faithwingNavigate(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  rng: () => number,
): GrapplePlan | null {
  const { targetPos } = faithwingPolicy(player, state, match, profile, rng);
  return planGrapple(player, targetPos, state);
}

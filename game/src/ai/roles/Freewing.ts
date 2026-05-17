// Freewing role policy — the artist, antispinward side.
// The franchise's signature; ticket sales have Freewings' names on them.
// Attacks the Free end — the hardest, most beautiful, most valuable direction.
// Decision tick: 10 Hz.
//
// Responsibilities:
//   - Attack the Free (antispinward) ring for Rise (5) and Loop (7) plays.
//   - Go high (near axis) to exploit Coriolis curves.
//   - On defense: contest from the cross-spin position, force bad throws.

import type { Vec3 } from '../../sim/vec';
import type { PlayerSim, SimState, MatchState } from '../../sim/types';
import type { TeamProfile } from '../../league/teams';
import { REG } from '../../sim/RegConstants';
import { planGrapple, type GrapplePlan } from '../nav/GrapplePlanner';
import { attackRingX, defendRingX } from '../Orientation';

// Freewings operate HIGH — near the axis to exploit Coriolis curves.
const FREEWING_ATTACK_RADIUS = REG.R * 0.22;
const FREEWING_HOLD_RADIUS = REG.R * 0.38;

export interface FreewingIntent {
  targetPos: Vec3;
  intent: 'free-attack' | 'loop-receive' | 'high-hold' | 'cross-contest';
  isAttacking: boolean;
}

/** Stable per-commitment style draws (C2). Supplied → deterministic angle;
 *  absent → CURRENT inline rng() behavior (byte-identical for direct callers
 *  like freewingNavigate and the planner tests). The freeEndBias COIN draw is
 *  a read decision, NOT an angle/radius, so it stays on inline rng() in both
 *  modes — preserving the style-divergence test's call sequence. */
export interface RoleStyle {
  angle: number;
  radius: number;
}

export function freewingPolicy(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  rng: () => number,
  style?: RoleStyle,
): FreewingIntent {
  const sAngle = () => (style ? style.angle : rng());
  const bell = state.bell;
  const possession = match.possession === player.team;
  // ORIENTATION-CORRECT: the Freewing still plays HIGH/near-axis for the Free
  // (5-pt) curve game, but its DIRECTION is the ring THIS team attacks.
  const freeX = attackRingX(player.team);
  const defX = defendRingX(player.team);

  if (possession) {
    const bellDistToFree = Math.abs(bell.p.x - freeX);

    // Strong free-end bias: aggressively go for the Free ring.
    if (profile.freeEndBias > 0.55 || rng() < profile.freeEndBias) {
      if (bellDistToFree < REG.L * 0.35) {
        // Free-end attack: position near axis (cross-spin curve territory).
        const attackX = freeX + (freeX > 0 ? -20 : 20);
        const angle = Math.PI * (0.05 + sAngle() * 0.12);
        return {
          targetPos: {
            x: attackX,
            y: FREEWING_ATTACK_RADIUS * Math.cos(angle),
            z: FREEWING_ATTACK_RADIUS * Math.sin(angle),
          },
          intent: 'free-attack',
          isAttacking: true,
        };
      }

      // Loop receive: near axis, ready for a loop-setter feed.
      if (profile.loopPropensity > 0.4) {
        const loopX = bell.p.x + (freeX > 0 ? 80 : -80) * profile.loopPropensity;
        const angle = sAngle() * Math.PI * 0.1;
        return {
          targetPos: {
            x: loopX,
            y: FREEWING_ATTACK_RADIUS * Math.cos(angle),
            z: FREEWING_ATTACK_RADIUS * Math.sin(angle),
          },
          intent: 'loop-receive',
          isAttacking: true,
        };
      }
    }

    // Midfield high hold: wait for the pass.
    const midX = bell.p.x + (freeX > 0 ? 60 : -60) * 0.6;
    const angle = Math.PI * (0.15 + sAngle() * 0.15);
    return {
      targetPos: {
        x: midX,
        y: FREEWING_HOLD_RADIUS * Math.cos(angle),
        z: FREEWING_HOLD_RADIUS * Math.sin(angle),
      },
      intent: 'high-hold',
      isAttacking: false,
    };
  } else {
    // Defense: cross-spin contest position — force the thrower to go Faith-side.
    const oppHolder = state.players.find(p => p.id === bell.heldBy);
    if (oppHolder) {
      // Position between the holder and OUR defended ring, near axis.
      const freeOffset = defX > 0 ? 15 : -15;
      return {
        targetPos: {
          x: oppHolder.p.x + freeOffset,
          y: oppHolder.p.y * 0.5,
          z: oppHolder.p.z * 0.5,
        },
        intent: 'cross-contest',
        isAttacking: false,
      };
    }

    // Guard OUR OWN ring.
    const guardX = defX + (defX > 0 ? -30 : 30);
    return {
      targetPos: { x: guardX, y: FREEWING_HOLD_RADIUS, z: 0 },
      intent: 'cross-contest',
      isAttacking: false,
    };
  }
}

export function freewingNavigate(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  rng: () => number,
): GrapplePlan | null {
  const { targetPos } = freewingPolicy(player, state, match, profile, rng);
  return planGrapple(player, targetPos, state);
}

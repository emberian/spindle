// Anchor role policy — plays deep/heavy, near the rim end of the gradient.
// The Anchor almost never floats high; when caught axis-side the team is exposed.
// Decision tick: 10 Hz (every 6 sim-ticks at 60 Hz).
//
// Responsibilities:
//   - On offense: position deep to receive from Spinners; anchor the loop play.
//   - On defense: guard Faith end, pressure throwers going deep.
//   - Never voluntarily goes higher than ANCHOR_MAX_RADIUS from axis.

import type { Vec3 } from '../../sim/vec';
import type { PlayerSim, SimState, MatchState } from '../../sim/types';
import type { TeamProfile } from '../../league/teams';
import { REG, GATE_X } from '../../sim/RegConstants';
import { planGrapple, type GrapplePlan } from '../nav/GrapplePlanner';

// The Anchor wants to be at ~70% skin radius: deep but not skinned.
const ANCHOR_TARGET_RADIUS = REG.R * 0.72;
const ANCHOR_MAX_RADIUS = REG.R * 0.85;

export interface RoleIntent {
  targetPos: Vec3;
  intent: 'guard-faith' | 'deep-receive' | 'hold-depth' | 'support-throw';
}

/**
 * Compute the Anchor's desired position and intent.
 * Decision is pure — no side effects.
 */
export function anchorPolicy(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  _profile: TeamProfile,
  rng: () => number,
): RoleIntent {
  const bell = state.bell;
  const onOffense = bell.heldBy !== null &&
    state.players.find(p => p.id === bell.heldBy)?.team === player.team;
  const possession = match.possession === player.team;

  const faithX = match.faithEnd === '+x' ? GATE_X : -GATE_X;

  if (possession || onOffense) {
    // Offense: stay deep near the Faith end to receive passes or contest loops.
    const deepX = faithX + (faithX > 0 ? -80 : 80); // 80 m from the ring
    const angle = Math.PI * (0.4 + rng() * 0.2);
    const targetY = ANCHOR_TARGET_RADIUS * Math.cos(angle);
    const targetZ = ANCHOR_TARGET_RADIUS * Math.sin(angle);
    return {
      targetPos: { x: deepX, y: targetY, z: targetZ },
      intent: 'deep-receive',
    };
  } else {
    // Defense: guard the Faith end goal ring.
    const guardX = faithX + (faithX > 0 ? -25 : 25);
    const guardY = ANCHOR_TARGET_RADIUS * 0.6;
    return {
      targetPos: { x: guardX, y: guardY, z: 0 },
      intent: 'guard-faith',
    };
  }
}

/**
 * Compute a grapple navigation plan for the Anchor toward its role target.
 */
export function anchorNavigate(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  rng: () => number,
): GrapplePlan | null {
  const { targetPos } = anchorPolicy(player, state, match, profile, rng);

  // If already too deep (above radius limit), aim inward first.
  const r = Math.sqrt(player.p.y ** 2 + player.p.z ** 2);
  if (r > ANCHOR_MAX_RADIUS) {
    const inward: Vec3 = { x: player.p.x, y: player.p.y * 0.5, z: player.p.z * 0.5 };
    return planGrapple(player, inward, state);
  }

  return planGrapple(player, targetPos, state);
}

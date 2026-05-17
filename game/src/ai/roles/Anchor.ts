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
import { REG } from '../../sim/RegConstants';
import { planGrapple, type GrapplePlan } from '../nav/GrapplePlanner';
import { attackRingX, defendRingX } from '../Orientation';

// The Anchor wants to be at ~70% skin radius: deep but not skinned.
const ANCHOR_TARGET_RADIUS = REG.R * 0.72;
const ANCHOR_MAX_RADIUS = REG.R * 0.85;

export interface RoleIntent {
  targetPos: Vec3;
  intent: 'guard-faith' | 'deep-receive' | 'hold-depth' | 'support-throw';
}

/** Stable per-commitment style draws (C2). Supplied → deterministic angle;
 *  absent → CURRENT inline rng() behavior (byte-identical for direct callers
 *  like anchorNavigate and the planner tests). */
export interface RoleStyle {
  angle: number;
  radius: number;
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
  style?: RoleStyle,
): RoleIntent {
  const sAngle = () => (style ? style.angle : rng());
  const bell = state.bell;
  const onOffense = bell.heldBy !== null &&
    state.players.find(p => p.id === bell.heldBy)?.team === player.team;
  const possession = match.possession === player.team;

  // ORIENTATION-CORRECT: position relative to the ring THIS team attacks /
  // defends (home → +X attack, away → -X attack), never faithEnd.
  const atkX = attackRingX(player.team);
  const defX = defendRingX(player.team);

  if (possession || onOffense) {
    // Offense: stay deep near OUR attacking ring to receive / contest loops.
    const deepX = atkX + (atkX > 0 ? -80 : 80); // 80 m short of the ring
    const angle = Math.PI * (0.4 + sAngle() * 0.2);
    const targetY = ANCHOR_TARGET_RADIUS * Math.cos(angle);
    const targetZ = ANCHOR_TARGET_RADIUS * Math.sin(angle);
    return {
      targetPos: { x: deepX, y: targetY, z: targetZ },
      intent: 'deep-receive',
    };
  } else {
    // Defense: guard OUR OWN ring.
    const guardX = defX + (defX > 0 ? -25 : 25);
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

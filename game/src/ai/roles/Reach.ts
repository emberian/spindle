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
import { REG } from '../../sim/RegConstants';
import { rk4Step } from '../../sim/trajectory';
import type { PointState } from '../../sim/trajectory';
import { planGrapple, type GrapplePlan } from '../nav/GrapplePlanner';
import { attackRingX, defendRingX } from '../Orientation';

// Reach patrols at moderate radius, close to the goal ring.
const REACH_ATTACK_RADIUS = REG.R * 0.35;

// Defensive keeper geometry.
const KEEP_STANDOFF = 14;            // m in front of the ring, field-side
const KEEP_HORIZON = 6;              // s — predict an incoming shot this far
const APERTURE = REG.gateRadius;     // 8 m: a score only counts inside this

/**
 * Where will the bell's REAL Coriolis arc cross the defended ring plane, and
 * is it actually heading there? Axial x is inertial (accel.x = 0, canon), so
 * the crossing time is exact; (y, z) are integrated with the same rk4
 * predictor the bell uses. Returns null if the bell isn't inbound in time.
 */
function predictRingCrossing(
  bp: Vec3, bv: Vec3, defendX: number,
): { y: number; z: number; t: number } | null {
  if (Math.abs(bv.x) < 1e-4) return null;
  const t = (defendX - bp.x) / bv.x;
  if (t <= 0 || t > KEEP_HORIZON) return null;
  let st: PointState = { p: { ...bp }, v: { ...bv } };
  const H = 1 / 60;
  for (let acc = 0; acc < t; ) {
    const h = Math.min(H, t - acc);
    st = rk4Step(st, REG.omega, h);
    acc += h;
  }
  return { y: st.p.y, z: st.p.z, t };
}

export interface ReachIntent {
  targetPos: Vec3;
  /** Whether to fire line at the ring itself (only the Reach may do this). */
  clipRing: boolean;
  intent: 'defend-mouth' | 'pivot-ring' | 'contest-approach' | 'clear-zone';
}

/** Stable per-commitment style draws (C2). Supplied → deterministic angle;
 *  absent → CURRENT inline rng() behavior (byte-identical for direct callers
 *  like reachNavigate and the planner tests). */
export interface RoleStyle {
  angle: number;
  radius: number;
}

export function reachPolicy(
  player: PlayerSim,
  state: SimState,
  match: MatchState,
  _profile: TeamProfile,
  rng: () => number,
  style?: RoleStyle,
): ReachIntent {
  const sAngle = () => (style ? style.angle : rng());
  const bell = state.bell;
  const possession = match.possession === player.team;
  // ORIENTATION-CORRECT: defend OUR ring, contest near the ring WE attack.
  const ourAttackX = attackRingX(player.team);
  const ourDefendX = defendRingX(player.team);

  if (!possession) {
    // Defense: actually MAN the scoring lane so a thrown bell can't just
    // sail through the hoop. Predict where the bell's Coriolis arc crosses
    // our ring plane and put the keeper's body THERE, inside the aperture,
    // a short standoff in front — so it intercepts the shot (and trips the
    // contest) instead of floating uselessly outside the scoring hole.
    const defendX = ourDefendX;
    const fieldSign = defendX > 0 ? -1 : 1; // toward midfield from the ring
    const cross = predictRingCrossing(bell.p, bell.v, defendX);

    if (cross) {
      const r = Math.hypot(cross.y, cross.z);
      // A shot only scores inside APERTURE. If the predicted crossing is
      // wildly wide it's no threat — hold a centred ready post. If it's a
      // real threat, sit right on its line, clamped into the aperture.
      const threat = r < APERTURE * 2.2;
      const k = r > 1e-3 ? Math.min(1, (APERTURE * 0.85) / r) : 0;
      const gy = threat ? cross.y * k : 0;
      const gz = threat ? cross.z * k : 0;

      if (cross.t < 0.9) {
        // Imminent: pivot ON the ring at the crossing angle (extended reach).
        const a = Math.atan2(cross.z, cross.y);
        return {
          targetPos: {
            x: defendX,
            y: APERTURE * Math.cos(a),
            z: APERTURE * Math.sin(a),
          },
          clipRing: true,
          intent: 'pivot-ring',
        };
      }
      return {
        targetPos: { x: defendX + fieldSign * KEEP_STANDOFF, y: gy, z: gz },
        clipRing: false,
        intent: 'contest-approach',
      };
    }

    // No inbound shot: hold a centred ready post in the mouth (near the
    // axis, in the aperture) — not floating outside it.
    return {
      targetPos: {
        x: defendX + fieldSign * KEEP_STANDOFF,
        y: APERTURE * 0.4 * Math.cos(Math.PI * sAngle()),
        z: APERTURE * 0.4 * Math.sin(Math.PI * sAngle()),
      },
      clipRing: false,
      intent: 'defend-mouth',
    };
  } else {
    // Offense: position near the ring WE attack to contest/clean up.
    const attackX = ourAttackX + (ourAttackX > 0 ? -20 : 20);
    const angle = Math.PI * (0.2 + sAngle() * 0.2);
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

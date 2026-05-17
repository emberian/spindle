// GateSolve — does a real scoring throw exist from here toward MY ring?
//
// Physics (canon): axial (x) motion is INERTIAL — constant vx. So a throw from
// x0 reaches the attacking ring at x = ringX after t = |ringX - x0| / |vx|.
// During that time the cross-section ζ = y + i·z evolves under the roulette
// ODE  ζ̈ = ω²ζ − 2iω ζ̇  whose EXACT solution (see sim/trajectory.ts → analytic
// and rig-core/src/analysis.rs) is
//
//     ζ(t) = (ζ₀ + v_rot·t)·e^{−iωt},     v_rot = (vy − ω·z₀) + i(vz + ω·y₀)
//
// Because e^{−iωt} is a pure rotation, the cross-section RADIUS at any time is
//
//     |ζ(t)| = |ζ₀ + v_rot·t|
//
// — the modulus of a *linear* function of t.  Therefore the ring-plane radius
// is ZEROED exactly when the rotating-frame launch velocity satisfies
//
//     v_rot = −ζ₀ / t            (t = dx / |vx|, the inertial flight time)
//
// which, undoing the v_rot definition, gives a CLOSED-FORM world launch:
//
//     vy = −y₀/t + ω·z₀
//     vz = −z₀/t − ω·y₀
//
// This is the analytic gate solution the previous grid-search version missed:
// the old code chose a perpendicular nudge that cancelled |ζ₀| *linearly* but
// ignored the centrifugal sweep ω·|ζ₀| baked into v_rot, so its throws skinned
// out / sailed wide and (per analysis.rs) almost never threaded the ring.
//
// We sweep only the axial speed vx (which sets the flight time); for each vx
// the cross-plane launch is the closed form above. We keep the skin-safe
// solution whose arrival radius is smallest. Deterministic — no RNG.

import type { Vec3 } from '../../sim/vec';
import type { TeamSide } from '../../sim/types';
import { rk4Step, type PointState } from '../../sim/trajectory';
import { REG, GATE_X } from '../../sim/RegConstants';
import { attackRingX, attackSign } from '../Orientation';

// The sim launches a thrown bell at  v_bell = thrower.v + dir·speed  with
// THROW_MIN ≤ speed ≤ THROW_MAX (rig-core/src/sim_world.rs). To realise our
// desired free-bell velocity v0 EXACTLY, the caller must aim along
// (v0 − thrower.v) and charge so the release speed equals |v0 − thrower.v|.
// We surface that here so the solver only returns physically achievable shots.
export const THROW_MIN_SPEED = 9.0;
export const THROW_MAX_SPEED = 34.0;

export interface GateSolution {
  /** Launch velocity (world) that threads the attacking ring. */
  v0: Vec3;
  /** Throw vector the player must impart: v0 − thrower.v (aim = its unit). */
  throwVec: Vec3;
  /** Release speed required = |throwVec| (already ≤ THROW_MAX_SPEED). */
  releaseSpeed: number;
  /** Predicted arrival radius at the ring plane (m). */
  arriveRho: number;
  /** Flight time to the ring (s). */
  flightTime: number;
  /** Axial speed component used. */
  vx: number;
}

const SOLVE_H = 1 / 240; // match the canonical RIG sim timestep

/**
 * Verify a candidate launch in the SAME rk4 integrator the sim uses: return
 * the cross-section radius at the moment x crosses `ringX`, and the max radius
 * over the flight (skin-safety). Returns null if it never reaches the plane.
 */
function verifyAtRing(
  p0: Vec3,
  v0: Vec3,
  omega: number,
  ringX: number,
  sign: 1 | -1,
  tFlight: number,
): { rho: number; maxR: number; t: number } | null {
  if (sign * v0.x <= 0.01) return null;
  let s: PointState = { p: { ...p0 }, v: { ...v0 } };
  let prevP = { ...p0 };
  let maxR = Math.sqrt(p0.y * p0.y + p0.z * p0.z);
  const maxSteps = Math.ceil(tFlight * 1.4 * 240) + 8;
  for (let i = 1; i <= maxSteps; i++) {
    const prev = s;
    s = rk4Step(s, omega, SOLVE_H);
    const r = Math.sqrt(s.p.y * s.p.y + s.p.z * s.p.z);
    if (r > maxR) maxR = r;
    const crossed =
      (sign > 0 && prevP.x < ringX && s.p.x >= ringX) ||
      (sign < 0 && prevP.x > ringX && s.p.x <= ringX);
    if (crossed) {
      // Linear-interpolate to the exact crossing for an accurate rho.
      const denom = s.p.x - prev.p.x;
      const frac =
        Math.abs(denom) > 1e-12 ? (ringX - prev.p.x) / denom : 0.5;
      const cy = prev.p.y + frac * (s.p.y - prev.p.y);
      const cz = prev.p.z + frac * (s.p.z - prev.p.z);
      return { rho: Math.sqrt(cy * cy + cz * cz), maxR, t: i * SOLVE_H };
    }
    prevP = s.p;
  }
  return null;
}

/**
 * Search for a launch velocity from `from` that threads the team's attacking
 * ring. For each candidate axial speed the cross-plane launch is the exact
 * closed-form that nulls the ring-plane radius (v_rot = −ζ₀/t). Returns the
 * best skin-safe ring solution, or null if none reaches the ring in a
 * believable flight.
 *
 * Deterministic — no RNG. Symmetric in team (uses Orientation only).
 */
export function solveGateThrow(
  from: Vec3,
  team: TeamSide,
  omega: number,
  baseSpeed: number,
  _loopPropensity: number,
  /** Thrower's current velocity (the sim ADDS it to the launch — we must
   *  cancel it). Defaults to zero for callers that don't supply it. */
  fromVel: Vec3 = { x: 0, y: 0, z: 0 },
): GateSolution | null {
  const ringX = attackRingX(team);
  const sign = attackSign(team);
  const toRingX = ringX - from.x;
  const dx = Math.abs(toRingX);

  // Need real axial separation to fly through the plane; if we're basically
  // on the ring plane there is no meaningful gate throw to solve.
  if (dx < 6) return null;
  // Too far for any believable flight (whole field is 640 m).
  if (dx > REG.L * 1.05) return null;

  const y0 = from.y;
  const z0 = from.z;

  // Axial-speed sweep. Slower → longer flight → more Coriolis winding (the
  // loop/curl scoring regime); faster → flatter Fall/Rise. All of them thread
  // the ring by construction; we keep the smallest-rho skin-safe one whose
  // flight comfortably finishes before the dead-ball timeout.
  //
  // HARD CONSTRAINT: the sim kills a free bell after MAX_FREE_TICKS = 7200
  // ticks = 30 s (rig-core/src/sim_world.rs). A throw that arrives later than
  // that is a dead ball and never scores — so we cap flight time well under
  // it and SCORE candidates to prefer shorter (more robust) flights.
  const MAX_FLIGHT = 24; // s — safe margin below the 30 s dead-ball limit
  // Axial speed must keep the REQUIRED release speed |v0 − thrower.v| within
  // the sim's [THROW_MIN, THROW_MAX] band, else the shot is unthrowable.
  // Slower axial speeds → longer flight → SMALLER perpendicular launch
  // (v_rot = −ζ₀/t shrinks as t grows) → the bell stays near the axis (the
  // canon "calm") instead of bulging to the skin. Include slow speeds so the
  // elegant near-axial Loop solutions are in the candidate set.
  const speeds = [10, 12, 14, 16, 18, 20, 22, 25, 28, 30, 32, baseSpeed].filter(
    s => s > 6 && s < THROW_MAX_SPEED + 8,
  );

  let best: GateSolution | null = null;
  let bestCost = Infinity;

  for (const vx of speeds) {
    const t = dx / vx;
    // Keep flight time physically sane: long enough to be a real arc, and
    // safely shorter than the dead-ball timeout.
    if (t < 1.0 || t > MAX_FLIGHT) continue;

    // EXACT closed-form cross-plane launch that zeroes |ζ| at the ring plane.
    //   v_rot = −ζ₀/t  ⇒  vy = −y₀/t + ω·z₀ ,  vz = −z₀/t − ω·y₀
    const v0: Vec3 = {
      x: Math.sign(toRingX) * vx,
      y: -y0 / t + omega * z0,
      z: -z0 / t - omega * y0,
    };

    // The player must impart (v0 − thrower.v); the sim adds thrower.v back.
    const throwVec: Vec3 = {
      x: v0.x - fromVel.x,
      y: v0.y - fromVel.y,
      z: v0.z - fromVel.z,
    };
    const releaseSpeed = Math.sqrt(
      throwVec.x * throwVec.x +
        throwVec.y * throwVec.y +
        throwVec.z * throwVec.z,
    );
    // Unthrowable by the sim's release-speed band → skip this candidate.
    if (releaseSpeed < THROW_MIN_SPEED || releaseSpeed > THROW_MAX_SPEED) {
      continue;
    }

    const hit = verifyAtRing(from, v0, omega, ringX, sign, t);
    if (!hit) continue;
    // Reject if the flight ever touches the skin.
    if (hit.maxR >= REG.R) continue;

    // Cost: thread accuracy first, then strongly prefer a trajectory that
    // STAYS IN THE CALM (small max cross-section radius) rather than one that
    // flings out near the skin — RIG is "the game played in the calm", the
    // near-axial Loop is the canon-elegant shot, and a skin-hugging bell read
    // badly on screen. All candidates thread the ring (rho≈0) by construction,
    // so maxR is the real discriminator; a light flight-time term keeps it
    // safely under the dead-ball timeout among similar-radius options.
    const cost = hit.rho + hit.maxR * 0.6 + hit.t * 0.05;
    if (cost < bestCost) {
      bestCost = cost;
      best = {
        v0,
        throwVec,
        releaseSpeed,
        arriveRho: hit.rho,
        flightTime: hit.t,
        vx,
      };
    }
  }

  // Only return if it actually threads the ring (rho ≤ gate radius). The
  // closed form makes this essentially always true for a reachable dx; the
  // tolerance only guards rk4 integration error.
  if (best && best.arriveRho <= REG.gateRadius) return best;
  return null;
}

export const GATE_RADIUS = REG.gateRadius;
export const GATE_HALF = GATE_X;

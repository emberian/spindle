// Coriolis lead-pass solver via Newton iteration on trajectory.ts predictor.
// Given a thrower position, bell velocity, a moving receiver, and omega,
// find the initial velocity v0 such that the bell arrives at the receiver's
// predicted position. Returns null if no solution found within `maxIter`.

import type { Vec3 } from '../../sim/vec';
import { vsub, vscale, vlen, vnorm } from '../../sim/vec';
import { rk4Step } from '../../sim/trajectory';
import type { PointState } from '../../sim/trajectory';

export interface LeadResult {
  /** Initial bell velocity to throw from `throwPos` */
  v0: Vec3;
  /** Predicted receiver intercept position */
  intercept: Vec3;
  /** Estimated flight time in seconds */
  flightTime: number;
}

// Integration step for lead solving (coarser than sim — good enough for AI).
const SOLVE_H = 1 / 30;
const MAX_FLIGHT_STEPS = 300; // 10 s max
const NEWTON_ITERS = 8;
const CONVERGE_SQ = 0.25; // 0.5 m tolerance

/**
 * Predict receiver position at time t via simple linear extrapolation.
 * The receiver's acceleration in the calm is not modelled here — we treat
 * them as moving at constant velocity from their current state.
 */
function extrapolateReceiver(p: Vec3, v: Vec3, t: number): Vec3 {
  return { x: p.x + v.x * t, y: p.y + v.y * t, z: p.z + v.z * t };
}

/**
 * Solve for the throw velocity that leads a moving receiver.
 *
 * Algorithm:
 *   1. Estimate flight time from straight-line distance / throwSpeed.
 *   2. Compute receiver intercept position at that time.
 *   3. Build initial v0 aimed at that intercept.
 *   4. Propagate the bell; measure error at receiver.
 *   5. Newton-iterate: shift aim point opposite to error.
 *
 * Deterministic — no RNG involved.
 */
export function solveLeadVelocity(
  throwPos: Vec3,
  throwSpeed: number,
  receiverPos: Vec3,
  receiverVel: Vec3,
  omega: number,
): LeadResult | null {
  // Initial guess: aim straight at receiver, estimate time by distance / speed.
  let aimPt: Vec3 = { ...receiverPos };
  let bestError = Infinity;
  let bestResult: LeadResult | null = null;

  for (let iter = 0; iter < NEWTON_ITERS; iter++) {
    // Build direction from throwPos to aimPt.
    const raw = vsub(aimPt, throwPos);
    const dist = vlen(raw);
    if (dist < 0.01) return null;
    const dir = vnorm(raw);
    const v0: Vec3 = vscale(dir, throwSpeed);

    // Estimate flight time: dist / speed (straight line, fast enough for iteration).
    const estTime = dist / throwSpeed;
    const steps = Math.min(MAX_FLIGHT_STEPS, Math.ceil(estTime / SOLVE_H) + 2);

    // Find the tick where the bell is closest to the predicted intercept.
    let s: PointState = { p: { ...throwPos }, v: { ...v0 } };
    let closestDist = Infinity;
    let closestPos: Vec3 = { ...throwPos };
    let closestTime = 0;

    for (let step = 1; step <= steps; step++) {
      s = rk4Step(s, omega, SOLVE_H);
      const t = step * SOLVE_H;
      const rcv = extrapolateReceiver(receiverPos, receiverVel, t);
      const err = vsub(s.p, rcv);
      const d = vlen(err);
      if (d < closestDist) {
        closestDist = d;
        closestPos = { ...s.p };
        closestTime = t;
      }
      if (d < Math.sqrt(CONVERGE_SQ)) {
        // Close enough — accept.
        if (d < bestError) {
          bestError = d;
          const intercept = extrapolateReceiver(receiverPos, receiverVel, t);
          bestResult = { v0, intercept, flightTime: t };
        }
        break;
      }
    }

    if (bestError < Math.sqrt(CONVERGE_SQ)) break;

    // Record best regardless of convergence.
    if (closestDist < bestError) {
      bestError = closestDist;
      const intercept = extrapolateReceiver(receiverPos, receiverVel, closestTime);
      bestResult = { v0, intercept, flightTime: closestTime };
    }

    // Newton correction: shift aimPt in opposite direction to position error.
    const rcvAtClose = extrapolateReceiver(receiverPos, receiverVel, closestTime);
    const errVec = vsub(closestPos, rcvAtClose);
    aimPt = vsub(aimPt, vscale(errVec, 0.65));
  }

  return bestResult;
}

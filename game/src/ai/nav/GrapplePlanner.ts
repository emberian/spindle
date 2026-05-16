// GrapplePlanner — AI movement via grapple-only locomotion.
// In the calm there is no walking. Players change vector only by:
//   1. Firing the line at an anchor (spar, skin, teammate) and reeling.
//   2. Pushing off something solid.
//
// Strategy: greedy ≤3-ply lookahead over candidate anchors.
// Each candidate is scored on how much closer to the target it gets the
// player after one swing+reel cycle. The best anchor is selected each tick.
//
// IMPORTANT: the skin is a VALID anchor for opponents (and used by AI to
// swing off), but the planner must NEVER target the skin as a self-anchor
// when the player is already outside the safe-zone or when the trajectory
// would push them through the skin (grounding).
//
// Candidate anchors:
//   - Spars along the spin axis (every SPAR_SPACING meters axially, at y=z=0)
//   - The skin (at radius R) — only if not already near skin and target is
//     axis-side of the player
//   - Teammates (their current positions, invMass ≠ 0)
//
// Output: one PlayerInput fragment (only the grapple fields are set).

import type { Vec3 } from '../../sim/vec';
import { vsub, vadd, vscale, vlen, vnorm, vdot, v3 } from '../../sim/vec';
import type { PlayerInput, PlayerSim, SimState } from '../../sim/types';
import type { PointState } from '../../sim/trajectory';
import { rk4Step } from '../../sim/trajectory';
import { REG } from '../../sim/RegConstants';

// ── Spar geometry ─────────────────────────────────────────────────────────────
const SPAR_SPACING = 40; // m along X axis
const SPAR_COUNT = Math.ceil(REG.L / SPAR_SPACING) + 1;

/** All spar positions: on the spin axis (y=0, z=0) at fixed X intervals. */
export function sparPositions(): Vec3[] {
  const out: Vec3[] = [];
  const start = -REG.L / 2;
  for (let i = 0; i < SPAR_COUNT; i++) {
    out.push({ x: start + i * SPAR_SPACING, y: 0, z: 0 });
  }
  return out;
}

const SPARS = sparPositions();

// ── Simulation helpers ────────────────────────────────────────────────────────
const PLAN_H = 1 / 15; // planning integration step
const PLAN_STEPS = 45;  // simulate 3 s of flight
const REEL_RATE = 14;   // m/s (mirrors Grapple.ts canon)
const TETHER_MIN = 3;
const SKIN_BUFFER = 4;  // m inside skin — don't fire at skin if within this

export interface GrapplePlan {
  /** The world-space anchor point to fire the line at. */
  anchorPos: Vec3;
  /** Whether to reel in (-1) or let swing (0) after firing. */
  reel: -1 | 0;
  /** Estimated closest approach to target (m) after PLAN_STEPS ticks. */
  projectedDist: number;
  /** true if the anchor is a spar, false = skin / player */
  isSpar: boolean;
}

/**
 * Simulate one player grapple cycle: fire line at anchor, reel in, return
 * the closest distance to `target` achieved over `steps` ticks.
 */
function simulateGrappleSwing(
  pos: Vec3,
  vel: Vec3,
  anchorPos: Vec3,
  omega: number,
  reel: -1 | 0,
  target: Vec3,
  steps: number,
  h: number,
): number {
  let p = { ...pos };
  let v = { ...vel };
  let restLen = vlen(vsub(p, anchorPos));
  let closestDist = vlen(vsub(p, target));

  for (let i = 0; i < steps; i++) {
    // Coriolis free-flight step.
    const s: PointState = rk4Step({ p, v }, omega, h);
    p = s.p;
    v = s.v;

    // Grapple constraint: remove separating radial velocity, preserve tangential.
    const d = vsub(p, anchorPos);
    const len = vlen(d);
    if (len > 1e-6) {
      const n = vscale(d, 1 / len);
      const vRad = vdot(v, n);
      if (len >= restLen && vRad > 0) {
        // Taut — cancel outward radial component.
        v = vsub(v, vscale(n, vRad));
      }
      // Reel: shorten restLen, conserve angular momentum.
      if (reel === -1 && len >= TETHER_MIN) {
        const target_len = Math.max(TETHER_MIN, restLen - REEL_RATE * h);
        if (target_len < restLen) {
          const vRadCurrent = vdot(v, n);
          const scale = restLen > 1e-6 ? restLen / target_len : 1;
          // Scale tangential component.
          const vTan = vsub(v, vscale(n, vRadCurrent));
          v = vadd(vscale(vTan, scale), vscale(n, vRadCurrent));
          restLen = target_len;
        }
      }
    }

    const dist = vlen(vsub(p, target));
    if (dist < closestDist) closestDist = dist;
  }

  return closestDist;
}

/**
 * Main planner entry: choose the best grapple anchor to approach `target`.
 *
 * Returns a GrapplePlan (anchorPos + reel), or null if already close enough.
 */
export function planGrapple(
  player: PlayerSim,
  target: Vec3,
  state: SimState,
): GrapplePlan | null {
  const pos = player.p;
  const vel = player.v;
  const omega = state.omega;

  // Already close enough — no grapple needed.
  const directDist = vlen(vsub(pos, target));
  if (directDist < 3.0) return null;

  const candidates: GrapplePlan[] = [];

  // ── Candidate 1: Spar anchors ──────────────────────────────────────────────
  for (const spar of SPARS) {
    // Only consider spars within useful range.
    const sparDist = vlen(vsub(pos, spar));
    if (sparDist < 2 || sparDist > 80) continue;

    // Don't fire back the way we came (would reduce progress).
    const toSpar = vnorm(vsub(spar, pos));
    const toTarget = vnorm(vsub(target, pos));
    if (vdot(toSpar, toTarget) < -0.7) continue;

    const projDist = simulateGrappleSwing(pos, vel, spar, omega, -1, target, PLAN_STEPS, PLAN_H);
    candidates.push({ anchorPos: spar, reel: -1, projectedDist: projDist, isSpar: true });
  }

  // ── Candidate 2: Skin anchor ───────────────────────────────────────────────
  // Use the skin point closest to the player (at radius R in the (y,z) plane).
  const playerRadius = Math.sqrt(pos.y * pos.y + pos.z * pos.z);
  const isTooCloseSkin = playerRadius > REG.R - SKIN_BUFFER;
  const targetRadius = Math.sqrt(target.y * target.y + target.z * target.z);
  // Only use skin if target is axis-side of the player (we'd swing inward).
  const skinUseful = !isTooCloseSkin && targetRadius < playerRadius;

  if (skinUseful) {
    const yzLen = playerRadius > 1e-6 ? playerRadius : 1;
    const skinPoint: Vec3 = {
      x: pos.x,
      y: (pos.y / yzLen) * REG.R,
      z: (pos.z / yzLen) * REG.R,
    };
    const projDist = simulateGrappleSwing(pos, vel, skinPoint, omega, 0, target, PLAN_STEPS, PLAN_H);
    candidates.push({ anchorPos: skinPoint, reel: 0, projectedDist: projDist, isSpar: false });
  }

  // ── Candidate 3: Teammate anchors ─────────────────────────────────────────
  for (const p of state.players) {
    if (p.id === player.id) continue;
    if (p.team !== player.team) continue;
    const tmDist = vlen(vsub(pos, p.p));
    if (tmDist < 3 || tmDist > 60) continue;

    const toTm = vnorm(vsub(p.p, pos));
    const toTarget = vnorm(vsub(target, pos));
    if (vdot(toTm, toTarget) < -0.5) continue;

    const projDist = simulateGrappleSwing(pos, vel, p.p, omega, -1, target, PLAN_STEPS, PLAN_H);
    candidates.push({ anchorPos: p.p, reel: -1, projectedDist: projDist, isSpar: false });
  }

  if (candidates.length === 0) {
    // Fallback: nearest spar.
    let nearestSpar = SPARS[0];
    let nearestDist = Infinity;
    for (const spar of SPARS) {
      const d = vlen(vsub(pos, spar));
      if (d < nearestDist) { nearestDist = d; nearestSpar = spar; }
    }
    return { anchorPos: nearestSpar, reel: -1, projectedDist: directDist, isSpar: true };
  }

  // Pick candidate with smallest projected distance to target.
  candidates.sort((a, b) => a.projectedDist - b.projectedDist);
  return candidates[0];
}

/**
 * Convert a GrapplePlan into PlayerInput fields.
 * The caller merges other fields (aim, throw*, etc.) around this.
 */
export function planToInput(plan: GrapplePlan | null, aim: Vec3): Partial<PlayerInput> {
  if (!plan) {
    return {
      aim,
      fireLineAt: null,
      reel: 0,
      release: false,
      pushoff: false,
      throwCharge: 0,
      throwReleased: false,
      throwSpin: 0,
      thrumbler: v3(),
    };
  }

  return {
    aim,
    fireLineAt: plan.anchorPos,
    reel: plan.reel,
    release: false,
    pushoff: false,
    throwCharge: 0,
    throwReleased: false,
    throwSpin: 0,
    thrumbler: v3(),
  };
}

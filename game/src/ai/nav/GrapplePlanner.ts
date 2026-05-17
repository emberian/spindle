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

// Nav-anchor hysteresis: a freshly-considered anchor must beat the currently
// committed ("sticky") anchor's cost by this margin before we switch to it.
// Mirrors the throw-target SWITCH_MARGIN pattern in RiggerAI. Without this,
// two near-equal-cost anchors flip frame-to-frame and the rigger is yanked
// (fireLineAt + reel toggling). Tuned: ~6 m of projected-distance cost — large
// enough to kill dithering between adjacent spars, small enough that a
// genuinely better anchor (closer approach, defender avoidance) still wins.
const ANCHOR_SWITCH_MARGIN = 6;

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
 * Defender-aware swing simulation: like simulateGrappleSwing but also returns
 * the closest the swung path comes to any opponent (so the planner can avoid
 * stranding the player on top of a defender or swinging through coverage).
 */
function simulateGrappleSwingDef(
  pos: Vec3,
  vel: Vec3,
  anchorPos: Vec3,
  omega: number,
  reel: -1 | 0,
  target: Vec3,
  steps: number,
  h: number,
  opponents: { x: number; y: number; z: number }[],
): { closestDist: number; minOppDist: number } {
  let p = { ...pos };
  let v = { ...vel };
  let restLen = vlen(vsub(p, anchorPos));
  let closestDist = vlen(vsub(p, target));
  let minOppDist = Infinity;

  for (let i = 0; i < steps; i++) {
    const s: PointState = rk4Step({ p, v }, omega, h);
    p = s.p;
    v = s.v;

    const d = vsub(p, anchorPos);
    const len = vlen(d);
    if (len > 1e-6) {
      const n = vscale(d, 1 / len);
      const vRad = vdot(v, n);
      if (len >= restLen && vRad > 0) v = vsub(v, vscale(n, vRad));
      if (reel === -1 && len >= TETHER_MIN) {
        const target_len = Math.max(TETHER_MIN, restLen - REEL_RATE * h);
        if (target_len < restLen) {
          const vRadCurrent = vdot(v, n);
          const scale = restLen > 1e-6 ? restLen / target_len : 1;
          const vTan = vsub(v, vscale(n, vRadCurrent));
          v = vadd(vscale(vTan, scale), vscale(n, vRadCurrent));
          restLen = target_len;
        }
      }
    }

    const dist = vlen(vsub(p, target));
    if (dist < closestDist) closestDist = dist;

    // Sample opponent proximity every few steps (cheap).
    if ((i & 3) === 0) {
      for (const o of opponents) {
        const od = vlen(vsub(p, o));
        if (od < minOppDist) minOppDist = od;
      }
    }
  }

  return { closestDist, minOppDist };
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
  avoidDefenders = true,
  sticky?: { pos: Vec3; reel: -1 | 0 } | null,
): GrapplePlan | null {
  const pos = player.p;
  const vel = player.v;
  const omega = state.omega;

  // Already close enough — no grapple needed.
  const directDist = vlen(vsub(pos, target));
  if (directDist < 3.0) return null;

  // Opponent positions for defender-aware path scoring. We avoid swings whose
  // path strands us inside DEFENDER_DANGER of an opponent.
  const opponents: Vec3[] = avoidDefenders
    ? state.players.filter(p => p.team !== player.team).map(p => p.p)
    : [];
  const DEFENDER_DANGER = 6; // m — within this is "swung into coverage"

  // Cost wrapper: projected distance to target + penalty for grazing a
  // defender. Keeps the planner from sailing the player into a mark.
  const scorePlan = (
    anchor: Vec3,
    reel: -1 | 0,
  ): { projectedDist: number; cost: number } => {
    if (opponents.length === 0) {
      const d = simulateGrappleSwing(
        pos, vel, anchor, omega, reel, target, PLAN_STEPS, PLAN_H,
      );
      return { projectedDist: d, cost: d };
    }
    const r = simulateGrappleSwingDef(
      pos, vel, anchor, omega, reel, target, PLAN_STEPS, PLAN_H, opponents,
    );
    const danger =
      r.minOppDist < DEFENDER_DANGER
        ? (DEFENDER_DANGER - r.minOppDist) * 3.5
        : 0;
    return { projectedDist: r.closestDist, cost: r.closestDist + danger };
  };

  const candidates: (GrapplePlan & { cost: number })[] = [];

  // ── Candidate 1: Spar anchors ──────────────────────────────────────────────
  for (const spar of SPARS) {
    // Only consider spars within useful range.
    const sparDist = vlen(vsub(pos, spar));
    if (sparDist < 2 || sparDist > 80) continue;

    // Don't fire back the way we came (would reduce progress).
    const toSpar = vnorm(vsub(spar, pos));
    const toTarget = vnorm(vsub(target, pos));
    if (vdot(toSpar, toTarget) < -0.7) continue;

    const sc = scorePlan(spar, -1);
    candidates.push({ anchorPos: spar, reel: -1, projectedDist: sc.projectedDist, isSpar: true, cost: sc.cost });
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
    const sc = scorePlan(skinPoint, 0);
    candidates.push({ anchorPos: skinPoint, reel: 0, projectedDist: sc.projectedDist, isSpar: false, cost: sc.cost });
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

    const sc = scorePlan(p.p, -1);
    candidates.push({ anchorPos: p.p, reel: -1, projectedDist: sc.projectedDist, isSpar: false, cost: sc.cost });
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

  // Pick the candidate with the smallest *cost* (projected distance plus a
  // penalty for swinging through / stranding near a defender). Tie-break on
  // raw projected distance, then anchor x for full determinism.
  candidates.sort((a, b) => {
    if (a.cost !== b.cost) return a.cost - b.cost;
    if (a.projectedDist !== b.projectedDist) return a.projectedDist - b.projectedDist;
    return a.anchorPos.x - b.anchorPos.x;
  });
  const best = candidates[0];

  // ── Nav-anchor hysteresis ─────────────────────────────────────────────────
  // If the caller passed the previously-committed anchor and it is STILL a
  // valid candidate under the SAME filters used to admit fresh candidates,
  // re-score it with the SAME scorer and KEEP it unless the new best beats it
  // by ANCHOR_SWITCH_MARGIN. This mirrors the throw-target hysteresis in
  // RiggerAI (decideThrow ~649-667) and is deterministic: no rng, pure
  // geometry / replayed-physics scoring of the injected state.
  if (sticky) {
    const sPos = sticky.pos;
    const stickyDist = vlen(vsub(pos, sPos));
    // Determine whether the sticky anchor is the skin point (high radius in
    // the y,z plane and matching the player's current skin projection) vs a
    // spar/teammate-style point — so we apply the matching admission filter.
    const sRadius = Math.sqrt(sPos.y * sPos.y + sPos.z * sPos.z);
    const isSkinSticky = sRadius > REG.R * 0.85;

    let stickyValid = false;
    if (isSkinSticky) {
      // Same gate as the skin candidate above.
      stickyValid = skinUseful && sticky.reel === 0;
    } else {
      // Same gate as spar / teammate candidates: in useful range and not
      // fired back the way we came relative to the target.
      const toAnchor = vnorm(vsub(sPos, pos));
      const toTarget = vnorm(vsub(target, pos));
      stickyValid =
        stickyDist >= 2 &&
        stickyDist <= 80 &&
        vdot(toAnchor, toTarget) >= -0.7 &&
        sticky.reel === -1;
    }

    if (stickyValid) {
      const sc = scorePlan(sPos, sticky.reel);
      // Keep sticky unless the fresh best is clearly better.
      if (best.cost >= sc.cost - ANCHOR_SWITCH_MARGIN) {
        return {
          anchorPos: sPos,
          reel: sticky.reel,
          projectedDist: sc.projectedDist,
          // isSpar only affects callers' cosmetics; reflect geometry.
          isSpar: !isSkinSticky && sRadius < 1,
        };
      }
    }
  }

  return {
    anchorPos: best.anchorPos,
    reel: best.reel,
    projectedDist: best.projectedDist,
    isSpar: best.isSpar,
  };
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

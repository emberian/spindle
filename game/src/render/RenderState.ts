// Render-only interpolation. Pure function: given the previous and current
// fixed-step sim snapshots and a fractional alpha in [0,1], produce a blended
// view in the SAME widened shape WasmSim.snapshot() returns. This NEVER feeds
// back into the deterministic sim — it only smooths what the renderer sees,
// killing the accel spikes that came from feeding the raw 240 Hz snapshot
// straight to the cel rig.
//
// Continuous numeric fields (positions, velocities, chime, dvBudget, …) are
// lerped; the bell orientation quaternion is slerped (local pure helper, since
// sim/vec.ts is frozen and has no slerp). ALL discrete / identity fields are
// taken verbatim from `cur` so the gameplay-truth (tick, possession, scores,
// roster identity, line topology, loop tier) is exactly the latest sim state.
//
// Teleport guard: re-grips, set resets and any non-1-tick gap snap the
// affected entity to `cur` so we never glide a body across the cylinder.

import type { SimState, PlayerSim, BellState } from '../sim/types';
import type { Vec3, Quat } from '../sim/vec';

/** The widened snapshot shape WasmSim.snapshot() returns. */
export type RenderView = SimState & {
  loopTurn: number;
  loopUntouched: boolean;
  loopTier: 'loop' | 'curl' | 'none';
};

// Any entity that moved more than this (metres) between prev→cur is treated as
// a discontinuity (catch/throw/reset relocation) and snapped, not glided.
const TELEPORT_DIST = 8;
const TELEPORT_DIST_SQ = TELEPORT_DIST * TELEPORT_DIST;

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpV3(a: Vec3, b: Vec3, t: number): Vec3 {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), z: lerp(a.z, b.z, t) };
}

function distSq(a: Vec3, b: Vec3): number {
  const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  return dx * dx + dy * dy + dz * dz;
}

/**
 * Local pure quaternion slerp (sim/vec.ts is frozen and has none). Operates on
 * the {x,y,z,w} convention used everywhere in the codebase. Falls back to nlerp
 * for nearly-parallel inputs to stay numerically stable.
 */
function slerpQuat(a: Quat, b: Quat, t: number): Quat {
  let bx = b.x, by = b.y, bz = b.z, bw = b.w;
  let cos = a.x * bx + a.y * by + a.z * bz + a.w * bw;
  // Take the shorter arc.
  if (cos < 0) {
    cos = -cos;
    bx = -bx; by = -by; bz = -bz; bw = -bw;
  }
  let s0: number, s1: number;
  if (cos > 0.9995) {
    // Nearly identical — linear interpolate then normalise.
    s0 = 1 - t;
    s1 = t;
  } else {
    const theta = Math.acos(cos);
    const sinTheta = Math.sin(theta);
    s0 = Math.sin((1 - t) * theta) / sinTheta;
    s1 = Math.sin(t * theta) / sinTheta;
  }
  const x = s0 * a.x + s1 * bx;
  const y = s0 * a.y + s1 * by;
  const z = s0 * a.z + s1 * bz;
  const w = s0 * a.w + s1 * bw;
  const l = Math.sqrt(x * x + y * y + z * z + w * w) || 1;
  return { x: x / l, y: y / l, z: z / l, w: w / l };
}

function blendBell(prev: BellState, cur: BellState, t: number, snap: boolean): BellState {
  // ALL discrete / identity fields always come from cur.
  const base: BellState = {
    p: cur.p,
    v: cur.v,
    q: cur.q,
    w: cur.w,
    chime: cur.chime,
    heldBy: cur.heldBy,
    thrownBy: cur.thrownBy,
    touchedSinceThrow: cur.touchedSinceThrow,
    releasePos: cur.releasePos,
    releaseTick: cur.releaseTick,
    passChain: cur.passChain,
  };
  if (snap) return base;
  return {
    ...base,
    p: lerpV3(prev.p, cur.p, t),
    v: lerpV3(prev.v, cur.v, t),
    q: slerpQuat(prev.q, cur.q, t),
    w: lerpV3(prev.w, cur.w, t),
    chime: lerp(prev.chime, cur.chime, t),
  };
}

function blendPlayer(prev: PlayerSim | undefined, cur: PlayerSim, t: number, gapOk: boolean): PlayerSim {
  // Discrete / identity / topology fields always come from cur.
  const base: PlayerSim = {
    id: cur.id,
    team: cur.team,
    role: cur.role,
    p: cur.p,
    v: cur.v,
    q: cur.q,
    line: cur.line,
    dvBudget: cur.dvBudget,
    contactRef: cur.contactRef,
    grounded: cur.grounded,
  };
  if (!prev || !gapOk || distSq(prev.p, cur.p) > TELEPORT_DIST_SQ) return base;
  return {
    ...base,
    p: lerpV3(prev.p, cur.p, t),
    v: lerpV3(prev.v, cur.v, t),
    q: slerpQuat(prev.q, cur.q, t),
    dvBudget: lerp(prev.dvBudget, cur.dvBudget, t),
  };
}

/**
 * Interpolate the renderer-facing view between two fixed-step snapshots.
 *
 * @param prev  snapshot at sim tick N-1 (FixedStepDriver.prev)
 * @param cur   snapshot at sim tick N   (FixedStepDriver.cur)
 * @param alpha fractional position into the current tick, clamped to [0,1]
 */
export function renderState(prev: RenderView, cur: RenderView, alpha: number): RenderView {
  const t = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;

  // A clean glide is only valid when prev→cur is exactly one tick. Any other
  // gap (re-arm/reset, dropped substeps resync) means we should not glide.
  const tickGapOk = cur.tick - prev.tick === 1;
  // Possession change ⇒ the bell was re-gripped/thrown: snap it to cur.
  const bellSnap = !tickGapOk || cur.bell.heldBy !== prev.bell.heldBy;

  const prevById = new Map<string, PlayerSim>();
  for (const p of prev.players) prevById.set(p.id, p);

  return {
    // Discrete clock & identity straight from cur.
    tick: cur.tick,
    omega: cur.omega,
    rngCursor: cur.rngCursor,
    loopTurn: cur.loopTurn,
    loopUntouched: cur.loopUntouched,
    loopTier: cur.loopTier,
    bell: blendBell(prev.bell, cur.bell, t, bellSnap),
    players: cur.players.map((cp) =>
      blendPlayer(prevById.get(cp.id), cp, t, tickGapOk),
    ),
  };
}

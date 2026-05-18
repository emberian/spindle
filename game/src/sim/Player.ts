// Player kinematics. A rigger is a point mass that moves ONLY by: free
// ballistic drift (rotating frame), pushing off structure, the rig line,
// or small capped thrumbler nudges. No flight, ever (canon).
//
// Fluidity Spike: every instantaneous response is now CONTINUOUS.
//   • grounding  = penetration push-out spring + exponential tangential
//                  friction bleed via the PRECOMPUTED FRICTION_DECAY constant
//                  (no per-step exp()); `grounded` latches so player_skinned
//                  still fires exactly once.
//   • contact    = a stiff stick-spring toward the latched contact anchor
//                  (replaces the invMass = 0 freeze toggle).
//   • pushoff /
//     thrumbler  = the SAME total Δv, eased over a fixed number of ticks;
//                  the dv-budget is consumed immediately so cap/determinism
//                  semantics are unchanged.
// Arithmetic is byte-identical to rig-core/src/player.rs.

import type { Vec3 } from './vec';
import { vadd, vscale, vlen, vsub } from './vec';
import { rk4Step, type PointState } from './trajectory';
import { resolveLine, type Body, type Line } from './Grapple';
import { REG, FEEL } from './RegConstants';

export const PLAYER_MASS = 78; // kg
export const THRUMBLER_CAP = 2.0; // m/s delta-v per possession

interface Ramp {
  perTick: Vec3;
  ticks: number;
}
const RAMP_NONE: Ramp = { perTick: { x: 0, y: 0, z: 0 }, ticks: 0 };

export interface PlayerBody {
  p: Vec3;
  v: Vec3;
  invMass: number; // 1/PLAYER_MASS (contact freeze is now a stick-spring)
  line: Line | null;
  // GRAPPLE LATENCY (mirrors rig-core player.rs): sim tick at/after which a
  // new line may be fired. A fire is ignored while sim.tick < this OR a line
  // is already present (claw in flight / attached). Integer, no wall clock.
  refireReadyTick: number;
  contact: boolean; // clipped to a spar/ring/teammate
  grounded: boolean; // touched the skin — LATCHED (one-shot event)
  dvBudget: number;
  // internal continuous-feel state (NOT in the snapshot)
  contactAnchor: Vec3;
  contactLatched: boolean;
  pushoffRamp: Ramp;
  thrumblerRamp: Ramp;
}

export function makePlayer(p: Vec3): PlayerBody {
  return {
    p: { ...p },
    v: { x: 0, y: 0, z: 0 },
    invMass: 1 / PLAYER_MASS,
    line: null,
    refireReadyTick: 0,
    contact: false,
    grounded: false,
    dvBudget: THRUMBLER_CAP,
    contactAnchor: { x: 0, y: 0, z: 0 },
    contactLatched: false,
    pushoffRamp: { ...RAMP_NONE },
    thrumblerRamp: { ...RAMP_NONE },
  };
}

export function pushOff(pl: PlayerBody, dir: Vec3, speed: number): void {
  if (!pl.contact) return;
  const l = vlen(dir) || 1e-12;
  const total = vscale(dir, speed / l); // identical total Δv to old impulse
  pl.pushoffRamp = {
    perTick: vscale(total, 1 / FEEL.PUSHOFF_RAMP_TICKS),
    ticks: FEEL.PUSHOFF_RAMP_TICKS,
  };
  pl.contact = false;
  pl.contactLatched = false;
}

export function thrumbler(pl: PlayerBody, dv: Vec3): void {
  let mag = vlen(dv);
  if (mag < 1e-6 || pl.dvBudget <= 0) return;
  let used: number;
  if (mag > pl.dvBudget) {
    dv = vscale(dv, pl.dvBudget / mag);
    used = pl.dvBudget;
  } else {
    used = mag;
  }
  // Accumulate onto any in-flight ramp so rapid taps still sum exactly.
  const remainingPrev = vscale(pl.thrumblerRamp.perTick, pl.thrumblerRamp.ticks);
  const total = vadd(remainingPrev, dv);
  pl.thrumblerRamp = {
    perTick: vscale(total, 1 / FEEL.THRUMBLER_RAMP_TICKS),
    ticks: FEEL.THRUMBLER_RAMP_TICKS,
  };
  pl.dvBudget = Math.max(0, pl.dvBudget - used);
}

export function newPossession(pl: PlayerBody): void {
  pl.dvBudget = THRUMBLER_CAP;
}

function applyRamp(ramp: Ramp): Vec3 {
  if (ramp.ticks === 0) return { x: 0, y: 0, z: 0 };
  ramp.ticks -= 1;
  return ramp.perTick;
}

// Integrate one fixed sub-step.
export function stepPlayer(pl: PlayerBody, h: number, reel: -1 | 0 | 1): void {
  // 1. Eased actuators.
  const dv = vadd(applyRamp(pl.pushoffRamp), applyRamp(pl.thrumblerRamp));
  pl.v = vadd(pl.v, dv);

  // 2. Free-flight (canon, untouched). Suppressed only while grounded.
  if (!pl.grounded) {
    const s: PointState = { p: pl.p, v: pl.v };
    const next = rk4Step(s, REG.omega, h);
    pl.p = next.p;
    pl.v = next.v;
  }

  // 3. Line constraint.
  if (pl.line) {
    const body: Body = { p: pl.p, v: pl.v, invMass: pl.invMass };
    resolveLine(body, pl.line, reel, h);
    pl.p = body.p;
    pl.v = body.v;
  }

  // 4. Contact stick-spring (replaces the invMass = 0 freeze).
  if (pl.contact) {
    if (!pl.contactLatched) {
      pl.contactAnchor = { ...pl.p };
      pl.contactLatched = true;
    }
    const d = vsub(pl.p, pl.contactAnchor);
    const fx = -FEEL.CONTACT_K * d.x - FEEL.CONTACT_C * pl.v.x;
    const fy = -FEEL.CONTACT_K * d.y - FEEL.CONTACT_C * pl.v.y;
    const fz = -FEEL.CONTACT_K * d.z - FEEL.CONTACT_C * pl.v.z;
    pl.v = {
      x: pl.v.x + fx * pl.invMass * h,
      y: pl.v.y + fy * pl.invMass * h,
      z: pl.v.z + fz * pl.invMass * h,
    };
  } else {
    pl.contactLatched = false;
  }

  // 5. Soft grounding: penetration push-out spring + precomputed-decay
  //    tangential friction bleed (no per-step exp()).
  const rho = Math.hypot(pl.p.y, pl.p.z);
  if (rho >= REG.R) {
    pl.grounded = true; // latch (one-shot player_skinned)
    const invRho = 1 / Math.max(rho, 1e-12);
    const ny = pl.p.y * invRho;
    const nz = pl.p.z * invRho;
    const pen = rho - REG.R;
    const vRad = pl.v.y * ny + pl.v.z * nz;
    const f = -FEEL.GROUND_K * pen - FEEL.GROUND_C * vRad;
    pl.v = {
      x: pl.v.x,
      y: pl.v.y + f * ny * pl.invMass * h,
      z: pl.v.z + f * nz * pl.invMass * h,
    };
    const vRad2 = pl.v.y * ny + pl.v.z * nz;
    const vtY = pl.v.y - vRad2 * ny;
    const vtZ = pl.v.z - vRad2 * nz;
    pl.v = {
      x: pl.v.x * FEEL.FRICTION_DECAY,
      y: vRad2 * ny + vtY * FEEL.FRICTION_DECAY,
      z: vRad2 * nz + vtZ * FEEL.FRICTION_DECAY,
    };
  }
}

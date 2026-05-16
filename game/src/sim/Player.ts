// Player kinematics. A rigger is a point mass that moves ONLY by: free
// ballistic drift (rotating frame), pushing off structure, the rig line,
// or small capped thrumbler nudges. No flight, ever (canon).

import type { Vec3 } from './vec';
import { vadd, vscale, vlen } from './vec';
import { rk4Step, type PointState } from './trajectory';
import { resolveLine, type Body, type Line } from './Grapple';
import { REG } from './RegConstants';

export const PLAYER_MASS = 78; // kg
export const THRUMBLER_CAP = 2.0; // m/s delta-v per possession

export interface PlayerBody {
  p: Vec3;
  v: Vec3;
  invMass: number; // 1/PLAYER_MASS, or 0 while clipped (immovable anchor)
  line: Line | null;
  contact: boolean; // clipped to a spar/ring/teammate
  grounded: boolean; // touched the skin — out of the calm
  dvBudget: number;
}

export function makePlayer(p: Vec3): PlayerBody {
  return {
    p: { ...p },
    v: { x: 0, y: 0, z: 0 },
    invMass: 1 / PLAYER_MASS,
    line: null,
    contact: false,
    grounded: false,
    dvBudget: THRUMBLER_CAP,
  };
}

export function pushOff(pl: PlayerBody, dir: Vec3, speed: number): void {
  if (!pl.contact) return;
  const l = vlen(dir) || 1;
  pl.v = vadd(pl.v, vscale(dir, speed / l));
  pl.contact = false;
}

export function thrumbler(pl: PlayerBody, dv: Vec3): void {
  let mag = vlen(dv);
  if (mag < 1e-6 || pl.dvBudget <= 0) return;
  if (mag > pl.dvBudget) {
    dv = vscale(dv, pl.dvBudget / mag);
    mag = pl.dvBudget;
  }
  pl.v = vadd(pl.v, dv);
  pl.dvBudget = Math.max(0, pl.dvBudget - mag);
}

export function newPossession(pl: PlayerBody): void {
  pl.dvBudget = THRUMBLER_CAP;
}

// Integrate one fixed sub-step: free-flight (unless clipped), then the line
// constraint, then skin clamp (grounding).
export function stepPlayer(pl: PlayerBody, h: number, reel: -1 | 0 | 1): void {
  if (!pl.contact && !pl.grounded) {
    const s: PointState = { p: pl.p, v: pl.v };
    const next = rk4Step(s, REG.omega, h);
    pl.p = next.p;
    pl.v = next.v;
  }
  if (pl.line) {
    const body: Body = { p: pl.p, v: pl.v, invMass: pl.contact ? 0 : pl.invMass };
    resolveLine(body, pl.line, reel, h);
    pl.p = body.p;
    pl.v = body.v;
  }
  const rho = Math.hypot(pl.p.y, pl.p.z);
  if (rho >= REG.R) {
    pl.grounded = true; // skinned — out until re-entry from your end
    const s = REG.R / (rho || 1);
    pl.p = { x: pl.p.x, y: pl.p.y * s, z: pl.p.z * s };
    pl.v = { x: 0, y: 0, z: 0 };
  }
}

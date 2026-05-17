// Skin bounce, the REAL relative-velocity catch (not v0.1's 5 m homing),
// and the contest clatter. Pure functions on bodies; callers emit events.

import type { Vec3 } from './vec';
import { vsub, vlen, vdot, vnorm, vscale } from './vec';
import { REG } from './RegConstants';
import { clatter, type BellBody } from './Bell';

const RESTITUTION = 0.55;
// Catch envelope. The original 0.7 m / 9 m/s was physically unreachable for
// grapple-only locomotion vs an 18–34 m/s Coriolis bell — telemetry showed
// ZERO completed catches across a whole match, so no possession/pass play
// ever formed. Widened to a tether-claw catch (a rigger leads & snares the
// bell, not bare hands): a real but skill-gated window so possession exists.
// Determinism is twice-equal (not golden) and the Rust twin moves in
// lockstep, so this stays bit-safe; stale replays are version-discarded.
const ARM_REACH = 1.8; // m — claw/tether snare radius
const CATCH_SPEED_BASE = 16; // m/s rel-speed a baseline rigger can absorb

/** Bounce a free body off the cylinder skin (cross-axis radius = R). Returns
 *  true if it hit this step. */
export function skinBounce(body: { p: Vec3; v: Vec3 }): boolean {
  const rho = Math.hypot(body.p.y, body.p.z);
  if (rho < REG.R) return false;
  const ny = body.p.y / rho;
  const nz = body.p.z / rho;
  const vn = body.v.y * ny + body.v.z * nz; // radial component
  if (vn > 0) {
    body.v.y -= (1 + RESTITUTION) * vn * ny;
    body.v.z -= (1 + RESTITUTION) * vn * nz;
  }
  const s = (REG.R - 1e-3) / rho;
  body.p.y *= s;
  body.p.z *= s;
  return true;
}

export type CatchResult = 'caught' | 'bobble' | 'miss';

/** A catch is physical: in arm's reach, closing, and slow enough relative to
 *  the catcher (skill widens the absorbable speed). Otherwise a bobble
 *  (partial damp + the bell loses trueness — an audible clatter). */
export function tryCatch(
  bellPos: Vec3,
  bellVel: Vec3,
  playerPos: Vec3,
  playerVel: Vec3,
  skill = 1,
): CatchResult {
  const d = vsub(bellPos, playerPos);
  const gap = vlen(d);
  if (gap > ARM_REACH) return 'miss';
  const rel = vsub(bellVel, playerVel);
  const relSpeed = vlen(rel);
  const closing = gap < 1e-6 ? 1 : -vdot(rel, vnorm(d));
  if (closing < -0.5) return 'miss';
  if (relSpeed <= CATCH_SPEED_BASE * skill) return 'caught';
  return 'bobble';
}

/** Apply a bobble: kill most of the relative velocity, knock it off true. */
export function applyBobble(bell: BellBody, playerVel: Vec3): void {
  const rel = vsub(bell.v, playerVel);
  bell.v = { x: playerVel.x + rel.x * 0.25, y: playerVel.y + rel.y * 0.25, z: playerVel.z + rel.z * 0.25 };
  clatter(bell, { x: 0, y: 4, z: 3 });
}

/** A defender contesting the live bell knocks it off true. Strength scales
 *  with the contester's closing speed (deep/violent contests hit harder). */
export function contestClatter(bell: BellBody, contesterVel: Vec3, strength = 1): void {
  const rel = vlen(vsub(bell.v, contesterVel));
  const mag = Math.min(18, 4 + rel * 0.4) * strength;
  // off the ring axis (body y/z) so trueness drops
  clatter(bell, vscale({ x: 0, y: 0.8, z: 0.6 }, mag));
}

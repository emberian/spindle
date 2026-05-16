// The rig line — THE core mechanic, and v0.1's biggest failure (a dumb
// max-distance position clamp with no momentum). Here it is a velocity-level
// impulse distance constraint:
//   • slack  (len < restLen): no force — free ballistic swing within radius
//   • taut   (len ≥ restLen): remove ONLY the separating radial velocity,
//              preserving all tangential velocity (the swing). Equal & opposite
//              on a player anchor ⇒ momentum-conserving slingshot, no special case.
//   • reel   : shrink restLen; angular momentum about the anchor is conserved
//              ⇒ tangential speed rises ⇒ emergent slingshot (real, not scripted).

import type { Vec3 } from './vec';
import { vsub, vdot, vscale, vadd, vlen, vnorm } from './vec';

export interface Body {
  p: Vec3;
  v: Vec3;
  invMass: number; // 0 = immovable anchor (spar / skin / ring)
}

export interface Line {
  anchorPos: Vec3; // used when the anchor is static
  anchorBody: Body | null; // used when the anchor is another player
  restLen: number;
  taut: boolean;
}

export const TETHER_MIN = 3;
export const TETHER_MAX = 60;
export const REEL_RATE = 14; // m/s of rope length change

function anchorP(line: Line): Vec3 {
  return line.anchorBody ? line.anchorBody.p : line.anchorPos;
}
function anchorV(line: Line): Vec3 {
  return line.anchorBody ? line.anchorBody.v : { x: 0, y: 0, z: 0 };
}

// One constraint solve for one line. `reel` ∈ {-1,0,1} (in/none/out).
export function resolveLine(player: Body, line: Line, reel: -1 | 0 | 1, h: number): void {
  const aP = anchorP(line);
  const d = vsub(player.p, aP);
  const len = vlen(d);
  if (len < 1e-6) {
    line.taut = false;
    return;
  }
  const n = vscale(d, 1 / len); // unit, anchor -> player

  // Reel: change rope length, conserving angular momentum about the anchor
  // (L = m·v_t·r ⇒ shrinking r raises v_t — the slingshot, emergent).
  if (reel !== 0) {
    const target = Math.max(
      TETHER_MIN,
      Math.min(TETHER_MAX, line.restLen + reel * REEL_RATE * h),
    );
    if (target !== line.restLen) {
      const relV = vsub(player.v, anchorV(line));
      const vRad = vdot(relV, n);
      const vTanVec = vsub(relV, vscale(n, vRad)); // tangential component
      const scale = line.restLen > 1e-6 ? line.restLen / target : 1;
      const newTan = vscale(vnorm(vTanVec), vlen(vTanVec) * scale);
      // closing/opening radial velocity follows the reel
      const newRad = -reel * REEL_RATE;
      const newRelV = vadd(newTan, vscale(n, newRad));
      player.v = vadd(anchorV(line), newRelV);
      player.p = vadd(aP, vscale(n, target));
      line.restLen = target;
      line.taut = true;
      return;
    }
  }

  // Slack: inside the radius, the line does nothing (free swing) — this is
  // what gives the arena-shooter arc feel.
  if (len < line.restLen - 1e-4) {
    line.taut = false;
    return;
  }

  // Taut: cancel only the separating radial velocity, preserve tangential.
  line.taut = true;
  const rv = vsub(player.v, anchorV(line));
  const vrel = vdot(rv, n); // >0 = separating
  if (vrel <= 0) return; // swinging inward / along: nothing to do
  const aBody = line.anchorBody;
  const invSum = player.invMass + (aBody ? aBody.invMass : 0);
  if (invSum <= 0) return;
  const j = -vrel / invSum; // impulse magnitude along n
  player.v = vadd(player.v, vscale(n, j * player.invMass));
  if (aBody) aBody.v = vadd(aBody.v, vscale(n, -j * aBody.invMass));

  // Soft Baumgarte position correction so it can't slowly drift past restLen.
  const err = len - line.restLen;
  if (err > 0) {
    const corr = (0.2 * err) / Math.max(h, 1e-4);
    player.v = vadd(player.v, vscale(n, -corr * (player.invMass / invSum) * h));
  }
}

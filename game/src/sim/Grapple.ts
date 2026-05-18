// The rig line — THE core mechanic. Continuous (Fluidity Spike) model: a
// one-sided radial spring-damper. TENSION ONLY — it never pushes inward when
// slack, so deep slack is a true free ballistic swing:
//   • deep slack  (stretch ≤ −SLACK_BAND): no force.
//   • slack band  (−SLACK_BAND < stretch ≤ 0): a gentle pre-tension spring
//              eases the line into tautness instead of snapping.
//   • taut        (stretch > 0): F = −LINE_K·stretch − LINE_C·v_radial,
//              clamped so it can only pull toward the anchor; equal & opposite
//              on a player anchor ⇒ momentum-conserving slingshot.
// The reel ONLY eases restLen toward target (no position set, no v_tan
// rescale) ⇒ the angular-momentum slingshot is fully EMERGENT.
//
// Semi-implicit (symplectic) Euler; ω_n·h ≈ 0.018 ≪ 2 ⇒ stable at 1/240 s.
// Arithmetic is byte-identical to rig-core/src/grapple.rs (same literals,
// same operation order).

import type { Vec3 } from './vec';
import { vsub, vdot, vscale, vadd, vlen } from './vec';
import { FEEL } from './RegConstants';

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
  // GRAPPLE LATENCY (mirrors rig-core grapple.rs Line): `attached` is false
  // while the claw is in flight (line committed to target, NO constraint
  // force) and true once it lands. `attachTick` is the sim tick at/after
  // which SimWorld flips `attached` true. Integer tick count, no wall clock.
  attached: boolean;
  attachTick: number;
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
  // GRAPPLE LATENCY: an in-flight claw exerts NO force — the constraint is
  // inert until SimWorld flips `attached` true (sim.tick >= attachTick).
  // Byte-identical to the legacy path the instant a line attaches.
  if (!line.attached) {
    line.taut = false;
    return;
  }
  const aP = anchorP(line);
  const aV = anchorV(line);
  const aInvMass = line.anchorBody ? line.anchorBody.invMass : 0;
  const d = vsub(player.p, aP);
  const len = vlen(d);
  if (len < 1e-6) {
    line.taut = false;
    return;
  }
  const n = vscale(d, 1 / len); // unit, anchor -> player

  // Reel: ONLY ease restLen toward the target. No position set, no v_tan
  // rescale — the slingshot must emerge from the spring.
  if (reel !== 0) {
    const target = Math.max(
      TETHER_MIN,
      Math.min(TETHER_MAX, line.restLen + reel * REEL_RATE * h),
    );
    line.restLen = target;
  }

  const stretch = len - line.restLen;

  // Deep slack: the line does nothing (free ballistic swing).
  if (stretch <= -FEEL.LINE_SLACK_BAND) {
    line.taut = false;
    return;
  }

  const invSum = player.invMass + aInvMass;
  if (invSum <= 0) {
    line.taut = stretch > 0;
    return;
  }

  // Radial relative velocity (positive ⇒ separating from the anchor).
  const rv = vsub(player.v, aV);
  const vRad = vdot(rv, n);

  // Scalar radial force along n. Tension pulls toward the anchor ⇒ ≤ 0.
  // Clamp to ≤ 0 so the rope can NEVER push outward.
  let f: number;
  if (stretch > 0) {
    line.taut = true;
    const raw = -FEEL.LINE_K * stretch - FEEL.LINE_C * vRad;
    f = Math.min(0, raw);
  } else {
    line.taut = false;
    f = FEEL.LINE_SLACK_K * stretch;
  }

  // Semi-implicit Euler velocity update; equal-and-opposite on the anchor.
  const impulse = f * h; // N·s along n
  player.v = vadd(player.v, vscale(n, impulse * player.invMass));
  if (line.anchorBody) {
    line.anchorBody.v = vadd(line.anchorBody.v, vscale(n, -impulse * line.anchorBody.invMass));
  }
}

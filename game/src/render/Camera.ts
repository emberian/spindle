// The camera is the #1 legibility fix. The old single dark side-chase made
// the game unreadable. This frames the three things you must always see —
// the bell, YOUR rigger, and the goal you're attacking — looking down the
// spine so depth/length read, with a dramatic broadside loop-cam.
//
// World frame: +X = spin axis (goals at x = ±GATE). (y,z) = rotating
// cross-section, axis at y=z=0, skin at radius R. There is no global "up":
// "up" everywhere points toward the axis (−radial). All smoothing is
// exponential + clamped so it never disorients.

import * as THREE from 'three';

const V = THREE.Vector3;
type V3 = { x: number; y: number; z: number };

function damp(cur: THREE.Vector3, target: THREE.Vector3, lambda: number, dt: number): void {
  // frame-rate-independent exponential approach
  const k = 1 - Math.exp(-lambda * dt);
  cur.x += (target.x - cur.x) * k;
  cur.y += (target.y - cur.y) * k;
  cur.z += (target.z - cur.z) * k;
}

export class GameCamera {
  private pos = new V();
  private aim = new V();
  private up = new V(0, 1, 0);
  private fov = 60;
  private inited = false;
  private loopBlend = 0;

  constructor(private cam: THREE.PerspectiveCamera) {}

  /**
   * @param bell    bell world position
   * @param p1      your rigger's world position
   * @param attackX +X goal you attack (e.g. +GATE for home, −GATE for away)
   * @param loopGlow 0..1 — a Loop developing → swing to the loop-cam
   * @param skinR   cylinder radius (for framing scale)
   */
  update(bell: V3, p1: V3, attackX: number, loopGlow: number, skinR: number, dt: number): void {
    const dir = Math.sign(attackX) || 1; // +1 attack +x, −1 attack −x

    // Focus = weighted blend of the bell (lead) and your rigger, nudged
    // toward the attacked goal so you always see where you're going.
    const focus = new V(
      bell.x * 0.62 + p1.x * 0.28 + attackX * 0.10,
      bell.y * 0.6 + p1.y * 0.4,
      bell.z * 0.6 + p1.z * 0.4,
    );

    // Cross-section radius of the focus → keeps the camera outside the action.
    const rho = Math.max(2, Math.hypot(focus.y, focus.z));
    // Radial outward unit (from axis to focus) in the (y,z) plane.
    const ry = focus.y / rho;
    const rz = focus.z / rho;

    let camTarget: THREE.Vector3;
    let aimTarget: THREE.Vector3;
    let fovTarget: number;
    let upTarget: THREE.Vector3;

    const wantLoop = loopGlow > 0.15;
    this.loopBlend += ((wantLoop ? 1 : 0) - this.loopBlend) * (1 - Math.exp(-4 * dt));

    if (this.loopBlend > 0.02) {
      // ── LOOP-CAM: broadside, so the closed Coriolis arc reads AS a loop.
      // Stand off the bell along the cross-section radial, pulled back a
      // little down-spine, looking square at the arc plane. Slow push-in.
      const standoff = skinR * (2.0 - 0.5 * loopGlow);
      camTarget = new V(
        bell.x - dir * skinR * 0.5,
        ry * standoff,
        rz * standoff,
      );
      aimTarget = new V(bell.x, bell.y * 0.5, bell.z * 0.5);
      fovTarget = 46 - 6 * loopGlow;
      upTarget = new V(1, 0, 0).multiplyScalar(dir); // spine reads as horizon
    } else {
      // ── BROADCAST/FOLLOW: behind the action, lifted toward the axis,
      // looking down-spine at the goal so length & depth are obvious.
      const back = skinR * 1.7; // behind, opposite the attack direction
      const lift = skinR * 0.55; // toward the axis (so we look "down" the tube)
      const side = skinR * 0.45; // slight tangential offset for parallax
      // tangent unit in (y,z): perpendicular to radial
      const ty = -rz,
        tz = ry;
      camTarget = new V(
        focus.x - dir * back,
        focus.y - ry * lift + ty * side,
        focus.z - rz * lift + tz * side,
      );
      // look toward the goal-ward action
      aimTarget = new V(focus.x + dir * skinR * 0.9, focus.y * 0.7, focus.z * 0.7);
      fovTarget = 62;
      // up = toward the axis (−radial), so the horizon never tumbles
      upTarget = new V(0, -ry, -rz);
    }

    if (!this.inited) {
      this.pos.copy(camTarget);
      this.aim.copy(aimTarget);
      this.up.copy(upTarget);
      this.fov = fovTarget;
      this.inited = true;
    }
    // gentle, clamped damping (loop-cam a touch snappier)
    const lam = this.loopBlend > 0.02 ? 5 : 3.2;
    damp(this.pos, camTarget, lam, dt);
    damp(this.aim, aimTarget, lam * 1.2, dt);
    damp(this.up, upTarget, 4, dt);
    this.fov += (fovTarget - this.fov) * (1 - Math.exp(-3 * dt));

    this.cam.position.copy(this.pos);
    this.cam.up.copy(this.up.lengthSq() < 1e-4 ? new V(0, 1, 0) : this.up.clone().normalize());
    this.cam.lookAt(this.aim);
    if (Math.abs(this.cam.fov - this.fov) > 0.01) {
      this.cam.fov = this.fov;
      this.cam.updateProjectionMatrix();
    }
  }

  /** Snap (no damping) — call when (re)entering a match so frame 1 is framed. */
  reset(): void {
    this.inited = false;
    this.loopBlend = 0;
  }
}

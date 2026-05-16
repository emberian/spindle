// AimModel: pointer-lock mouse-look → a 3D aim direction in camera-local
// frame; raycast against candidate anchors (spar points + the skin surface)
// → a snapped anchor reticle with valid/invalid/out-of-range status.
//
// Pure-ish (no DOM events — the InputManager feeds deltaMouse). Designed so
// it can be tested without a DOM.

import * as THREE from 'three';
import type { Vec3 } from '../sim/vec';
import { REG } from '../sim/RegConstants';

// ── Config ────────────────────────────────────────────────────────────────

/** Mouse sensitivity (radians per pixel). */
const SENS_X = 0.0018;
const SENS_Y = 0.0018;

/** Pitch clamp (avoid gimbal lock). */
const MAX_PITCH = Math.PI * 0.49;

/** Max range for anchor snapping (metres). */
const SNAP_RANGE = 120;

/** Snapping cone half-angle (screen-space, radians). */
const SNAP_CONE = 0.08; // ~4.6°

// ── Types ─────────────────────────────────────────────────────────────────

export interface AnchorResult {
  pos: Vec3;
  valid: boolean;   // in range + clear line of sight (simple here)
  outOfRange: boolean;
}

// ── AimModel ─────────────────────────────────────────────────────────────

export class AimModel {
  /** Yaw accumulator (radians). */
  private yaw   = 0;
  /** Pitch accumulator (radians). */
  private pitch = 0;

  // Working Three.js objects (reused to avoid GC)
  private _dir    = new THREE.Vector3(0, 0, -1);
  private _ray    = new THREE.Ray();
  private _tmpV   = new THREE.Vector3();

  /**
   * Integrate mouse delta into aim angles.
   * @param dx  Raw pointer-lock movementX (pixels)
   * @param dy  Raw pointer-lock movementY (pixels)
   */
  update(dx: number, dy: number): void {
    this.yaw   -= dx * SENS_X;
    this.pitch -= dy * SENS_Y;
    this.pitch  = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, this.pitch));
  }

  /**
   * Current aim direction in world space, derived from camera orientation +
   * accumulated yaw/pitch offsets.
   */
  aimDir(camera: THREE.Camera): Vec3 {
    // Start from camera's forward, then apply our own accumulated yaw/pitch
    // (so the aim can diverge from the camera if camera is chase-locked).
    const euler = new THREE.Euler(this.pitch, this.yaw, 0, 'YXZ');
    this._dir.set(0, 0, -1).applyEuler(euler).applyQuaternion(camera.quaternion);
    return { x: this._dir.x, y: this._dir.y, z: this._dir.z };
  }

  /**
   * Find the best snap-able anchor from `spars` list + the skin cylinder.
   * Returns null if no candidate is within cone + range.
   *
   * @param spars    World-space spar positions (Vec3[])
   * @param camera   Active camera
   * @param playerPos  The local player's position (origin of the rig line)
   */
  anchorTarget(
    spars: Vec3[],
    camera: THREE.Camera,
    playerPos: Vec3,
  ): AnchorResult | null {
    const dir = this.aimDir(camera);
    const origin = new THREE.Vector3(playerPos.x, playerPos.y, playerPos.z);
    this._ray.set(origin, new THREE.Vector3(dir.x, dir.y, dir.z));

    let bestDot = -Infinity;
    let bestPos: THREE.Vector3 | null = null;
    let bestDist = Infinity;

    const camPos = new THREE.Vector3();
    camera.getWorldPosition(camPos);
    const aimW = new THREE.Vector3(dir.x, dir.y, dir.z).normalize();

    // ── Test spar candidates ──────────────────────────────────────────────
    for (const sp of spars) {
      this._tmpV.set(sp.x, sp.y, sp.z);
      const toSpar = this._tmpV.clone().sub(camPos).normalize();
      const dot = toSpar.dot(aimW);
      if (dot < Math.cos(SNAP_CONE)) continue; // outside cone

      const dist = origin.distanceTo(this._tmpV);
      if (dist > SNAP_RANGE) continue;

      if (dot > bestDot) {
        bestDot  = dot;
        bestPos  = this._tmpV.clone();
        bestDist = dist;
      }
    }

    // ── Test skin cylinder ────────────────────────────────────────────────
    // Approximate: skin is a sphere of radius R about world origin for
    // ray-test purposes (the actual cylinder is infinite along X; we just
    // test the yz cross-section intersection via a circle in the ray's plane).
    // Full analytic cylinder intersection:
    {
      const ox = this._ray.origin.y;
      const oy = this._ray.origin.z;
      const dx2 = this._ray.direction.y;
      const dy2 = this._ray.direction.z;
      const a = dx2 * dx2 + dy2 * dy2;
      if (a > 1e-10) {
        const bCoef = 2 * (ox * dx2 + oy * dy2);
        const c = ox * ox + oy * oy - REG.R * REG.R;
        const disc = bCoef * bCoef - 4 * a * c;
        if (disc >= 0) {
          const sqrtD = Math.sqrt(disc);
          // We want the intersection behind us (we're inside), take positive t
          const t1 = (-bCoef + sqrtD) / (2 * a);
          const t2 = (-bCoef - sqrtD) / (2 * a);
          const t = t1 > 0 ? t1 : (t2 > 0 ? t2 : -1);
          if (t > 0 && t < SNAP_RANGE) {
            const hitX = this._ray.origin.x + this._ray.direction.x * t;
            if (hitX >= -320 && hitX <= 320) { // within calm bounds
              const hitY = this._ray.origin.y + this._ray.direction.y * t;
              const hitZ = this._ray.origin.z + this._ray.direction.z * t;
              const skinPos = new THREE.Vector3(hitX, hitY, hitZ);
              const toSkin = skinPos.clone().sub(camPos).normalize();
              const dot = toSkin.dot(aimW);
              if (dot > bestDot || bestPos === null) {
                // skin always loses to a spar if one is in cone
                if (bestPos === null) {
                  bestDot  = dot;
                  bestPos  = skinPos;
                  bestDist = t;
                }
              }
            }
          }
        }
      }
    }

    if (!bestPos) return null;

    const outOfRange = bestDist > SNAP_RANGE;
    return {
      pos: { x: bestPos.x, y: bestPos.y, z: bestPos.z },
      valid: !outOfRange,
      outOfRange,
    };
  }

  /** Reset aim back to neutral (use on pointer-lock release). */
  reset(): void {
    this.yaw   = 0;
    this.pitch = 0;
  }
}

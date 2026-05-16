// AimModel — mouse-look → 3D aim direction; anchor snapping with generous cone.
//
// Works in two modes:
//   Pointer-lock: integrated raw movementX/Y deltas → yaw/pitch accumulator.
//   Free-mouse:   canvas-relative cursor position → direction via unproject.
//
// Anchor snapping uses a wide screen-space cone (generous — no pixel-perfect).
// Spar candidates beat the skin cylinder when both land in cone.
// Aim-assist toward the Faith ring (+x) is applied as a gentle bias when the
// caller requests it (throwing mode) — never hard-locks, just nudges.

import * as THREE from 'three';
import type { Vec3 } from '../sim/vec';
import { REG, GATE_X } from '../sim/RegConstants';

// ── Config ────────────────────────────────────────────────────────────────────

/** Pointer-lock sensitivity (radians per pixel). Tuned for ~800 dpi mouse. */
const SENS_X = 0.0016;
const SENS_Y = 0.0016;

/** Smoothing: low-pass on pointer-lock delta (0 = instant, 1 = frozen). */
const SMOOTH = 0.28;

/** Pitch clamp — avoid full gimbal flip. */
const MAX_PITCH = Math.PI * 0.48;

/** Anchor snap cone half-angle (radians) — ~8°, very forgiving. */
const SNAP_CONE = 0.14;

/** Max tether range (metres). */
const SNAP_RANGE = 130;

/** Aim-assist pull strength toward target when throwing (0 = none, 1 = full). */
const ASSIST_STRENGTH = 0.06;

// ── Types ─────────────────────────────────────────────────────────────────────

export type ReticleState = 'free' | 'anchor-lock' | 'charging' | 'out_of_range';

export interface AnchorResult {
  pos:        Vec3;
  valid:      boolean;       // in range
  outOfRange: boolean;
  locked:     boolean;       // snapped to a discrete spar (vs. skin fallback)
}

// ── AimModel ──────────────────────────────────────────────────────────────────

export class AimModel {
  // Pointer-lock mode accumulators
  private yaw   = 0;
  private pitch = 0;

  // Smoothed delta accumulators
  private smoothDX = 0;
  private smoothDY = 0;

  // Free-mouse mode: NDC [-1,1] of cursor, or null when not tracked
  private cursorNDC: { x: number; y: number } | null = null;

  // Working Three.js objects (reused)
  private _dir  = new THREE.Vector3(0, 0, -1);
  private _ray  = new THREE.Ray();
  private _tmpV = new THREE.Vector3();

  // ── External setters ────────────────────────────────────────────────────────

  /** Set canvas-relative cursor position for free-mouse aim (no pointer lock). */
  setFreeMouse(nx: number, ny: number): void {
    // nx, ny ∈ [-1,1] NDC
    this.cursorNDC = { x: nx, y: ny };
  }

  /** Clear free-mouse position (reverts to pointer-lock aim). */
  clearFreeMouse(): void {
    this.cursorNDC = null;
  }

  // ── Update ───────────────────────────────────────────────────────────────────

  /**
   * Integrate pointer-lock mouse delta.
   * @param dx  raw movementX
   * @param dy  raw movementY
   */
  update(dx: number, dy: number): void {
    // Exponential smooth before integrating — kills jitter on high-dpi.
    this.smoothDX = this.smoothDX * SMOOTH + dx * (1 - SMOOTH);
    this.smoothDY = this.smoothDY * SMOOTH + dy * (1 - SMOOTH);

    this.yaw   -= this.smoothDX * SENS_X;
    this.pitch -= this.smoothDY * SENS_Y;
    this.pitch  = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, this.pitch));
  }

  // ── Aim direction ────────────────────────────────────────────────────────────

  /**
   * Current world-space aim direction.
   * In pointer-lock mode: camera quaternion × accumulated yaw/pitch.
   * In free-mouse mode: un-project NDC cursor through camera.
   *
   * @param aimAssistTarget  Optional world position to bias toward (weak pull).
   */
  aimDir(camera: THREE.Camera, aimAssistTarget?: Vec3): Vec3 {
    let dir: THREE.Vector3;

    if (this.cursorNDC) {
      // Free-mouse: cast a ray through the NDC cursor position
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(
        new THREE.Vector2(this.cursorNDC.x, this.cursorNDC.y),
        camera,
      );
      dir = raycaster.ray.direction.clone().normalize();
    } else {
      // Pointer-lock: apply accumulated yaw/pitch relative to camera
      const euler = new THREE.Euler(this.pitch, this.yaw, 0, 'YXZ');
      dir = this._dir.clone().set(0, 0, -1).applyEuler(euler).applyQuaternion(camera.quaternion).normalize();
    }

    // Gentle aim-assist bias toward a target (throwing mode only)
    if (aimAssistTarget) {
      const camPos = new THREE.Vector3();
      camera.getWorldPosition(camPos);
      const toTarget = new THREE.Vector3(
        aimAssistTarget.x - camPos.x,
        aimAssistTarget.y - camPos.y,
        aimAssistTarget.z - camPos.z,
      ).normalize();
      // Only assist when we're already roughly facing the target (within ~60°)
      if (dir.dot(toTarget) > 0.5) {
        dir.lerp(toTarget, ASSIST_STRENGTH).normalize();
      }
    }

    return { x: dir.x, y: dir.y, z: dir.z };
  }

  // ── Anchor snapping ──────────────────────────────────────────────────────────

  /**
   * Find the best snap-able anchor: spars > skin cylinder.
   *
   * @param spars      World-space spar positions
   * @param camera     Active camera
   * @param playerPos  Fire origin (player world pos)
   */
  anchorTarget(
    spars:     Vec3[],
    camera:    THREE.Camera,
    playerPos: Vec3,
  ): AnchorResult | null {
    const dir    = this.aimDir(camera);
    const origin = new THREE.Vector3(playerPos.x, playerPos.y, playerPos.z);
    this._ray.set(origin, new THREE.Vector3(dir.x, dir.y, dir.z).normalize());

    const camPos = new THREE.Vector3();
    camera.getWorldPosition(camPos);
    const aimW = new THREE.Vector3(dir.x, dir.y, dir.z).normalize();

    let bestSparDot = -Infinity;
    let bestSparPos: THREE.Vector3 | null = null;

    // ── Spar candidates ───────────────────────────────────────────────────────
    for (const sp of spars) {
      this._tmpV.set(sp.x, sp.y, sp.z);
      const toSpar = this._tmpV.clone().sub(camPos).normalize();
      const dot    = toSpar.dot(aimW);
      if (dot < Math.cos(SNAP_CONE)) continue; // outside generous cone

      const dist = origin.distanceTo(this._tmpV);
      if (dist > SNAP_RANGE) continue;  // filtered: all stored spars are in range

      if (dot > bestSparDot) {
        bestSparDot = dot;
        bestSparPos = this._tmpV.clone();
      }
    }

    // Spars always win over the skin cylinder
    if (bestSparPos) {
      return {
        pos:        { x: bestSparPos.x, y: bestSparPos.y, z: bestSparPos.z },
        valid:      true, // dist checked above
        outOfRange: false,
        locked:     true,
      };
    }

    // ── Skin cylinder fallback ────────────────────────────────────────────────
    // Analytic infinite-cylinder intersection (axis = X, radius = REG.R).
    // We solve in the yz plane: (oy + t·dy)² + (oz + t·dz)² = R²
    {
      const oy = this._ray.origin.y;
      const oz = this._ray.origin.z;
      const dy = this._ray.direction.y;
      const dz = this._ray.direction.z;
      const a  = dy * dy + dz * dz;
      if (a > 1e-10) {
        const b    = 2 * (oy * dy + oz * dz);
        const c    = oy * oy + oz * oz - REG.R * REG.R;
        const disc = b * b - 4 * a * c;
        if (disc >= 0) {
          const sqrtD = Math.sqrt(disc);
          const t1    = (-b + sqrtD) / (2 * a);
          const t2    = (-b - sqrtD) / (2 * a);
          // Pick the positive intersection (we're inside the cylinder)
          const t     = (t1 > 0.5) ? t1 : (t2 > 0.5 ? t2 : -1);
          if (t > 0 && t < SNAP_RANGE) {
            const hitX = this._ray.origin.x + this._ray.direction.x * t;
            if (hitX >= -GATE_X && hitX <= GATE_X) {
              const hitY = oy + dy * t;
              const hitZ = oz + dz * t;
              return {
                pos:        { x: hitX, y: hitY, z: hitZ },
                valid:      true, // t < SNAP_RANGE already checked above
                outOfRange: false,
                locked:     false,
              };
            }
          }
        }
      }
    }

    return null;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────────

  /**
   * Gentle aim-assist: bias the aim direction toward the Faith (+x) ring
   * center of mass in the yz plane. Call this from InputManager when
   * holding the bell to guide the thrower toward the scoring ring.
   *
   * Returns the biased aim direction Vec3 (unit-ish).
   */
  assistedThrowDir(
    camera:       THREE.Camera,
    _playerPos:   Vec3,
    targetRingX:  number,
  ): Vec3 {
    // Faith ring is at (targetRingX, 0, 0) — aim there as assist target
    const assistTarget: Vec3 = { x: targetRingX, y: 0, z: 0 };
    return this.aimDir(camera, assistTarget);
  }

  /** Reset aim to neutral — call on pointer-lock release. */
  reset(): void {
    this.yaw      = 0;
    this.pitch    = 0;
    this.smoothDX = 0;
    this.smoothDY = 0;
  }
}

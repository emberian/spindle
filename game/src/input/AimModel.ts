// AimModel — mouse-look → 3D aim direction; anchor snapping with generous cone.
//
// Works in two modes:
//   Pointer-lock: integrated raw movementX/Y deltas → yaw/pitch accumulator.
//   Free-mouse:   canvas-relative cursor position → direction via unproject.
//
// FEEL pass: the two modes now feel the SAME. Pointer-lock uses a critically
// damped follow (no mushy heavy low-pass lag, no high-dpi jitter) and free-
// mouse smooths the unprojected ray identically, so aim is crisp and
// predictable either way. Anchor snapping has hysteresis ("sticky lock"):
// once you've snapped to a spar it takes a wider break angle to let go, so
// the reticle doesn't chatter between candidates. Aim-assist is decoupled
// from the snap (it should never fight your manual aim).

import * as THREE from 'three';
import type { Vec3 } from '../sim/vec';
import { REG, GATE_X } from '../sim/RegConstants';

// ── Config ────────────────────────────────────────────────────────────────────

/** Pointer-lock sensitivity (radians per pixel). Tuned for ~800 dpi mouse. */
const SENS_X = 0.0019;
const SENS_Y = 0.0019;

/**
 * Pointer-lock follow rate. We integrate raw deltas directly (so 1:1, no
 * accumulating lag) but ease the *reported* yaw/pitch toward the integrated
 * target — fast enough to feel instant, just enough to kill 1-pixel jitter.
 */
const FOLLOW_LAMBDA = 38;

/** Free-mouse ray smoothing — matched to pointer-lock so modes feel alike. */
const FREE_LAMBDA = 30;

/** Pitch clamp — avoid full gimbal flip. */
const MAX_PITCH = Math.PI * 0.48;

/** Anchor snap cone half-angle (radians) — ~20°, HEAVY snap so the mouse
 *  reliably grabs an anchor without precise aiming. */
const SNAP_CONE = 0.35;
/** Once locked, only break past this much wider angle (~30°) — very sticky. */
const SNAP_BREAK = 0.52;

/** Max tether range (metres). */
const SNAP_RANGE = 130;

/** Aim-assist pull strength toward target when throwing (0 = none, 1 = full). */
const ASSIST_STRENGTH = 0.05;

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
  // Pointer-lock: raw integrated target, plus eased "shown" angles.
  private yawT = 0;
  private pitchT = 0;
  private yaw = 0;
  private pitch = 0;

  // Free-mouse mode: NDC [-1,1] of cursor, or null when not tracked
  private cursorNDC: { x: number; y: number } | null = null;
  // Smoothed free-mouse aim direction (world space) — matched feel to lock.
  private freeDir = new THREE.Vector3(0, 0, -1);
  private freeInited = false;

  // dt for this tick (set by update()); falls back to ~60fps if unset.
  private dt = 1 / 60;

  // Sticky-lock memory: index of currently locked spar (-1 = none).
  private lockedSpar = -1;

  // Working Three.js objects (reused)
  private _dir  = new THREE.Vector3(0, 0, -1);
  private _ray  = new THREE.Ray();
  private _tmpV = new THREE.Vector3();
  private _q    = new THREE.Quaternion();
  private _e    = new THREE.Euler(0, 0, 0, 'YXZ');

  // ── External setters ────────────────────────────────────────────────────────

  /** Set canvas-relative cursor position for free-mouse aim (no pointer lock). */
  setFreeMouse(nx: number, ny: number): void {
    // nx, ny ∈ [-1,1] NDC
    this.cursorNDC = { x: nx, y: ny };
  }

  /** Clear free-mouse position (reverts to pointer-lock aim). */
  clearFreeMouse(): void {
    this.cursorNDC = null;
    this.freeInited = false;
  }

  // ── Update ───────────────────────────────────────────────────────────────────

  /**
   * Integrate pointer-lock mouse delta.
   * @param dx     raw movementX
   * @param dy     raw movementY
   * @param dtSec  seconds this tick (for frame-rate-independent smoothing)
   */
  update(dx: number, dy: number, dtSec?: number): void {
    if (dtSec && dtSec > 1e-5) this.dt = Math.min(0.1, dtSec);

    // Integrate raw deltas 1:1 into the target — zero accumulating lag.
    this.yawT -= dx * SENS_X;
    this.pitchT -= dy * SENS_Y;
    this.pitchT = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, this.pitchT));

    // Ease shown angles toward target: crisp but de-jittered.
    const k = 1 - Math.exp(-FOLLOW_LAMBDA * this.dt);
    this.yaw += (this.yawT - this.yaw) * k;
    this.pitch += (this.pitchT - this.pitch) * k;
  }

  // ── View-agency hooks for the chase camera ──────────────────────────────────

  /**
   * Normalised look intent for GameCamera.setLook(). Maps the accumulated
   * yaw/pitch into [-1,1] of a comfortable look budget so the player can
   * glance around without the framing guarantee ever breaking. Returns
   * whether the player is actively looking (non-trivial offset).
   */
  lookIntent(): { yaw: number; pitch: number; active: boolean } {
    // ~40° of yaw / ~28° of pitch saturates the budget.
    const YB = 0.7;
    const PB = 0.5;
    const y = Math.max(-1, Math.min(1, this.yaw / YB));
    const p = Math.max(-1, Math.min(1, this.pitch / PB));
    const active = Math.hypot(this.yawT, this.pitchT) > 0.02;
    return { yaw: y, pitch: p, active };
  }

  // ── Aim direction ────────────────────────────────────────────────────────────

  /**
   * Current world-space aim direction.
   * In pointer-lock mode: camera quaternion × accumulated yaw/pitch.
   * In free-mouse mode: un-project NDC cursor through camera (smoothed).
   *
   * @param aimAssistTarget  Optional world position to bias toward (weak pull).
   */
  aimDir(camera: THREE.Camera, aimAssistTarget?: Vec3): Vec3 {
    let dir: THREE.Vector3;

    if (this.cursorNDC) {
      // Free-mouse: cast a ray through the NDC cursor position.
      this._tmpV.set(this.cursorNDC.x, this.cursorNDC.y, 0.5);
      this._tmpV.unproject(camera);
      const camPos = new THREE.Vector3();
      camera.getWorldPosition(camPos);
      this._tmpV.sub(camPos).normalize();
      if (!this.freeInited) {
        this.freeDir.copy(this._tmpV);
        this.freeInited = true;
      } else {
        // Slerp-ish ease, matched to pointer-lock follow feel.
        const k = 1 - Math.exp(-FREE_LAMBDA * this.dt);
        this.freeDir.lerp(this._tmpV, k).normalize();
      }
      dir = this.freeDir.clone();
    } else {
      // Pointer-lock: apply accumulated yaw/pitch relative to camera.
      this._e.set(this.pitch, this.yaw, 0, 'YXZ');
      this._q.setFromEuler(this._e);
      dir = this._dir.clone().set(0, 0, -1).applyQuaternion(this._q).applyQuaternion(camera.quaternion).normalize();
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
   * Sticky lock: a spar we're already locked to keeps the lock until the
   * aim wanders past SNAP_BREAK (vs SNAP_CONE to acquire) — no chatter.
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
    let bestSparIdx = -1;
    let bestSparPos: THREE.Vector3 | null = null;

    const acquireCos = Math.cos(SNAP_CONE);
    const breakCos   = Math.cos(SNAP_BREAK);

    // ── Spar candidates ───────────────────────────────────────────────────────
    for (let i = 0; i < spars.length; i++) {
      const sp = spars[i];
      this._tmpV.set(sp.x, sp.y, sp.z);
      const toSpar = this._tmpV.clone().sub(camPos).normalize();
      const dot    = toSpar.dot(aimW);

      // Sticky: the currently-locked spar gets the wider break cone.
      const thresh = i === this.lockedSpar ? breakCos : acquireCos;
      if (dot < thresh) continue;

      const dist = origin.distanceTo(this._tmpV);
      if (dist > SNAP_RANGE) continue;

      // Bias toward the held lock so we don't flicker between two near spars.
      const score = dot + (i === this.lockedSpar ? 0.05 : 0);
      if (score > bestSparDot) {
        bestSparDot = score;
        bestSparIdx = i;
        bestSparPos = this._tmpV.clone();
      }
    }

    // Spars always win over the skin cylinder
    if (bestSparPos) {
      this.lockedSpar = bestSparIdx;
      return {
        pos:        { x: bestSparPos.x, y: bestSparPos.y, z: bestSparPos.z },
        valid:      true, // dist checked above
        outOfRange: false,
        locked:     true,
      };
    }
    this.lockedSpar = -1;

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
          if (t > 0) {
            const hitX = this._ray.origin.x + this._ray.direction.x * t;
            if (hitX >= -GATE_X && hitX <= GATE_X) {
              const hitY = oy + dy * t;
              const hitZ = oz + dz * t;
              const inRange = t <= SNAP_RANGE;
              return {
                pos:        { x: hitX, y: hitY, z: hitZ },
                valid:      inRange,
                outOfRange: !inRange,
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
    this.yawT     = 0;
    this.pitchT   = 0;
    this.lockedSpar = -1;
    this.freeInited = false;
  }
}

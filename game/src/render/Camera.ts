// BROADCAST RAIL CAMERA — fixed orientation, slow single-axis dolly.
//
// History: every dynamic version (follow/lead/cut/loop-cam) was called
// "moving around rapidly, hard to understand" by real players — multi-axis
// reframing destroys the spatial model. A fully static whole-tube shot
// fixed that but left the action a few unreadable pixels. The resolution
// is the standard sports-broadcast solution: the camera NEVER rotates,
// NEVER zooms, NEVER cuts — its orientation and FOV are constant — it only
// glides slowly along ONE axis (the tube length) so the action stays
// framed and big. Predictable like a touchline rail, not the old chaos.
//
// Public API (update / cinematic / setLook / cycleCinematicAngle / reset)
// is preserved so main.ts and the spectate path are unchanged.

import * as THREE from 'three';

type V3 = { x: number; y: number; z: number };

// Side vantage: close enough that riggers/bell read, far enough that a
// good span of tube + the cross-section are in frame. Constant forever.
const CAM_Y = 58;       // height above the axis (constant pitch)
const CAM_Z = -150;     // side standoff (constant — tighter than the old -395)
const CAM_FOV = 56;     // constant
const RAIL_CLAMP = 300; // don't dolly past the very ends (rings at ±320)
const RAIL_LAMBDA = 1.6; // slow, smooth follow along x — never jerky

export class GameCamera {
  private camX = 0;       // current dolly position along the tube
  private inited = false;

  constructor(private cam: THREE.PerspectiveCamera) {
    this.cam.fov = CAM_FOV;
    this.cam.updateProjectionMatrix();
    this._apply();
  }

  /**
   * Dolly the FIXED-orientation camera toward `focusX` along the tube.
   * Orientation, height, standoff and FOV never change — only x glides,
   * slowly and smoothly, so it is never disorienting.
   */
  private rail(focusX: number, dt: number): void {
    const target = Math.max(-RAIL_CLAMP, Math.min(RAIL_CLAMP, focusX));
    if (!this.inited) {
      this.camX = target;
      this.inited = true;
    } else {
      const k = 1 - Math.exp(-RAIL_LAMBDA * Math.max(1 / 240, Math.min(dt, 1 / 15)));
      this.camX += (target - this.camX) * k;
    }
    this._apply();
  }

  private _apply(): void {
    this.cam.position.set(this.camX, CAM_Y, CAM_Z);
    this.cam.up.set(0, 1, 0);
    this.cam.lookAt(this.camX, 0, 0); // look straight across at the axis line
    if (Math.abs(this.cam.fov - CAM_FOV) > 0.01) {
      this.cam.fov = CAM_FOV;
      this.cam.updateProjectionMatrix();
    }
  }

  // ── Preserved API ─────────────────────────────────────────────────────────

  /** Play camera: rail gently follows YOUR rigger along the tube. */
  update(
    _bell: V3, p1: V3, _attackX: number,
    _loopGlow: number, _skinR: number, dt: number,
  ): void {
    this.rail(p1.x, dt);
  }

  /** Spectate camera: rail gently follows the bell. */
  cinematic(
    bell: V3,
    _players: { id: string; p: V3; team: 'home' | 'away' }[],
    _attackX: number, _loopGlow: number, _skinR: number, dt: number,
  ): void {
    this.rail(bell.x, dt);
  }

  /** View-agency removed (was part of the disorientation). No-op. */
  setLook(_yaw: number, _pitch: number, _active: boolean): void {
    /* the camera does not rotate */
  }

  /** One angle only. No-op. */
  cycleCinematicAngle(): void {
    /* nothing to cycle */
  }

  /** Snap the dolly on (re)entry so frame 1 is already framed. */
  reset(): void {
    this.inited = false;
  }
}

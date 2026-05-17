// AUTO-FOLLOW RAIL CAMERA.
//
// The player wants to directly drive their rigger — the camera should
// just keep them in view, not be another thing to manage. So: fixed
// orientation / height / zoom that NEVER rotates, zooms or cuts; it only
// glides along the tube's length to keep the followed subject (your
// rigger in play, the bell in spectate) framed. Predictable, hands-off.
//
// Public API (update / cinematic / setControl / setLook /
// cycleCinematicAngle / reset) preserved so main.ts is unchanged.

import * as THREE from 'three';

type V3 = { x: number; y: number; z: number };

const CAM_Y = 46;       // constant height above the axis
const CAM_Z = -115;     // closer side standoff so the tube fills the frame
const CAM_FOV = 58;     // constant
const FOLLOW_LAMBDA = 4; // smooth glide along x — never jerky

export class GameCamera {
  private camX = 0;
  private inited = false;

  constructor(private cam: THREE.PerspectiveCamera) {
    this.cam.fov = CAM_FOV;
    this.cam.up.set(0, 1, 0);
    this.cam.updateProjectionMatrix();
    this._apply();
  }

  private follow(focusX: number, dt: number): void {
    const t = Math.max(-300, Math.min(300, focusX));
    if (!this.inited) { this.camX = t; this.inited = true; }
    else {
      const k = 1 - Math.exp(-FOLLOW_LAMBDA * Math.max(1 / 240, Math.min(dt, 1 / 15)));
      this.camX += (t - this.camX) * k;
    }
    this._apply();
  }

  private _apply(): void {
    this.cam.position.set(this.camX, CAM_Y, CAM_Z);
    this.cam.up.set(0, 1, 0);
    this.cam.lookAt(this.camX, 0, 0);
    if (Math.abs(this.cam.fov - CAM_FOV) > 0.01) {
      this.cam.fov = CAM_FOV;
      this.cam.updateProjectionMatrix();
    }
  }

  // ── Preserved API ─────────────────────────────────────────────────────────
  /** Play: follow YOUR rigger along the tube. */
  update(
    _bell: V3, p1: V3, _attackX: number,
    _loopGlow: number, _skinR: number, dt: number,
  ): void {
    this.follow(p1.x, dt);
  }

  /** Spectate: follow the bell. */
  cinematic(
    bell: V3,
    _players: { id: string; p: V3; team: 'home' | 'away' }[],
    _attackX: number, _loopGlow: number, _skinR: number, dt: number,
  ): void {
    this.follow(bell.x, dt);
  }

  setControl(_c: { yaw: number; pitch: number; dist: number; follow: 'player' | 'ball' }): void {
    /* camera is hands-off auto-follow now */
  }
  setLook(_yaw: number, _pitch: number, _active: boolean): void { /* no-op */ }
  cycleCinematicAngle(): void { /* no-op */ }
  reset(): void { this.inited = false; }
}

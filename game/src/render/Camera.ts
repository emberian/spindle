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
const CAM_FOV = 58;     // resting FOV
const FOLLOW_LAMBDA = 4; // smooth glide along x — never jerky

// Speed-coupled FOV liveness: a static frame flattens the new athletic motion +
// blazing trail, so widen the lens *slightly* with subject speed to sell pace.
// Strictly bounded & heavily eased — NO rotation, dolly, shake or cut, ever.
const FOV_MAX_GAIN = 5;        // hard cap: 58 → at most 63
const SPEED_FULL = 26;         // m/s mapping to full gain
const SPEED_LAMBDA = 2.2;      // heavy speed smoothing (slower than the glide)
const FOV_LAMBDA = 2.6;        // heavy FOV smoothing — no snap on speed spikes

export class GameCamera {
  private camX = 0;
  private inited = false;
  private fov = CAM_FOV;
  private spd = 0;
  private prevFocus: V3 | null = null;

  constructor(private cam: THREE.PerspectiveCamera) {
    this.cam.fov = CAM_FOV;
    this.cam.up.set(0, 1, 0);
    this.cam.updateProjectionMatrix();
    this._apply();
  }

  /** Read-only 3D speed estimate of the followed subject (no sim coupling). */
  private trackSpeed(focus: V3, dt: number): void {
    const h = Math.max(1 / 240, Math.min(dt, 1 / 15));
    if (this.prevFocus) {
      const dx = focus.x - this.prevFocus.x;
      const dy = focus.y - this.prevFocus.y;
      const dz = focus.z - this.prevFocus.z;
      let inst = Math.sqrt(dx * dx + dy * dy + dz * dz) / h;
      if (inst > 80) inst = this.spd;          // ignore teleport/re-grip jumps
      const ks = 1 - Math.exp(-SPEED_LAMBDA * h);
      this.spd += (inst - this.spd) * ks;
    }
    this.prevFocus = { x: focus.x, y: focus.y, z: focus.z };
    const targetFov = CAM_FOV + FOV_MAX_GAIN * Math.min(1, this.spd / SPEED_FULL);
    const kf = 1 - Math.exp(-FOV_LAMBDA * h);
    this.fov += (targetFov - this.fov) * kf;
  }

  private follow(focus: V3, dt: number): void {
    this.trackSpeed(focus, dt);
    const t = Math.max(-300, Math.min(300, focus.x));
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
    if (Math.abs(this.cam.fov - this.fov) > 0.01) {
      this.cam.fov = this.fov;
      this.cam.updateProjectionMatrix();
    }
  }

  // ── Preserved API ─────────────────────────────────────────────────────────
  /** Play: follow YOUR rigger along the tube. */
  update(
    _bell: V3, p1: V3, _attackX: number,
    _loopGlow: number, _skinR: number, dt: number,
  ): void {
    this.follow(p1, dt);
  }

  /** Spectate: follow the bell. */
  cinematic(
    bell: V3,
    _players: { id: string; p: V3; team: 'home' | 'away' }[],
    _attackX: number, _loopGlow: number, _skinR: number, dt: number,
  ): void {
    this.follow(bell, dt);
  }

  setControl(_c: { yaw: number; pitch: number; dist: number; follow: 'player' | 'ball' }): void {
    /* camera is hands-off auto-follow now */
  }
  setLook(_yaw: number, _pitch: number, _active: boolean): void { /* no-op */ }
  cycleCinematicAngle(): void { /* no-op */ }
  reset(): void { this.inited = false; this.prevFocus = null; this.spd = 0; this.fov = CAM_FOV; }
}

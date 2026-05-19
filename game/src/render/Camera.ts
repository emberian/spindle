// GAME CAMERA — auto-follow rail + freefly mode.
//
// Default: hands-off rail that glides along the tube's length to keep the
// followed subject (your rigger in play, the bell in spectate) framed.
//
// Freefly (toggle with C key): full 6DOF camera. WASD = move, mouse = look,
// Shift = fast, Space/Ctrl = up/down. Press C again to snap back to auto.
//
// Public API (update / cinematic / setControl / setLook /
// cycleCinematicAngle / reset) preserved so main.ts is unchanged.

import * as THREE from 'three';

type V3 = { x: number; y: number; z: number };

const CAM_Y = 46;
const CAM_Z = -128;
const CAM_FOV = 60;
const FOLLOW_LAMBDA = 4;

const FOV_MAX_GAIN = 5;
const SPEED_FULL = 26;
const SPEED_LAMBDA = 2.2;
const FOV_LAMBDA = 2.6;

// Freefly settings
const FLY_SPEED = 60;       // m/s base
const FLY_FAST = 180;       // m/s with shift
const MOUSE_SENS = 0.002;   // rad/px

export class GameCamera {
  private camX = 0;
  private inited = false;
  private fov = CAM_FOV;
  private spd = 0;
  private prevFocus: V3 | null = null;

  // Freefly state
  private _freefly = false;
  private _yaw = 0;
  private _pitch = 0;
  private _keys = new Set<string>();
  private _pointerLocked = false;
  private _canvas: HTMLElement | null = null;

  get freefly(): boolean { return this._freefly; }

  constructor(private cam: THREE.PerspectiveCamera) {
    this.cam.fov = CAM_FOV;
    this.cam.up.set(0, 1, 0);
    this.cam.updateProjectionMatrix();
    this._apply();
    this._bindFreefly();
  }

  private _bindFreefly(): void {
    addEventListener('keydown', (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;

      if (e.code === 'KeyC' && !e.repeat) {
        e.preventDefault();
        this._freefly = !this._freefly;
        if (this._freefly) {
          this._yaw = 0;
          this._pitch = 0;
          const euler = new THREE.Euler().setFromQuaternion(this.cam.quaternion, 'YXZ');
          this._yaw = euler.y;
          this._pitch = euler.x;
          this._requestPointerLock();
        } else {
          this._releasePointerLock();
          this.inited = false;
        }
        return;
      }
      this._keys.add(e.code);
    });
    addEventListener('keyup', (e: KeyboardEvent) => {
      this._keys.delete(e.code);
    });
    addEventListener('mousemove', (e: MouseEvent) => {
      if (!this._freefly || !this._pointerLocked) return;
      this._yaw -= e.movementX * MOUSE_SENS;
      this._pitch -= e.movementY * MOUSE_SENS;
      this._pitch = Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, this._pitch));
    });
    addEventListener('pointerlockchange', () => {
      this._pointerLocked = document.pointerLockElement != null;
      if (!this._pointerLocked && this._freefly) {
        // Lost lock without pressing C — stay in freefly, user can click to re-lock
      }
    });
  }

  private _requestPointerLock(): void {
    const el = this._canvas ?? document.querySelector('canvas');
    if (el) {
      this._canvas = el;
      el.requestPointerLock?.();
    }
  }

  private _releasePointerLock(): void {
    if (document.pointerLockElement) document.exitPointerLock?.();
  }

  private _flyTick(dt: number): void {
    const speed = this._keys.has('ShiftLeft') || this._keys.has('ShiftRight') ? FLY_FAST : FLY_SPEED;
    const move = new THREE.Vector3();

    if (this._keys.has('KeyW')) move.z -= 1;
    if (this._keys.has('KeyS')) move.z += 1;
    if (this._keys.has('KeyA')) move.x -= 1;
    if (this._keys.has('KeyD')) move.x += 1;
    if (this._keys.has('Space')) move.y += 1;
    if (this._keys.has('ControlLeft') || this._keys.has('ControlRight')) move.y -= 1;

    if (move.lengthSq() > 0) {
      move.normalize().multiplyScalar(speed * dt);
      const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(this._pitch, this._yaw, 0, 'YXZ'));
      move.applyQuaternion(q);
      this.cam.position.add(move);
    }

    this.cam.quaternion.setFromEuler(new THREE.Euler(this._pitch, this._yaw, 0, 'YXZ'));
  }

  private trackSpeed(focus: V3, dt: number): void {
    const h = Math.max(1 / 240, Math.min(dt, 1 / 15));
    if (this.prevFocus) {
      const dx = focus.x - this.prevFocus.x;
      const dy = focus.y - this.prevFocus.y;
      const dz = focus.z - this.prevFocus.z;
      let inst = Math.sqrt(dx * dx + dy * dy + dz * dz) / h;
      if (inst > 80) inst = this.spd;
      const ks = 1 - Math.exp(-SPEED_LAMBDA * h);
      this.spd += (inst - this.spd) * ks;
    }
    this.prevFocus = { x: focus.x, y: focus.y, z: focus.z };
    const targetFov = CAM_FOV + FOV_MAX_GAIN * Math.min(1, this.spd / SPEED_FULL);
    const kf = 1 - Math.exp(-FOV_LAMBDA * h);
    this.fov += (targetFov - this.fov) * kf;
  }

  private follow(focus: V3, dt: number): void {
    if (this._freefly) { this._flyTick(dt); return; }
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
  update(
    _bell: V3, p1: V3, _attackX: number,
    _loopGlow: number, _skinR: number, dt: number,
  ): void {
    this.follow(p1, dt);
  }

  cinematic(
    bell: V3,
    _players: { id: string; p: V3; team: 'home' | 'away' }[],
    _attackX: number, _loopGlow: number, _skinR: number, dt: number,
  ): void {
    this.follow(bell, dt);
  }

  setControl(_c: { yaw: number; pitch: number; dist: number; follow: 'player' | 'ball' }): void { }
  setLook(_yaw: number, _pitch: number, _active: boolean): void { }
  cycleCinematicAngle(): void {
    this._freefly = !this._freefly;
    if (this._freefly) {
      const euler = new THREE.Euler().setFromQuaternion(this.cam.quaternion, 'YXZ');
      this._yaw = euler.y;
      this._pitch = euler.x;
      this._requestPointerLock();
    } else {
      this._releasePointerLock();
      this.inited = false;
    }
  }
  reset(): void { this.inited = false; this.prevFocus = null; this.spd = 0; this.fov = CAM_FOV; this._freefly = false; }
}

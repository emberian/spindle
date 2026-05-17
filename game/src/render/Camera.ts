// PLAYER-CONTROLLED ORBIT CAMERA.
//
// Every autonomous model failed ("moving around rapidly", "can't judge
// distance"). Per Kanzo's explicit ask, the camera now has real controls
// and NEVER moves on its own: the player orbits it (mouse), zooms it
// (wheel) and switches its follow target (C = your rigger ↔ the bell).
// It only tracks the focus *position* (smoothly, so the subject stays in
// frame); the angle/zoom are entirely the player's and stay put until
// they change them. Constant world-up so distance reads normally.
//
// Public API (update / cinematic / setLook / cycleCinematicAngle / reset)
// is preserved; control comes via setControl(), called each frame from
// InputManager.cameraState.

import * as THREE from 'three';

type V3 = { x: number; y: number; z: number };

export class GameCamera {
  private yaw = 0.5;
  private pitch = 0.32;
  private dist = 70;
  private follow: 'player' | 'ball' = 'player';
  private fx = 0;   // smoothed focus (so a moving target stays framed)
  private fy = 0;
  private fz = 0;
  private inited = false;

  constructor(private cam: THREE.PerspectiveCamera) {
    this.cam.fov = 55;
    this.cam.up.set(0, 1, 0);
    this.cam.updateProjectionMatrix();
  }

  /** Player camera control (from InputManager.cameraState) each frame. */
  setControl(c: { yaw: number; pitch: number; dist: number; follow: 'player' | 'ball' }): void {
    this.yaw = c.yaw;
    this.pitch = c.pitch;
    this.dist = c.dist;
    this.follow = c.follow;
  }

  private orbit(player: V3, ball: V3, dt: number): void {
    const f = this.follow === 'ball' ? ball : player;
    if (!this.inited) {
      this.fx = f.x; this.fy = f.y; this.fz = f.z;
      this.inited = true;
    } else {
      // Track the focus position only — smooth, never jerky; the ANGLE is
      // the player's and is not touched here.
      const k = 1 - Math.exp(-6 * Math.max(1 / 240, Math.min(dt, 1 / 15)));
      this.fx += (f.x - this.fx) * k;
      this.fy += (f.y - this.fy) * k;
      this.fz += (f.z - this.fz) * k;
    }
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    this.cam.position.set(
      this.fx + cp * sy * this.dist,
      this.fy + sp * this.dist,
      this.fz + cp * cy * this.dist,
    );
    this.cam.up.set(0, 1, 0);
    this.cam.lookAt(this.fx, this.fy, this.fz);
  }

  // ── Preserved API ─────────────────────────────────────────────────────────
  update(
    bell: V3, p1: V3, _attackX: number,
    _loopGlow: number, _skinR: number, dt: number,
  ): void {
    this.orbit(p1, bell, dt);
  }

  cinematic(
    bell: V3,
    players: { id: string; p: V3; team: 'home' | 'away' }[],
    _attackX: number, _loopGlow: number, _skinR: number, dt: number,
  ): void {
    // Spectate: no human rigger — follow the bell (or first player if
    // the user toggled to 'player').
    const p = players[0]?.p ?? bell;
    this.orbit(p, bell, dt);
  }

  setLook(_yaw: number, _pitch: number, _active: boolean): void {
    /* superseded by setControl() */
  }

  cycleCinematicAngle(): void {
    this.follow = this.follow === 'player' ? 'ball' : 'player';
  }

  reset(): void {
    this.inited = false;
  }
}

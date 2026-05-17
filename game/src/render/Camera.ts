// FULLY STATIC BROADCAST CAMERA.
//
// Every motion model (follow/lead/cut/loop-cam, and even a slow rail
// dolly) was reported as disorienting by real players. Since the assisted
// shot now draws a bold arc to the ring and the HUD/marker carry the
// feedback, the camera does NOT need to chase anything. So it is a single
// constant pose: it never moves, rotates, zooms or cuts — ever. A wide
// side vantage that comfortably shows the whole active span of the tube.
//
// The public API (update / cinematic / setLook / cycleCinematicAngle /
// reset) is preserved so main.ts and the spectate path are unchanged;
// every one is a no-op around the one constant pose.

import * as THREE from 'three';

type V3 = { x: number; y: number; z: number };

const CAM_POS = new THREE.Vector3(0, 110, -300); // constant, forever
const CAM_LOOK = new THREE.Vector3(0, 0, 0);
const CAM_FOV = 54;

export class GameCamera {
  constructor(private cam: THREE.PerspectiveCamera) {
    this._apply();
  }

  private _apply(): void {
    this.cam.position.copy(CAM_POS);
    this.cam.up.set(0, 1, 0);
    this.cam.lookAt(CAM_LOOK);
    if (Math.abs(this.cam.fov - CAM_FOV) > 0.01) {
      this.cam.fov = CAM_FOV;
      this.cam.updateProjectionMatrix();
    }
  }

  // ── Preserved API — all are the one constant pose / no-ops ────────────────
  update(
    _bell: V3, _p1: V3, _attackX: number,
    _loopGlow: number, _skinR: number, _dt: number,
  ): void {
    this._apply();
  }

  cinematic(
    _bell: V3,
    _players: { id: string; p: V3; team: 'home' | 'away' }[],
    _attackX: number, _loopGlow: number, _skinR: number, _dt: number,
  ): void {
    this._apply();
  }

  setLook(_yaw: number, _pitch: number, _active: boolean): void {
    /* the camera does not move */
  }

  cycleCinematicAngle(): void {
    /* nothing to cycle */
  }

  reset(): void {
    this._apply();
  }
}

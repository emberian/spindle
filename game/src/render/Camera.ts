// FIXED BROADCAST CAMERA — deliberately NOT dynamic.
//
// Every prior version followed / led / cut / loop-cammed the action. Two
// real players independently said the camera "moving around rapidly makes
// it hard to understand" — it was the #1 reason the game was unreadable.
// You cannot build a spatial model of a 3D zero-g sport when the frame
// constantly reframes itself. So the autonomous camera is RIPPED OUT: a
// single static stadium vantage of the whole calm that never moves. The
// public API (update / cinematic / setLook / cycleCinematicAngle / reset)
// is preserved so main.ts is unchanged; every one now yields, or is a
// no-op around, the one fixed pose.

import * as THREE from 'three';

type V3 = { x: number; y: number; z: number };

export class GameCamera {
  constructor(private cam: THREE.PerspectiveCamera) {
    this.fixedBroadcast();
  }

  /**
   * The one and only camera pose. Idempotent → zero motion. A side-on,
   * lifted 3/4 stadium view of the whole tube (x ∈ [−320, 320], radius 45)
   * looking at its centre: length reads left↔right (field progress), the
   * cross-section reads with depth, both rings + every rigger + the bell
   * are always in a frame that never changes. Tuned by eye.
   */
  fixedBroadcast(): void {
    this.cam.position.set(0, 120, -395);
    this.cam.up.set(0, 1, 0);
    this.cam.lookAt(0, 0, 0);
    if (Math.abs(this.cam.fov - 60) > 0.01) {
      this.cam.fov = 60;
      this.cam.updateProjectionMatrix();
    }
  }

  // ── Preserved API — all collapse to the fixed pose ────────────────────────

  /** Play camera. Dynamic chase/lead/loop REMOVED — fixed pose. */
  update(
    _bell: V3, _p1: V3, _attackX: number,
    _loopGlow: number, _skinR: number, _dt: number,
  ): void {
    this.fixedBroadcast();
  }

  /** Spectate camera. Dynamic director REMOVED — fixed pose. */
  cinematic(
    _bell: V3,
    _players: { id: string; p: V3; team: 'home' | 'away' }[],
    _attackX: number, _loopGlow: number, _skinR: number, _dt: number,
  ): void {
    this.fixedBroadcast();
  }

  /** Player view-agency removed (it was part of the disorientation). No-op. */
  setLook(_yaw: number, _pitch: number, _active: boolean): void {
    /* intentionally empty — the camera does not move */
  }

  /** Camera-cycle button now does nothing (only one fixed angle). No-op. */
  cycleCinematicAngle(): void {
    /* intentionally empty — there is one fixed angle */
  }

  /** Nothing to reset on a fixed camera; re-assert the pose for safety. */
  reset(): void {
    this.fixedBroadcast();
  }
}

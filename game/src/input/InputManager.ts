// InputManager — attaches to a canvas; produces the human's PlayerInput for
// P1 each tick from devices. Fixes v0.1's blind-tether / invisible-charge by
// exposing reticleState, chargeLevel, and a predicted Coriolis ghost arc.
//
// Device bindings:
//   Mouse aim (pointer-lock)   → aim direction via AimModel
//   LMB                        → fireLineAt snapped anchor / start bell-charge
//   Hold LMB (on bell)         → charge throw (0..1), release = throw
//   A / D  or  scroll          → throwSpin [-1..1]
//   SHIFT                      → pushoff
//   W / S                      → reel in/out
//   SPACE                      → alt action (thrumbler nudge)

import * as THREE from 'three';
import { AimModel, type AnchorResult } from './AimModel';
import { predictPath } from '../sim/trajectory';
import { REG } from '../sim/RegConstants';
import type { PlayerInput } from '../sim/types';
import type { Vec3 } from '../sim/vec';

// ── Config ────────────────────────────────────────────────────────────────

const CHARGE_RATE     = 1.2;  // full charge in ~0.83 s of hold
const THROW_SPEED_MAX = 28.0; // m/s at full charge
const THROW_SPEED_MIN =  4.0;
const SPIN_SCROLL_K   = 0.003; // scroll delta → spin
const SPIN_DECAY      = 0.92;  // per-tick spin decay

// Arc ghost: number of trajectory samples and step size
const GHOST_STEPS = 28;
const GHOST_H     = 0.04; // seconds per step

// ── Reticle states ────────────────────────────────────────────────────────

export type ReticleState = 'valid' | 'invalid' | 'out_of_range' | 'none';

// ── InputView exposed to HUD ──────────────────────────────────────────────

export interface InputView {
  reticleState: ReticleState;
  reticlePos:   Vec3 | null;
  chargeLevel:  number;        // [0,1]
  ghostArc:     Vec3[];        // predicted Coriolis path samples
}

// ── InputManager ─────────────────────────────────────────────────────────

export class InputManager {
  private aim = new AimModel();
  private canvas: HTMLElement;
  private camera: THREE.Camera;

  // Key state
  private keys = new Set<string>();
  // Mouse buttons
  private lmbDown = false;
  // Accumulated scroll spin
  private scrollSpin = 0;
  // Charge (0..1 while holding LMB on bell)
  private _charge  = 0;
  private charging = false;
  // Was LMB just released this tick?
  private lmbReleased = false;
  // Pointer-lock mouse delta (accumulated per tick)
  private mouseDX = 0;
  private mouseDY = 0;
  // Whether the bell is currently held by P1 (set externally each frame)
  private holdingBell = false;
  // Player world position (set externally each frame)
  private playerPos: Vec3 = { x: 0, y: 0, z: 0 };
  // Player velocity (for arc ghost origin)
  private playerVel: Vec3 = { x: 0, y: 0, z: 0 };
  // Spar list for snapping (set externally)
  private spars: Vec3[] = [];

  // Computed this tick
  private _reticle: AnchorResult | null = null;
  private _ghostArc: Vec3[] = [];

  constructor(canvas: HTMLElement, camera: THREE.Camera) {
    this.canvas = canvas;
    this.camera = camera;
    this._bindEvents();
  }

  // ── External setters (call before get() each frame) ───────────────────

  setPlayerState(pos: Vec3, vel: Vec3, holdingBell: boolean): void {
    this.playerPos  = pos;
    this.playerVel  = vel;
    this.holdingBell = holdingBell;
  }

  setSpars(spars: Vec3[]): void {
    this.spars = spars;
  }

  // ── Tick ──────────────────────────────────────────────────────────────

  /**
   * Advance one simulation tick (dtSec = seconds elapsed this step).
   * Returns the PlayerInput to feed into the sim.
   */
  get(dtSec: number): PlayerInput {
    // Integrate mouse
    this.aim.update(this.mouseDX, this.mouseDY);
    this.mouseDX = 0;
    this.mouseDY = 0;

    // Aim direction
    const aimDir = this.aim.aimDir(this.camera);

    // Snapped anchor
    this._reticle = this.aim.anchorTarget(this.spars, this.camera, this.playerPos);

    // Charge: only while LMB held AND holding bell
    let throwReleased = false;
    if (this.lmbDown && this.holdingBell) {
      this.charging  = true;
      this._charge   = Math.min(1.0, this._charge + CHARGE_RATE * dtSec);
    }
    if (this.lmbReleased && this.charging) {
      throwReleased  = true;
      this.charging  = false;
    }
    if (!this.holdingBell) {
      this._charge  = 0;
      this.charging = false;
    }

    // Throw spin from A/D or scroll
    const keySpinL = this.keys.has('KeyA') || this.keys.has('ArrowLeft');
    const keySpinR = this.keys.has('KeyD') || this.keys.has('ArrowRight');
    if (keySpinL)  this.scrollSpin = Math.max(-1, this.scrollSpin - 0.04);
    if (keySpinR)  this.scrollSpin = Math.min( 1, this.scrollSpin + 0.04);
    if (!keySpinL && !keySpinR) this.scrollSpin *= SPIN_DECAY;
    const throwSpin = Math.max(-1, Math.min(1, this.scrollSpin));

    // Reel
    const reelIn  = this.keys.has('KeyW') || this.keys.has('ArrowUp');
    const reelOut = this.keys.has('KeyS') || this.keys.has('ArrowDown');
    const reel: -1 | 0 | 1 = reelIn ? -1 : reelOut ? 1 : 0;

    // FireLineAt: LMB pressed this tick while NOT charging a throw
    const fireLineAt: Vec3 | null =
      (this.lmbReleased && !throwReleased && this._reticle?.valid)
        ? this._reticle.pos
        : null;

    // Pushoff: SHIFT
    const pushoff = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');

    // SPACE thrumbler nudge in aim direction
    const spaceHeld = this.keys.has('Space');
    const thrumbler: Vec3 = spaceHeld
      ? { x: aimDir.x * 0.4, y: aimDir.y * 0.4, z: aimDir.z * 0.4 }
      : { x: 0, y: 0, z: 0 };

    // Ghost arc: sample predicted Coriolis path from current throw state
    if (this.holdingBell && this._charge > 0.01) {
      const speed = THROW_SPEED_MIN + (THROW_SPEED_MAX - THROW_SPEED_MIN) * this._charge;
      const v0: Vec3 = {
        x: this.playerVel.x + aimDir.x * speed,
        y: this.playerVel.y + aimDir.y * speed,
        z: this.playerVel.z + aimDir.z * speed,
      };
      this._ghostArc = predictPath(this.playerPos, v0, REG.omega, GHOST_H, GHOST_STEPS);
    } else {
      this._ghostArc = [];
    }

    // Consume LMB-released flag
    this.lmbReleased = false;

    return {
      id:           'P1',
      aim:          aimDir,
      fireLineAt,
      reel,
      release:      false,
      pushoff,
      throwCharge:  this._charge,
      throwReleased,
      throwSpin,
      thrumbler,
    };
  }

  /** View data for the HUD — call after get(). */
  get view(): InputView {
    let reticleState: ReticleState = 'none';
    if (this._reticle) {
      reticleState = this._reticle.outOfRange
        ? 'out_of_range'
        : this._reticle.valid
          ? 'valid'
          : 'invalid';
    }
    return {
      reticleState,
      reticlePos:  this._reticle?.pos ?? null,
      chargeLevel: this._charge,
      ghostArc:    this._ghostArc,
    };
  }

  // ── Pointer lock ──────────────────────────────────────────────────────

  requestPointerLock(): void {
    this.canvas.requestPointerLock?.();
  }

  // ── Event binding ─────────────────────────────────────────────────────

  private _bindEvents(): void {
    // Pointer lock mouse move
    document.addEventListener('mousemove', (e: MouseEvent) => {
      if (document.pointerLockElement === this.canvas) {
        this.mouseDX += e.movementX;
        this.mouseDY += e.movementY;
      }
    });

    // Mouse buttons
    this.canvas.addEventListener('mousedown', (e: MouseEvent) => {
      if (e.button === 0) {
        this.lmbDown    = true;
        this.lmbReleased = false;
        if (!document.pointerLockElement) this.requestPointerLock();
      }
    });
    this.canvas.addEventListener('mouseup', (e: MouseEvent) => {
      if (e.button === 0) {
        this.lmbDown    = false;
        this.lmbReleased = true;
      }
    });

    // Scroll → throwSpin
    this.canvas.addEventListener('wheel', (e: WheelEvent) => {
      e.preventDefault();
      this.scrollSpin = Math.max(-1, Math.min(1,
        this.scrollSpin + e.deltaY * SPIN_SCROLL_K,
      ));
    }, { passive: false });

    // Keyboard
    window.addEventListener('keydown', (e: KeyboardEvent) => {
      this.keys.add(e.code);
      e.preventDefault();
    });
    window.addEventListener('keyup', (e: KeyboardEvent) => {
      this.keys.delete(e.code);
    });

    // Pointer lock change
    document.addEventListener('pointerlockchange', () => {
      if (!document.pointerLockElement) {
        this.aim.reset();
      }
    });
  }

  dispose(): void {
    // In a real teardown we would removeEventListener — omitted for brevity
    // since RIG's game loop owns this for the session lifetime.
  }
}

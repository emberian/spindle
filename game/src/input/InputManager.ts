// InputManager — attaches to a canvas; produces P1's PlayerInput each tick.
//
// Design goals (rewrite):
//   • Forgiving aim: wide snap cone, no pixel-perfect required.
//   • Works WITHOUT pointer-lock (falls back to free-mouse via cursor NDC).
//     Pointer-lock is optional — requested on canvas click only when available.
//   • Ghost arc always visible once bell is held (not just when charging),
//     biased toward the Faith ring to teach the Coriolis curve.
//   • Throw aim-assist: gentle pull toward the attacking ring.
//   • Clean charge feel: smooth ramp, orange overflow pulse at max.
//   • Reticle states: free / anchor-lock / charging — conveyed via InputView.
//   • All keys debounced; sensible WASD + scroll defaults.
//
// Device bindings:
//   Mouse aim (pointer-lock or free)  → aim via AimModel
//   LMB (not holding bell)            → fire grapple line at snapped anchor
//   LMB hold (holding bell)           → charge throw (0..1)
//   LMB release (charging)            → throw
//   A / D  or  scroll                 → throwSpin [-1..1]
//   W / S                             → reel in/out
//   SHIFT                             → pushoff (when in contact)
//   SPACE                             → thrumbler nudge in aim direction

import * as THREE from 'three';
import { AimModel } from './AimModel';
import type { AnchorResult, ReticleState } from './AimModel';
import { predictPath } from '../sim/trajectory';
import { REG, GATE_X } from '../sim/RegConstants';
import type { PlayerInput } from '../sim/types';
import type { Vec3 } from '../sim/vec';

// Re-export ReticleState so HUD/orchestrator can import it from here.
export type { ReticleState } from './AimModel';

// ── Config ────────────────────────────────────────────────────────────────────

/** Throw charge: full charge in ~0.75 s. */
const CHARGE_RATE     = 1.33;

/** Throw speed range (m/s). Min throw is still useful for short passes. */
const THROW_SPEED_MIN =  5.0;
const THROW_SPEED_MAX = 30.0;

/** Scroll wheel → spin sensitivity. */
const SPIN_SCROLL_K   = 0.004;

/** A/D key ramp per tick (when held). */
const SPIN_KEY_RATE   = 0.05;

/** Spin decay per tick when no input. */
const SPIN_DECAY      = 0.88;

/** Ghost arc: sample count and step size (seconds). More steps = longer trail. */
const GHOST_STEPS = 32;
const GHOST_H     = 0.045;

/** Minimum ghost arc charge level — show arc as soon as bell is held. */
const GHOST_MIN_CHARGE = 0.0;

/** Minimum throw charge for the thrown bell to actually release. */
const THROW_MIN_CHARGE = 0.05;

// ── InputView exposed to HUD ──────────────────────────────────────────────────

export interface InputView {
  /** Reticle display state. */
  reticleState: ReticleState;
  /** World-space position of the snapped anchor (or null). */
  reticlePos:   Vec3 | null;
  /** Throw charge level [0,1]. */
  chargeLevel:  number;
  /** Predicted Coriolis arc samples (world space) — draw as ghost trail. */
  ghostArc:     Vec3[];
  /** True if pointer-lock is currently active. */
  pointerLocked: boolean;
}

// ── InputManager ──────────────────────────────────────────────────────────────

export class InputManager {
  private aim    = new AimModel();
  private canvas: HTMLElement;
  private camera: THREE.Camera;

  // ── Device state ────────────────────────────────────────────────────────────
  private keys        = new Set<string>();
  private lmbDown     = false;
  private lmbPressed  = false;   // edge: just went down this tick
  private lmbReleased = false;   // edge: just went up this tick
  private scrollSpin  = 0;
  private mouseDX     = 0;
  private mouseDY     = 0;
  private pointerLocked = false;

  // ── Player state (set by orchestrator each frame) ─────────────────────────
  private holdingBell = false;
  private playerPos:  Vec3 = { x: 0, y: 0, z: 0 };
  private playerVel:  Vec3 = { x: 0, y: 0, z: 0 };
  /** Faith ring X-position (attacking direction). Default = +GATE_X. */
  private faithRingX: number = GATE_X;

  // ── Spar list ────────────────────────────────────────────────────────────
  private spars: Vec3[] = [];

  // ── Charge state ─────────────────────────────────────────────────────────
  private _charge   = 0;
  private charging  = false;

  // ── Computed per-tick ────────────────────────────────────────────────────
  private _reticle:   AnchorResult | null = null;
  private _ghostArc:  Vec3[] = [];

  // ── Event cleanup refs ────────────────────────────────────────────────────
  private _offFns: Array<() => void> = [];

  constructor(canvas: HTMLElement, camera: THREE.Camera) {
    this.canvas = canvas;
    this.camera = camera;
    this._bindEvents();
  }

  // ── External setters (call before get() each frame) ───────────────────────

  setPlayerState(pos: Vec3, vel: Vec3, holdingBell: boolean): void {
    this.playerPos   = pos;
    this.playerVel   = vel;
    this.holdingBell = holdingBell;
  }

  setSpars(spars: Vec3[]): void {
    this.spars = spars;
  }

  /** Optionally update which ring is the Faith (attacking) end. */
  setFaithRingX(x: number): void {
    this.faithRingX = x;
  }

  // ── Tick ──────────────────────────────────────────────────────────────────

  /**
   * Advance one tick. Returns the PlayerInput to feed into the sim.
   * dtSec = seconds elapsed this step.
   */
  get(dtSec: number): PlayerInput {
    // 1. Integrate pointer-lock mouse movement
    this.aim.update(this.mouseDX, this.mouseDY);
    this.mouseDX = 0;
    this.mouseDY = 0;

    // 2. Aim direction: assist toward Faith ring when holding bell
    const aimDir = this.holdingBell
      ? this.aim.assistedThrowDir(this.camera, this.playerPos, this.faithRingX)
      : this.aim.aimDir(this.camera);

    // 3. Anchor snapping (always, for reticle feedback)
    this._reticle = this.aim.anchorTarget(this.spars, this.camera, this.playerPos);

    // 4. Charge logic: only while LMB held AND holding bell
    let throwReleased = false;
    if (this.lmbDown && this.holdingBell) {
      this.charging = true;
      this._charge  = Math.min(1.0, this._charge + CHARGE_RATE * dtSec);
    }
    if (this.lmbReleased && this.charging) {
      if (this._charge >= THROW_MIN_CHARGE) {
        throwReleased = true;
      }
      this.charging = false;
    }
    // Reset charge when bell leaves hand
    if (!this.holdingBell) {
      this._charge  = 0;
      this.charging = false;
    }

    // 5. Throw spin from A/D or scroll
    const keySpinL = this.keys.has('KeyA') || this.keys.has('ArrowLeft');
    const keySpinR = this.keys.has('KeyD') || this.keys.has('ArrowRight');
    if (keySpinL)              this.scrollSpin = Math.max(-1, this.scrollSpin - SPIN_KEY_RATE);
    else if (keySpinR)         this.scrollSpin = Math.min( 1, this.scrollSpin + SPIN_KEY_RATE);
    else                       this.scrollSpin *= SPIN_DECAY;
    const throwSpin = Math.max(-1, Math.min(1, this.scrollSpin));

    // 6. Reel
    const reelIn  = this.keys.has('KeyW') || this.keys.has('ArrowUp');
    const reelOut = this.keys.has('KeyS') || this.keys.has('ArrowDown');
    const reel: -1 | 0 | 1 = reelIn ? -1 : reelOut ? 1 : 0;

    // 7. FireLineAt: LMB pressed this tick while NOT holding the bell
    //    (we fire on press, not release, so grapple feels instant)
    const fireLineAt: Vec3 | null =
      (this.lmbPressed && !this.holdingBell && this._reticle?.valid)
        ? this._reticle.pos
        : null;

    // 8. Pushoff: SHIFT (only when in contact with something)
    const pushoff = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');

    // 9. Thrumbler nudge (SPACE) in aim direction
    const spaceHeld = this.keys.has('Space');
    const NUDGE = 0.35;
    const thrumbler: Vec3 = spaceHeld
      ? { x: aimDir.x * NUDGE, y: aimDir.y * NUDGE, z: aimDir.z * NUDGE }
      : { x: 0, y: 0, z: 0 };

    // 10. Ghost arc: predict Coriolis path from current throw state.
    //     Show whenever bell is held (even before charge starts) so the
    //     player can see where their *minimum* throw will go.
    if (this.holdingBell) {
      const charge = Math.max(GHOST_MIN_CHARGE, this._charge);
      const speed  = THROW_SPEED_MIN + (THROW_SPEED_MAX - THROW_SPEED_MIN) * charge;
      const v0: Vec3 = {
        x: this.playerVel.x + aimDir.x * speed,
        y: this.playerVel.y + aimDir.y * speed,
        z: this.playerVel.z + aimDir.z * speed,
      };
      this._ghostArc = predictPath(this.playerPos, v0, REG.omega, GHOST_H, GHOST_STEPS);
    } else {
      this._ghostArc = [];
    }

    // Consume edge flags
    this.lmbPressed  = false;
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

  // ── InputView for HUD (call after get()) ──────────────────────────────────

  get view(): InputView {
    let reticleState: ReticleState;

    if (this.holdingBell && this.charging) {
      reticleState = 'charging';
    } else if (this._reticle) {
      if (this._reticle.outOfRange) {
        reticleState = 'out_of_range';
      } else if (this._reticle.locked) {
        reticleState = 'anchor-lock';
      } else {
        reticleState = 'free';
      }
    } else {
      reticleState = 'free';
    }

    return {
      reticleState,
      reticlePos:    this._reticle?.pos ?? null,
      chargeLevel:   this._charge,
      ghostArc:      this._ghostArc,
      pointerLocked: this.pointerLocked,
    };
  }

  // ── Pointer lock ──────────────────────────────────────────────────────────

  requestPointerLock(): void {
    this.canvas.requestPointerLock?.();
  }

  // ── Dispose ───────────────────────────────────────────────────────────────

  dispose(): void {
    for (const off of this._offFns) off();
    this._offFns = [];
  }

  // ── Event binding ─────────────────────────────────────────────────────────

  private _bindEvents(): void {
    // addEventListener wrapper that registers a cleanup function.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const on = (el: EventTarget, type: string, fn: (e: any) => void, opts?: AddEventListenerOptions) => {
      el.addEventListener(type, fn, opts);
      this._offFns.push(() => el.removeEventListener(type, fn, opts));
    };

    // ── Pointer-lock mouse delta ──────────────────────────────────────────
    on(document, 'mousemove', (e: MouseEvent) => {
      if (this.pointerLocked) {
        this.mouseDX += e.movementX;
        this.mouseDY += e.movementY;
      }
    });

    // ── Free-mouse: track cursor over canvas when NOT pointer-locked ──────
    on(this.canvas, 'mousemove', (e: MouseEvent) => {
      if (!this.pointerLocked) {
        const rect = this.canvas.getBoundingClientRect();
        const nx = ((e.clientX - rect.left) / rect.width)  * 2 - 1;
        const ny = ((e.clientY - rect.top)  / rect.height) * -2 + 1;
        this.aim.setFreeMouse(nx, ny);
      }
    });

    // ── Mouse buttons ─────────────────────────────────────────────────────
    on(this.canvas, 'mousedown', (e: MouseEvent) => {
      if (e.button === 0) {
        this.lmbDown    = true;
        this.lmbPressed = true;
        if (!this.pointerLocked) this.canvas.requestPointerLock?.();
      }
    });

    on(this.canvas, 'mouseup', (e: MouseEvent) => {
      if (e.button === 0) {
        this.lmbDown     = false;
        this.lmbReleased = true;
      }
    });

    // ── Scroll → throwSpin ────────────────────────────────────────────────
    on(this.canvas, 'wheel', (e: WheelEvent) => {
      e.preventDefault();
      this.scrollSpin = Math.max(-1, Math.min(1,
        this.scrollSpin + e.deltaY * SPIN_SCROLL_K,
      ));
    }, { passive: false });

    // ── Keyboard ─────────────────────────────────────────────────────────
    // Only call preventDefault for game keys — don't swallow browser shortcuts.
    const GAME_KEYS = new Set([
      'KeyW','KeyA','KeyS','KeyD',
      'ArrowUp','ArrowDown','ArrowLeft','ArrowRight',
      'ShiftLeft','ShiftRight','Space',
    ]);
    on(window, 'keydown', (e: KeyboardEvent) => {
      this.keys.add(e.code);
      if (GAME_KEYS.has(e.code)) e.preventDefault();
    });
    on(window, 'keyup', (e: KeyboardEvent) => {
      this.keys.delete(e.code);
    });

    // ── Pointer-lock state ────────────────────────────────────────────────
    on(document, 'pointerlockchange', () => {
      this.pointerLocked = document.pointerLockElement === this.canvas;
      if (!this.pointerLocked) {
        // Exited lock — reset pointer-lock aim; free-mouse takes over on next move
        this.aim.reset();
        this.aim.clearFreeMouse();
        this.mouseDX = 0;
        this.mouseDY = 0;
      } else {
        // Entered lock — suppress free-mouse
        this.aim.clearFreeMouse();
      }
    });
  }
}

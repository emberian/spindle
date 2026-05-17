// InputManager — attaches to a canvas; produces P1's PlayerInput each tick.
//
// Design goals (FEEL pass):
//   • Forgiving aim: wide sticky snap cone, no pixel-perfect required.
//   • Works WITHOUT pointer-lock (free-mouse via cursor NDC), and the two
//     modes feel identical. Pointer-lock is OWNED here (single source of
//     truth — main.ts no longer requests it). It's acquired on a deliberate
//     gesture, rejection is swallowed, and free-mouse is a first-class path.
//   • Ghost arc always visible once bell is held (not just when charging),
//     biased toward the Faith ring to teach the Coriolis curve; length &
//     fidelity scale with charge so the curve is learnable.
//   • Throw aim-assist: gentle pull toward the attacking ring.
//   • Clean charge feel: ease-out ramp so taps = short passes, holds = bombs.
//   • Spin self-centres with a detent so neutral is easy to find.
//   • Reticle states: free / anchor-lock / charging — conveyed via InputView.
//   • View agency: exposes lookIntent for GameCamera.setLook (orchestrator
//     wires it) so the player can glance around the chase cam.
//   • All keys debounced; sensible WASD + scroll defaults.
//
// Device bindings:
//   Mouse aim (pointer-lock or free)  → aim via AimModel
//   RMB press                         → fire grapple at snapped anchor
//                                       (works WHILE carrying the bell)
//   RMB release                       → release the line (slingshot off)
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

/** Throw charge: full charge in ~0.7 s of linear time, but the *curve* is
 *  ease-out (see below) so a quick tap already gives a usable short pass and
 *  the top end takes deliberate commitment — far more readable than linear. */
const CHARGE_RATE     = 1.45;

/** Throw speed range (m/s). Min throw is still useful for short passes. */
const THROW_SPEED_MIN =  5.0;
const THROW_SPEED_MAX = 30.0;

/** Scroll wheel → spin sensitivity. */
const SPIN_SCROLL_K   = 0.004;

/** A/D spin ramp (units per second — frame-rate independent now). */
const SPIN_KEY_RATE   = 2.6;

/** Spin self-centring rate per second when no input (pulls toward 0). */
const SPIN_RETURN     = 7.0;

/** Spin detent: snap tiny residual spin to exactly 0 so neutral is findable. */
const SPIN_DETENT     = 0.04;

/** Ghost arc: sample count and step size (seconds). Length grows with charge
 *  so a soft toss shows a short near arc and a full bomb shows the whole
 *  Coriolis sweep — the curve becomes learnable by watching it change. */
const GHOST_STEPS_MIN = 22;
const GHOST_STEPS_MAX = 46;
const GHOST_H         = 0.045;

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
  private lmbDown     = false;   // held → charge throw (while holding bell)
  private lmbReleased = false;   // edge: just went up this tick → throw
  // RMB = grapple (works WHILE carrying the bell — the carrier was otherwise
  // immobile: grapple used to be gated on !holdingBell and LMB is throw).
  private rmbPressed  = false;   // edge → fire line
  private rmbReleased = false;   // edge → release line (slingshot off)
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
  // _chargeRaw = linear hold time [0,1]; _charge = ease-out shaped value the
  // sim/HUD see (so the same dwell maps to a more controllable speed band).
  private _chargeRaw = 0;
  private _charge    = 0;
  private charging   = false;

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
    // 1. Integrate pointer-lock mouse movement (dt-aware smoothing).
    this.aim.update(this.mouseDX, this.mouseDY, dtSec);
    this.mouseDX = 0;
    this.mouseDY = 0;

    // 2. Aim direction: assist toward Faith ring when holding bell
    const aimDir = this.holdingBell
      ? this.aim.assistedThrowDir(this.camera, this.playerPos, this.faithRingX)
      : this.aim.aimDir(this.camera);

    // 3. Anchor snapping (always, for reticle feedback)
    this._reticle = this.aim.anchorTarget(this.spars, this.camera, this.playerPos);

    // 4. Charge logic: only while LMB held AND holding bell.
    //    Linear dwell → ease-out shaped charge: charge = 1-(1-r)^1.7. A
    //    short tap already yields a controllable medium pass; the top of
    //    the band needs a deliberate hold, which reads far better.
    let throwReleased = false;
    if (this.lmbDown && this.holdingBell) {
      this.charging   = true;
      this._chargeRaw = Math.min(1.0, this._chargeRaw + CHARGE_RATE * dtSec);
    }
    this._charge = 1 - Math.pow(1 - this._chargeRaw, 1.7);
    if (this.lmbReleased && this.charging) {
      if (this._charge >= THROW_MIN_CHARGE) {
        throwReleased = true;
      }
      this.charging = false;
    }
    // Reset charge when bell leaves hand
    if (!this.holdingBell) {
      this._chargeRaw = 0;
      this._charge    = 0;
      this.charging   = false;
    }

    // 5. Throw spin from A/D or scroll. Frame-rate-independent ramp; when no
    //    key is held it eases back to 0 (self-centring) and a small detent
    //    snaps the last sliver to exactly neutral so 0 is easy to hit.
    const keySpinL = this.keys.has('KeyA') || this.keys.has('ArrowLeft');
    const keySpinR = this.keys.has('KeyD') || this.keys.has('ArrowRight');
    if (keySpinL) {
      this.scrollSpin = Math.max(-1, this.scrollSpin - SPIN_KEY_RATE * dtSec);
    } else if (keySpinR) {
      this.scrollSpin = Math.min(1, this.scrollSpin + SPIN_KEY_RATE * dtSec);
    } else {
      this.scrollSpin *= Math.exp(-SPIN_RETURN * dtSec);
      if (Math.abs(this.scrollSpin) < SPIN_DETENT) this.scrollSpin = 0;
    }
    const throwSpin = Math.max(-1, Math.min(1, this.scrollSpin));

    // 6. Reel
    const reelIn  = this.keys.has('KeyW') || this.keys.has('ArrowUp');
    const reelOut = this.keys.has('KeyS') || this.keys.has('ArrowDown');
    const reel: -1 | 0 | 1 = reelIn ? -1 : reelOut ? 1 : 0;

    // 7. FireLineAt: RMB pressed this tick (fire on press → instant), at the
    //    snapped anchor. Works WHETHER OR NOT we hold the bell — you carry
    //    the bell down the calm by grappling, then throw with LMB. RMB
    //    release lets the line go (slingshot off).
    const fireLineAt: Vec3 | null =
      (this.rmbPressed && this._reticle?.valid)
        ? this._reticle.pos
        : null;
    const releaseLine = this.rmbReleased;

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
      // Trail length tracks charge so the player *sees* the Coriolis sweep
      // grow as they wind up — that change is what makes the curve learnable.
      const steps = Math.round(
        GHOST_STEPS_MIN + (GHOST_STEPS_MAX - GHOST_STEPS_MIN) * charge,
      );
      this._ghostArc = predictPath(this.playerPos, v0, REG.omega, GHOST_H, steps);
    } else {
      this._ghostArc = [];
    }

    // Consume edge flags
    this.lmbReleased = false;
    this.rmbPressed  = false;
    this.rmbReleased = false;

    return {
      id:           'P1',
      aim:          aimDir,
      fireLineAt,
      reel,
      release:      releaseLine,
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

  // ── Camera view-agency hook ───────────────────────────────────────────────

  /**
   * Normalised player look intent for GameCamera.setLook(). The orchestrator
   * should call `cam.setLook(li.yaw, li.pitch, li.active)` each frame with
   * this (PLAY mode only). Additive to the frozen contract — purely a
   * read; safe to ignore if unwired (camera just won't have view agency).
   */
  get lookIntent(): { yaw: number; pitch: number; active: boolean } {
    return this.aim.lookIntent();
  }

  // ── Pointer lock ──────────────────────────────────────────────────────────

  /**
   * Public, idempotent pointer-lock request. Single source of truth — main.ts
   * must NOT request lock itself (the orchestrator removes that). Rejection
   * (no user gesture / browser refusal) is swallowed; free-mouse keeps working.
   */
  requestPointerLock(): void {
    this._tryLock();
  }

  private _lockPending = false;

  private _tryLock(): void {
    if (this.pointerLocked || this._lockPending) return;
    const fn = this.canvas.requestPointerLock;
    if (!fn) return; // unsupported → free-mouse path, no error
    this._lockPending = true;
    try {
      // Some browsers return a promise; swallow rejection silently.
      const r = fn.call(this.canvas) as unknown as Promise<void> | undefined;
      if (r && typeof r.then === 'function') {
        r.then(
          () => { this._lockPending = false; },
          () => { this._lockPending = false; },
        );
      } else {
        // No promise API — clear the guard shortly after; pointerlockchange
        // will confirm success either way.
        setTimeout(() => { this._lockPending = false; }, 250);
      }
    } catch {
      this._lockPending = false; // refused — stay in free-mouse, no jank
    }
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
      } else if (e.button === 2) {
        this.rmbPressed = true;
      }
      // Any deliberate click acquires pointer-lock (idempotent, sole
      // requester). Free-mouse aim is used until lock engages so nothing
      // feels gated on it; LMB throws / RMB grapples the same tick.
      this._tryLock();
    });

    // RMB is grapple now (a gameplay action) — just suppress the context
    // menu; do NOT drop pointer-lock here (Esc still exits lock for UI).
    on(this.canvas, 'contextmenu', (e: MouseEvent) => {
      e.preventDefault();
    });

    on(this.canvas, 'mouseup', (e: MouseEvent) => {
      if (e.button === 0) {
        this.lmbDown     = false;
        this.lmbReleased = true;
      } else if (e.button === 2) {
        this.rmbReleased = true;
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
      const wasLocked = this.pointerLocked;
      this.pointerLocked = document.pointerLockElement === this.canvas;
      this._lockPending = false;
      this.mouseDX = 0;
      this.mouseDY = 0;
      if (!this.pointerLocked && wasLocked) {
        // Exited lock. DON'T reset aim — that would snap the view. Seed the
        // free-mouse cursor to screen-centre so the aim direction is
        // continuous (centre ≈ forward, matching the just-held lock aim).
        this.aim.reset();
        this.aim.setFreeMouse(0, 0);
      } else if (this.pointerLocked && !wasLocked) {
        // Entered lock — drop free-mouse, start the yaw/pitch accumulator
        // at neutral. Centre-screen free-mouse ≈ forward, and neutral
        // pointer-lock ≈ forward too, so the handoff is visually seamless.
        this.aim.reset();
        this.aim.clearFreeMouse();
      }
    });
  }
}

// Rigger renderer — CEL / GRAPHIC-NOVEL zero-g sport athletes.
//
// Each rigger is an articulated humanoid (RiggerFigure): slim suit + chest
// grapple-rig, flat team-colour fill, bold black ink outlines (inverted-hull),
// 3-band quantised toon shade. Team reads from fill colour (home cyan / away
// orange), role/team accent from the chest plate + visor. The human player
// (p1Id) gets a tasteful halo + ▼ beacon highlight on the new model.
//
// Motion is procedural and fluid: every render frame sync() derives velocity &
// acceleration from kept prior positions and drives a damped pose —
//   • lean/bank into acceleration,
//   • a smooth zero-g swim/stroke cycle whose amplitude scales with speed,
//   • the right "grapple" arm extends toward travel/line when a line is active
//     (its animated hand is published for RigLine to originate the line from),
//   • secondary follow-through on torso/limbs (eased, never snaps),
//   • a calm idle drift when nearly still.
// All easing is frame-rate-independent (exponential smoothing on dt).
//
// World scale matched to RegConstants: R = 45 m calm; figures are ~1.9 m
// humans. Footprint kept close to the previous placeholder so the spectate
// camera framing and in-world placement are unchanged.

import * as THREE from 'three';
import type { PlayerSim } from '../sim/types';
import { PAL, ROLE_TINT } from '../ui/palette';
import { RiggerFigure } from './RiggerFigure';
import { publishGrappleHand } from './RigGrapple';

const MAX_RIGGERS = 12;

// Speed (m/s) at which the stroke cycle reaches full amplitude.
const STROKE_FULL_SPEED = 14;
// Stroke phase advance per unit distance travelled (rad/m) — couples the swim
// rhythm to actual motion so it never looks like a treadmill.
const STROKE_PER_METRE = 0.5;
// Idle drift angular speed (rad/s).
const IDLE_RATE = 1.1;

// ── Helpers ──────────────────────────────────────────────────────────────────

function lerpHex(a: number, b: number, t: number): number {
  const ri = (a >> 16) & 0xff, gi = (a >> 8) & 0xff, bi = a & 0xff;
  const rf = (b >> 16) & 0xff, gf = (b >> 8) & 0xff, bf = b & 0xff;
  return (((ri + (rf - ri) * t) | 0) << 16)
       | (((gi + (gf - gi) * t) | 0) << 8)
       |  ((bi + (bf - bi) * t) | 0);
}

function teamBase(team: string): number {
  return team === 'home' ? PAL.cyan : PAL.orange;
}

function bodyColor(team: string, grounded: boolean): number {
  const base = teamBase(team);
  return grounded ? lerpHex(base, PAL.dim, 0.65) : base;
}

/** Frame-rate-independent exponential smoothing toward `target`. */
function damp(cur: number, target: number, lambda: number, dt: number): number {
  return target + (cur - target) * Math.exp(-lambda * dt);
}

// Scratch (no per-frame alloc in the hot path).
const _fwd = new THREE.Vector3();
const _up = new THREE.Vector3();
const _right = new THREE.Vector3();
const _basisM = new THREE.Matrix4();
const _basisQ = new THREE.Quaternion();
const _hand = new THREE.Vector3();
const _accLocal = new THREE.Vector3();

// ── Per-figure instance ──────────────────────────────────────────────────────

class RiggerInstance {
  /**
   * Optional, purely-additive posing emphasis per player id (0..1). The
   * orchestrator MAY set this (e.g. bell carrier / throw windup) via
   * Riggers.setPoseEmphasis() for a richer reach pose. Untouched = 0 =
   * current behaviour.
   */
  static readonly emphasis = new Map<string, number>();

  readonly figure = new RiggerFigure();

  // Kept state for motion derivation + damping.
  private px = 0; private py = 0; private pz = 0;
  private vx = 0; private vy = 0; private vz = 0;
  private hasPrev = false;

  private lean = 0;
  private bank = 0;
  private strokeAmp = 0;
  private strokePhase = 0;
  private reach = 0;
  private idlePhase = Math.random() * Math.PI * 2;

  get root(): THREE.Group {
    return this.figure.root;
  }

  hide(): void {
    this.figure.root.visible = false;
  }

  /**
   * Build an orientation with NO global up: "up" points toward the cylinder
   * axis (−radial in the (y,z) cross-section, since centrifugal "down" is
   * radially outward from the +X spin axis). Facing follows travel direction;
   * falls back to the sim quat's forward when nearly still so a parked rigger
   * doesn't spin randomly.
   */
  private orient(ps: PlayerSim, speed: number): void {
    // Radial-inward = local up. On-axis fallback → +Y.
    const ry = ps.p.y, rz = ps.p.z;
    const rl = Math.hypot(ry, rz);
    if (rl > 1e-4) {
      _up.set(0, -ry / rl, -rz / rl);
    } else {
      _up.set(0, 1, 0);
    }

    // Facing: velocity direction if moving, else sim-quat forward (+Z).
    if (speed > 0.4) {
      _fwd.set(this.vx, this.vy, this.vz).multiplyScalar(1 / speed);
    } else {
      _fwd.set(0, 0, 1).applyQuaternion(
        _basisQ.set(ps.q.x, ps.q.y, ps.q.z, ps.q.w).normalize(),
      );
    }

    // Orthonormalise: right = fwd × up, then re-derive fwd = up × right.
    _right.crossVectors(_fwd, _up);
    if (_right.lengthSq() < 1e-6) {
      // fwd nearly parallel to up — nudge with world +X (the spin axis).
      _right.set(1, 0, 0).cross(_up);
    }
    _right.normalize();
    _fwd.crossVectors(_up, _right).normalize();

    // Basis columns: X=right, Y=up, Z=fwd → figure faces +Z, stands +Y.
    _basisM.makeBasis(_right, _up, _fwd);
    this.figure.root.quaternion.setFromRotationMatrix(_basisM);
  }

  apply(ps: PlayerSim, isP1: boolean, dt: number, now: number): void {
    this.figure.root.visible = true;
    this.figure.root.position.set(ps.p.x, ps.p.y, ps.p.z);

    // ── Motion derivation (render-frame deltas) ─────────────────────────────
    let ax = 0, ay = 0, az = 0;
    if (this.hasPrev && dt > 1e-5) {
      const nvx = (ps.p.x - this.px) / dt;
      const nvy = (ps.p.y - this.py) / dt;
      const nvz = (ps.p.z - this.pz) / dt;
      ax = (nvx - this.vx) / dt;
      ay = (nvy - this.vy) / dt;
      az = (nvz - this.vz) / dt;
      // Blend derived velocity with sim velocity for stability.
      this.vx = nvx * 0.5 + ps.v.x * 0.5;
      this.vy = nvy * 0.5 + ps.v.y * 0.5;
      this.vz = nvz * 0.5 + ps.v.z * 0.5;
    } else {
      this.vx = ps.v.x; this.vy = ps.v.y; this.vz = ps.v.z;
    }
    this.px = ps.p.x; this.py = ps.p.y; this.pz = ps.p.z;
    this.hasPrev = true;

    const speed = Math.hypot(this.vx, this.vy, this.vz);

    // Orientation (no global up).
    this.orient(ps, speed);

    // ── Pose targets from motion, expressed in figure-local space ───────────
    // Acceleration in local frame → lean (local Z accel) & bank (local X).
    _accLocal.set(ax, ay, az).applyQuaternion(
      _basisQ.copy(this.figure.root.quaternion).invert(),
    );
    const leanTarget = THREE.MathUtils.clamp(_accLocal.z * 0.05, -0.6, 0.6);
    const bankTarget = THREE.MathUtils.clamp(-_accLocal.x * 0.05, -0.5, 0.5);

    const ampTarget = THREE.MathUtils.clamp(speed / STROKE_FULL_SPEED, 0, 1);
    // Reach: active line OR clipped contact → extend grapple arm toward travel.
    // Optional emphasis (bell-carry / throw windup) lets the orchestrator push
    // the reach arm out further without touching the frozen sync() signature.
    const emphasis = RiggerInstance.emphasis.get(ps.id) ?? 0;
    const reachTarget = Math.max(
      ps.line ? 1 : (ps.contactRef !== null ? 0.5 : 0),
      emphasis,
    );

    // ── Damp everything (frame-rate independent) ────────────────────────────
    this.lean = damp(this.lean, leanTarget, 8, dt);
    this.bank = damp(this.bank, bankTarget, 8, dt);
    this.strokeAmp = damp(this.strokeAmp, ampTarget, 5, dt);
    this.reach = damp(this.reach, reachTarget, 7, dt);

    // Stroke phase advances with distance travelled (+ a slow idle baseline so
    // a parked rigger still breathes/floats rather than freezing).
    this.strokePhase += speed * dt * STROKE_PER_METRE + dt * 1.4;
    this.idlePhase += dt * IDLE_RATE;

    // ── Colours ─────────────────────────────────────────────────────────────
    const base = teamBase(ps.team);
    const bodyCol = bodyColor(ps.team, ps.grounded);
    const accent = isP1 ? PAL.paper : (ROLE_TINT[ps.role] ?? lerpHex(base, PAL.paper, 0.4));
    const emissiveHex = isP1 ? base : (ps.grounded ? 0x000000 : base);
    const emissiveInt = isP1 ? (ps.grounded ? 0.4 : 0.9) : (ps.grounded ? 0.0 : 0.28);
    this.figure.setColors(bodyCol, accent, emissiveHex, emissiveInt);

    // ── Drive the pose ──────────────────────────────────────────────────────
    this.figure.pose(
      this.lean, this.bank,
      this.strokePhase, this.strokeAmp,
      this.reach, this.idlePhase,
    );

    // ── Publish animated grapple hand for RigLine coordination ──────────────
    this.figure.grappleHandWorld(_hand);
    publishGrappleHand(ps.id, _hand);

    // ── P1 highlight ────────────────────────────────────────────────────────
    this.figure.setHighlight(isP1, PAL.cyan, now * 0.001);
  }
}

// ── Public API (FROZEN SIGNATURES) ───────────────────────────────────────────

/** Manages a pool of rigger figures. Call sync() every rendered frame. */
export class Riggers {
  private pool: RiggerInstance[] = [];
  private group = new THREE.Group();
  private lastNow = 0;

  constructor(scene: THREE.Scene) {
    for (let i = 0; i < MAX_RIGGERS; i++) {
      const inst = new RiggerInstance();
      this.pool.push(inst);
      // Stylised scale-up: a true-1.9 m human against the 45 m calm, filmed
      // from the broadcast camera, is a few pixels — the cel art was
      // invisible. Riggers are larger-than-life so the figures actually read
      // (uniform root scale; the per-frame pose math is untouched).
      inst.root.scale.setScalar(5);
      this.group.add(inst.root);
    }
    scene.add(this.group);
  }

  /**
   * Sync visuals to the current (interpolated) player list.
   * @param players  Interpolated PlayerSim array for this render frame.
   * @param p1Id     Local human player's id; '' = spectate/replay (no highlight).
   */
  sync(players: PlayerSim[], p1Id: string): void {
    const now = performance.now();
    let dt = this.lastNow ? (now - this.lastNow) / 1000 : 1 / 60;
    // Clamp dt so a tab-stall / first frame can't make the pose explode.
    dt = THREE.MathUtils.clamp(dt, 1 / 240, 1 / 15);
    this.lastNow = now;

    for (const inst of this.pool) inst.hide();

    const n = Math.min(players.length, MAX_RIGGERS);
    for (let i = 0; i < n; i++) {
      this.pool[i].apply(players[i], players[i].id === p1Id, dt, now);
    }
  }

  /**
   * OPTIONAL additive hook (not part of the frozen contract — safe to ignore).
   * Sets a 0..1 posing emphasis for a player id so the orchestrator can later
   * make e.g. the bell carrier / a thrower wind the grapple arm out further.
   * Damped like everything else, so toggling it never pops.
   */
  setPoseEmphasis(id: string, amount: number): void {
    RiggerInstance.emphasis.set(id, THREE.MathUtils.clamp(amount, 0, 1));
  }
}

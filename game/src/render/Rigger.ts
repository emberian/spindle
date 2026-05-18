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

// Speed (m/s) at which the (now SMALL, slow) idle-stroke reaches its modest
// full amplitude. A moving rigger should look like a person GLIDING/HAULING
// on a line in freefall, NOT a sprinter — so the stroke is a faint, slow
// drift, not a speed-pumped leg thrash. Kept generous so even a slow drifter
// shows only the gentle stroke and nothing kicks in at travel speed.
const STROKE_FULL_SPEED = 6;
// Stroke phase advance per unit distance travelled (rad/m). DRASTICALLY cut
// from the old 0.5 so a fast haul does NOT spin the limbs faster (no
// mid-air-jogging cadence) — the stroke stays a slow, subtle background sway.
const STROKE_PER_METRE = 0.06;
// Idle drift angular speed (rad/s).
const IDLE_RATE = 1.1;
// Speed (m/s) at which "effort" saturates. Effort now drives ONLY the
// streamline (arrowed, limbs tucked/trailing) — NOT cadence or leg amplitude.
const EFFORT_FULL_SPEED = 7;
// Decel (m/s²) magnitude (opposing velocity) at which the brace/settle pose
// is fully expressed — an athlete planting a stop. Tunable.
const BRACE_FULL_DECEL = 22;

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
  // Lift the in-play body toward paper so each figure reads as a bright,
  // distinct body against the calm (legibility WINS over darkness) while
  // keeping the team hue. Grounded stays dimmed (out of play).
  return grounded
    ? lerpHex(base, PAL.dim, 0.65)
    : lerpHex(base, PAL.paper, 0.30);
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
const _toAnchor = new THREE.Vector3();
const _velDir = new THREE.Vector3();

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

  // Kept state for motion derivation + damping. We retain the interpolated
  // velocity across frames to derive acceleration (frame-to-frame Δv).
  private vx = 0; private vy = 0; private vz = 0;
  private hasPrev = false;

  private lean = 0;
  private bank = 0;
  private strokeAmp = 0;
  private strokePhase = 0;
  private reach = 0;
  private idlePhase = Math.random() * Math.PI * 2;

  // Athletic-motion damped state.
  private effort = 0;
  private brace = 0;
  private wind = 0;
  private swingPhase = Math.random() * Math.PI * 2;
  // For swing-rate derivation: prior unit vector hand→anchor (the line we
  // whip around). We advance swingPhase by the angle swept per frame so the
  // haul cadence is welded to how fast the rigger actually orbits the anchor.
  private prevAx = 0; private prevAy = 0; private prevAz = 0;
  private hadAnchor = false;
  private wasLine = false;
  private prevEmph = 0;
  private throwTimer = 0; // s remaining in a windup→release throw envelope

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

    // ── Motion derivation ───────────────────────────────────────────────────
    // ps.v is now the cleanly RenderState-interpolated sim velocity, so use it
    // directly instead of double-differencing render-frame positions (that old
    // path turned every 240 Hz sim snap into an accel spike that lurched the
    // cel pose). Acceleration = frame-to-frame delta of the interpolated v.
    let ax = 0, ay = 0, az = 0;
    if (this.hasPrev && dt > 1e-5) {
      ax = (ps.v.x - this.vx) / dt;
      ay = (ps.v.y - this.vy) / dt;
      az = (ps.v.z - this.vz) / dt;
    }
    this.vx = ps.v.x; this.vy = ps.v.y; this.vz = ps.v.z;
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

    // Stroke amplitude is now CAPPED LOW and INVERTED vs effort: the faster a
    // rigger travels the SLEEKER/STILLER it gets (streamlined glide), not the
    // harder it pumps. A slow drift shows a small subtle stroke; at haul speed
    // the stroke fades toward a near-still arrow.
    const effortTarget = THREE.MathUtils.clamp(speed / EFFORT_FULL_SPEED, 0, 1);
    const ampTarget = THREE.MathUtils.clamp(speed / STROKE_FULL_SPEED, 0, 1)
      * (1 - effortTarget * 0.8) * 0.35;
    // Reach: active line OR clipped contact → extend grapple arm toward travel.
    // Optional emphasis (bell-carry / throw windup) lets the orchestrator push
    // the reach arm out further without touching the frozen sync() signature.
    const emphasis = RiggerInstance.emphasis.get(ps.id) ?? 0;
    const reachTarget = Math.max(
      ps.line ? 1 : (ps.contactRef !== null ? 0.5 : 0),
      emphasis,
    );

    // ── Effort (computed above) now drives the STREAMLINE only: body aligned
    // to velocity/grapple direction with limbs tucked and trailing — a person
    // flying/hauling on a line in freefall, never jogging.

    // ── Brace/settle: project acceleration onto the OPPOSITE of velocity.
    // A large component fighting the direction of travel = the rigger is
    // killing speed → plant a brace pose. (Only meaningful while moving.)
    let braceTarget = 0;
    if (speed > 1.5) {
      _velDir.set(this.vx, this.vy, this.vz).multiplyScalar(1 / speed);
      const decel = -(ax * _velDir.x + ay * _velDir.y + az * _velDir.z);
      braceTarget = THREE.MathUtils.clamp(decel / BRACE_FULL_DECEL, 0, 1);
    }

    // ── Throw envelope: detect the bell leaving the hand. The orchestrator
    // raises emphasis on the carrier/thrower; the frame emphasis DROPS sharply
    // (carry → released) we fire a windup→release whip. Also fires on a line
    // release (let the bell go on a cross-line). Pure visual, no sim read.
    const lineReleased = this.wasLine && !ps.line;
    const carryDrop = this.prevEmph - emphasis;
    if (this.throwTimer <= 0 && (carryDrop > 0.35 || (lineReleased && this.prevEmph > 0.2))) {
      this.throwTimer = 0.42; // s: brief wind-up then explosive release
    }
    this.wasLine = !!ps.line;
    this.prevEmph = emphasis;
    let windTarget = 0;
    if (this.throwTimer > 0) {
      this.throwTimer -= dt;
      // First ~45% = cock back (negative), remainder = whip across (positive),
      // decaying to 0 (follow-through settle).
      const u = 1 - Math.max(0, this.throwTimer) / 0.42; // 0→1 over envelope
      windTarget = u < 0.45
        ? -(u / 0.45)                       // -1 … 0  (wind up)
        :  Math.pow(1 - (u - 0.45) / 0.55, 0.6); // 1 → 0 (release + follow)
      if (u < 0.45) windTarget = -Math.pow(u / 0.45, 0.7);
    }

    // ── Damp everything (frame-rate independent) ────────────────────────────
    this.lean = damp(this.lean, leanTarget, 8, dt);
    this.bank = damp(this.bank, bankTarget, 8, dt);
    this.strokeAmp = damp(this.strokeAmp, ampTarget, 5, dt);
    this.reach = damp(this.reach, reachTarget, 7, dt);
    this.effort = damp(this.effort, effortTarget, 4, dt);
    this.brace = damp(this.brace, braceTarget, 9, dt);
    // Wind whips fast (snappy) but never pops because the envelope itself is
    // smooth; a quick damp removes any frame jitter.
    this.wind = damp(this.wind, windTarget, 22, dt);

    // ── Swing/haul cadence: when on a line, advance swingPhase by the angle
    // actually swept around the anchor this frame (welded to the swing), plus
    // a speed baseline so a fast straight haul still pumps. Off-line it idles.
    const swingRate = 1.0 + speed * 0.10;          // rad/s baseline (off-line)
    if (ps.line) {
      _toAnchor.set(
        ps.line.anchorPos.x - ps.p.x,
        ps.line.anchorPos.y - ps.p.y,
        ps.line.anchorPos.z - ps.p.z,
      );
      const al = _toAnchor.length();
      if (al > 1e-3) {
        _toAnchor.multiplyScalar(1 / al);
        if (this.hadAnchor) {
          const dot = THREE.MathUtils.clamp(
            this.prevAx * _toAnchor.x + this.prevAy * _toAnchor.y + this.prevAz * _toAnchor.z,
            -1, 1,
          );
          const swept = Math.acos(dot);            // angle orbited this frame
          // 2 haul beats per radian swept feels like driving the swing.
          this.swingPhase += swept * 2.0 + speed * dt * 0.15;
        }
        this.prevAx = _toAnchor.x; this.prevAy = _toAnchor.y; this.prevAz = _toAnchor.z;
        this.hadAnchor = true;
      }
    } else {
      this.hadAnchor = false;
      this.swingPhase += swingRate * dt;
    }

    // Stroke phase advances at a SLOW, near-constant tempo. The old code
    // ramped cadence with speed AND effort (the "sprinting in mid-air" bug):
    // a fast hauler whipped its limbs. Now it is a gentle ~0.45 rad/s drift
    // with only a tiny distance contribution, and effort SLOWS it further
    // (a streamlined glider is stiller, not faster).
    this.strokePhase += dt * 0.45 * (1 - this.effort * 0.6)
      + speed * dt * STROKE_PER_METRE;
    this.idlePhase += dt * IDLE_RATE;

    // ── Colours ─────────────────────────────────────────────────────────────
    const base = teamBase(ps.team);
    const bodyCol = bodyColor(ps.team, ps.grounded);
    const accent = isP1 ? PAL.paper : (ROLE_TINT[ps.role] ?? lerpHex(base, PAL.paper, 0.4));
    const emissiveHex = isP1 ? base : (ps.grounded ? 0x000000 : base);
    // Raised non-P1 in-play emissive 0.28 → 0.55 so each of the 8 riggers
    // self-lights into a clearly legible body even on its shadow side, without
    // blowing past bloom threshold (kept well under the P1 0.9).
    const emissiveInt = isP1 ? (ps.grounded ? 0.4 : 0.9) : (ps.grounded ? 0.0 : 0.55);
    this.figure.setColors(bodyCol, accent, emissiveHex, emissiveInt);

    // ── Drive the pose ──────────────────────────────────────────────────────
    this.figure.pose(
      this.lean, this.bank,
      this.strokePhase, this.strokeAmp,
      this.reach, this.idlePhase,
      this.effort, this.swingPhase, this.brace, this.wind,
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

  // Figure scale is VIEW-dependent: the far spectate/cinematic camera needs
  // big larger-than-life figures to read at all; the close human chase-cam
  // needs near-human scale or the figures fill the screen and you can't tell
  // anything. Orchestrator picks via setViewScale(); default = spectate.
  private viewScale = 5;

  constructor(scene: THREE.Scene) {
    for (let i = 0; i < MAX_RIGGERS; i++) {
      const inst = new RiggerInstance();
      this.pool.push(inst);
      inst.root.scale.setScalar(this.viewScale);
      this.group.add(inst.root);
    }
    scene.add(this.group);
  }

  /** Set the figure world-scale for the current view (play ≈ 2, spectate = 5). */
  setViewScale(k: number): void {
    if (k === this.viewScale) return;
    this.viewScale = k;
    for (const inst of this.pool) inst.root.scale.setScalar(k);
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

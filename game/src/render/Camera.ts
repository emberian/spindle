// The camera is the #1 legibility fix. The old single dark side-chase made
// the game unreadable. This frames the three things you must always see —
// the bell, YOUR rigger, and the goal you're attacking — looking down the
// spine so depth/length read, with a dramatic broadside loop-cam.
//
// World frame: +X = spin axis (goals at x = ±GATE). (y,z) = rotating
// cross-section, axis at y=z=0, skin at radius R. There is no global "up":
// "up" everywhere points toward the axis (−radial). All smoothing is
// exponential + clamped so it never disorients.
//
// FEEL pass: the player now has gentle *view agency* — a look-orbit offset
// fed in via setLook() that swings the chase cam around the focus without
// ever breaking the "you see bell + self + goal" guarantee, and without
// touching the frozen update() signature. Camera leads toward motion so
// grapple traversal feels fast and intentional, not floaty. The up-vector
// is slerped on the sphere (not lerped through the origin) so it can never
// shrink/tumble near the axis. cinematic() is the spectate director.

import * as THREE from 'three';

const V = THREE.Vector3;
type V3 = { x: number; y: number; z: number };

function damp(cur: THREE.Vector3, target: THREE.Vector3, lambda: number, dt: number): void {
  // frame-rate-independent exponential approach
  const k = 1 - Math.exp(-lambda * dt);
  cur.x += (target.x - cur.x) * k;
  cur.y += (target.y - cur.y) * k;
  cur.z += (target.z - cur.z) * k;
}

function dampScalar(cur: number, target: number, lambda: number, dt: number): number {
  return cur + (target - cur) * (1 - Math.exp(-lambda * dt));
}

// Slerp a *direction* on the unit sphere — never shrinks toward origin, never
// tumbles when the two dirs get near-antiparallel (degrades to a clean lerp+
// renormalise). This is the correct way to ease an up-vector near the axis.
const _a = new V();
const _b = new V();
function dampDir(cur: THREE.Vector3, target: THREE.Vector3, lambda: number, dt: number): void {
  _a.copy(cur);
  if (_a.lengthSq() < 1e-9) _a.set(0, 1, 0);
  _a.normalize();
  _b.copy(target);
  if (_b.lengthSq() < 1e-9) return;
  _b.normalize();
  const k = 1 - Math.exp(-lambda * dt);
  let d = _a.dot(_b);
  d = Math.max(-1, Math.min(1, d));
  if (d > 0.9995) {
    // already aligned — tiny lerp + renormalise
    cur.copy(_a).lerp(_b, k).normalize();
    return;
  }
  const theta = Math.acos(d) * k;
  // orthogonal component of b relative to a
  _b.addScaledVector(_a, -d);
  if (_b.lengthSq() < 1e-9) {
    cur.copy(_a).lerp(target, k).normalize();
    return;
  }
  _b.normalize();
  cur.copy(_a).multiplyScalar(Math.cos(theta)).addScaledVector(_b, Math.sin(theta)).normalize();
}

export type CinematicAngle = 'broadcast' | 'low' | 'goal' | 'wide';

export class GameCamera {
  private pos = new V();
  private aim = new V();
  private up = new V(0, 1, 0);
  private fov = 60;
  private inited = false;
  private loopBlend = 0;

  // Smoothed lead vector (camera anticipates where the action is heading).
  private lead = new V();
  private prevFocus = new V();
  private hasPrev = false;

  // ── Player view agency (PLAY only) ──────────────────────────────────────
  // yaw orbits around the axis-aligned "up"; pitch tilts toward/away the
  // spine. Both are gently clamped & spring back to centre so the three
  // guaranteed objects can never leave frame.
  private lookYaw = 0;
  private lookPitch = 0;
  private lookActive = false;

  // ── Cinematic director ──────────────────────────────────────────────────
  private cineAngle: CinematicAngle = 'broadcast';
  // Only the two cross-axis angles are in the cycle. 'goal' and 'wide' both
  // look back DOWN the 640 m spine, which stacks the bell + ring + spars +
  // axis glow into a bloom white-out that per-knob tuning can't reliably
  // beat (and headless can't be trusted to judge). broadcast + low both
  // read cleanly with full volume context; the loop-cam (broadside, also
  // cross-axis) still triggers separately for the money shot.
  private cineOrder: CinematicAngle[] = ['broadcast', 'low'];
  private cineT = 0; // running clock for slow drift moves

  // Smoothed attack direction (−1..+1). Possession flips pivot the rig over
  // ~1 s instead of snapping 180°/64 m — the big "schizophrenic" cause.
  private dirS = 0;
  // Loop-cam hysteresis latch so frequent CURLS (loopGlow≈0.4) never trip the
  // hero broadside cut; only a real LOOP (≈1.0) does, and it holds.
  private loopLatch = false;

  constructor(private cam: THREE.PerspectiveCamera) {}

  /** Smoothed attack direction; eases across possession flips. */
  private smoothDir(attackX: number, dt: number): number {
    const raw = Math.sign(attackX) || 1;
    if (this.dirS === 0) this.dirS = raw;
    else this.dirS = dampScalar(this.dirS, raw, 1.6, dt);
    // keep magnitude ~1 so offsets don't shrink mid-pivot
    const m = Math.abs(this.dirS);
    return m < 0.25 ? (this.dirS < 0 ? -0.25 : 0.25) : this.dirS;
  }

  /** Loop-cam hysteresis: enter only on a true loop, hold through the dip. */
  private wantLoopCam(loopGlow: number): boolean {
    this.loopLatch = this.loopLatch ? loopGlow > 0.55 : loopGlow > 0.85;
    return this.loopLatch;
  }

  /**
   * Gentle player look. Pass normalised look intent each frame BEFORE
   * update(); (0,0) springs back to the framed default. Magnitude is the
   * fraction of the soft limit the player is requesting. This never breaks
   * the bell+self+goal guarantee because the orbit radius/limits are small
   * and it springs home when released.
   *
   * @param yaw    [-1,1] tangential orbit (left/right around the action)
   * @param pitch  [-1,1] tilt toward(+)/away(−) the spine
   * @param active true while the player is actively looking (else recenters)
   */
  setLook(yaw: number, pitch: number, active: boolean): void {
    this.lookActive = active;
    if (active) {
      this.lookYaw = Math.max(-1, Math.min(1, yaw));
      this.lookPitch = Math.max(-1, Math.min(1, pitch));
    }
  }

  /**
   * @param bell    bell world position
   * @param p1      your rigger's world position
   * @param attackX +X goal you attack (e.g. +GATE for home, −GATE for away)
   * @param loopGlow 0..1 — a Loop developing → swing to the loop-cam
   * @param skinR   cylinder radius (for framing scale)
   */
  update(bell: V3, p1: V3, attackX: number, loopGlow: number, skinR: number, dt: number): void {
    const dir = this.smoothDir(attackX, dt); // smoothed; no possession snap

    // Focus = weighted blend of the bell (lead) and your rigger. (No raw
    // attackX pull — it lurched the frame 64 m on every turnover; the aim
    // target below already leans goal-ward via `dir`.)
    const focus = new V(
      bell.x * 0.66 + p1.x * 0.34,
      bell.y * 0.58 + p1.y * 0.42,
      bell.z * 0.58 + p1.z * 0.42,
    );

    // Estimate focus velocity → lead the camera so traversal reads as fast
    // and intentional rather than the world sliding under a static frame.
    if (this.hasPrev && dt > 1e-5) {
      const inst = new V(
        (focus.x - this.prevFocus.x) / dt,
        (focus.y - this.prevFocus.y) / dt,
        (focus.z - this.prevFocus.z) / dt,
      );
      // Lead distance saturates so a fast slingshot doesn't fling the cam.
      const sp = inst.length();
      const leadMag = Math.min(skinR * 0.7, sp * 0.35);
      if (sp > 1e-4) inst.multiplyScalar(leadMag / sp);
      else inst.set(0, 0, 0);
      damp(this.lead, inst, 3.5, dt);
    }
    this.prevFocus.copy(focus);
    this.hasPrev = true;

    // Cross-section radius of the focus → keeps the camera outside the action.
    const rho = Math.max(2, Math.hypot(focus.y, focus.z));
    // Radial outward unit (from axis to focus) in the (y,z) plane.
    const ry = focus.y / rho;
    const rz = focus.z / rho;
    // tangent unit in (y,z): perpendicular to radial
    const ty = -rz,
      tz = ry;

    let camTarget: THREE.Vector3;
    let aimTarget: THREE.Vector3;
    let fovTarget: number;
    let upTarget: THREE.Vector3;

    const wantLoop = this.wantLoopCam(loopGlow);
    this.loopBlend += ((wantLoop ? 1 : 0) - this.loopBlend) * (1 - Math.exp(-4 * dt));

    // Ease player look toward request, spring to 0 when not actively looking.
    const lyTarget = this.lookActive ? this.lookYaw : 0;
    const lpTarget = this.lookActive ? this.lookPitch : 0;
    // recenter faster than we follow input so it never feels sticky
    const lookLam = this.lookActive ? 9 : 5;
    this.lookYaw = dampScalar(this.lookYaw, lyTarget, lookLam, dt);
    this.lookPitch = dampScalar(this.lookPitch, lpTarget, lookLam, dt);
    // suppress player look entirely during the loop-cam (it's a hero shot)
    const lookGain = 1 - this.loopBlend;
    const yawA = this.lookYaw * 0.6 * lookGain; // ≤ ~34° tangential swing
    const pitchA = this.lookPitch * 0.45 * lookGain;

    if (this.loopBlend > 0.02) {
      // ── LOOP-CAM: broadside, so the closed Coriolis arc reads AS a loop.
      // Stand off the bell along the cross-section radial, pulled back a
      // little down-spine, looking square at the arc plane. Slow push-in.
      const standoff = skinR * (2.0 - 0.5 * loopGlow);
      camTarget = new V(
        bell.x - dir * skinR * 0.5,
        ry * standoff,
        rz * standoff,
      );
      aimTarget = new V(bell.x, bell.y * 0.5, bell.z * 0.5);
      fovTarget = 46 - 6 * loopGlow;
      upTarget = new V(1, 0, 0).multiplyScalar(dir); // spine reads as horizon
    } else {
      // ── BROADCAST/FOLLOW: behind the action, lifted toward the axis,
      // looking down-spine at the goal so length & depth are obvious.
      const back = skinR * 1.7; // behind, opposite the attack direction
      const lift = skinR * 0.55; // toward the axis (so we look "down" the tube)
      const side = skinR * 0.45; // slight tangential offset for parallax

      // Player look: add a tangential orbit (yaw) + radial pitch on top of
      // the framed default. Limited so bell/self/goal stay in frame.
      const sideTotal = side + yawA * skinR * 1.3;
      const liftTotal = lift + pitchA * skinR * 0.8;
      const backTotal = back * (1 - pitchA * 0.25); // tilt up → ease in a bit

      const fx = focus.x + this.lead.x;
      const fy = focus.y + this.lead.y;
      const fz = focus.z + this.lead.z;

      camTarget = new V(
        fx - dir * backTotal,
        fy - ry * liftTotal + ty * sideTotal,
        fz - rz * liftTotal + tz * sideTotal,
      );
      // look toward the goal-ward action (lead a touch so motion sits ahead)
      aimTarget = new V(
        focus.x + dir * skinR * 0.9 + this.lead.x * 0.5,
        focus.y * 0.7 + this.lead.y * 0.4,
        focus.z * 0.7 + this.lead.z * 0.4,
      );
      fovTarget = 62;
      // up = toward the axis (−radial), so the horizon never tumbles
      upTarget = new V(0, -ry, -rz);
    }

    if (!this.inited) {
      this.pos.copy(camTarget);
      this.aim.copy(aimTarget);
      this.up.copy(upTarget);
      this.fov = fovTarget;
      this.inited = true;
    }
    // Snappier settle than before so traversal feels intentional, but still
    // exponential + clamped so it never snap-jars. Aim tracks a touch
    // tighter than position (classic chase-cam readability).
    const lam = this.loopBlend > 0.02 ? 5.5 : 4.6;
    damp(this.pos, camTarget, lam, dt);
    damp(this.aim, aimTarget, lam * 1.25, dt);
    dampDir(this.up, upTarget, 5, dt);
    this.fov = dampScalar(this.fov, fovTarget, 4, dt);

    this._apply();
  }

  /**
   * Spectate/cinematic director. Both teams are AI — no single player to
   * follow — so frame the bell + the cluster of nearby players + the
   * attacked goal. Same no-global-up discipline; dramatic broadside
   * loop-cam when loopGlow>0.15. Cut/preset chosen via cycleCinematicAngle.
   */
  cinematic(
    bell: V3,
    players: { id: string; p: V3; team: 'home' | 'away' }[],
    attackX: number,
    loopGlow: number,
    skinR: number,
    dt: number,
  ): void {
    this.cineT += dt;
    const dir = this.smoothDir(attackX, dt);

    // Cluster of players near the bell → centroid + spread (for framing).
    let cx = bell.x,
      cy = bell.y,
      cz = bell.z,
      wsum = 1;
    let maxd = skinR * 0.5;
    for (const pl of players) {
      const dx = pl.p.x - bell.x,
        dy = pl.p.y - bell.y,
        dz = pl.p.z - bell.z;
      const d = Math.hypot(dx, dy, dz);
      // weight nearer players more (the action is around the bell)
      const w = 1 / (1 + (d / (skinR * 1.5)) ** 2);
      cx += pl.p.x * w;
      cy += pl.p.y * w;
      cz += pl.p.z * w;
      wsum += w;
      if (d < skinR * 2.5) maxd = Math.max(maxd, d);
    }
    const focus = new V(
      (cx / wsum) * 0.6 + bell.x * 0.4,
      (cy / wsum) * 0.6 + bell.y * 0.4,
      (cz / wsum) * 0.6 + bell.z * 0.4,
    );

    const rho = Math.max(2, Math.hypot(focus.y, focus.z));
    const ry = focus.y / rho;
    const rz = focus.z / rho;
    const ty = -rz,
      tz = ry;

    let camTarget: THREE.Vector3;
    let aimTarget: THREE.Vector3;
    let fovTarget: number;
    let upTarget: THREE.Vector3;

    const wantLoop = this.wantLoopCam(loopGlow);
    this.loopBlend += ((wantLoop ? 1 : 0) - this.loopBlend) * (1 - Math.exp(-4 * dt));

    // framing scale grows with the spread so the cluster never overflows.
    // Min raised so we always sit far enough back to read a good chunk of
    // the cylinder cross-section — gives volume context so a bell near the
    // skin reads as "inside the tube" rather than "at the edge".
    const frame = Math.max(skinR * 1.8, maxd * 1.6);

    if (this.loopBlend > 0.02) {
      // Hero broadside loop-cam (shared with PLAY) — never gimbal-locks
      // because up = ±spine here, fully orthogonal to the radial standoff.
      const standoff = skinR * (2.1 - 0.5 * loopGlow);
      const drift = Math.sin(this.cineT * 0.12) * skinR * 0.12; // gentle track
      camTarget = new V(bell.x - dir * (skinR * 0.5 + drift), ry * standoff, rz * standoff);
      aimTarget = new V(bell.x, bell.y * 0.5, bell.z * 0.5);
      fovTarget = 44 - 6 * loopGlow;
      upTarget = new V(1, 0, 0).multiplyScalar(dir);
    } else {
      switch (this.cineAngle) {
        case 'low': {
          // Dramatic low/near-skin, looking up-spine across the cluster.
          const standoff = Math.min(skinR * 0.96, rho + frame * 0.25);
          camTarget = new V(
            focus.x - dir * frame * 1.1,
            ry * standoff + ty * frame * 0.2,
            rz * standoff + tz * frame * 0.2,
          );
          aimTarget = new V(focus.x + dir * skinR * 0.8, focus.y, focus.z);
          fovTarget = 56;
          upTarget = new V(0, -ry, -rz);
          break;
        }
        case 'goal': {
          // Goal-cam: BEHIND & above the attacked ring, off-axis, looking
          // back at the incoming play. Deliberately NOT down the spine —
          // staring along the axis stacked the additive sunline + ring to a
          // white-out. The 3/4 behind-the-goal angle keeps the scoring drama
          // (ring + incoming bell in frame) without nuking the lens.
          camTarget = new V(
            attackX + dir * frame * 0.45,
            ry * skinR * 0.85,
            rz * skinR * 0.85,
          );
          aimTarget = new V(
            focus.x * 0.45 + attackX * 0.55,
            focus.y * 0.4,
            focus.z * 0.4,
          );
          fovTarget = 50;
          upTarget = new V(0, -ry, -rz);
          break;
        }
        case 'wide': {
          // Establishing wide: high toward the axis, whole calm in view.
          const lift = skinR * 0.9;
          camTarget = new V(
            focus.x - dir * frame * 2.2,
            -ry * lift,
            -rz * lift,
          );
          aimTarget = new V(focus.x + dir * skinR * 0.6, focus.y * 0.5, focus.z * 0.5);
          fovTarget = 70;
          upTarget = new V(0, -ry, -rz);
          break;
        }
        case 'broadcast':
        default: {
          // Broadcast-follow: behind & lifted, slow parallax track. Lifted
          // more toward the axis so the cross-section / volume stays in frame.
          const back = frame * 1.7;
          const lift = skinR * 0.72;
          const side = skinR * 0.45 + Math.sin(this.cineT * 0.12) * skinR * 0.08;
          camTarget = new V(
            focus.x - dir * back,
            focus.y - ry * lift + ty * side,
            focus.z - rz * lift + tz * side,
          );
          aimTarget = new V(focus.x + dir * skinR * 0.9, focus.y * 0.5, focus.z * 0.5);
          fovTarget = 64;
          upTarget = new V(0, -ry, -rz);
          break;
        }
      }
    }

    if (!this.inited) {
      this.pos.copy(camTarget);
      this.aim.copy(aimTarget);
      this.up.copy(upTarget);
      this.fov = fovTarget;
      this.inited = true;
    }
    // Director moves are smooth & languid (except the snappier loop-cam).
    const lam = this.loopBlend > 0.02 ? 4.5 : 2.4;
    damp(this.pos, camTarget, lam, dt);
    damp(this.aim, aimTarget, lam * 1.2, dt);
    dampDir(this.up, upTarget, 4, dt);
    this.fov = dampScalar(this.fov, fovTarget, 3, dt);

    this._apply();
  }

  /** Cycle director preset (broadcast → low → goal → wide → …). */
  cycleCinematicAngle(): void {
    const i = this.cineOrder.indexOf(this.cineAngle);
    this.cineAngle = this.cineOrder[(i + 1) % this.cineOrder.length];
  }

  private _apply(): void {
    this.cam.position.copy(this.pos);
    this.cam.up.copy(this.up.lengthSq() < 1e-4 ? new V(0, 1, 0) : this.up.clone().normalize());
    this.cam.lookAt(this.aim);
    if (Math.abs(this.cam.fov - this.fov) > 0.01) {
      this.cam.fov = this.fov;
      this.cam.updateProjectionMatrix();
    }
  }

  /** Snap (no damping) — call when (re)entering a match so frame 1 is framed. */
  reset(): void {
    this.inited = false;
    this.loopBlend = 0;
    this.lead.set(0, 0, 0);
    this.hasPrev = false;
    this.lookYaw = 0;
    this.lookPitch = 0;
    this.lookActive = false;
    this.cineT = 0;
    this.dirS = 0;
    this.loopLatch = false;
  }
}

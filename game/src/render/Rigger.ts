// Rigger renderer — bold, emissive, instantly readable figures.
// Team-coloured (home = cyan #1aa6b7, away = orange #d4602a) with role accent.
// P1 (the human) is unmistakably YOU: brightest emissive + halo ring + ▼ beacon above.
// Grounded = dimmed + skin-tint. contactRef = clip-bracket glint.
// Motion streak when moving fast (juice). Pooled for perf.

import * as THREE from 'three';
import type { PlayerSim } from '../sim/types';
import { PAL, ROLE_TINT } from '../ui/palette';

// ── Geometry constants — sized for a ~45 m calm radius ───────────────────────

// Players live in a 90 m diameter cylinder. We need figures visible from
// camera distances of ~20-60 m. Scale accordingly.
const BODY_RADIUS  = 1.2;  // m — thick trunk, easy to read
const BODY_LENGTH  = 2.8;  // m — capsule cylinder half-length
const HEAD_RADIUS  = 0.85; // m — faceted gem head
const AIM_LEN      = 3.2;  // m — facing spike length
const AIM_BASE_R   = 0.22; // m — base radius of aim cone

// P1 distinction
const HALO_OUTER   = 3.2;  // m — ring outer
const HALO_TUBE    = 0.18; // m
const BEACON_SIZE  = 1.6;  // m — ▼ tetrahedron arrow above player
const BEACON_Y     = BODY_LENGTH * 0.5 + HEAD_RADIUS * 2.0 + BEACON_SIZE * 1.1;

// Contact clip bracket
const CLIP_OUTER   = 2.2;
const CLIP_TUBE    = 0.14;

// Streak tail — shown when speed > threshold
const STREAK_SPEED_SQ = 6 * 6; // m/s squared threshold
const STREAK_LEN      = 5.0;

const MAX_RIGGERS = 12;

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Blend two hex colours by t (0=a, 1=b). */
function lerpHex(a: number, b: number, t: number): number {
  const ri = a >> 16 & 0xff, gi = a >> 8 & 0xff, bi = a & 0xff;
  const rf = b >> 16 & 0xff, gf = b >> 8 & 0xff, bf = b & 0xff;
  const r = (ri + (rf - ri) * t) | 0;
  const g = (gi + (gf - gi) * t) | 0;
  const bl = (bi + (bf - bi) * t) | 0;
  return (r << 16) | (g << 8) | bl;
}

function teamBase(team: string): number {
  return team === 'home' ? PAL.cyan : PAL.orange;
}

/** Body colour: full team when airborne, dimmed toward skin-tint when grounded. */
function bodyColor(team: string, grounded: boolean): number {
  const base = teamBase(team);
  if (!grounded) return base;
  return lerpHex(base, PAL.dim, 0.70);
}

/** Apply sim Quat to a THREE.Quaternion in-place. */
function applySimQuat(
  out: THREE.Quaternion,
  q: { x: number; y: number; z: number; w: number },
): void {
  out.set(q.x, q.y, q.z, q.w).normalize();
}

// Reusable scratch vectors (avoid alloc in hot path)
const _vel = new THREE.Vector3();
const _negVel = new THREE.Vector3();

// ── Per-instance visual group ─────────────────────────────────────────────────

class RiggerInstance {
  readonly root = new THREE.Group();

  // Core body parts
  private bodyMesh: THREE.Mesh;
  private headMesh: THREE.Mesh;
  // Emissive shell over body — separate so we can tune glow independently
  private glowShell: THREE.Mesh;

  // Facing indicator
  private aimMesh: THREE.Mesh;

  // Role ring at chest
  private accentRing: THREE.Mesh;

  // P1-only extras (always in scene, visibility toggled)
  private haloRing: THREE.Mesh;
  private beacon: THREE.Mesh;     // ▼ arrow above

  // Contact clip bracket
  private clipGlint: THREE.Mesh;

  // Motion streak
  private streakLine: THREE.Line;
  private streakPos: Float32Array;
  private streakGeo: THREE.BufferGeometry;
  private streakMat: THREE.LineBasicMaterial;

  constructor() {
    // ── Body: capsule, bold & readable ───────────────────────────────────────
    const bodyGeo = new THREE.CapsuleGeometry(BODY_RADIUS, BODY_LENGTH, 6, 12);
    this.bodyMesh = new THREE.Mesh(
      bodyGeo,
      new THREE.MeshStandardMaterial({
        color: PAL.dim,
        roughness: 0.55,
        metalness: 0.10,
        // emissive set dynamically
      }),
    );
    this.bodyMesh.castShadow = false;

    // Translucent glow shell — slightly larger, additive
    this.glowShell = new THREE.Mesh(
      new THREE.CapsuleGeometry(BODY_RADIUS * 1.14, BODY_LENGTH, 4, 8),
      new THREE.MeshBasicMaterial({
        color: PAL.dim,
        transparent: true,
        opacity: 0.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.BackSide,
      }),
    );

    // ── Head: faceted gem ─────────────────────────────────────────────────────
    this.headMesh = new THREE.Mesh(
      new THREE.OctahedronGeometry(HEAD_RADIUS, 1),
      new THREE.MeshStandardMaterial({
        color: PAL.dim,
        roughness: 0.45,
        metalness: 0.25,
        emissive: new THREE.Color(0x000000),
        emissiveIntensity: 0,
      }),
    );
    // Sits atop the body capsule
    this.headMesh.position.y = BODY_LENGTH * 0.5 + BODY_RADIUS + HEAD_RADIUS * 0.9;

    // ── Aim indicator: tapered cone in local +Y (facing direction) ────────────
    const aimGeo = new THREE.CylinderGeometry(0, AIM_BASE_R, AIM_LEN, 6, 1);
    this.aimMesh = new THREE.Mesh(
      aimGeo,
      new THREE.MeshBasicMaterial({
        color: PAL.paper,
        transparent: true,
        opacity: 0.7,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    // Position: tip points forward, base at head level
    this.aimMesh.position.y = this.headMesh.position.y + HEAD_RADIUS + AIM_LEN * 0.5;

    // ── Role accent ring at chest ─────────────────────────────────────────────
    this.accentRing = new THREE.Mesh(
      new THREE.TorusGeometry(BODY_RADIUS * 0.85, 0.14, 4, 16),
      new THREE.MeshBasicMaterial({
        color: PAL.dim,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        opacity: 0.85,
      }),
    );
    this.accentRing.position.y = BODY_LENGTH * 0.12;
    this.accentRing.rotation.x = Math.PI / 2;

    // ── P1 halo ring (horizontal disc at mid-body) ────────────────────────────
    this.haloRing = new THREE.Mesh(
      new THREE.TorusGeometry(HALO_OUTER, HALO_TUBE, 6, 24),
      new THREE.MeshBasicMaterial({
        color: PAL.paper,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        opacity: 0.0,
      }),
    );
    this.haloRing.rotation.x = Math.PI / 2;
    this.haloRing.position.y = 0;

    // ── P1 beacon: ▼ tetrahedron arrow floating above ─────────────────────────
    // ConeGeometry pointing down (tip at -y)
    const beaconGeo = new THREE.ConeGeometry(BEACON_SIZE * 0.55, BEACON_SIZE, 4, 1);
    // Rotate so tip faces down toward the player
    beaconGeo.rotateZ(Math.PI);
    this.beacon = new THREE.Mesh(
      beaconGeo,
      new THREE.MeshBasicMaterial({
        color: PAL.paper,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        opacity: 0.0,
      }),
    );
    this.beacon.position.y = BEACON_Y;

    // ── Contact clip bracket (torus at equator, team-tinted) ──────────────────
    this.clipGlint = new THREE.Mesh(
      new THREE.TorusGeometry(CLIP_OUTER, CLIP_TUBE, 4, 20),
      new THREE.MeshBasicMaterial({
        color: PAL.paper,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        opacity: 0.0,
      }),
    );
    this.clipGlint.rotation.x = Math.PI / 2;
    this.clipGlint.position.y = BODY_LENGTH * 0.5;

    // ── Motion streak: 2-point line behind the player ─────────────────────────
    this.streakPos = new Float32Array(6); // 2 points × 3 components
    this.streakGeo = new THREE.BufferGeometry();
    this.streakGeo.setAttribute('position', new THREE.BufferAttribute(this.streakPos, 3));
    this.streakMat = new THREE.LineBasicMaterial({
      color: PAL.dim,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.streakLine = new THREE.Line(this.streakGeo, this.streakMat);
    this.streakLine.frustumCulled = false;

    this.root.add(
      this.glowShell,
      this.bodyMesh,
      this.headMesh,
      this.aimMesh,
      this.accentRing,
      this.haloRing,
      this.beacon,
      this.clipGlint,
      this.streakLine,
    );
    this.root.visible = false;
  }

  /** Update this instance for a given PlayerSim. isP1 = the human player. */
  apply(ps: PlayerSim, isP1: boolean): void {
    this.root.visible = true;

    // ── Position & orientation ────────────────────────────────────────────────
    this.root.position.set(ps.p.x, ps.p.y, ps.p.z);
    applySimQuat(this.root.quaternion, ps.q);

    const base    = teamBase(ps.team);
    const bodyCol = bodyColor(ps.team, ps.grounded);

    // ── Body & head colours ───────────────────────────────────────────────────
    const bodyMat = this.bodyMesh.material as THREE.MeshStandardMaterial;
    bodyMat.color.setHex(bodyCol);

    const headMat = this.headMesh.material as THREE.MeshStandardMaterial;
    headMat.color.setHex(bodyCol);

    // Emissive glow — P1 is brightest, airborne > grounded
    const emissiveHex = ps.grounded ? 0x000000 : bodyCol;
    const emissiveInt = isP1
      ? (ps.grounded ? 0.45 : 1.2)
      : (ps.grounded ? 0.0  : 0.4);

    bodyMat.emissive.setHex(isP1 ? base : emissiveHex);
    bodyMat.emissiveIntensity = emissiveInt;
    headMat.emissive.setHex(isP1 ? base : (ps.grounded ? 0x000000 : bodyCol));
    headMat.emissiveIntensity = isP1 ? emissiveInt * 0.8 : emissiveInt * 0.5;

    // Glow shell
    const shellMat = this.glowShell.material as THREE.MeshBasicMaterial;
    shellMat.color.setHex(base);
    shellMat.opacity = isP1 ? 0.28 : (ps.grounded ? 0.0 : 0.10);

    // ── Aim indicator ─────────────────────────────────────────────────────────
    const aimMat = this.aimMesh.material as THREE.MeshBasicMaterial;
    aimMat.color.setHex(isP1 ? PAL.paper : lerpHex(base, PAL.paper, 0.25));
    aimMat.opacity = isP1 ? 0.95 : 0.40;

    // ── Role accent ring ──────────────────────────────────────────────────────
    const roleTint = ROLE_TINT[ps.role] ?? PAL.dim;
    (this.accentRing.material as THREE.MeshBasicMaterial).color.setHex(roleTint);
    (this.accentRing.material as THREE.MeshBasicMaterial).opacity = 0.90;

    // ── P1 halo & beacon ──────────────────────────────────────────────────────
    const haloMat   = this.haloRing.material as THREE.MeshBasicMaterial;
    const beaconMat = this.beacon.material   as THREE.MeshBasicMaterial;
    if (isP1) {
      haloMat.color.setHex(PAL.cyan);
      haloMat.opacity = 0.90;
      beaconMat.color.setHex(PAL.paper);
      beaconMat.opacity = 0.95;
      // Gentle bob: beacon oscillates slightly on local Y (world-space via root)
      const t = performance.now() * 0.0015;
      this.beacon.position.y = BEACON_Y + Math.sin(t) * 0.4;
    } else {
      haloMat.opacity   = 0.0;
      beaconMat.opacity = 0.0;
    }

    // ── Contact clip bracket ──────────────────────────────────────────────────
    const clipMat = this.clipGlint.material as THREE.MeshBasicMaterial;
    if (ps.contactRef !== null) {
      clipMat.color.setHex(lerpHex(base, PAL.paper, 0.50));
      clipMat.opacity = isP1 ? 0.95 : 0.65;
    } else {
      clipMat.opacity = 0.0;
    }

    // ── Motion streak ─────────────────────────────────────────────────────────
    const vx = ps.v.x, vy = ps.v.y, vz = ps.v.z;
    const speedSq = vx * vx + vy * vy + vz * vz;
    if (speedSq > STREAK_SPEED_SQ) {
      const invSpeed = 1.0 / Math.sqrt(speedSq);
      // In local (root) space — tail is behind body
      // The root's quaternion transforms world->local, but for streak
      // we work in world space directly (streakLine is a child of root,
      // so we must express positions relative to root origin = ps.p)
      _vel.set(vx, vy, vz);
      _negVel.set(-vx * invSpeed * STREAK_LEN, -vy * invSpeed * STREAK_LEN, -vz * invSpeed * STREAK_LEN);
      // Transform into root-local space (root.quaternion is the inverse of world->local for pure rotation)
      _negVel.applyQuaternion(this.root.quaternion.clone().invert());

      // Point 0 = body centre (local origin)
      this.streakPos[0] = 0;
      this.streakPos[1] = 0;
      this.streakPos[2] = 0;
      // Point 1 = tail
      this.streakPos[3] = _negVel.x;
      this.streakPos[4] = _negVel.y;
      this.streakPos[5] = _negVel.z;
      this.streakGeo.attributes.position.needsUpdate = true;

      this.streakMat.color.setHex(base);
      // Opacity ramps with speed, capped at 0.75
      this.streakMat.opacity = Math.min(0.75, (speedSq - STREAK_SPEED_SQ) / (30 * 30 - STREAK_SPEED_SQ) * 0.75);
    } else {
      this.streakMat.opacity = 0.0;
    }
  }

  hide(): void {
    this.root.visible = false;
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

/** Manages a pool of rigger visual instances. Call sync() every rendered frame. */
export class Riggers {
  private pool: RiggerInstance[] = [];
  private group = new THREE.Group();

  constructor(scene: THREE.Scene) {
    for (let i = 0; i < MAX_RIGGERS; i++) {
      const inst = new RiggerInstance();
      this.pool.push(inst);
      this.group.add(inst.root);
    }
    scene.add(this.group);
  }

  /**
   * Sync visuals to the current (interpolated) player list.
   * @param players  Interpolated PlayerSim array for this render frame.
   * @param p1Id     The local human player's id — rendered unmistakably bright.
   */
  sync(players: PlayerSim[], p1Id: string): void {
    for (const inst of this.pool) inst.hide();

    const n = Math.min(players.length, MAX_RIGGERS);
    for (let i = 0; i < n; i++) {
      this.pool[i].apply(players[i], players[i].id === p1Id);
    }
  }
}

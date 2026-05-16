// Procedural low-poly rigger figures — capsule body + facing indicator +
// role-tint accent. Instanced where sensible. Positioned/oriented from
// interpolated PlayerSim (p, q). Grounded riggers rendered dim at the skin;
// contactRef state shown via a faint clip-glow on the nearest spar.

import * as THREE from 'three';
import type { PlayerSim } from '../sim/types';
import { PAL, ROLE_TINT } from '../ui/palette';

// ── Geometry constants ────────────────────────────────────────────────────

const BODY_RADIUS  = 0.55; // m
const BODY_LENGTH  = 1.40; // m (capsule half-length excluding caps)
const HEAD_RADIUS  = 0.40; // m
const AIM_LEN      = 1.60; // m  aim-indicator spike
const AIM_RADIUS   = 0.07;

// Max simultaneous riggers on the field (both teams).
const MAX_RIGGERS = 12;

// ── Helpers ───────────────────────────────────────────────────────────────

function teamColor(team: string, grounded: boolean): number {
  const base = team === 'home' ? PAL.cyan : PAL.orange;
  if (!grounded) return base;
  // grounded: mix toward dim
  const r = ((base >> 16 & 0xff) * 0.45 + (PAL.dim >> 16 & 0xff) * 0.55) | 0;
  const g = ((base >>  8 & 0xff) * 0.45 + (PAL.dim >>  8 & 0xff) * 0.55) | 0;
  const b = ((base       & 0xff) * 0.45 + (PAL.dim       & 0xff) * 0.55) | 0;
  return (r << 16) | (g << 8) | b;
}

/** Convert sim Quat to THREE.Quaternion in-place. */
function applySimQuat(out: THREE.Quaternion, q: { x: number; y: number; z: number; w: number }): void {
  out.set(q.x, q.y, q.z, q.w).normalize();
}

// ── Per-instance visual group ─────────────────────────────────────────────

class RiggerInstance {
  readonly root = new THREE.Group();
  private bodyMesh: THREE.Mesh;
  private headMesh: THREE.Mesh;
  private aimMesh:  THREE.Mesh;
  private accentMesh: THREE.Mesh;
  private glowRing: THREE.Mesh;

  constructor() {
    // Body: low-poly capsule (8 radial segments keeps it cheap)
    const bodyGeo = new THREE.CapsuleGeometry(BODY_RADIUS, BODY_LENGTH, 4, 8);
    this.bodyMesh = new THREE.Mesh(
      bodyGeo,
      new THREE.MeshStandardMaterial({ color: PAL.dim, roughness: 0.8, metalness: 0.05 }),
    );
    // Head sits atop the body
    this.headMesh = new THREE.Mesh(
      new THREE.OctahedronGeometry(HEAD_RADIUS, 1), // faceted low-poly
      new THREE.MeshStandardMaterial({ color: PAL.dim, roughness: 0.7, metalness: 0 }),
    );
    this.headMesh.position.y = BODY_LENGTH * 0.5 + BODY_RADIUS + HEAD_RADIUS * 0.85;

    // Aim indicator: a thin tapered spike along local +Y (player facing)
    const aimGeo = new THREE.CylinderGeometry(0, AIM_RADIUS, AIM_LEN, 5, 1);
    this.aimMesh = new THREE.Mesh(
      aimGeo,
      new THREE.MeshBasicMaterial({ color: PAL.paper, transparent: true, opacity: 0.6 }),
    );
    this.aimMesh.position.y = BODY_LENGTH * 0.5 + BODY_RADIUS + AIM_LEN * 0.5;

    // Role accent: a tiny disc / ring at chest level
    this.accentMesh = new THREE.Mesh(
      new THREE.TorusGeometry(BODY_RADIUS * 0.75, 0.08, 4, 12),
      new THREE.MeshBasicMaterial({ color: PAL.dim }),
    );
    this.accentMesh.position.y = BODY_LENGTH * 0.1;
    this.accentMesh.rotation.x = Math.PI / 2;

    // Contact glow ring — faint, shown when contactRef != null
    this.glowRing = new THREE.Mesh(
      new THREE.TorusGeometry(BODY_RADIUS * 1.35, 0.06, 4, 16),
      new THREE.MeshBasicMaterial({
        color: PAL.cyan,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    this.glowRing.rotation.x = Math.PI / 2;

    this.root.add(this.bodyMesh, this.headMesh, this.aimMesh, this.accentMesh, this.glowRing);
    this.root.visible = false;
  }

  /** Update this instance for a given PlayerSim. isP1 = the human player. */
  apply(ps: PlayerSim, isP1: boolean): void {
    this.root.visible = true;

    // Position
    this.root.position.set(ps.p.x, ps.p.y, ps.p.z);

    // Orientation from sim quaternion
    applySimQuat(this.root.quaternion, ps.q);

    // Body colour — team + grounded state
    const bodyCol = teamColor(ps.team, ps.grounded);
    (this.bodyMesh.material as THREE.MeshStandardMaterial).color.setHex(bodyCol);
    (this.headMesh.material as THREE.MeshStandardMaterial).color.setHex(bodyCol);

    // Role accent tint
    const roleTint = ROLE_TINT[ps.role] ?? PAL.dim;
    (this.accentMesh.material as THREE.MeshBasicMaterial).color.setHex(roleTint);

    // Aim indicator color — P1 gets bright paper; others get dim
    const aimCol = isP1 ? PAL.paper : PAL.dim;
    (this.aimMesh.material as THREE.MeshBasicMaterial).color.setHex(aimCol);
    (this.aimMesh.material as THREE.MeshBasicMaterial).opacity = isP1 ? 0.85 : 0.30;

    // P1 highlight: faint emissive boost on body
    const emInt = isP1 ? 0.35 : 0.0;
    (this.bodyMesh.material as THREE.MeshStandardMaterial).emissive.setHex(isP1 ? bodyCol : 0x000000);
    (this.bodyMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = emInt;

    // Contact glow
    const hasContact = ps.contactRef !== null;
    (this.glowRing.material as THREE.MeshBasicMaterial).opacity = hasContact ? 0.55 : 0;
    const contactTint = ps.team === 'home' ? PAL.cyan : PAL.orange;
    (this.glowRing.material as THREE.MeshBasicMaterial).color.setHex(contactTint);
  }

  hide(): void {
    this.root.visible = false;
  }
}

// ── Public API ────────────────────────────────────────────────────────────

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
   * Sync the visuals to the current (interpolated) player list.
   * @param players  Interpolated PlayerSim array for this render frame.
   * @param p1Id     The local human player's id (highlighted differently).
   */
  sync(players: PlayerSim[], p1Id: string): void {
    // Hide all then re-show used slots
    for (const inst of this.pool) inst.hide();

    const n = Math.min(players.length, MAX_RIGGERS);
    for (let i = 0; i < n; i++) {
      this.pool[i].apply(players[i], players[i].id === p1Id);
    }
  }
}

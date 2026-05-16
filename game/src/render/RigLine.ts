// Rig-line renderer — draws each player's grapple line when player.line is
// set. Taut lines are straight and bright; slack lines sag with a cosmetic
// catenary curve. A small anchor glint marks the attach point.

import * as THREE from 'three';
import type { PlayerSim } from '../sim/types';
import { PAL } from '../ui/palette';

// ── Constants ─────────────────────────────────────────────────────────────

// Catenary samples — enough curve points for a smooth sag
const SAG_SEGMENTS = 20;

// Line width hint (ignored by WebGL but noted for WebGPU / future)
const LINE_OPTS = { transparent: true, depthWrite: false };

// Anchor glint sphere size (world metres)
const GLINT_RADIUS = 0.38;

// ── Catenary helper ───────────────────────────────────────────────────────

/**
 * Produce `segments+1` points along a cosmetic catenary from `a` to `b`.
 * `sag` is the peak perpendicular displacement at the midpoint in metres.
 * If sag ≈ 0 the output is a straight line.
 */
function catenary(
  a: THREE.Vector3,
  b: THREE.Vector3,
  sag: number,
  segments: number,
  out: Float32Array,
): void {
  // "Down" in the calm is away from the axis (centrifugal).
  // We approximate it as the direction from axis centre toward the midpoint.
  const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
  // radial outward from spin axis (y,z plane); axis is at x-only
  const radial = new THREE.Vector3(0, mid.y, mid.z).normalize();
  if (radial.lengthSq() < 1e-6) radial.set(0, 1, 0); // fallback if exactly on axis

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    // lerp along the chord
    const px = a.x + (b.x - a.x) * t;
    const py = a.y + (b.y - a.y) * t;
    const pz = a.z + (b.z - a.z) * t;
    // catenary sag parabola: sag * 4 * t * (1-t) at midpoint
    const drop = sag * 4.0 * t * (1.0 - t);
    // outward displacement
    out[i * 3 + 0] = px + radial.x * drop;
    out[i * 3 + 1] = py + radial.y * drop;
    out[i * 3 + 2] = pz + radial.z * drop;
  }
}

// ── Per-player line object ────────────────────────────────────────────────

class LineInstance {
  readonly group = new THREE.Group();
  private geo  = new THREE.BufferGeometry();
  private posArr: Float32Array;
  private line: THREE.Line;
  private glint: THREE.Mesh;
  private mat: THREE.LineBasicMaterial;

  constructor() {
    this.posArr = new Float32Array((SAG_SEGMENTS + 1) * 3);
    this.geo.setAttribute(
      'position',
      new THREE.BufferAttribute(this.posArr, 3),
    );
    this.mat = new THREE.LineBasicMaterial({
      color: PAL.dim,
      ...LINE_OPTS,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
    });
    this.line = new THREE.Line(this.geo, this.mat);
    this.line.frustumCulled = false;

    this.glint = new THREE.Mesh(
      new THREE.OctahedronGeometry(GLINT_RADIUS, 0),
      new THREE.MeshBasicMaterial({
        color: PAL.paper,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        opacity: 0.9,
      }),
    );

    this.group.add(this.line, this.glint);
    this.group.visible = false;
  }

  apply(ps: PlayerSim): void {
    const lineState = ps.line;
    if (!lineState) {
      this.group.visible = false;
      return;
    }
    this.group.visible = true;

    const ap = lineState.anchorPos;
    const pp = ps.p;

    const from = new THREE.Vector3(pp.x, pp.y, pp.z);
    const to   = new THREE.Vector3(ap.x, ap.y, ap.z);
    const chord = from.distanceTo(to);

    // Slack sag: zero when taut, grows as estimated droop
    const sag = lineState.taut ? 0 : Math.min(chord * 0.08, 4.0);

    catenary(from, to, sag, SAG_SEGMENTS, this.posArr);
    this.geo.attributes.position.needsUpdate = true;
    this.geo.setDrawRange(0, SAG_SEGMENTS + 1);

    // Team tint: bright when taut, dimmer when slack
    const teamCol = ps.team === 'home' ? PAL.cyan : PAL.orange;
    const opacity = lineState.taut ? 0.90 : 0.40;
    this.mat.color.setHex(lineState.taut ? teamCol : PAL.dim);
    this.mat.opacity = opacity;

    // Anchor glint position + team tint
    this.glint.position.set(ap.x, ap.y, ap.z);
    (this.glint.material as THREE.MeshBasicMaterial).color.setHex(teamCol);
  }

  hide(): void {
    this.group.visible = false;
  }
}

// ── Public API ────────────────────────────────────────────────────────────

/** Manages one LineInstance per player. Call sync() every rendered frame. */
export class RigLines {
  private instances = new Map<string, LineInstance>();
  private group = new THREE.Group();

  constructor(scene: THREE.Scene) {
    scene.add(this.group);
  }

  /**
   * Sync rig-line visuals to the current player list.
   * @param players  Interpolated PlayerSim array.
   */
  sync(players: PlayerSim[]): void {
    // Mark all existing instances for potential hide
    for (const inst of this.instances.values()) inst.hide();

    for (const ps of players) {
      let inst = this.instances.get(ps.id);
      if (!inst) {
        inst = new LineInstance();
        this.group.add(inst.group);
        this.instances.set(ps.id, inst);
      }
      inst.apply(ps);
    }
  }
}

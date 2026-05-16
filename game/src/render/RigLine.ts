// Rig-line renderer — the grapple line is the core mechanic; make it read.
// Taut = bright, energized straight line in team colour; P1's is brightest.
// Slack = soft catenary sag, dimmer, using the radial-outward "gravity" direction.
// Anchor glint = vivid glowing octahedron at the anchorPos.
// Player end of the line attaches at the rigger's "hand" (offset from body centre).

import * as THREE from 'three';
import type { PlayerSim } from '../sim/types';
import { PAL } from '../ui/palette';

// ── Constants ─────────────────────────────────────────────────────────────────

// Number of segments for the catenary curve
const SAG_SEGMENTS = 24;

// Anchor glint size (world metres) — large enough to be visible as a focal point
const GLINT_RADIUS = 0.72;

// "Hand" offset from body centre along local +Y (body up/forward direction).
// Positions the line player-end at roughly arm height rather than the centre of mass.
const HAND_OFFSET_LOCAL = new THREE.Vector3(0, 1.4, 0.9);

// Taut line width boost: we can't set linewidth >1 in WebGL but we compensate
// with brightness + a thin parallel "glow" overlay
const TAUT_OPACITY   = 0.95;
const SLACK_OPACITY  = 0.42;
const P1_OPACITY_BOOST = 0.05; // added on top

// Glow overlay for taut lines (additive, slightly thicker via duplicate)
const GLOW_OPACITY_TAUT  = 0.50;
const GLOW_OPACITY_SLACK = 0.0;

// ── Helpers ───────────────────────────────────────────────────────────────────

const _handWorld = new THREE.Vector3();
const _handOffsetRotated = new THREE.Vector3();
const _quat = new THREE.Quaternion();

/** World-space position of a player's "hand" given their position and orientation. */
function handWorldPos(ps: PlayerSim): THREE.Vector3 {
  _handOffsetRotated.copy(HAND_OFFSET_LOCAL);
  _quat.set(ps.q.x, ps.q.y, ps.q.z, ps.q.w).normalize();
  _handOffsetRotated.applyQuaternion(_quat);
  _handWorld.set(ps.p.x + _handOffsetRotated.x, ps.p.y + _handOffsetRotated.y, ps.p.z + _handOffsetRotated.z);
  return _handWorld;
}

/**
 * Fill `out` (Float32Array, length = (segments+1)*3) with positions along a
 * cosmetic catenary from `a` to `b`. `sag` is the peak radial-outward
 * displacement at the midpoint (metres). sag=0 gives a straight line.
 *
 * In the calm, centrifugal "down" is radially outward from the spin axis
 * (+X). We approximate the sag direction as the radial outward at the midpoint
 * in the (y,z) plane.
 */
function fillCatenary(
  a: THREE.Vector3,
  b: THREE.Vector3,
  sag: number,
  segments: number,
  out: Float32Array,
): void {
  const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
  // Radially outward from the spin axis (+X), living in the (y,z) plane
  const radial = new THREE.Vector3(0, mid.y, mid.z);
  const radLen = radial.length();
  if (radLen > 1e-6) {
    radial.multiplyScalar(1.0 / radLen);
  } else {
    radial.set(0, 1, 0); // fallback if exactly on axis
  }

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    // Lerp along chord
    const px = a.x + (b.x - a.x) * t;
    const py = a.y + (b.y - a.y) * t;
    const pz = a.z + (b.z - a.z) * t;
    // Parabolic sag: peaks at t=0.5, zero at endpoints
    const drop = sag * 4.0 * t * (1.0 - t);
    out[i * 3 + 0] = px + radial.x * drop;
    out[i * 3 + 1] = py + radial.y * drop;
    out[i * 3 + 2] = pz + radial.z * drop;
  }
}

// ── Per-player line object ─────────────────────────────────────────────────────

class LineInstance {
  readonly group = new THREE.Group();

  // Primary line
  private geo: THREE.BufferGeometry;
  private posArr: Float32Array;
  private lineMesh: THREE.Line;
  private lineMat: THREE.LineBasicMaterial;

  // Glow overlay (duplicate, additive, slightly wider visually due to bloom)
  private glowGeo: THREE.BufferGeometry;
  private glowLine: THREE.Line;
  private glowMat: THREE.LineBasicMaterial;

  // Anchor glint
  private glint: THREE.Mesh;
  private glintInner: THREE.Mesh; // bright core

  constructor() {
    const count = (SAG_SEGMENTS + 1) * 3;
    this.posArr = new Float32Array(count);

    // Primary line
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.posArr, 3));
    this.lineMat = new THREE.LineBasicMaterial({
      color: PAL.dim,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.lineMesh = new THREE.Line(this.geo, this.lineMat);
    this.lineMesh.frustumCulled = false;

    // Glow overlay — shares same position array (geometry copy for independence)
    this.glowGeo = new THREE.BufferGeometry();
    this.glowGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count), 3));
    this.glowMat = new THREE.LineBasicMaterial({
      color: PAL.dim,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.glowLine = new THREE.Line(this.glowGeo, this.glowMat);
    this.glowLine.frustumCulled = false;

    // Anchor glint: large glowing octahedron
    this.glint = new THREE.Mesh(
      new THREE.OctahedronGeometry(GLINT_RADIUS, 1),
      new THREE.MeshBasicMaterial({
        color: PAL.paper,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        opacity: 0.0,
      }),
    );
    // Inner bright core (smaller, opaque-ish)
    this.glintInner = new THREE.Mesh(
      new THREE.OctahedronGeometry(GLINT_RADIUS * 0.45, 0),
      new THREE.MeshBasicMaterial({
        color: PAL.paper,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        opacity: 0.0,
      }),
    );

    this.group.add(this.lineMesh, this.glowLine, this.glint, this.glintInner);
    this.group.visible = false;
  }

  apply(ps: PlayerSim, isP1: boolean): void {
    const lineState = ps.line;
    if (!lineState) {
      this.group.visible = false;
      return;
    }
    this.group.visible = true;

    const ap  = lineState.anchorPos;
    const to  = new THREE.Vector3(ap.x, ap.y, ap.z);
    const from = handWorldPos(ps);

    const chord = from.distanceTo(to);
    const taut  = lineState.taut;

    // Sag: zero when taut, grows proportionally when slack
    // Capped to look physically plausible (not a huge loop)
    const sag = taut ? 0.0 : Math.min(chord * 0.12, 6.0);

    // Fill primary line geometry
    fillCatenary(from, to, sag, SAG_SEGMENTS, this.posArr);
    this.geo.attributes.position.needsUpdate = true;
    this.geo.setDrawRange(0, SAG_SEGMENTS + 1);

    // Copy into glow geo
    const glowBuf = this.glowGeo.attributes.position.array as Float32Array;
    glowBuf.set(this.posArr);
    this.glowGeo.attributes.position.needsUpdate = true;
    this.glowGeo.setDrawRange(0, SAG_SEGMENTS + 1);

    // Team colour
    const teamCol = ps.team === 'home' ? PAL.cyan : PAL.orange;

    // Line colour: full team when taut, dimmer blend toward dim when slack
    const lineHex = taut ? teamCol : blendHex(teamCol, PAL.dim, 0.55);
    const baseOpacity = (taut ? TAUT_OPACITY : SLACK_OPACITY) + (isP1 ? P1_OPACITY_BOOST : 0);

    this.lineMat.color.setHex(lineHex);
    this.lineMat.opacity = baseOpacity;

    // Glow overlay: only when taut (the "energized" look)
    this.glowMat.color.setHex(teamCol);
    this.glowMat.opacity = taut ? (GLOW_OPACITY_TAUT + (isP1 ? 0.20 : 0)) : GLOW_OPACITY_SLACK;

    // Anchor glint: bright when taut, softer when slack
    const glintOpacity = taut
      ? (isP1 ? 1.0 : 0.85)
      : (isP1 ? 0.60 : 0.40);

    const glintMat  = this.glint.material as THREE.MeshBasicMaterial;
    const innerMat  = this.glintInner.material as THREE.MeshBasicMaterial;

    glintMat.color.setHex(taut ? blendHex(teamCol, PAL.paper, 0.45) : teamCol);
    glintMat.opacity = glintOpacity;

    innerMat.color.setHex(PAL.paper);
    innerMat.opacity = taut ? glintOpacity * 0.90 : 0.0;

    // Rotate glint slowly for sparkle (cosmetic)
    const t = performance.now() * 0.001;
    this.glint.position.set(ap.x, ap.y, ap.z);
    this.glint.rotation.y = t * 0.8;
    this.glint.rotation.z = t * 0.5;
    this.glintInner.position.copy(this.glint.position);
  }

  hide(): void {
    this.group.visible = false;
  }
}

// ── Colour util (local, avoids import) ───────────────────────────────────────

/** Linearly blend two hex colours; t=0 → a, t=1 → b. */
function blendHex(a: number, b: number, t: number): number {
  const ar = a >> 16 & 0xff, ag = a >> 8 & 0xff, ab = a & 0xff;
  const br = b >> 16 & 0xff, bg = b >> 8 & 0xff, bb = b & 0xff;
  return (((ar + (br - ar) * t) | 0) << 16)
       | (((ag + (bg - ag) * t) | 0) << 8)
       |  ((ab + (bb - ab) * t) | 0);
}

// ── Public API ────────────────────────────────────────────────────────────────

/** Manages one LineInstance per player. Call sync() every rendered frame. */
export class RigLines {
  private instances = new Map<string, LineInstance>();
  private group = new THREE.Group();
  // P1 id is inferred on first sync (the home player whose id stays constant).
  // We identify P1 as the first home player seen (main.ts always places P1 in
  // the home team with id 'P1'). Stored so we can apply the brightening in
  // subsequent frames without touching the fixed sync API.
  private p1Id: string | null = null;

  constructor(scene: THREE.Scene) {
    scene.add(this.group);
  }

  /**
   * Sync rig-line visuals to the current player list.
   * @param players  Interpolated PlayerSim array.
   */
  sync(players: PlayerSim[]): void {
    // Infer P1 id: first player with id 'P1', falling back to first home player
    if (this.p1Id === null && players.length > 0) {
      const explicit = players.find((p) => p.id === 'P1');
      this.p1Id = explicit ? explicit.id : (players.find((p) => p.team === 'home')?.id ?? players[0].id);
    }

    // Hide all, re-show active ones
    for (const inst of this.instances.values()) inst.hide();

    for (const ps of players) {
      let inst = this.instances.get(ps.id);
      if (!inst) {
        inst = new LineInstance();
        this.group.add(inst.group);
        this.instances.set(ps.id, inst);
      }
      inst.apply(ps, ps.id === this.p1Id);
    }
  }
}

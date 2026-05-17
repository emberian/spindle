// Rig-line renderer — graphic-novel grapple line.
//
// The line originates from the rigger's ANIMATED grapple hand (published by
// Rigger.ts via RigGrapple), so it stays welded to the posed reach-arm instead
// of a fixed body offset. Taut = a clean, bold, straight energized line in team
// colour with an ink edge; slack = a soft catenary sag along the radial-outward
// ("centrifugal down") direction, dimmer. A vivid faceted glint marks the
// anchor. P1's line is the brightest.
//
// Ink-edged look: a thin black underlay line drawn behind a slightly thinner
// team-colour line → reads as an inked stroke at any distance (matches the
// figures' inverted-hull ink outline). No new deps, no per-frame alloc in the
// hot path.

import * as THREE from 'three';
import type { PlayerSim } from '../sim/types';
import { PAL } from '../ui/palette';
import { INK } from './RiggerToon';
import { grappleHand } from './RigGrapple';

const SAG_SEGMENTS = 24;
const GLINT_RADIUS = 0.55;

// Fallback hand offset (figure-local) if Rigger hasn't published a hand yet
// (e.g. first frame ordering): roughly chest-rig / reach-arm height, +Z front.
const HAND_FALLBACK_LOCAL = new THREE.Vector3(0.42, 1.55, 0.55);

const TAUT_OPACITY = 0.95;
const SLACK_OPACITY = 0.45;
const P1_BOOST = 0.05;

// ── Scratch (no hot-path alloc) ──────────────────────────────────────────────
const _from = new THREE.Vector3();
const _to = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _radial = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _off = new THREE.Vector3();

function blendHex(a: number, b: number, t: number): number {
  const ar = (a >> 16) & 0xff, ag = (a >> 8) & 0xff, ab = a & 0xff;
  const br = (b >> 16) & 0xff, bg = (b >> 8) & 0xff, bb = b & 0xff;
  return (((ar + (br - ar) * t) | 0) << 16)
       | (((ag + (bg - ag) * t) | 0) << 8)
       |  ((ab + (bb - ab) * t) | 0);
}

/** World-space line origin: the animated grapple hand, else a posed fallback. */
function lineOrigin(ps: PlayerSim, out: THREE.Vector3): void {
  const h = grappleHand(ps.id);
  if (h) {
    out.copy(h);
    return;
  }
  _off.copy(HAND_FALLBACK_LOCAL);
  _quat.set(ps.q.x, ps.q.y, ps.q.z, ps.q.w).normalize();
  _off.applyQuaternion(_quat);
  out.set(ps.p.x + _off.x, ps.p.y + _off.y, ps.p.z + _off.z);
}

/** Cosmetic catenary from a→b; sag = peak radial-outward droop at midpoint. */
function fillCatenary(
  a: THREE.Vector3, b: THREE.Vector3, sag: number,
  segments: number, out: Float32Array,
): void {
  _mid.addVectors(a, b).multiplyScalar(0.5);
  _radial.set(0, _mid.y, _mid.z);
  const rl = _radial.length();
  if (rl > 1e-6) _radial.multiplyScalar(1 / rl);
  else _radial.set(0, 1, 0);

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const px = a.x + (b.x - a.x) * t;
    const py = a.y + (b.y - a.y) * t;
    const pz = a.z + (b.z - a.z) * t;
    const drop = sag * 4 * t * (1 - t);
    out[i * 3 + 0] = px + _radial.x * drop;
    out[i * 3 + 1] = py + _radial.y * drop;
    out[i * 3 + 2] = pz + _radial.z * drop;
  }
}

// ── Per-player line ──────────────────────────────────────────────────────────

class LineInstance {
  readonly group = new THREE.Group();

  private posArr: Float32Array;
  private inkArr: Float32Array;

  private geo: THREE.BufferGeometry;
  private lineMat: THREE.LineBasicMaterial;

  // Ink underlay: same path, slightly radial-offset, black, drawn first.
  private inkGeo: THREE.BufferGeometry;
  private inkMat: THREE.LineBasicMaterial;
  private inkLine: THREE.Line;

  private glint: THREE.Mesh;
  private glintCore: THREE.Mesh;

  constructor() {
    const count = (SAG_SEGMENTS + 1) * 3;
    this.posArr = new Float32Array(count);
    this.inkArr = new Float32Array(count);

    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.posArr, 3));
    this.lineMat = new THREE.LineBasicMaterial({
      color: PAL.cyan, transparent: true, opacity: 0, depthWrite: false,
    });
    const lineMesh = new THREE.Line(this.geo, this.lineMat);
    lineMesh.frustumCulled = false;
    lineMesh.renderOrder = 1;

    this.inkGeo = new THREE.BufferGeometry();
    this.inkGeo.setAttribute('position', new THREE.BufferAttribute(this.inkArr, 3));
    this.inkMat = new THREE.LineBasicMaterial({
      color: INK, transparent: true, opacity: 0, depthWrite: false,
    });
    this.inkLine = new THREE.Line(this.inkGeo, this.inkMat);
    this.inkLine.frustumCulled = false;
    this.inkLine.renderOrder = 0;

    this.glint = new THREE.Mesh(
      new THREE.OctahedronGeometry(GLINT_RADIUS, 0),
      new THREE.MeshBasicMaterial({
        color: PAL.paper, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }),
    );
    this.glintCore = new THREE.Mesh(
      new THREE.OctahedronGeometry(GLINT_RADIUS * 0.4, 0),
      new THREE.MeshBasicMaterial({
        color: PAL.paper, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }),
    );

    this.group.add(this.inkLine, lineMesh, this.glint, this.glintCore);
    this.group.visible = false;
  }

  apply(ps: PlayerSim, isP1: boolean, now: number): void {
    const ls = ps.line;
    if (!ls) {
      this.group.visible = false;
      return;
    }
    this.group.visible = true;

    lineOrigin(ps, _from);
    _to.set(ls.anchorPos.x, ls.anchorPos.y, ls.anchorPos.z);

    const chord = _from.distanceTo(_to);
    const taut = ls.taut;
    const sag = taut ? 0 : Math.min(chord * 0.12, 6);

    fillCatenary(_from, _to, sag, SAG_SEGMENTS, this.posArr);
    this.geo.attributes.position.needsUpdate = true;
    this.geo.setDrawRange(0, SAG_SEGMENTS + 1);

    // Ink underlay: same curve, nudged slightly radial-outward so the black
    // edge sits just behind/under the colour stroke = inked-line read.
    _radial.set(0, (_from.y + _to.y) * 0.5, (_from.z + _to.z) * 0.5);
    const rl = _radial.length();
    if (rl > 1e-6) _radial.multiplyScalar(0.06 / rl);
    else _radial.set(0, 0.06, 0);
    for (let i = 0; i < this.posArr.length; i += 3) {
      this.inkArr[i + 0] = this.posArr[i + 0] + _radial.x;
      this.inkArr[i + 1] = this.posArr[i + 1] + _radial.y;
      this.inkArr[i + 2] = this.posArr[i + 2] + _radial.z;
    }
    this.inkGeo.attributes.position.needsUpdate = true;
    this.inkGeo.setDrawRange(0, SAG_SEGMENTS + 1);

    const teamCol = ps.team === 'home' ? PAL.cyan : PAL.orange;
    const lineHex = taut ? teamCol : blendHex(teamCol, PAL.dim, 0.5);
    const op = (taut ? TAUT_OPACITY : SLACK_OPACITY) + (isP1 ? P1_BOOST : 0);

    this.lineMat.color.setHex(lineHex);
    this.lineMat.opacity = op;
    this.inkMat.opacity = (taut ? 0.85 : 0.55) + (isP1 ? P1_BOOST : 0);

    // Anchor glint — bright/faceted when taut, soft when slack.
    const glintOp = taut ? (isP1 ? 1.0 : 0.85) : (isP1 ? 0.55 : 0.4);
    const gm = this.glint.material as THREE.MeshBasicMaterial;
    const cm = this.glintCore.material as THREE.MeshBasicMaterial;
    gm.color.setHex(taut ? blendHex(teamCol, PAL.paper, 0.5) : teamCol);
    gm.opacity = glintOp;
    cm.color.setHex(PAL.paper);
    cm.opacity = taut ? glintOp * 0.9 : 0;

    const t = now * 0.001;
    this.glint.position.copy(_to);
    this.glint.rotation.set(t * 0.6, t * 0.8, 0);
    this.glintCore.position.copy(_to);
  }

  hide(): void {
    this.group.visible = false;
  }
}

// ── Public API (FROZEN SIGNATURE) ────────────────────────────────────────────

/** Manages one LineInstance per player. Call sync() every rendered frame. */
export class RigLines {
  private instances = new Map<string, LineInstance>();
  private group = new THREE.Group();
  private p1Id: string | null = null;

  constructor(scene: THREE.Scene) {
    scene.add(this.group);
  }

  /**
   * Sync rig-line visuals to the current player list.
   * @param players Interpolated PlayerSim array.
   */
  sync(players: PlayerSim[]): void {
    if (this.p1Id === null && players.length > 0) {
      const explicit = players.find((p) => p.id === 'P1');
      this.p1Id = explicit
        ? explicit.id
        : (players.find((p) => p.team === 'home')?.id ?? players[0].id);
    }

    const now = performance.now();
    for (const inst of this.instances.values()) inst.hide();

    for (const ps of players) {
      let inst = this.instances.get(ps.id);
      if (!inst) {
        inst = new LineInstance();
        this.group.add(inst.group);
        this.instances.set(ps.id, inst);
      }
      inst.apply(ps, ps.id === this.p1Id, now);
    }
  }
}

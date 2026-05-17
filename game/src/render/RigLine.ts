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
const _dir = new THREE.Vector3();
const _perp = new THREE.Vector3();

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

  // Dynamics state.
  private active = false;     // had a line last frame? (for fire-snap detect)
  private fire = 0;           // 0→1 snap-out progress on a fresh fire
  private prevChord = 0;      // last frame chord length (reel-in detection)
  private ta15 = false;       // prev taut (for snap-to-taut shimmer kick)
  private snapKick = 0;       // brief shimmer burst when a line goes taut

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

  apply(ps: PlayerSim, isP1: boolean, now: number, dt: number): void {
    const ls = ps.line;
    if (!ls) {
      this.group.visible = false;
      this.active = false;
      this.fire = 0;
      return;
    }
    this.group.visible = true;

    lineOrigin(ps, _from);
    _to.set(ls.anchorPos.x, ls.anchorPos.y, ls.anchorPos.z);

    const chord = _from.distanceTo(_to);
    const taut = ls.taut;

    // ── Fire-snap: a brand-new line shoots out from the hand to the anchor
    // in a fast ease-out (~90 ms) instead of just appearing welded.
    if (!this.active) {
      this.fire = 0;
      this.prevChord = chord;
    }
    this.active = true;
    this.fire = Math.min(1, this.fire + dt / 0.09);
    const fireEase = 1 - Math.pow(1 - this.fire, 3); // fast out, eases in

    // ── Snap-to-taut shimmer kick: the instant slack→taut, the line cracks
    // tight; give it a brief high-freq tension shimmer that decays.
    if (taut && !this.ta15) this.snapKick = 1;
    this.ta15 = taut;
    this.snapKick = Math.max(0, this.snapKick - dt * 4.5);

    // Reeling-in reads as the chord shrinking frame-to-frame: tighten the
    // line visibly (sag collapses faster, slight extra brightness).
    const reelingIn = chord < this.prevChord - 0.02;
    this.prevChord = chord;

    // Sag: taut = laser straight; slack = lazy catenary; reeling pulls it in.
    let sag = taut ? 0 : Math.min(chord * 0.12, 6);
    if (reelingIn) sag *= 0.55;

    // The animated endpoint races out along the chord during the fire-snap.
    _to.lerpVectors(_from, _to, fireEase);

    fillCatenary(_from, _to, sag, SAG_SEGMENTS, this.posArr);

    // ── Tension shimmer: a taut line is *energized* — overlay a tiny, fast
    // perpendicular ripple (sub-cm, scaled by chord) so it sings rather than
    // sitting dead-straight. Strongest right after it cracks taut, then a
    // faint idle hum. None when slack (a slack line is lazy, not humming).
    if (taut && this.fire >= 1) {
      _dir.subVectors(_to, _from);
      const dl = _dir.length();
      if (dl > 1e-4) {
        _dir.multiplyScalar(1 / dl);
        _perp.set(0, _dir.z, -_dir.y); // a stable perpendicular (in y,z)
        if (_perp.lengthSq() < 1e-6) _perp.set(0, 1, 0);
        else _perp.normalize();
        const hum = 0.012 + this.snapKick * 0.06;
        const amp = hum * Math.min(dl, 30) * 0.06;
        const tt = now * 0.001;
        for (let i = 1; i < SAG_SEGMENTS; i++) {
          const u = i / SAG_SEGMENTS;
          // standing-wave-ish: nodes at the ends, fast travelling ripple
          const env = Math.sin(u * Math.PI);
          const w = Math.sin(u * 26 - tt * 34) * env * amp;
          this.posArr[i * 3 + 0] += _perp.x * w;
          this.posArr[i * 3 + 1] += _perp.y * w;
          this.posArr[i * 3 + 2] += _perp.z * w;
        }
      }
    }

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
    // Taut = energized: push the colour toward bright paper (hot wire) and
    // pulse it subtly with the snap-kick so a freshly-tensioned line cracks
    // visibly. Slack = lazy & dim. (No additive on the line — opacity only,
    // so it never contributes to a screen-white mass.)
    const tautHot = blendHex(teamCol, PAL.paper, 0.30 + this.snapKick * 0.25);
    const lineHex = taut ? tautHot : blendHex(teamCol, PAL.dim, 0.5);
    const op = (taut ? TAUT_OPACITY : SLACK_OPACITY)
      + (isP1 ? P1_BOOST : 0)
      + (taut ? this.snapKick * 0.04 : 0);

    this.lineMat.color.setHex(lineHex);
    this.lineMat.opacity = Math.min(1, op);
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
    this.active = false;
    this.fire = 0;
  }
}

// ── Public API (FROZEN SIGNATURE) ────────────────────────────────────────────

/** Manages one LineInstance per player. Call sync() every rendered frame. */
export class RigLines {
  private instances = new Map<string, LineInstance>();
  private group = new THREE.Group();
  private p1Id: string | null = null;
  private lastNow = 0;

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
    let dt = this.lastNow ? (now - this.lastNow) / 1000 : 1 / 60;
    dt = THREE.MathUtils.clamp(dt, 1 / 240, 1 / 15);
    this.lastNow = now;

    for (const inst of this.instances.values()) inst.hide();

    for (const ps of players) {
      let inst = this.instances.get(ps.id);
      if (!inst) {
        inst = new LineInstance();
        this.group.add(inst.group);
        this.instances.set(ps.id, inst);
      }
      inst.apply(ps, ps.id === this.p1Id, now, dt);
    }
  }
}

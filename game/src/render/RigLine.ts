// Rig-line renderer — graphic-novel grapple line.
//
// THE core mechanic is the grapple: a rigger fires a line to a spar/teammate
// and HAULS along it. That has to be unmistakable on screen. A 1 px
// THREE.Line (hardware-capped width) was invisible at the lore-scale
// spectate camera — "I don't see RIGGING". So the line is now a real
// world-space TUBE with thickness, a bold ink edge, a bright bitten anchor
// node and a node at the rigger's hand, so you can read: who fired, to
// where, taut (hauling — thick/bright/straight) vs slack (lazy/dim/sagging),
// and the reel snapping it in.
//
// The line originates from the rigger's ANIMATED grapple hand (published by
// Rigger.ts) so it stays welded to the posed reach-arm. Render-only; ~8
// short tubes rebuilt per frame is trivial off the 240 Hz sim path.

import * as THREE from 'three';
import type { PlayerSim } from '../sim/types';
import { PAL } from '../ui/palette';
import { INK } from './RiggerToon';
import { grappleHand } from './RigGrapple';

const SAG_SEGMENTS = 24;

// World-space line radii (m) — tuned to read clearly at the wide lore camera.
const TAUT_RADIUS = 0.42;   // hauling: a taut, energized cable
const SLACK_RADIUS = 0.26;  // lazy rope
const P1_RADIUS_BOOST = 0.12;

// Fallback hand offset (figure-local) if Rigger hasn't published a hand yet.
const HAND_FALLBACK_LOCAL = new THREE.Vector3(0.42, 1.55, 0.55);

// ── Scratch (no hot-path alloc) ──────────────────────────────────────────────
const _from = new THREE.Vector3();
const _to = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _radial = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _off = new THREE.Vector3();
const _pts: THREE.Vector3[] = [];
for (let i = 0; i <= SAG_SEGMENTS; i++) _pts.push(new THREE.Vector3());

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
  if (h) { out.copy(h); return; }
  _off.copy(HAND_FALLBACK_LOCAL);
  _quat.set(ps.q.x, ps.q.y, ps.q.z, ps.q.w).normalize();
  _off.applyQuaternion(_quat);
  out.set(ps.p.x + _off.x, ps.p.y + _off.y, ps.p.z + _off.z);
}

/** Fill _pts[0..SAG_SEGMENTS] with a cosmetic catenary a→b (radial-out sag). */
function fillCatenary(a: THREE.Vector3, b: THREE.Vector3, sag: number): void {
  _mid.addVectors(a, b).multiplyScalar(0.5);
  _radial.set(0, _mid.y, _mid.z);
  const rl = _radial.length();
  if (rl > 1e-6) _radial.multiplyScalar(1 / rl);
  else _radial.set(0, 1, 0);
  for (let i = 0; i <= SAG_SEGMENTS; i++) {
    const t = i / SAG_SEGMENTS;
    const drop = sag * 4 * t * (1 - t);
    _pts[i].set(
      a.x + (b.x - a.x) * t + _radial.x * drop,
      a.y + (b.y - a.y) * t + _radial.y * drop,
      a.z + (b.z - a.z) * t + _radial.z * drop,
    );
  }
}

// ── Per-player line ──────────────────────────────────────────────────────────

class LineInstance {
  readonly group = new THREE.Group();

  private rope: THREE.Mesh;       // bright team-colour tube
  private ink: THREE.Mesh;        // slightly fatter dark tube = bold ink edge
  private anchor: THREE.Mesh;     // the bite point (bright, faceted)
  private anchorCore: THREE.Mesh; // hot white core
  private hand: THREE.Mesh;       // node where the line leaves the rigger
  private curGeo: THREE.TubeGeometry | null = null;
  private inkGeo: THREE.TubeGeometry | null = null;

  private active = false;
  private fire = 0;
  private prevChord = 0;
  private wasTaut = false;
  private snapKick = 0;

  constructor() {
    this.ink = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshBasicMaterial({
        color: INK, transparent: true, opacity: 0, depthWrite: false,
      }),
    );
    this.ink.frustumCulled = false;
    this.ink.renderOrder = 0;

    this.rope = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshBasicMaterial({
        color: PAL.cyan, transparent: true, opacity: 0, depthWrite: false,
        // Additive so the tether GLOWS against the dark calm (the line was
        // "too dark"). Safe: a thin ~0.4 m tube is a bounded emitter — the
        // old blowout was huge additive markers / loop bloom, not this.
        blending: THREE.AdditiveBlending,
      }),
    );
    this.rope.frustumCulled = false;
    this.rope.renderOrder = 1;

    this.anchor = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.9, 0),
      new THREE.MeshBasicMaterial({
        color: PAL.paper, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }),
    );
    this.anchor.renderOrder = 2;
    this.anchorCore = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.42, 0),
      new THREE.MeshBasicMaterial({
        color: PAL.paper, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }),
    );
    this.anchorCore.renderOrder = 3;

    this.hand = new THREE.Mesh(
      new THREE.SphereGeometry(0.45, 8, 6),
      new THREE.MeshBasicMaterial({
        color: PAL.paper, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }),
    );
    this.hand.renderOrder = 2;

    this.group.add(this.ink, this.rope, this.anchor, this.anchorCore, this.hand);
    this.group.visible = false;
  }

  private rebuild(radius: number): void {
    const curve = new THREE.CatmullRomCurve3(_pts, false, 'catmullrom', 0);
    const g = new THREE.TubeGeometry(curve, SAG_SEGMENTS, radius, 6, false);
    const gi = new THREE.TubeGeometry(curve, SAG_SEGMENTS, radius * 1.7, 6, false);
    this.rope.geometry = g;
    this.ink.geometry = gi;
    this.curGeo?.dispose();
    this.inkGeo?.dispose();
    this.curGeo = g;
    this.inkGeo = gi;
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

    if (!this.active) { this.fire = 0; this.prevChord = chord; }
    this.active = true;
    this.fire = Math.min(1, this.fire + dt / 0.09);
    const fireEase = 1 - Math.pow(1 - this.fire, 3);

    if (taut && !this.wasTaut) this.snapKick = 1;
    this.wasTaut = taut;
    this.snapKick = Math.max(0, this.snapKick - dt * 4.5);

    const reelingIn = chord < this.prevChord - 0.02;
    this.prevChord = chord;

    let sag = taut ? 0 : Math.min(chord * 0.12, 6);
    if (reelingIn) sag *= 0.55;

    // The endpoint races out along the chord during the fire-snap.
    _to.lerpVectors(_from, _to, fireEase);
    fillCatenary(_from, _to, sag);

    const baseR = taut ? TAUT_RADIUS : SLACK_RADIUS;
    const radius = baseR
      + (isP1 ? P1_RADIUS_BOOST : 0)
      + this.snapKick * 0.14
      + (reelingIn ? 0.06 : 0);
    this.rebuild(radius);

    const teamCol = ps.team === 'home' ? PAL.cyan : PAL.orange;
    // Taut = hauling: hot toward paper-white and pulses on the snap. Slack =
    // lazy & dim. Opacity is high either way so the rope is always legible.
    // Bright even slack (it was "too dark"): taut runs hot toward white,
    // slack stays a vivid team colour rather than dimmed-out.
    const ropeHex = taut
      ? blendHex(teamCol, PAL.paper, 0.5 + this.snapKick * 0.3)
      : blendHex(teamCol, PAL.paper, 0.18);
    (this.rope.material as THREE.MeshBasicMaterial).color.setHex(ropeHex);
    (this.rope.material as THREE.MeshBasicMaterial).opacity =
      (taut ? 1.0 : 0.85) + (isP1 ? 0.0 : 0);
    (this.ink.material as THREE.MeshBasicMaterial).opacity = taut ? 0.7 : 0.45;

    // Anchor bite-point — big & hot when taut (you're hauling on it), softer
    // when slack. This is the "where the claw bit" read.
    const an = this.anchor.material as THREE.MeshBasicMaterial;
    const ac = this.anchorCore.material as THREE.MeshBasicMaterial;
    an.color.setHex(taut ? blendHex(teamCol, PAL.paper, 0.55) : teamCol);
    an.opacity = taut ? (isP1 ? 1.0 : 0.9) : 0.5;
    ac.opacity = taut ? 0.95 : 0.25;
    const t = now * 0.001;
    this.anchor.position.copy(_to);
    this.anchor.rotation.set(t * 0.6, t * 0.8, 0);
    const aScale = 1 + this.snapKick * 0.6;
    this.anchor.scale.setScalar(aScale);
    this.anchorCore.position.copy(_to);

    // Hand node: the line visibly LEAVES the rigger.
    const hm = this.hand.material as THREE.MeshBasicMaterial;
    hm.color.setHex(blendHex(teamCol, PAL.paper, 0.4));
    hm.opacity = taut ? 0.85 : 0.55;
    this.hand.position.copy(_from);
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

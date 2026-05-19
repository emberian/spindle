// Rig-line renderer — graphic-novel grapple line.
//
// THE core mechanic is the grapple: a rigger fires a line to a spar/skin/
// teammate and HAULS along it. With the powered hook ALL 8 riggers grapple
// at once, so the OLD thick-bloomed-double-tube + fat 0.9 octahedron read
// smeared 8 tethers into one glowing mass that fought the bell trail.
//
// This renderer is built for LEGIBILITY UNDER 8 SIMULTANEOUS TETHERS:
//   - each tether is ONE thin, taut, crisp line (no 1.7× ink halo, low
//     additive contribution) so eight of them stay separable and never
//     out-glow the bell;
//   - the anchor node is SMALL and shaped BY ANCHOR TYPE so a watcher can
//     read WHAT was grappled at a glance:
//       spar   → octahedron  (hard structural bite)
//       skin   → flat ring   (free-space skin point)
//       ring   → cube        (the goal ring)
//       player → the tether runs all the way to the LIVE target rigger and
//                gets a node at BOTH ends, so rigger→rigger coordination is
//                visually unmistakable;
//   - the bell + BellTrail stay the brightest, most saturated thing — the
//     tethers are deliberately recessive against it.
//
// GRAPPLE LATENCY: a fired line is NOT instantly a cable. While the sim
// reports `line.attached === false` the claw is still in flight, so this
// renderer draws a small travelling BITE (octahedron) with a thin pay-out
// thread behind it — NOT the taut/slack hauling cable. It switches to the
// cable rendering only once `attached` flips true. Flight progress is
// render-only (CLAW_SPEED · elapsed along hand→anchor) and never re-enters
// the sim.
//
// The line originates from the rigger's ANIMATED grapple hand (published by
// Rigger.ts). Render-only; ~8 short tubes rebuilt per frame is trivial off
// the 240 Hz sim path.

import * as THREE from 'three';
import type { PlayerSim, GrappleState } from '../sim/types';
import { PAL } from '../ui/palette';
import { FEEL } from '../sim/RegConstants';
import { grappleHand } from './RigGrapple';

const SAG_SEGMENTS = 24;

// World-space line radii (m). DELIBERATELY THIN: a taut hauling cable reads
// as a crisp stroke, not a rope. Eight of these must not merge into a blob,
// so the radius is a fraction of the old 0.42 and there is no halo tube.
// Tethers are CONTEXT, not focus: the riggers + bell must dominate. These are
// a fraction of even the previous "thin" values so 8 simultaneous hauls read
// as faint guide-threads, never a glowing web.
const TAUT_RADIUS = 0.075;  // hauling: a taut, energised cable (still recessive)
const SLACK_RADIUS = 0.035; // slack/inactive: near-invisible thread
const P1_RADIUS_BOOST = 0.035;

// Chord length (m) over which a tether fades toward minimum opacity: a long
// haul spans the playspace and would dominate, so longer = more recessive.
const LEN_FADE_NEAR = 30;   // ≤ this: full (already-low) strength
const LEN_FADE_FAR  = 260;  // ≥ this: faded to the long-haul floor

// Small typed anchor sizes (m) — unobtrusive context markers, shrunk further
// so the node never competes with a rigger body or the bell.
const ANCHOR_SIZE = 0.22;
const HAND_SIZE = 0.13;

// GRAPPLE LATENCY (Gap 1): a fired-but-not-yet-attached claw is a TRAVELLING
// PROJECTILE, not a hauling cable. While `!ls.attached` the sim is paying out
// flight — there is no constraint force yet — so we draw a small claw head
// flying hand→anchor with a thin pay-out line trailing it. Progress along the
// flight is RENDER-ONLY: derived from CLAW_SPEED + the hand→anchor distance +
// render time since the unattached line first appeared. It never re-enters
// sim.step (the established hard rule); it only has to LOOK like the same
// latency the sim is enforcing (ceil(dist / CLAW_SPEED / h) ticks).
const CLAW_HEAD_SIZE = 0.16; // m — the flying bite (slightly bigger than HAND)
const FLIGHT_RADIUS = 0.03;  // m — pay-out thread: thinner even than SLACK

// Fallback hand offset (figure-local) if Rigger hasn't published a hand yet.
const HAND_FALLBACK_LOCAL = new THREE.Vector3(0.42, 1.55, 0.55);

// ── Scratch (no hot-path alloc) ──────────────────────────────────────────────
const _from = new THREE.Vector3();
const _to = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _radial = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _off = new THREE.Vector3();
const _claw = new THREE.Vector3();
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

/** Fill _pts[0..SAG_SEGMENTS] with a straight a→b run (the taut pay-out
 *  thread behind a flying claw — no sag; it's a line being dragged out). */
function fillStraight(a: THREE.Vector3, b: THREE.Vector3): void {
  for (let i = 0; i <= SAG_SEGMENTS; i++) {
    const t = i / SAG_SEGMENTS;
    _pts[i].set(
      a.x + (b.x - a.x) * t,
      a.y + (b.y - a.y) * t,
      a.z + (b.z - a.z) * t,
    );
  }
}

// ── Per-player line ──────────────────────────────────────────────────────────

class LineInstance {
  readonly group = new THREE.Group();

  private rope: THREE.Mesh;        // thin team-colour tube (the tether)
  private spar: THREE.Mesh;        // octahedron anchor (structural bite)
  private skin: THREE.Mesh;        // flat ring anchor (free-space skin point)
  private ring: THREE.Mesh;        // cube anchor (the goal ring)
  private hand: THREE.Mesh;        // node where the line leaves the rigger
  private endNode: THREE.Mesh;     // node at the FAR end for player↔player
  private clawHead: THREE.Mesh;    // the in-flight bite (GRAPPLE LATENCY)
  private curGeo: THREE.TubeGeometry | null = null;

  private active = false;
  private fire = 0;
  private prevChord = 0;
  private wasTaut = false;
  private snapKick = 0;

  // Render-only flight clock: seconds the line has existed UNATTACHED. Reset
  // whenever the line is gone or has attached, so a fresh shot always flies
  // from the hand. Pure cosmetic; never feeds the sim.
  private flightT = 0;
  private wasAttached = true;

  constructor() {
    // Tether: one thin tube. NOT additive — a normal translucent stroke so
    // eight of them stay crisp and recessive instead of blooming into a
    // glowing mass that competes with the bell.
    // vertexColors: lets us fade the tube toward the ANCHOR end (the rigger
    // end stays readable as "this line belongs to that figure"; the far end
    // dissolves so a long haul reads as a hint, not a hard cable).
    this.rope = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshBasicMaterial({
        color: PAL.cyan, transparent: true, opacity: 0, depthWrite: false,
        vertexColors: true,
      }),
    );
    this.rope.frustumCulled = false;
    this.rope.renderOrder = 1;

    // Typed anchor nodes — small, additive (a single small node is a bounded
    // emitter), only the one matching anchorType is shown per frame.
    const anchorMat = () =>
      new THREE.MeshBasicMaterial({
        color: PAL.paper, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false,
      });

    this.spar = new THREE.Mesh(new THREE.OctahedronGeometry(ANCHOR_SIZE, 0), anchorMat());
    this.skin = new THREE.Mesh(
      new THREE.TorusGeometry(ANCHOR_SIZE * 0.95, ANCHOR_SIZE * 0.22, 6, 14),
      anchorMat(),
    );
    this.ring = new THREE.Mesh(
      new THREE.BoxGeometry(ANCHOR_SIZE * 1.3, ANCHOR_SIZE * 1.3, ANCHOR_SIZE * 1.3),
      anchorMat(),
    );
    this.spar.renderOrder = 2;
    this.skin.renderOrder = 2;
    this.ring.renderOrder = 2;

    this.hand = new THREE.Mesh(new THREE.SphereGeometry(HAND_SIZE, 8, 6), anchorMat());
    this.hand.renderOrder = 2;
    // Far-end node — only lit for player↔player tethers, so rigger→rigger
    // coordination reads as two clearly linked figures.
    this.endNode = new THREE.Mesh(new THREE.SphereGeometry(HAND_SIZE * 1.25, 8, 6), anchorMat());
    this.endNode.renderOrder = 2;

    // The flying claw head: an octahedron (a structural BITE, same shape
    // language as a spar anchor) — small, crisp, additive, only lit while
    // the line is unattached. It reads as "thrown, not yet bitten".
    this.clawHead = new THREE.Mesh(
      new THREE.OctahedronGeometry(CLAW_HEAD_SIZE, 0), anchorMat(),
    );
    this.clawHead.renderOrder = 2;

    this.group.add(
      this.rope, this.spar, this.skin, this.ring, this.hand, this.endNode,
      this.clawHead,
    );
    this.group.visible = false;
  }

  private rebuild(radius: number): void {
    const curve = new THREE.CatmullRomCurve3(_pts, false, 'catmullrom', 0);
    // 4 radial segments: a thin tube doesn't need 6, fewer verts for 8 tethers.
    const RADIAL = 4;
    const g = new THREE.TubeGeometry(curve, SAG_SEGMENTS, radius, RADIAL, false);
    // Per-vertex brightness fade along the length: ~1 at the rigger (hand)
    // end, fading to a faint tail at the anchor end so the line points back
    // to its figure but never reads as a hard structural cable across the
    // whole playspace. TubeGeometry orders verts ring-by-ring along the path.
    const ringCount = SAG_SEGMENTS + 1;
    const vpr = RADIAL + 1; // verts per ring
    const total = ringCount * vpr;
    const col = new Float32Array(total * 3);
    for (let r = 0; r < ringCount; r++) {
      const t = r / SAG_SEGMENTS;            // 0 = hand end, 1 = anchor end
      // Stay bright over the first third, then ease down to a 0.15 tail.
      const f = 0.15 + 0.85 * Math.pow(1 - THREE.MathUtils.smoothstep(t, 0.25, 1), 1.1);
      for (let k = 0; k < vpr; k++) {
        const i = (r * vpr + k) * 3;
        col[i] = f; col[i + 1] = f; col[i + 2] = f;
      }
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.rope.geometry = g;
    this.curGeo?.dispose();
    this.curGeo = g;
  }

  /** @param targetPos live world pos of the grappled player, or null. */
  apply(
    ps: PlayerSim,
    isP1: boolean,
    now: number,
    dt: number,
    targetPos: THREE.Vector3 | null,
  ): void {
    const ls = ps.line;
    if (!ls) {
      this.group.visible = false;
      this.active = false;
      this.fire = 0;
      this.flightT = 0;
      this.wasAttached = true;
      this.clawHead.visible = false;
      return;
    }
    this.group.visible = true;

    const kind = ls.anchorType;
    const isPlayerAnchor = kind === 'player' && targetPos !== null;

    lineOrigin(ps, _from);
    // For a player anchor, draw to the LIVE target rigger so the link is
    // unmistakable; otherwise to the bound anchor point.
    if (isPlayerAnchor) _to.copy(targetPos as THREE.Vector3);
    else _to.set(ls.anchorPos.x, ls.anchorPos.y, ls.anchorPos.z);
    const chord = _from.distanceTo(_to);
    const taut = ls.taut;

    // ── GRAPPLE LATENCY: claw in flight (Gap 1) ─────────────────────────────
    // The sim has fired but not yet attached: there is NO hauling cable yet,
    // only a claw travelling toward anchorPos. Draw it as a projectile and
    // bail before any of the taut/slack cable machinery runs. Progress is
    // render-only — CLAW_SPEED·elapsed along hand→anchor, clamped — so it
    // visually tracks the sim's ceil(dist / CLAW_SPEED / h)-tick latency
    // WITHOUT ever feeding sim.step.
    if (!ls.attached) {
      // Fresh shot (or just-fired) → restart the cosmetic flight clock so the
      // claw always launches from the hand, never mid-air.
      if (this.wasAttached || !this.active) this.flightT = 0;
      this.active = true;
      this.wasAttached = false;
      this.wasTaut = false;
      this.flightT += dt;

      // Param along hand→anchor from the render clock. A real shot lands in
      // chord / CLAW_SPEED s; clamp <1 so the head never visually "arrives"
      // before the sim flips attached (the cable then takes over).
      const flightDur = Math.max(chord / FEEL.CLAW_SPEED, 1e-3);
      const u = Math.min(0.985, this.flightT / flightDur);
      _claw.lerpVectors(_from, _to, u);

      // Thin pay-out thread: hand → claw head only (the line is being
      // dragged out behind the bite; nothing past the head exists yet).
      fillStraight(_from, _claw);
      this.rebuild(FLIGHT_RADIUS + (isP1 ? P1_RADIUS_BOOST * 0.4 : 0));

      const teamColF = ps.team === 'home' ? PAL.cyan : PAL.orange;
      const rmF = this.rope.material as THREE.MeshBasicMaterial;
      rmF.color.setHex(teamColF);
      rmF.opacity = 0.22; // a faint flying thread — recessive, but trackable

      // The head: crisp, additive, paper-tinted so the BITE reads against the
      // dim thread; a gentle spin + slight scale pulse sells "in flight".
      const tF = now * 0.001;
      const cm = this.clawHead.material as THREE.MeshBasicMaterial;
      cm.color.setHex(blendHex(teamColF, PAL.paper, 0.55));
      cm.opacity = 0.7;
      this.clawHead.position.copy(_claw);
      this.clawHead.rotation.set(tF * 5.0, tF * 6.5, 0);
      this.clawHead.scale.setScalar(1 + 0.12 * Math.sin(tF * 22));
      this.clawHead.visible = true;

      // No cable, no anchor node, no hand node while flying — JUST the
      // thread + the travelling bite, so the throw reads cleanly.
      this.spar.visible = false;
      this.skin.visible = false;
      this.ring.visible = false;
      this.endNode.visible = false;
      this.hand.visible = false;
      this.prevChord = chord;
      return;
    }
    // Attached: the claw has bitten — hand off to the cable renderer below.
    this.clawHead.visible = false;
    this.hand.visible = true;
    this.wasAttached = true;

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
      + this.snapKick * 0.04
      + (reelingIn ? 0.02 : 0);
    this.rebuild(radius);

    const teamCol = ps.team === 'home' ? PAL.cyan : PAL.orange;
    // Tether colour: a clean team-tinted stroke. It NO LONGER runs hot toward
    // paper-white (that's the bell's job) — taut just nudges a little brighter
    // and pulses faintly on the snap.
    const rm = this.rope.material as THREE.MeshBasicMaterial;
    rm.color.setHex(
      taut ? blendHex(teamCol, PAL.paper, 0.12 + this.snapKick * 0.18)
           : teamCol,
    );
    // RECESSIVE: only a TAUT, actively-hauled line shows clearly; a slack /
    // inactive line is near-invisible. Then fade further with chord length so
    // a long haul spanning the field doesn't dominate the riggers + bell.
    const lenT = THREE.MathUtils.clamp(
      (chord - LEN_FADE_NEAR) / (LEN_FADE_FAR - LEN_FADE_NEAR), 0, 1,
    );
    const lenFade = 1 - lenT * 0.70;            // long haul → 30% strength
    // Player↔player stays a touch more visible (that rigger↔rigger link is
    // the coordination we DO want to read) but is still secondary.
    const tautBase  = isPlayerAnchor ? 0.42 : 0.34;
    const slackBase = 0.07;                     // slack ≈ invisible context
    rm.opacity = (taut ? tautBase : slackBase) * lenFade;

    // ── Typed anchor: show ONLY the node matching anchorType ────────────────
    this.spar.visible = false;
    this.skin.visible = false;
    this.ring.visible = false;
    this.endNode.visible = false;

    const t = now * 0.001;
    const aScale = 1 + this.snapKick * 0.35;

    if (isPlayerAnchor) {
      // rigger→rigger: a node welded to the live target figure + the hand
      // node. Two linked dots on a thin line = readable coordination.
      const em = this.endNode.material as THREE.MeshBasicMaterial;
      em.color.setHex(blendHex(teamCol, PAL.paper, 0.45));
      em.opacity = (taut ? 0.6 : 0.3) * lenFade;
      this.endNode.position.copy(_to);
      this.endNode.scale.setScalar(aScale);
      this.endNode.visible = true;
    } else {
      const node =
        kind === 'spar' ? this.spar :
        kind === 'ring' ? this.ring :
        this.skin; // 'skin' (and player-with-no-target fallback)
      const nm = node.material as THREE.MeshBasicMaterial;
      nm.color.setHex(taut ? blendHex(teamCol, PAL.paper, 0.3) : teamCol);
      nm.opacity = (taut ? 0.5 : 0.18) * lenFade;
      node.position.copy(_to);
      node.rotation.set(t * 0.6, t * 0.8, 0);
      node.scale.setScalar(aScale);
      node.visible = true;
    }

    // Hand node: the line visibly LEAVES the rigger (small, dim — context,
    // not a focal point).
    const hm = this.hand.material as THREE.MeshBasicMaterial;
    hm.color.setHex(blendHex(teamCol, PAL.paper, 0.25));
    hm.opacity = (taut ? 0.4 : 0.15) * lenFade;
    this.hand.position.copy(_from);
  }

  hide(): void {
    this.group.visible = false;
    this.active = false;
    this.fire = 0;
    this.flightT = 0;
    this.wasAttached = true;
    this.clawHead.visible = false;
  }
}

// ── Public API (FROZEN SIGNATURE) ────────────────────────────────────────────

/** Manages one LineInstance per player. Call sync() every rendered frame. */
export class RigLines {
  private instances = new Map<string, LineInstance>();
  private group = new THREE.Group();
  private p1Id: string | null = null;
  private lastNow = 0;
  private _tgt = new THREE.Vector3();

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
      // Resolve a player-anchor to the LIVE target rigger position so a
      // rigger→rigger tether connects two figures, not a stale point.
      let targetPos: THREE.Vector3 | null = null;
      const ls: GrappleState | null = ps.line;
      if (ls && ls.anchorType === 'player' && ls.anchorRef) {
        const tgt = players.find((p) => p.id === ls.anchorRef);
        if (tgt) {
          this._tgt.set(tgt.p.x, tgt.p.y, tgt.p.z);
          targetPos = this._tgt;
        }
      }
      inst.apply(ps, ps.id === this.p1Id, now, dt, targetPos);
    }
  }
}

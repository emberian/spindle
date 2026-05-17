// An articulated, cel-shaded humanoid rigger athlete for RIG.
//
// A zero-g grapple-sport athlete: slim suit + a rig harness/spool across the
// chest that the lines fire from. Built from low-poly primitives parented into
// a small skeleton of pivot Groups so the procedural animator (RiggerFigure.pose)
// can drive lean / limb stroke / reach / idle drift purely from motion.
//
// Local frame convention (matches the rest of render/): the figure stands along
// local +Y ("up" = toward the cylinder axis once the root is oriented by the
// no-global-up radial frame) and faces local +Z (travel/aim direction). The
// root's orientation is set by the caller (Rigger.ts) from sim quat + velocity.
//
// Scale: human ~ 1.9 m tall, against the R = 45 m calm. Overall footprint is
// kept close to the previous placeholder so figures sit correctly in-world and
// the spectate camera framing is unchanged.

import * as THREE from 'three';
import { InkedPart } from './RiggerToon';

// ── Proportions (metres) ─────────────────────────────────────────────────────
// ~1.9 m athlete: legs + torso + head.
const HIP_Y      = 0.95;  // hip pivot height above figure origin
const TORSO_LEN  = 0.62;  // hip → shoulder
const SHOULDER_Y = HIP_Y + TORSO_LEN;
const HEAD_R     = 0.16;
const SHOULDER_W = 0.42;  // half-width between shoulders ×2 → ~0.42 across
const HIP_W      = 0.20;  // half-distance between hip pivots
const UPPER_ARM  = 0.34;
const FOREARM    = 0.34;
const THIGH      = 0.46;
const SHIN       = 0.46;
const LIMB_R     = 0.075; // limb tube radius

// Reusable low-poly geometry (shared across all figures via constructor args).
function capsule(r: number, len: number): THREE.CapsuleGeometry {
  return new THREE.CapsuleGeometry(r, len, 3, 8);
}

// One shared soft radial-gradient sprite texture — the per-rigger glow aura
// that keeps a tiny figure trackable in the vast lore-scale calm without
// zooming the camera in. Built once, tinted per figure via the sprite colour.
let _auraTex: THREE.Texture | null = null;
function auraTexture(): THREE.Texture {
  if (_auraTex) return _auraTex;
  const S = 64;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const g = cv.getContext('2d')!;
  const grad = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
  grad.addColorStop(0.0, 'rgba(255,255,255,0.95)');
  grad.addColorStop(0.35, 'rgba(255,255,255,0.45)');
  grad.addColorStop(1.0, 'rgba(255,255,255,0.0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, S, S);
  const tex = new THREE.CanvasTexture(cv);
  tex.needsUpdate = true;
  _auraTex = tex;
  return tex;
}

/** A two-bone limb (upper + lower) with a mid joint pivot. */
class Limb {
  /** Pivot at the shoulder/hip; rotate this for the whole-limb swing. */
  readonly root = new THREE.Group();
  /** Pivot at elbow/knee; rotate this for the bend. */
  private readonly joint = new THREE.Group();
  /** Empty marker at the hand/foot tip — for line/IK coordination. */
  private readonly tip = new THREE.Group();

  constructor(upperLen: number, lowerLen: number, color: number, endGeo?: THREE.BufferGeometry) {
    const upper = new InkedPart(capsule(LIMB_R, upperLen), color);
    upper.group.position.y = -upperLen * 0.5 - LIMB_R;
    this.root.add(upper.group);

    this.joint.position.y = -(upperLen + LIMB_R * 2);
    this.root.add(this.joint);

    const lower = new InkedPart(capsule(LIMB_R * 0.92, lowerLen), color);
    lower.group.position.y = -lowerLen * 0.5 - LIMB_R;
    this.joint.add(lower.group);

    this.tip.position.y = -(lowerLen + LIMB_R * 2);
    this.joint.add(this.tip);

    if (endGeo) {
      const end = new InkedPart(endGeo, color);
      end.group.position.copy(this.tip.position);
      this.joint.add(end.group);
    }
  }

  /** swing = whole-limb rotation about local X (forward+/back-). bend = joint flex. */
  set(swingX: number, bend: number, swingZ = 0): void {
    this.root.rotation.set(swingX, 0, swingZ);
    this.joint.rotation.x = bend;
  }

  /** World-space position of the limb tip (hand/foot). */
  tipWorld(out: THREE.Vector3): void {
    this.tip.getWorldPosition(out);
  }
}

export class RiggerFigure {
  /** Root — caller sets position + orientation each frame. */
  readonly root = new THREE.Group();

  private readonly torso: InkedPart;
  private readonly chestRig: InkedPart;  // grapple harness/spool plate
  private readonly head: InkedPart;
  private readonly visor: THREE.Mesh;    // accent (team/role read)

  private readonly armL: Limb;
  private readonly armR: Limb;           // R = grapple/reach arm
  private readonly legL: Limb;
  private readonly legR: Limb;

  // Sub-pivots for secondary motion.
  private readonly spine = new THREE.Group();   // hip→up; lean lives here
  private readonly shoulders = new THREE.Group(); // counter-rotate / follow-through

  // Always-on team-glow aura (every rigger, readable when small).
  private readonly aura: THREE.Sprite;

  // Stylized rigger kit — bold silhouette read at lore distance.
  private readonly backPack: InkedPart;
  private readonly paulL: InkedPart;
  private readonly paulR: InkedPart;

  // P1 highlight extras.
  private readonly halo: THREE.Mesh;
  private readonly beacon: THREE.Mesh;

  constructor() {
    // Spine pivot at the hips.
    this.spine.position.y = HIP_Y;
    this.root.add(this.spine);

    // Torso (tapered: capsule).
    this.torso = new InkedPart(capsule(0.17, TORSO_LEN), 0x808080);
    this.torso.group.position.y = TORSO_LEN * 0.5 + 0.05;
    this.spine.add(this.torso.group);

    // Chest rig: a flat-ish box plate across the chest = the spool the lines
    // fire from. Slightly proud of the torso, faces +Z.
    this.chestRig = new InkedPart(new THREE.BoxGeometry(0.40, 0.30, 0.16), 0x808080, 0.04);
    this.chestRig.group.position.set(0, TORSO_LEN * 0.62, 0.16);
    this.spine.add(this.chestRig.group);

    // Back rig-pack: the line spool/reel slung between the shoulders. A bold,
    // asymmetric block on the back so the silhouette reads as an equipped
    // rigger (not a stick figure) even a few pixels tall. Heavier ink line.
    this.backPack = new InkedPart(new THREE.BoxGeometry(0.34, 0.40, 0.20), 0x808080, 0.09);
    this.backPack.group.position.set(0, TORSO_LEN * 0.55, -0.18);
    this.spine.add(this.backPack.group);

    // Shoulder yoke pivot (top of spine) — arms + head hang off this so
    // follow-through twist propagates naturally.
    this.shoulders.position.y = TORSO_LEN + 0.06;
    this.spine.add(this.shoulders);

    // Head + visor accent.
    this.head = new InkedPart(new THREE.IcosahedronGeometry(HEAD_R, 1), 0x808080);
    this.head.group.position.y = HEAD_R + 0.10;
    this.shoulders.add(this.head.group);

    // Visor: a thin curved-ish band on the face (+Z). MeshBasicMaterial so the
    // accent reads as a bright glyph even at distance / in shadow band.
    this.visor = new THREE.Mesh(
      new THREE.BoxGeometry(HEAD_R * 1.7, HEAD_R * 0.6, 0.03),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    this.visor.position.set(0, HEAD_R + 0.10, HEAD_R * 0.92);
    this.shoulders.add(this.visor);

    // Arms — small "hand" cap on the end.
    const hand = (): THREE.BufferGeometry => new THREE.IcosahedronGeometry(LIMB_R * 1.4, 0);
    this.armL = new Limb(UPPER_ARM, FOREARM, 0x808080, hand());
    this.armR = new Limb(UPPER_ARM, FOREARM, 0x808080, hand());
    this.armL.root.position.set(-SHOULDER_W, -0.02, 0);
    this.armR.root.position.set( SHOULDER_W, -0.02, 0);
    this.shoulders.add(this.armL.root, this.armR.root);

    // Pauldrons: blocky shoulder caps. They square off the silhouette into a
    // stylized "broad-shouldered athlete in a rig" read at any distance.
    this.paulL = new InkedPart(new THREE.BoxGeometry(0.20, 0.16, 0.22), 0x808080, 0.08);
    this.paulR = new InkedPart(new THREE.BoxGeometry(0.20, 0.16, 0.22), 0x808080, 0.08);
    this.paulL.group.position.set(-SHOULDER_W, 0.04, 0);
    this.paulR.group.position.set( SHOULDER_W, 0.04, 0);
    this.shoulders.add(this.paulL.group, this.paulR.group);

    // Legs — small "boot" cap.
    const boot = (): THREE.BufferGeometry => new THREE.BoxGeometry(0.13, 0.10, 0.22);
    this.legL = new Limb(THIGH, SHIN, 0x808080, boot());
    this.legR = new Limb(THIGH, SHIN, 0x808080, boot());
    this.legL.root.position.set(-HIP_W, 0, 0);
    this.legR.root.position.set( HIP_W, 0, 0);
    // Legs hang off the root at hip height (not the leaning spine) so a lean
    // pivots the torso while legs trail from the body's true centre.
    const hips = new THREE.Group();
    hips.position.y = HIP_Y;
    hips.add(this.legL.root, this.legR.root);
    this.root.add(hips);

    // ── P1 highlight: small feet ring + ▼ beacon overhead ───────────────────
    // Always-on-top so it's never lost (depthTest:false + high renderOrder),
    // but DELIBERATELY small and NON-additive: the previous big additive
    // version bloomed the whole screen white. A crisp coloured marker reads
    // fine without being a glow bomb.
    this.halo = new THREE.Mesh(
      new THREE.TorusGeometry(0.55, 0.06, 6, 24),
      new THREE.MeshBasicMaterial({
        color: 0x6fe9ff, transparent: true, opacity: 0,
        depthWrite: false, depthTest: false,
      }),
    );
    this.halo.rotation.x = Math.PI / 2;
    this.halo.position.y = 0.02;
    this.halo.renderOrder = 9998;
    this.root.add(this.halo);

    const beaconGeo = new THREE.ConeGeometry(0.3, 0.75, 4, 1);
    beaconGeo.rotateZ(Math.PI);
    this.beacon = new THREE.Mesh(
      beaconGeo,
      new THREE.MeshBasicMaterial({
        color: 0x6fe9ff, transparent: true, opacity: 0,
        depthWrite: false, depthTest: false,
      }),
    );
    this.beacon.position.y = SHOULDER_Y + HEAD_R * 2 + 1.0;
    this.beacon.renderOrder = 9999; // always on top, but small & non-additive
    this.root.add(this.beacon);

    // Team-glow aura: a soft additive bloom centred on the torso, drawn under
    // (never occluded by) the figure so a small rigger is always a clear
    // coloured presence in the vast calm. Tinted/faded per state in setColors.
    this.aura = new THREE.Sprite(new THREE.SpriteMaterial({
      map: auraTexture(),
      color: 0xffffff,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: false,
    }));
    this.aura.scale.set(3.0, 3.0, 1);
    this.aura.position.y = HIP_Y + TORSO_LEN * 0.55;
    this.aura.renderOrder = 9990; // above scene, below the P1 halo/beacon
    this.root.add(this.aura);

    this.root.visible = false;
  }

  // ── Appearance ─────────────────────────────────────────────────────────────

  /** Team/grounded fill colour + emissive + accent colour. */
  setColors(bodyHex: number, accentHex: number, emissiveHex: number, emissiveInt: number): void {
    this.torso.setColor(bodyHex);
    this.head.setColor(bodyHex);
    this.chestRig.setColor(accentHex);
    this.torso.setEmissive(emissiveHex, emissiveInt);
    this.head.setEmissive(emissiveHex, emissiveInt);
    (this.visor.material as THREE.MeshBasicMaterial).color.setHex(accentHex);
    this.setLimbColor(bodyHex);

    // Stylized kit: pack/pauldrons in team body colour; the chest spool gets
    // a bright accent emissive so every rigger has a hot focal point.
    this.backPack.setColor(bodyHex);
    this.paulL.setColor(bodyHex);
    this.paulR.setColor(bodyHex);
    this.chestRig.setEmissive(accentHex, emissiveHex === 0 ? 0.0 : 0.85);

    // Team-glow aura: tint to the team emissive (which Rigger.ts already zeroes
    // for grounded/out-of-play), and keep a readable floor for anyone in play
    // so a small figure is always a clear coloured presence.
    const auraMat = this.aura.material as THREE.SpriteMaterial;
    if (emissiveHex === 0) {
      auraMat.opacity = 0;
    } else {
      auraMat.color.setHex(emissiveHex);
      auraMat.opacity = Math.max(0.42, Math.min(0.6, emissiveInt * 0.6));
    }
  }

  private setLimbColor(hex: number): void {
    // Limbs are InkedParts inside the Limb groups; recolour via traversal of
    // the fill meshes (cheap: a handful of meshes, no per-frame alloc).
    for (const limbRoot of [this.armL.root, this.armR.root, this.legL.root, this.legR.root]) {
      limbRoot.traverse((o) => {
        const m = (o as THREE.Mesh).material;
        if (m instanceof THREE.MeshToonMaterial) m.color.setHex(hex);
      });
    }
  }

  // ── Pose ───────────────────────────────────────────────────────────────────

  /**
   * Drive the procedural pose. All angles in radians; caller passes already-
   * damped/eased values (see Rigger.ts) so nothing pops here.
   *
   * @param lean      forward(+)/back(-) torso lean about local X
   * @param bank      side roll about local Z (banking into a turn)
   * @param strokeAng phase of the swim/stroke cycle (radians, advancing)
   * @param strokeAmp 0..1 amplitude of the stroke (scaled by speed)
   * @param reach     0..1 how far the R (grapple) arm extends forward toward
   *                  the line/travel direction
   * @param idle      idle-drift phase (radians) for the near-still calm sway
   * @param effort    0..1 athletic exertion: streamline + powered haul. Drives
   *                  the difference between "sliding" and "swinging hard".
   * @param swingPhase  rad, the haul cycle: the grapple arm rhythmically
   *                  HAULS the line in (pull → recover) when on a taut line.
   * @param brace     0..1 deceleration brace/settle (legs forward, body up,
   *                  arms wide to kill speed) — an athlete planting a stop.
   * @param wind      -1..+1 throw shaping: <0 = wind-up (arm cocked back),
   *                  >0 = release follow-through (arm whipped across). 0 = none.
   */
  pose(
    lean: number, bank: number,
    strokeAng: number, strokeAmp: number,
    reach: number, idle: number,
    effort = 0, swingPhase = 0, brace = 0, wind = 0,
  ): void {
    // ── Spine: lean + bank, plus an effort streamline (curl toward velocity)
    // and a brace-up (chest rises, hips drop forward) when killing speed.
    const idleSway = (1 - strokeAmp) * 0.05;
    const streamline = effort * 0.45;          // tuck forward when driving hard
    const braceArch  = -brace * 0.55;          // arch back, plant against motion
    // Throw torque: wind-up coils the spine away, release whips it across.
    const windCoil = wind < 0 ? wind * 0.30 : wind * 0.22;
    this.spine.rotation.set(
      lean + streamline + braceArch + Math.sin(idle) * idleSway,
      windCoil,
      bank + Math.cos(idle * 0.7) * idleSway * 0.6,
    );
    // Shoulder yoke: stroke counter-rotation + a strong haul/throw twist so
    // the whole upper body whips around the working arm (athletic, not sliding).
    const haul = Math.sin(swingPhase);                 // -1..1 pull/recover
    const haulTwist = haul * reach * (0.35 + effort * 0.45);
    this.shoulders.rotation.y = -Math.sin(strokeAng) * strokeAmp * 0.22
      + haulTwist - windCoil * 1.4;
    this.shoulders.rotation.x = -lean * 0.25 - effort * 0.30 + brace * 0.35;

    // ── Limb cadence: amplitude AND tempo scale with speed/effort so a fast
    // rigger thrashes hard; a settling one stiffens (brace damps the swim).
    const swimGate = (1 - brace * 0.8);
    const amp = (0.05 + strokeAmp * 1.05 + effort * 0.55) * swimGate;
    // How hard the limbs are actually swimming. Near 0 when parked so the
    // joint-flex oscillators below go quiet (no bobbing in place); when
    // moving it reaches 1 and they pump fully as before. The slow spine
    // idleSway is then the only idle motion — a faint, calm breath.
    const swimAmt = THREE.MathUtils.clamp(amp, 0, 1);
    const s = Math.sin(strokeAng);
    const c = Math.cos(strokeAng);

    // Left arm: swim stroke; a fast hard pump when effort is high. Under brace
    // it flares WIDE forward to kill momentum (an athlete's air-brake).
    const armBend = 0.35 + strokeAmp * 0.45 + effort * 0.25;
    const lSwim  =  s * amp * 0.7;
    const lBrace = -1.05;                       // forward, wide
    this.armL.set(
      THREE.MathUtils.lerp(lSwim, lBrace, brace),
      armBend + (s * 0.5 + 0.5) * 0.3 * swimAmt + brace * 0.2,
      -0.12 - c * 0.10 * swimAmt - brace * 0.55,
    );

    // ── Right (grapple) arm — the worker. Three blended intents:
    //   reach  → arm extended toward line/travel
    //   haul   → rhythmic powerful pull (bicep curl back toward the chest rig)
    //   wind   → cocked back (windup, wind<0) then whipped across (release>0)
    const reachSwing = -1.30;                   // straighter, higher, athletic
    const reachBend  = 0.14;
    // Haul: from a long extended catch (-1.5) to a hard pulled-in flex.
    const haulSwing  = THREE.MathUtils.lerp(-1.5, -0.35, haul * 0.5 + 0.5);
    const haulBend   = THREE.MathUtils.lerp(0.10, 1.35, haul * 0.5 + 0.5);
    // Reach pose is the base; the haul cycle modulates it by reach amount.
    let rSwing = THREE.MathUtils.lerp(-s * amp * 0.7, reachSwing, reach);
    let rBend  = THREE.MathUtils.lerp(armBend + (-s * 0.5 + 0.5) * 0.3 * swimAmt, reachBend, reach);
    rSwing = THREE.MathUtils.lerp(rSwing, haulSwing, reach * 0.7);
    rBend  = THREE.MathUtils.lerp(rBend,  haulBend,  reach * 0.7);
    let rFlare = THREE.MathUtils.lerp(0.12 + c * 0.10 * swimAmt, -0.05, reach);
    // Throw: wind-up cocks the arm way back+bent; release whips it forward.
    const wUp = Math.max(0, -wind), wRel = Math.max(0, wind);
    rSwing = THREE.MathUtils.lerp(rSwing, 1.7, wUp);     // cocked behind
    rBend  = THREE.MathUtils.lerp(rBend, 1.9, wUp);      // deep cock
    rSwing = THREE.MathUtils.lerp(rSwing, -2.0, wRel);   // whipped across+up
    rBend  = THREE.MathUtils.lerp(rBend, 0.05, wRel);    // snapped straight
    rFlare = THREE.MathUtils.lerp(rFlare, 0.35 * wind, Math.abs(wind));
    this.armR.set(rSwing, rBend, rFlare);

    // ── Legs: scissor opposite the arms; on a hard swing they drive off
    // (deep coil → extension); braced legs swing forward to plant the stop.
    const legBend = 0.30 + strokeAmp * 0.55 + effort * 0.30;
    const drive = haul * reach * 0.5;           // legs push as the arm hauls
    const lLeg = -s * amp * 0.55 - drive;
    const rLeg =  s * amp * 0.55 - drive;
    const braceLeg = 0.95;                       // both legs forward to brake
    this.legL.set(
      THREE.MathUtils.lerp(lLeg, braceLeg, brace),
      legBend + (-s * 0.5 + 0.5) * 0.35 * swimAmt + brace * 0.7,
    );
    this.legR.set(
      THREE.MathUtils.lerp(rLeg, braceLeg, brace),
      legBend + ( s * 0.5 + 0.5) * 0.35 * swimAmt + brace * 0.7,
    );
  }

  /**
   * World-space position of the right (grapple) hand — for line coordination.
   * Refreshes world matrices first since sync() runs before the renderer's
   * own scene-graph update.
   */
  grappleHandWorld(out: THREE.Vector3): THREE.Vector3 {
    this.root.updateMatrixWorld(true);
    this.armR.tipWorld(out);
    return out;
  }

  // ── P1 highlight ───────────────────────────────────────────────────────────

  setHighlight(on: boolean, color: number, t: number): void {
    const haloMat = this.halo.material as THREE.MeshBasicMaterial;
    const beaconMat = this.beacon.material as THREE.MeshBasicMaterial;
    if (on) {
      haloMat.color.setHex(0x6fe9ff);
      haloMat.opacity = 0.7;
      beaconMat.color.setHex(0x6fe9ff);
      beaconMat.opacity = 0.8;
      void color;
      this.beacon.position.y = SHOULDER_Y + HEAD_R * 2 + 1.0 + Math.sin(t * 2.2) * 0.25;
      this.beacon.rotation.y = t * 1.4;
    } else {
      haloMat.opacity = 0;
      beaconMat.opacity = 0;
    }
  }
}

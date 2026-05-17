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

    // ── P1 highlight: halo ring at the feet + ▼ beacon overhead ─────────────
    // The "YOU" marker MUST be unmistakable and never occluded by the other
    // (large) figures — so it renders on top (depthTest:false, high
    // renderOrder) and is big.
    this.halo = new THREE.Mesh(
      new THREE.TorusGeometry(1.5, 0.12, 8, 32),
      new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
      }),
    );
    this.halo.rotation.x = Math.PI / 2;
    this.halo.position.y = 0.02;
    this.halo.renderOrder = 9998;
    this.root.add(this.halo);

    const beaconGeo = new THREE.ConeGeometry(0.85, 1.9, 4, 1);
    beaconGeo.rotateZ(Math.PI);
    this.beacon = new THREE.Mesh(
      beaconGeo,
      new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
      }),
    );
    this.beacon.position.y = SHOULDER_Y + HEAD_R * 2 + 1.4;
    this.beacon.renderOrder = 9999; // always on top — never hidden behind figures
    this.root.add(this.beacon);

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
   */
  pose(
    lean: number, bank: number,
    strokeAng: number, strokeAmp: number,
    reach: number, idle: number,
  ): void {
    // Spine: lean + bank. Idle adds a slow breathing sway when amp is low.
    const idleSway = (1 - strokeAmp) * 0.05;
    this.spine.rotation.set(
      lean + Math.sin(idle) * idleSway,
      0,
      bank + Math.cos(idle * 0.7) * idleSway * 0.6,
    );
    // Shoulder yoke counter-rotates slightly = secondary follow-through.
    this.shoulders.rotation.y = -Math.sin(strokeAng) * strokeAmp * 0.22;
    this.shoulders.rotation.x = -lean * 0.25;

    // Zero-g swim stroke: arms & legs sweep in smooth opposed arcs. Amplitude
    // scales with speed; a gentle baseline keeps idle floaty (never frozen).
    const amp = 0.28 + strokeAmp * 0.95;
    const s = Math.sin(strokeAng);
    const c = Math.cos(strokeAng);

    // Arms sweep fore/aft (X) with a soft outward flare (Z); elbows trail.
    const armBend = 0.35 + strokeAmp * 0.45;
    this.armL.set( s * amp * 0.7,  armBend + (s * 0.5 + 0.5) * 0.3, -0.12 - c * 0.10);
    // Right arm: blend stroke with the reach pose toward +Z (travel/line).
    const reachSwing = -1.15;            // arm forward+up toward travel
    const reachBend  = 0.18;             // nearly straight when reaching
    const rSwing = THREE.MathUtils.lerp(-s * amp * 0.7, reachSwing, reach);
    const rBend  = THREE.MathUtils.lerp(armBend + (-s * 0.5 + 0.5) * 0.3, reachBend, reach);
    const rFlare = THREE.MathUtils.lerp(0.12 + c * 0.10, -0.05, reach);
    this.armR.set(rSwing, rBend, rFlare);

    // Legs scissor opposite the arms; knees trail through the stroke.
    const legBend = 0.30 + strokeAmp * 0.55;
    this.legL.set(-s * amp * 0.55, legBend + (-s * 0.5 + 0.5) * 0.35);
    this.legR.set( s * amp * 0.55, legBend + ( s * 0.5 + 0.5) * 0.35);
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
      haloMat.color.setHex(color);
      haloMat.opacity = 0.85;
      beaconMat.color.setHex(0xffffff);
      beaconMat.opacity = 0.95;
      this.beacon.position.y = SHOULDER_Y + HEAD_R * 2 + 1.4 + Math.sin(t * 2.2) * 0.35;
      this.beacon.rotation.y = t * 1.4;
    } else {
      haloMat.opacity = 0;
      beaconMat.opacity = 0;
    }
  }
}

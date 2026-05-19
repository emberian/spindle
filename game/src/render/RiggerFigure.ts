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
//
// ROLE SILHOUETTES: each role receives distinct body proportions + badge geometry
// so it reads at cinematic spectate distance (viewScale 5). Anchor = massive,
// Spinner = sleek default, Faithwing = armoured, Freewing = asymmetric artist,
// Reach = minimal/light keeper.

import * as THREE from 'three';
import { InkedPart } from './RiggerToon';
import type { RiggerRole } from '../sim/types';

// ── Base proportions (metres) ────────────────────────────────────────────────
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

// ── Role-specific proportion multipliers ─────────────────────────────────────
interface RoleProps {
  shoulderW: number;    // shoulder half-width
  paulScale: number;    // pauldron scale multiplier
  limbR: number;        // limb tube radius
  torsoR: number;       // torso capsule radius
  torsoLen: number;     // torso capsule length
  hipY: number;         // hip pivot height
  upperArm: number;     // upper arm length
  forearm: number;      // forearm length
  thigh: number;        // thigh length
  shin: number;         // shin length
  backPackScale: [number, number, number]; // xyz scale of back-pack
  chestRigScale: [number, number, number]; // xyz scale of chest rig plate
  hasBackPack: boolean;
  hasPaulL: boolean;
  hasPaulR: boolean;
}

function roleProps(role: RiggerRole): RoleProps {
  switch (role) {
    case 'anchor':
      // MASSIVE/planted: wide shoulders, thick limbs, heavy kit, slightly taller
      return {
        shoulderW: SHOULDER_W * 1.30,
        paulScale: 1.40,
        limbR: LIMB_R * 1.25,
        torsoR: 0.17 * 1.20,
        torsoLen: TORSO_LEN * 1.06,
        hipY: HIP_Y * 1.03,
        upperArm: UPPER_ARM * 1.05,
        forearm: FOREARM * 1.05,
        thigh: THIGH * 1.05,
        shin: SHIN * 1.05,
        backPackScale: [1.3, 1.2, 1.3],
        chestRigScale: [1.15, 1.1, 1.1],
        hasBackPack: true,
        hasPaulL: true,
        hasPaulR: true,
      };
    case 'spinner':
      // Sleek, agile: default proportions, slightly longer limbs for elegance
      return {
        shoulderW: SHOULDER_W,
        paulScale: 1.0,
        limbR: LIMB_R,
        torsoR: 0.17,
        torsoLen: TORSO_LEN,
        hipY: HIP_Y,
        upperArm: UPPER_ARM * 1.05,
        forearm: FOREARM * 1.05,
        thigh: THIGH * 1.04,
        shin: SHIN * 1.04,
        backPackScale: [1.0, 1.0, 1.0],
        chestRigScale: [1.0, 1.0, 1.0],
        hasBackPack: true,
        hasPaulL: true,
        hasPaulR: true,
      };
    case 'faithwing':
      // Broader than spinner, armoured: bigger chest rig plate, solid pauldrons
      return {
        shoulderW: SHOULDER_W * 1.12,
        paulScale: 1.25,
        limbR: LIMB_R * 1.10,
        torsoR: 0.17 * 1.08,
        torsoLen: TORSO_LEN * 1.02,
        hipY: HIP_Y,
        upperArm: UPPER_ARM,
        forearm: FOREARM,
        thigh: THIGH,
        shin: SHIN,
        backPackScale: [1.1, 1.1, 1.1],
        chestRigScale: [1.25, 1.20, 1.30],
        hasBackPack: true,
        hasPaulL: true,
        hasPaulR: true,
      };
    case 'freewing':
      // Narrow/lighter, longer forearms (artist reach), asymmetric (one pauldron)
      return {
        shoulderW: SHOULDER_W * 0.95,
        paulScale: 1.0,
        limbR: LIMB_R * 0.95,
        torsoR: 0.17 * 0.95,
        torsoLen: TORSO_LEN,
        hipY: HIP_Y,
        upperArm: UPPER_ARM,
        forearm: FOREARM * 1.18, // the artist has reach
        thigh: THIGH,
        shin: SHIN,
        backPackScale: [0.85, 0.9, 0.85],
        chestRigScale: [0.9, 0.95, 0.9],
        hasBackPack: true,
        hasPaulL: false,  // asymmetric: left pauldron removed
        hasPaulR: true,
      };
    case 'reach':
      // Smallest/lightest: short limbs, minimal kit, nimble on the ring
      return {
        shoulderW: SHOULDER_W * 0.90,
        paulScale: 0.75,
        limbR: LIMB_R * 0.88,
        torsoR: 0.17 * 0.90,
        torsoLen: TORSO_LEN * 0.92,
        hipY: HIP_Y * 0.95,
        upperArm: UPPER_ARM * 0.92,
        forearm: FOREARM * 0.92,
        thigh: THIGH * 0.92,
        shin: SHIN * 0.92,
        backPackScale: [0.55, 0.55, 0.55], // tiny or nearly absent
        chestRigScale: [0.80, 0.85, 0.80],
        hasBackPack: true,
        hasPaulL: true,
        hasPaulR: true,
      };
  }
}

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

  constructor(upperLen: number, lowerLen: number, limbR: number, color: number, endGeo?: THREE.BufferGeometry) {
    const upper = new InkedPart(capsule(limbR, upperLen), color);
    upper.group.position.y = -upperLen * 0.5 - limbR;
    this.root.add(upper.group);

    this.joint.position.y = -(upperLen + limbR * 2);
    this.root.add(this.joint);

    const lower = new InkedPart(capsule(limbR * 0.92, lowerLen), color);
    lower.group.position.y = -lowerLen * 0.5 - limbR;
    this.joint.add(lower.group);

    this.tip.position.y = -(lowerLen + limbR * 2);
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

  private torso!: InkedPart;
  private chestRig!: InkedPart;  // grapple harness/spool plate
  private head!: InkedPart;
  private visor!: THREE.Mesh;    // accent (team/role read)

  private armL!: Limb;
  private armR!: Limb;           // R = grapple/reach arm
  private legL!: Limb;
  private legR!: Limb;

  // Sub-pivots for secondary motion.
  private spine = new THREE.Group();   // hip→up; lean lives here
  private shoulders = new THREE.Group(); // counter-rotate / follow-through
  private hips = new THREE.Group();

  // Always-on team-glow aura (every rigger, readable when small).
  private aura!: THREE.Sprite;

  // Stylized rigger kit — bold silhouette read at lore distance.
  private backPack!: InkedPart;
  private paulL!: InkedPart;
  private paulR!: InkedPart;

  // Role-specific badge geometry (may be null for roles without one).
  private roleBadge: THREE.Group | null = null;

  // P1 highlight extras.
  private halo!: THREE.Mesh;
  private beacon!: THREE.Mesh;

  // Track current role so we only rebuild when it changes.
  private currentRole: RiggerRole | null = null;

  constructor() {
    // Build once with default spinner proportions; setRole() will rebuild
    // if a different role is assigned.
    this._build('spinner');

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
    this.aura.scale.set(7.5, 7.5, 1);
    this.aura.position.y = HIP_Y + TORSO_LEN * 0.55;
    this.aura.renderOrder = 9990; // above scene, below the P1 halo/beacon
    this.root.add(this.aura);

    this.root.visible = false;
  }

  // ── Role assignment ─────────────────────────────────────────────────────────

  /** Assign (or reassign) the role. Rebuilds body geometry when the role changes. */
  setRole(role: RiggerRole): void {
    if (role === this.currentRole) return;
    this._teardown();
    this._build(role);
  }

  /** Remove all role-specific body parts from the hierarchy. */
  private _teardown(): void {
    // Remove the spine (which holds torso, chest rig, back-pack, shoulders,
    // head, visor, arms, pauldrons) and the hips (legs).
    this.root.remove(this.spine);
    this.root.remove(this.hips);
    if (this.roleBadge) {
      this.root.remove(this.roleBadge);
      this.roleBadge = null;
    }
    // Create fresh pivots.
    this.spine = new THREE.Group();
    this.shoulders = new THREE.Group();
    this.hips = new THREE.Group();
  }

  /** Build the body from the given role's proportions. */
  private _build(role: RiggerRole): void {
    this.currentRole = role;
    const rp = roleProps(role);

    // Spine pivot at the hips.
    this.spine.position.y = rp.hipY;
    this.root.add(this.spine);

    // Torso (tapered: capsule).
    this.torso = new InkedPart(capsule(rp.torsoR, rp.torsoLen), 0x808080);
    this.torso.group.position.y = rp.torsoLen * 0.5 + 0.05;
    this.spine.add(this.torso.group);

    // Chest rig: a flat-ish box plate across the chest = the spool the lines
    // fire from. Slightly proud of the torso, faces +Z.
    const cw = 0.40 * rp.chestRigScale[0];
    const ch = 0.30 * rp.chestRigScale[1];
    const cd = 0.16 * rp.chestRigScale[2];
    this.chestRig = new InkedPart(new THREE.BoxGeometry(cw, ch, cd), 0x808080, 0.04);
    this.chestRig.group.position.set(0, rp.torsoLen * 0.62, 0.16);
    this.spine.add(this.chestRig.group);

    // Back rig-pack: the line spool/reel slung between the shoulders.
    if (rp.hasBackPack) {
      const bw = 0.34 * rp.backPackScale[0];
      const bh = 0.40 * rp.backPackScale[1];
      const bd = 0.20 * rp.backPackScale[2];
      this.backPack = new InkedPart(new THREE.BoxGeometry(bw, bh, bd), 0x808080, 0.09);
      this.backPack.group.position.set(0, rp.torsoLen * 0.55, -0.18);
      this.spine.add(this.backPack.group);
    } else {
      // Create a zero-size placeholder so setColors doesn't break
      this.backPack = new InkedPart(new THREE.BoxGeometry(0.001, 0.001, 0.001), 0x808080, 0);
      this.backPack.group.visible = false;
      this.spine.add(this.backPack.group);
    }

    // Shoulder yoke pivot (top of spine) — arms + head hang off this so
    // follow-through twist propagates naturally.
    this.shoulders.position.y = rp.torsoLen + 0.06;
    this.spine.add(this.shoulders);

    // Head + visor accent.
    this.head = new InkedPart(new THREE.IcosahedronGeometry(HEAD_R, 1), 0x808080);
    this.head.group.position.y = HEAD_R + 0.10;
    this.shoulders.add(this.head.group);

    // Visor: style varies by role. Reach gets a wide wraparound "lunatic" visor.
    const visorW = role === 'reach' ? HEAD_R * 2.2 : HEAD_R * 1.7;
    const visorH = role === 'reach' ? HEAD_R * 0.75 : HEAD_R * 0.6;
    const visorD = role === 'reach' ? 0.05 : 0.03;
    this.visor = new THREE.Mesh(
      new THREE.BoxGeometry(visorW, visorH, visorD),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    this.visor.position.set(0, HEAD_R + 0.10, HEAD_R * 0.92);
    // Reach visor wraps: slight curve via rotation of side panels simulated
    // with a wider + deeper geometry — already handled by wider dims above.
    this.shoulders.add(this.visor);

    // Arms — small "hand" cap on the end.
    const hand = (lr: number): THREE.BufferGeometry => new THREE.IcosahedronGeometry(lr * 1.4, 0);
    this.armL = new Limb(rp.upperArm, rp.forearm, rp.limbR, 0x808080, hand(rp.limbR));
    this.armR = new Limb(rp.upperArm, rp.forearm, rp.limbR, 0x808080, hand(rp.limbR));
    this.armL.root.position.set(-rp.shoulderW, -0.02, 0);
    this.armR.root.position.set( rp.shoulderW, -0.02, 0);
    this.shoulders.add(this.armL.root, this.armR.root);

    // Pauldrons: blocky shoulder caps.
    const pw = 0.20 * rp.paulScale;
    const ph = 0.16 * rp.paulScale;
    const pd = 0.22 * rp.paulScale;
    this.paulL = new InkedPart(new THREE.BoxGeometry(pw, ph, pd), 0x808080, 0.08);
    this.paulR = new InkedPart(new THREE.BoxGeometry(pw, ph, pd), 0x808080, 0.08);
    this.paulL.group.position.set(-rp.shoulderW, 0.04, 0);
    this.paulR.group.position.set( rp.shoulderW, 0.04, 0);
    this.paulL.group.visible = rp.hasPaulL;
    this.paulR.group.visible = rp.hasPaulR;
    this.shoulders.add(this.paulL.group, this.paulR.group);

    // Legs — small "boot" cap.
    const boot = (): THREE.BufferGeometry => new THREE.BoxGeometry(
      0.13 * (rp.limbR / LIMB_R),
      0.10 * (rp.limbR / LIMB_R),
      0.22 * (rp.limbR / LIMB_R),
    );
    this.legL = new Limb(rp.thigh, rp.shin, rp.limbR, 0x808080, boot());
    this.legR = new Limb(rp.thigh, rp.shin, rp.limbR, 0x808080, boot());
    this.legL.root.position.set(-HIP_W, 0, 0);
    this.legR.root.position.set( HIP_W, 0, 0);
    // Legs hang off the root at hip height (not the leaning spine) so a lean
    // pivots the torso while legs trail from the body's true centre.
    this.hips.position.y = rp.hipY;
    this.hips.add(this.legL.root, this.legR.root);
    this.root.add(this.hips);

    // ── Role badge geometry ─────────────────────────────────────────────────
    this._buildRoleBadge(role, rp);
  }

  /** Add a distinctive geometric badge/marking per role. */
  private _buildRoleBadge(role: RiggerRole, rp: RoleProps): void {
    switch (role) {
      case 'anchor': {
        // Thick collar/gorget: a torus ring at shoulder height — reads as
        // a heavy neck guard, unmistakably the biggest figure.
        const badge = new THREE.Group();
        const gorget = new InkedPart(
          new THREE.TorusGeometry(rp.shoulderW * 0.55, 0.07, 6, 16),
          0x808080, 0.10,
        );
        gorget.group.rotation.x = Math.PI / 2;
        gorget.group.position.y = rp.torsoLen + 0.02;
        badge.add(gorget.group);
        badge.position.y = rp.hipY;
        this.roleBadge = badge;
        this.root.add(badge);
        break;
      }
      case 'spinner': {
        // Dorsal fin/crest on the back-pack: a thin vertical wedge that makes
        // the spinner's back silhouette knife-edged and distinctive.
        const badge = new THREE.Group();
        const finGeo = new THREE.BoxGeometry(0.04, 0.28, 0.18);
        const fin = new InkedPart(finGeo, 0x808080, 0.08);
        fin.group.position.set(0, rp.torsoLen * 0.55 + 0.22, -0.28);
        badge.add(fin.group);
        badge.position.y = rp.hipY;
        this.roleBadge = badge;
        this.root.add(badge);
        break;
      }
      case 'faithwing': {
        // Armoured collar ridge: a flat angular plate across the upper chest
        // that visually doubles as extra armour, broader than the spinner fin.
        const badge = new THREE.Group();
        const plateGeo = new THREE.BoxGeometry(rp.shoulderW * 1.6, 0.08, 0.26);
        const plate = new InkedPart(plateGeo, 0x808080, 0.06);
        plate.group.position.set(0, rp.torsoLen * 0.82, 0.10);
        badge.add(plate.group);
        badge.position.y = rp.hipY;
        this.roleBadge = badge;
        this.root.add(badge);
        break;
      }
      case 'freewing': {
        // Asymmetric arm-blade on the right forearm — a small trailing vane
        // that reinforces the "artist's reach" and the missing left pauldron.
        const badge = new THREE.Group();
        const bladeGeo = new THREE.BoxGeometry(0.03, 0.22, 0.12);
        const blade = new InkedPart(bladeGeo, 0x808080, 0.06);
        // Positioned relative to right shoulder (the working arm).
        blade.group.position.set(rp.shoulderW, -rp.upperArm * 0.6, 0.08);
        badge.add(blade.group);
        badge.position.y = rp.hipY + rp.torsoLen + 0.06; // at shoulder height
        this.roleBadge = badge;
        this.root.add(badge);
        break;
      }
      case 'reach': {
        // Wide visor is already handled above. Add a small ring/hoop on each
        // forearm (clip anchors for pivoting on the goal ring). These tiny
        // hoops read as distinctive "cuffs" at distance.
        const badge = new THREE.Group();
        const cuffGeo = new THREE.TorusGeometry(rp.limbR * 2.2, 0.025, 6, 12);
        const cuffL = new InkedPart(cuffGeo, 0x808080, 0.06);
        const cuffR = new InkedPart(cuffGeo.clone(), 0x808080, 0.06);
        cuffL.group.rotation.x = Math.PI / 2;
        cuffR.group.rotation.x = Math.PI / 2;
        // Position at wrist area of each arm.
        cuffL.group.position.set(-rp.shoulderW, -(rp.upperArm + rp.forearm * 0.75), 0);
        cuffR.group.position.set( rp.shoulderW, -(rp.upperArm + rp.forearm * 0.75), 0);
        badge.add(cuffL.group, cuffR.group);
        badge.position.y = rp.hipY + rp.torsoLen + 0.06; // at shoulder height
        this.roleBadge = badge;
        this.root.add(badge);
        break;
      }
    }
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

    // Role badge colouring: traverse and apply body colour.
    if (this.roleBadge) {
      this.roleBadge.traverse((o) => {
        const m = (o as THREE.Mesh).material;
        if (m instanceof THREE.MeshToonMaterial) {
          m.color.setHex(bodyHex);
          m.emissive.setHex(emissiveHex);
          m.emissiveIntensity = emissiveInt;
        }
      });
    }

    // Team-glow aura: tint to the team emissive (which Rigger.ts already zeroes
    // for grounded/out-of-play), and keep a readable floor for anyone in play
    // so a small figure is always a clear coloured presence.
    const auraMat = this.aura.material as THREE.SpriteMaterial;
    if (emissiveHex === 0) {
      auraMat.opacity = 0.15;
      auraMat.color.setHex(bodyHex);
    } else {
      auraMat.color.setHex(emissiveHex);
      auraMat.opacity = Math.max(0.7, Math.min(1.0, emissiveInt * 0.8));
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
    // ── Spine: lean + bank, plus an effort STREAMLINE (the whole body lines
    // up along the velocity/grapple direction into an arrowed glide) and a
    // brace-up (chest rises, hips drop forward) when killing speed.
    const idleSway = (1 - strokeAmp) * 0.05;
    const streamline = effort * 0.30;          // gently align torso to travel
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

    // ── Limb cadence: a SLOW SUBTLE STROKE only. The old code scaled the
    // amplitude with effort (+0.55) so a fast rigger thrashed its legs and
    // read as sprinting in mid-air — absurd in zero-g. Now effort does NOT
    // feed amplitude at all; the stroke stays small, and effort tucks the
    // limbs in (streamlined freefall glide), the opposite of pumping.
    const swimGate = (1 - brace * 0.8) * (1 - effort * 0.7);
    const amp = (0.05 + strokeAmp * 0.55) * swimGate;
    // How hard the limbs are actually swimming. Near 0 when parked so the
    // joint-flex oscillators below go quiet (no bobbing in place); when
    // moving it reaches 1 and they pump fully as before. The slow spine
    // idleSway is then the only idle motion — a faint, calm breath.
    const swimAmt = THREE.MathUtils.clamp(amp, 0, 1);
    const s = Math.sin(strokeAng);
    const c = Math.cos(strokeAng);

    // Left arm: swim stroke; a fast hard pump when effort is high. Under brace
    // it flares WIDE forward to kill momentum (an athlete's air-brake).
    // Effort tucks the off (left) arm in tight against the body and trailing,
    // completing the arrowed glide silhouette instead of a swimming pump.
    const armBend = 0.35 + strokeAmp * 0.30 + effort * 0.55;
    const lSwim  = (s * amp * 0.7) * (1 - effort) + effort * 0.55;
    const lBrace = -1.05;                       // forward, wide
    this.armL.set(
      THREE.MathUtils.lerp(lSwim, lBrace, brace),
      armBend + (s * 0.5 + 0.5) * 0.3 * swimAmt + brace * 0.2,
      -0.12 - c * 0.10 * swimAmt - brace * 0.55 - effort * 0.10,
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

    // ── Legs: a faint scissor at rest, but as effort rises they TUCK and
    // TRAIL together behind the body — an arrowed, gliding freefall posture.
    // The old code ADDED effort to leg bend/scissor and a haul "drive" kick,
    // which is what made the riggers pump their legs like sprinters. The
    // drive kick is gone; the scissor amplitude is small and effort folds it
    // away into a streamlined trail (legs swept back, lightly bent together).
    const legBend = 0.20 + strokeAmp * 0.35 + effort * 0.18;
    const trail = effort * -0.85;                // both legs sweep back/together
    const lLeg = (-s * amp * 0.40) * (1 - effort) + trail;
    const rLeg = ( s * amp * 0.40) * (1 - effort) + trail;
    const braceLeg = 0.95;                       // both legs forward to brake
    this.legL.set(
      THREE.MathUtils.lerp(lLeg, braceLeg, brace),
      legBend + (-s * 0.5 + 0.5) * 0.25 * swimAmt + brace * 0.7,
    );
    this.legR.set(
      THREE.MathUtils.lerp(rLeg, braceLeg, brace),
      legBend + ( s * 0.5 + 0.5) * 0.25 * swimAmt + brace * 0.7,
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

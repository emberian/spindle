// DISTANT HABITAT VISTA — the world *outside* the calm.
//
// The iconic O'Neill-cylinder interior, seen FAR away beyond the play volume:
// the opposite curving land surface of the rotating habitat wrapping up and
// over into the "sky", distant terraced / agricultural / industrial banding,
// the axial sunline glow running down the spine — all dissolving into the
// existing `Calm` haze/fog so it reads as a vast lived-in rotating world
// without ever competing with gameplay.
//
// FRAME (mirrors Calm.ts): rendered in the CO-ROTATING frame. World +X = the
// 640 m spin axis; (y,z) = rotating cross-section; play skin at R=45. This
// vista is a set of huge concentric shells co-axial with +X, sitting FAR
// outside R=45 and well past the play volume — recessive background only.
//
// TASTE CONTRACT: distant, dim, low-contrast, heavily fog-faded; kept under
// the bloom threshold (0.7) so it never blooms; scrolls in the SAME sense as
// Calm's skin texture (slower) so it reinforces "you are inside a spinning
// habitat"; a handful of meshes only; geometry/material created once; no
// per-frame allocation; disposed on teardown. When unsure: dimmer / further.

import * as THREE from 'three';
import { REG } from '../sim/RegConstants';

// Vista geometry. These radii are FAR outside the R=45 play skin and the
// shell length over-runs the 640 m play volume so its ends never clip into
// gameplay; depth-correct ordering is carried by the scene's linear fog
// (anything this distant is already mostly haze).
const NEAR_SHELL_R = 320;            // the great curving land surface
const FAR_SHELL_R  = 560;            // a fainter banded layer further out
const SUNLINE_R    = 6.5;            // distant axial sunline tube radius
const SHELL_LEN    = REG.L * 2.2;    // over-runs the 640 m field generously

// ── Texture: the distant curving land — longitudinal habitat banding ─────────
// A tall, low-contrast canvas: terraced agricultural / industrial bands plus
// faint glazed window strips, painted dark so fog + low opacity keep it a
// whisper. wrapT repeats along the axis; wrapS wraps the hoop.
function makeLandTexture(): THREE.CanvasTexture {
  const W = 256, H = 1024;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d')!;

  // Base: very dark space-tinted ground (never near black — it is LIT land,
  // but distant). Stays well below the bloom threshold.
  g.fillStyle = '#161b26';
  g.fillRect(0, 0, W, H);

  // Eight longitudinal "valleys" around the hoop: alternating muted
  // agricultural greens / industrial greys, with two glazed window strips.
  const bands = 8;
  for (let i = 0; i < bands; i++) {
    const x0 = (i / bands) * W;
    const w  = W / bands;
    if (i % 4 === 1) {
      // glazed sky-window strip — faint cool glow (the cylinder's "windows")
      const grd = g.createLinearGradient(x0, 0, x0 + w, 0);
      grd.addColorStop(0,   'rgba(45,120,135,0.00)');
      grd.addColorStop(0.5, 'rgba(45,120,135,0.16)');
      grd.addColorStop(1,   'rgba(45,120,135,0.00)');
      g.fillStyle = grd;
      g.fillRect(x0, 0, w, H);
    } else {
      // land valley — muted green-grey, alternating tone
      g.fillStyle = i % 2 ? '#1b2530' : '#1d2722';
      g.fillRect(x0, 0, w, H);
    }
  }

  // Terraced / agricultural cross-banding along the axis — soft horizontal
  // strata, very low contrast (the "fields" of the far land receding).
  for (let y = 0; y < H; y += 22) {
    const tone = 18 + ((y * 7) % 16);
    g.fillStyle = `rgba(${tone + 14},${tone + 22},${tone + 18},0.16)`;
    g.fillRect(0, y, W, 11);
  }

  // Sparse industrial flecks — distant settlement lights, kept dim so they
  // never bloom (peak well under threshold 0.7 after fog + low opacity).
  g.fillStyle = 'rgba(150,170,180,0.10)';
  for (let i = 0; i < 220; i++) {
    const x = (Math.sin(i * 12.9898) * 43758.5453 % 1 + 1) % 1 * W;
    const y = (Math.sin(i * 78.233)  * 12543.1234 % 1 + 1) % 1 * H;
    g.fillRect(x, y, 1.4, 1.4);
  }

  const t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(3, 4);   // 3 around the far hoop, 4 along the long axis
  t.anisotropy = 8;
  return t;
}

// ── A big curving-land shell, viewed from inside, normal (fog-obeying) mat ────
function makeLandShell(
  radius: number,
  tex: THREE.Texture,
  opacity: number,
  tint: number,
): THREE.Mesh {
  const geo = new THREE.CylinderGeometry(radius, radius, SHELL_LEN, 80, 1, true);
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    color: tint,
    side: THREE.BackSide,     // we are inside the vast cylinder, looking out
    transparent: true,
    opacity,                  // very low — a whisper
    depthWrite: false,        // background: never occlude; fog carries order
    fog: true,                // CRUCIAL: heavily haze-faded into Calm's fog
  });
  const m = new THREE.Mesh(geo, mat);
  m.rotation.z = Math.PI / 2; // align cylinder axis with world +X (spin axis)
  m.renderOrder = -10;        // draw before gameplay (defensive; fog dominates)
  return m;
}

// ── Distant axial sunline — the bright weightless spine, far & dim ───────────
function makeDistantSunline(): THREE.Mesh {
  const geo = new THREE.CylinderGeometry(
    SUNLINE_R, SUNLINE_R, SHELL_LEN * 0.96, 16, 1, true,
  );
  const mat = new THREE.MeshBasicMaterial({
    color: 0xbfe2ec,
    transparent: true,
    opacity: 0.05,            // far under bloom threshold; pure haze hint
    side: THREE.BackSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    fog: true,
  });
  const m = new THREE.Mesh(geo, mat);
  m.rotation.z = Math.PI / 2;
  m.renderOrder = -9;
  return m;
}

// ─────────────────────────────────────────────────────────────────────────────
export class DistantHabitat {
  readonly group = new THREE.Group();

  private landTex?: THREE.CanvasTexture;
  private nearShell?: THREE.Mesh;
  private farShell?: THREE.Mesh;
  private sunline?: THREE.Mesh;

  /**
   * @param scene  scene to attach to (persistent, like Calm)
   * @param enabled quality flag — default ON (it IS the atmosphere) but the
   *                whole vista can be disabled to reclaim its (tiny) budget.
   */
  constructor(scene: THREE.Scene, enabled = true) {
    if (!enabled) return;

    this.landTex = makeLandTexture();

    // The great curving land surface wrapping up and over into the "sky".
    // Cool dim tint, very low opacity — recessive, heavily fog-faded.
    this.nearShell = makeLandShell(NEAR_SHELL_R, this.landTex, 0.22, 0x3a4654);
    this.group.add(this.nearShell);

    // A fainter banded layer further out for parallax depth — even dimmer,
    // its own scaled texture instance so the two layers drift independently.
    const farTex = this.landTex.clone();
    farTex.needsUpdate = true;
    farTex.repeat.set(2, 3);
    this.farShell = makeLandShell(FAR_SHELL_R, farTex, 0.10, 0x2b3340);
    this.group.add(this.farShell);

    // The axial sunline glow receding down the spine into haze.
    this.sunline = makeDistantSunline();
    this.group.add(this.sunline);

    scene.add(this.group);
  }

  /**
   * Slow parallax drift — SAME sense as Calm's skin scroll (V offset grows
   * with +omega), but markedly slower because it is far away (distant things
   * subtend less angular motion). No nausea, no fast motion, no allocation.
   */
  update(dt: number): void {
    if (!this.nearShell || !this.farShell) return;
    // Calm scrolls skinTex.offset.y by dt*omega*0.12. The far land is much
    // further out, so the same spin reads as a gentler crawl: 0.035 / 0.022.
    const near = this.nearShell.material as THREE.MeshBasicMaterial;
    const far  = this.farShell.material  as THREE.MeshBasicMaterial;
    if (near.map) near.map.offset.y = (near.map.offset.y + dt * REG.omega * 0.035) % 1;
    if (far.map)  far.map.offset.y  = (far.map.offset.y  + dt * REG.omega * 0.022) % 1;
  }

  /** Release all GPU resources (no leak on teardown). */
  dispose(): void {
    this.group.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
      const mat = m.material as THREE.MeshBasicMaterial | undefined;
      if (mat) {
        if (mat.map) mat.map.dispose();
        mat.dispose();
      }
    });
    this.group.removeFromParent();
  }
}

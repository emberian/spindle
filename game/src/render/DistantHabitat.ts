// DISTANT HABITAT VISTA — the world curving up overhead.
//
// The iconic O'Neill-cylinder interior: the opposite curving land surface
// wrapping up and over into the "sky", distant terraced / agricultural /
// industrial banding with warm settlement lights, the axial sunline glow
// running down the spine — the signature "you are inside a rotating world"
// moment. This should be VISIBLE. Not overpowering, but unmistakable:
// the first thing that makes someone go "whoa, I'm inside a cylinder."
//
// FRAME (mirrors Calm.ts): rendered in the CO-ROTATING frame. World +X = the
// 640 m spin axis; (y,z) = rotating cross-section; play skin at R=45. This
// vista is a set of huge concentric shells co-axial with +X, sitting FAR
// outside R=45 and well past the play volume.
//
// LAYERS (inside→out):
//   R=200  — atmosphere/cloud shell (depth/haze between player and land)
//   R=320  — the great curving land surface (near shell)
//   R=560  — a fainter banded layer further out (far shell, parallax)
//   R=6.5  — axial sunline tube (the cylinder's light source)
//   R=12   — sunline glow falloff (soft additive halo)

import * as THREE from 'three';
import { REG } from '../sim/RegConstants';

// Vista geometry radii.
const NEAR_SHELL_R = 320;            // the great curving land surface
const FAR_SHELL_R  = 560;            // a fainter banded layer further out
const CLOUD_SHELL_R = 200;           // atmosphere/cloud layer (between player & land)
const SUNLINE_R    = 6.5;            // distant axial sunline tube radius
const SUNLINE_GLOW_R = 14;           // soft glow halo around sunline
const SHELL_LEN    = REG.L * 2.2;    // over-runs the 640 m field generously

// ── Texture: the distant curving land — longitudinal habitat banding ─────────
// A tall canvas: terraced agricultural / industrial bands, glazed window strips,
// warm settlement light clusters. Painted bright enough to survive transparency
// and read clearly as "distant inhabited land."
function makeLandTexture(): THREE.CanvasTexture {
  const W = 256, H = 1024;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d')!;

  // Base: medium-dark land tone — NOT near-black. This is lit land seen from
  // across the cylinder interior. Think twilight-lit terraces.
  g.fillStyle = '#2a3545';
  g.fillRect(0, 0, W, H);

  // Eight longitudinal "valleys" around the hoop: varied biomes —
  // green agriculture, brown/amber fields, grey industry, with two glazed
  // window strips that let starlight in.
  const bands = 8;
  const biomeColors = [
    '#2e4538', // deep forest green
    '#3a4a3a', // muted green (agriculture)
    '#3d3828', // amber/brown (grain fields)
    '#354050', // grey-blue (industry/city)
    '#2a4035', // green valley
    '#4a4030', // warm brown (agriculture)
    '#384555', // industrial grey
    '#2e3d30', // darker forest
  ];
  for (let i = 0; i < bands; i++) {
    const x0 = (i / bands) * W;
    const w  = W / bands;
    if (i % 4 === 1) {
      // Glazed sky-window strip — visible cool-white glow (the cylinder's
      // windows to space). These should clearly READ as illuminated strips.
      const grd = g.createLinearGradient(x0, 0, x0 + w, 0);
      grd.addColorStop(0,   'rgba(140,200,220,0.00)');
      grd.addColorStop(0.3, 'rgba(140,200,220,0.35)');
      grd.addColorStop(0.5, 'rgba(180,225,240,0.45)');
      grd.addColorStop(0.7, 'rgba(140,200,220,0.35)');
      grd.addColorStop(1,   'rgba(140,200,220,0.00)');
      g.fillStyle = grd;
      g.fillRect(x0, 0, w, H);
    } else {
      // Land valley — varied biomes
      g.fillStyle = biomeColors[i];
      g.fillRect(x0, 0, w, H);
    }
  }

  // Terraced / agricultural cross-banding along the axis — horizontal strata
  // with visible contrast (the "fields" of the far land).
  for (let y = 0; y < H; y += 18) {
    const tone = 30 + ((y * 7) % 25);
    const green = tone + 8 + ((y * 3) % 12);
    g.fillStyle = `rgba(${tone + 10},${green},${tone + 5},0.22)`;
    g.fillRect(0, y, W, 9);
  }

  // ── Settlement lights: warm orange/yellow clusters that read as "cities" ──
  // These are the "people live there" sell. Varied: some bright clusters,
  // some dim singles, some tight groups.
  const rng = (seed: number) => {
    // Deterministic hash for reproducible placement
    const x = Math.sin(seed) * 43758.5453;
    return x - Math.floor(x);
  };

  // Large settlement clusters (bright, 8-15 lights grouped together)
  for (let cluster = 0; cluster < 14; cluster++) {
    const cx = rng(cluster * 17.31 + 5.7) * W;
    const cy = rng(cluster * 23.71 + 11.3) * H;
    const count = 8 + Math.floor(rng(cluster * 7.13) * 8);
    for (let j = 0; j < count; j++) {
      const dx = (rng(cluster * 100 + j * 3.7) - 0.5) * 14;
      const dy = (rng(cluster * 100 + j * 5.3 + 1) - 0.5) * 10;
      const brightness = 0.25 + rng(cluster * 100 + j * 2.1 + 2) * 0.35;
      const r = 180 + Math.floor(rng(cluster * 100 + j) * 60);
      const gb = 120 + Math.floor(rng(cluster * 100 + j + 50) * 60);
      g.fillStyle = `rgba(${r},${gb},${Math.floor(gb * 0.4)},${brightness})`;
      const size = 1.2 + rng(cluster * 100 + j * 1.1) * 1.8;
      g.fillRect(cx + dx, cy + dy, size, size);
    }
  }

  // Medium settlements (4-7 lights)
  for (let cluster = 0; cluster < 25; cluster++) {
    const cx = rng(cluster * 31.7 + 100) * W;
    const cy = rng(cluster * 41.3 + 200) * H;
    const count = 4 + Math.floor(rng(cluster * 11.7 + 300) * 4);
    for (let j = 0; j < count; j++) {
      const dx = (rng(cluster * 200 + j * 4.1) - 0.5) * 8;
      const dy = (rng(cluster * 200 + j * 6.7 + 1) - 0.5) * 6;
      const brightness = 0.20 + rng(cluster * 200 + j * 2.3 + 2) * 0.25;
      g.fillStyle = `rgba(220,160,60,${brightness})`;
      g.fillRect(cx + dx, cy + dy, 1.4, 1.4);
    }
  }

  // Scattered single lights (dim, isolated farmsteads / outposts)
  for (let i = 0; i < 180; i++) {
    const x = rng(i * 12.9898 + 500) * W;
    const y = rng(i * 78.233 + 600) * H;
    const brightness = 0.15 + rng(i * 3.14 + 700) * 0.20;
    g.fillStyle = `rgba(200,150,70,${brightness})`;
    g.fillRect(x, y, 1.2, 1.2);
  }

  // ── Horizon-line atmospheric haze at texture edges (hoop direction) ────────
  // The land curving away into the distance gets hazier at the edges,
  // simulating atmospheric perspective along the curve.
  const hazeGrad = g.createLinearGradient(0, 0, W, 0);
  hazeGrad.addColorStop(0, 'rgba(100,130,160,0.18)');
  hazeGrad.addColorStop(0.15, 'rgba(100,130,160,0.00)');
  hazeGrad.addColorStop(0.85, 'rgba(100,130,160,0.00)');
  hazeGrad.addColorStop(1, 'rgba(100,130,160,0.18)');
  g.fillStyle = hazeGrad;
  g.fillRect(0, 0, W, H);

  // Also add slight brightening at the top/bottom edges (axial haze)
  const axialHaze = g.createLinearGradient(0, 0, 0, H);
  axialHaze.addColorStop(0, 'rgba(80,110,140,0.14)');
  axialHaze.addColorStop(0.1, 'rgba(80,110,140,0.00)');
  axialHaze.addColorStop(0.9, 'rgba(80,110,140,0.00)');
  axialHaze.addColorStop(1, 'rgba(80,110,140,0.14)');
  g.fillStyle = axialHaze;
  g.fillRect(0, 0, W, H);

  const t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(3, 4);   // 3 around the far hoop, 4 along the long axis
  t.anisotropy = 8;
  return t;
}

// ── Cloud/atmosphere texture — wispy, soft, blue-white ───────────────────────
function makeCloudTexture(): THREE.CanvasTexture {
  const W = 256, H = 512;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d')!;

  // Transparent base
  g.clearRect(0, 0, W, H);

  // Wispy cloud patches — irregular soft blobs scattered across the canvas.
  // These should read as thin cirrus-like wisps, not solid overcast.
  const rng = (seed: number) => {
    const x = Math.sin(seed) * 43758.5453;
    return x - Math.floor(x);
  };

  for (let i = 0; i < 40; i++) {
    const cx = rng(i * 13.7 + 1) * W;
    const cy = rng(i * 29.3 + 2) * H;
    const rx = 15 + rng(i * 7.1 + 3) * 40;
    const ry = 5 + rng(i * 11.3 + 4) * 15;
    const alpha = 0.04 + rng(i * 5.7 + 5) * 0.08;

    const grad = g.createRadialGradient(cx, cy, 0, cx, cy, Math.max(rx, ry));
    grad.addColorStop(0, `rgba(200,220,240,${alpha})`);
    grad.addColorStop(0.5, `rgba(180,210,235,${alpha * 0.5})`);
    grad.addColorStop(1, 'rgba(180,210,235,0.00)');
    g.fillStyle = grad;
    g.save();
    g.translate(cx, cy);
    g.scale(rx / Math.max(rx, ry), ry / Math.max(rx, ry));
    g.translate(-cx, -cy);
    g.fillRect(cx - Math.max(rx, ry), cy - Math.max(rx, ry),
               Math.max(rx, ry) * 2, Math.max(rx, ry) * 2);
    g.restore();
  }

  // A few longer streak clouds (high-altitude cirrus feel)
  for (let i = 0; i < 12; i++) {
    const y = rng(i * 47.1 + 100) * H;
    const x0 = rng(i * 31.7 + 101) * W * 0.3;
    const len = 60 + rng(i * 19.3 + 102) * 120;
    const alpha = 0.03 + rng(i * 9.1 + 103) * 0.06;
    g.strokeStyle = `rgba(210,225,240,${alpha})`;
    g.lineWidth = 3 + rng(i * 3.3 + 104) * 5;
    g.lineCap = 'round';
    g.beginPath();
    g.moveTo(x0, y);
    g.quadraticCurveTo(
      x0 + len * 0.5, y + (rng(i * 7.7 + 105) - 0.5) * 20,
      x0 + len, y + (rng(i * 4.1 + 106) - 0.5) * 10,
    );
    g.stroke();
  }

  const t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(4, 3);
  t.anisotropy = 4;
  return t;
}

// ── A big curving-land shell, viewed from inside ─────────────────────────────
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
    opacity,
    depthWrite: false,
    fog: false,               // fog was eating these — disable it; shells are
                              // well within the fog near plane anyway (they sit
                              // at 270-510m from cam, fog near = 672m), but the
                              // linear fog math still dims them slightly. We
                              // handle depth via opacity + layering instead.
  });
  const m = new THREE.Mesh(geo, mat);
  m.rotation.z = Math.PI / 2; // align cylinder axis with world +X (spin axis)
  m.renderOrder = -10;        // draw before gameplay
  return m;
}

// ── Cloud/atmosphere shell ───────────────────────────────────────────────────
function makeCloudShell(tex: THREE.Texture): THREE.Mesh {
  const geo = new THREE.CylinderGeometry(
    CLOUD_SHELL_R, CLOUD_SHELL_R, SHELL_LEN * 0.9, 64, 1, true,
  );
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    side: THREE.BackSide,
    transparent: true,
    opacity: 0.10,
    depthWrite: false,
    fog: false,
    blending: THREE.NormalBlending,
  });
  const m = new THREE.Mesh(geo, mat);
  m.rotation.z = Math.PI / 2;
  m.renderOrder = -11;   // draw before land shells
  return m;
}

// ── Distant axial sunline — the cylinder's light source ─────────────────────
function makeDistantSunline(): THREE.Mesh {
  const geo = new THREE.CylinderGeometry(
    SUNLINE_R, SUNLINE_R, SHELL_LEN * 0.96, 16, 1, true,
  );
  const mat = new THREE.MeshBasicMaterial({
    color: 0xfff0d4,          // warm yellow-white — this IS the sun
    transparent: true,
    opacity: 0.22,            // clearly visible warm glow
    side: THREE.BackSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  const m = new THREE.Mesh(geo, mat);
  m.rotation.z = Math.PI / 2;
  m.renderOrder = -8;
  return m;
}

// ── Sunline glow falloff — soft halo around the sunline ─────────────────────
function makeSunlineGlow(): THREE.Mesh {
  const geo = new THREE.CylinderGeometry(
    SUNLINE_GLOW_R, SUNLINE_GLOW_R, SHELL_LEN * 0.94, 16, 1, true,
  );
  const mat = new THREE.MeshBasicMaterial({
    color: 0xffe8b0,          // slightly warmer/dimmer than the core
    transparent: true,
    opacity: 0.08,            // soft falloff halo
    side: THREE.BackSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    fog: false,
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
  private cloudTex?: THREE.CanvasTexture;
  private nearShell?: THREE.Mesh;
  private farShell?: THREE.Mesh;
  private cloudShell?: THREE.Mesh;
  private sunline?: THREE.Mesh;
  private sunlineGlow?: THREE.Mesh;

  /**
   * @param scene  scene to attach to (persistent, like Calm)
   * @param enabled quality flag — default ON (it IS the atmosphere) but the
   *                whole vista can be disabled to reclaim its (tiny) budget.
   */
  constructor(scene: THREE.Scene, enabled = true) {
    if (!enabled) return;

    this.landTex = makeLandTexture();

    // The great curving land surface wrapping up and over into the "sky".
    // This is THE signature visual — it must be clearly visible.
    // Tint is a muted slate-green to sell "land at a distance through air."
    this.nearShell = makeLandShell(NEAR_SHELL_R, this.landTex, 0.60, 0x5a7068);
    this.group.add(this.nearShell);

    // A fainter banded layer further out for parallax depth.
    const farTex = this.landTex.clone();
    farTex.needsUpdate = true;
    farTex.repeat.set(2, 3);
    this.farShell = makeLandShell(FAR_SHELL_R, farTex, 0.35, 0x4a5a6a);
    this.group.add(this.farShell);

    // Cloud/atmosphere layer — creates depth between player and distant land,
    // sells "there's air in this cylinder." Wispy, not overcast.
    this.cloudTex = makeCloudTexture();
    this.cloudShell = makeCloudShell(this.cloudTex);
    this.group.add(this.cloudShell);

    // The axial sunline — warm yellow-white, clearly visible as the light source.
    this.sunline = makeDistantSunline();
    this.group.add(this.sunline);

    // Soft glow halo around the sunline for natural falloff.
    this.sunlineGlow = makeSunlineGlow();
    this.group.add(this.sunlineGlow);

    scene.add(this.group);
  }

  /**
   * Slow parallax drift — SAME sense as Calm's skin scroll (V offset grows
   * with +omega), but markedly slower because it is far away (distant things
   * subtend less angular motion). The cloud layer drifts at its own rate
   * to create parallax separation.
   */
  update(dt: number): void {
    if (!this.nearShell || !this.farShell) return;
    const near = this.nearShell.material as THREE.MeshBasicMaterial;
    const far  = this.farShell.material  as THREE.MeshBasicMaterial;
    if (near.map) near.map.offset.y = (near.map.offset.y + dt * REG.omega * 0.035) % 1;
    if (far.map)  far.map.offset.y  = (far.map.offset.y  + dt * REG.omega * 0.022) % 1;

    // Cloud layer drifts slightly differently — sells independent atmosphere motion.
    if (this.cloudShell) {
      const cloud = this.cloudShell.material as THREE.MeshBasicMaterial;
      if (cloud.map) {
        cloud.map.offset.y = (cloud.map.offset.y + dt * REG.omega * 0.028) % 1;
        // Very slow axial drift for the clouds (wind along the cylinder)
        cloud.map.offset.x = (cloud.map.offset.x + dt * 0.003) % 1;
      }
    }
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

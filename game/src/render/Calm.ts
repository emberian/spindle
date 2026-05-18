// The calm, rendered: the interior of the rotating O'Neill cylinder.
// World +X = spin axis / 640 m length; (y,z) = rotating cross-section.
// Rendered in the CO-ROTATING frame — geometry static, spin sold by scrolling
// skin texture and particle dust (no nausea). Faith ring = CYAN at +x
// (spinward), Free ring = ORANGE at −x (antispinward). Gravity gradient:
// bright/cool at axis, warm/hazy at skin ("deep is faithful, high is free").

import * as THREE from 'three';
import { REG, GATE_X } from '../sim/RegConstants';

// ── Palette ──────────────────────────────────────────────────────────────────
const C_BG     = 0x11131a;
const C_CYAN   = 0x1aa6b7;
const C_ORANGE = 0xd4602a;
const C_PAPER  = 0xf4f1ea;
const C_DIM    = 0x6b7079;

// ── Skin texture: luminous land-stripe pattern scrolls along +V to sell spin ─
function makeSkinTexture(): THREE.CanvasTexture {
  const W = 128, H = 1024;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d')!;

  // Base: mid-dark teal-grey (not black — the skin is LIT land)
  g.fillStyle = '#1b2030';
  g.fillRect(0, 0, W, H);

  // Six longitudinal biomes: 4 "land" (warm/neutral) + 2 "window strip" (blue-glow)
  for (let i = 0; i < 6; i++) {
    const x0 = (i / 6) * W;
    const w  = W / 6;
    if (i % 3 === 0) {
      // bright window strip — glazed cyan glow
      const grd = g.createLinearGradient(x0, 0, x0 + w, 0);
      grd.addColorStop(0,   'rgba(26,166,183,0.0)');
      grd.addColorStop(0.3, 'rgba(26,166,183,0.22)');
      grd.addColorStop(0.7, 'rgba(26,166,183,0.22)');
      grd.addColorStop(1,   'rgba(26,166,183,0.0)');
      g.fillStyle = grd;
      g.fillRect(x0, 0, w, H);
    } else {
      // land panel — warm grey-green terrain
      g.fillStyle = i % 2 ? '#232a35' : '#1e2530';
      g.fillRect(x0, 0, w, H);
    }
  }

  // Axial lane ticks (faint, marking the 640 m length)
  g.strokeStyle = 'rgba(107,112,121,0.28)';
  g.lineWidth = 0.8;
  for (let y = 0; y < H; y += 64) {
    g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke();
  }
  // Fine grain — faint noise-like horizontal dashes for texture depth
  g.strokeStyle = 'rgba(180,190,200,0.06)';
  g.lineWidth = 0.5;
  for (let y = 8; y < H; y += 16) {
    for (let x = 0; x < W; x += 12) {
      if (Math.sin(x * 1.7 + y * 0.31) > 0.3) {
        g.beginPath(); g.moveTo(x, y); g.lineTo(x + 6, y); g.stroke();
      }
    }
  }

  const t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(6, 1);   // 6 repeats around hoop → 36 stripes total
  t.anisotropy = 8;
  return t;
}


// ── Axis "sunline" — the bright weightless core light tube ───────────────────
function makeAxisGlow(): THREE.Mesh {
  // Thin long cylinder along X, very slightly emissive, inside the calm.
  const geo = new THREE.CylinderGeometry(1.6, 1.6, REG.L * 0.96, 12, 1, true);
  const mat = new THREE.MeshBasicMaterial({
    color: 0xd4eef5,
    transparent: true,
    opacity: 0.06, // was 0.18 — looking down the 640 m axis the additive tube saturated to white
    side: THREE.BackSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const m = new THREE.Mesh(geo, mat);
  m.rotation.z = Math.PI / 2;
  return m;
}

// ── Goal ring assembly: torus + halo disc + inner throat + text sprite ────────
function makeGoalAssembly(
  x: number,
  color: number,
  label: string,
): THREE.Group {
  const grp = new THREE.Group();
  grp.position.x = x;

  // Primary scoring torus
  const torus = new THREE.Mesh(
    new THREE.TorusGeometry(REG.gateRadius, 0.7, 20, 80),
    new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 1.6,
      roughness: 0.2,
      metalness: 0.6,
    }),
  );
  torus.rotation.y = Math.PI / 2;
  grp.add(torus);

  // Bright inner ring for bloom pickup (thin, high emissive)
  const inner = new THREE.Mesh(
    new THREE.TorusGeometry(REG.gateRadius, 0.18, 10, 80),
    new THREE.MeshBasicMaterial({ color: C_PAPER, blending: THREE.AdditiveBlending }),
  );
  inner.rotation.y = Math.PI / 2;
  grp.add(inner);

  // Halo disc (soft additive backlight)
  const halo = new THREE.Mesh(
    new THREE.RingGeometry(REG.gateRadius - 1.5, REG.gateRadius + 4, 64, 1),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.28,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  halo.rotation.y = Math.PI / 2;
  grp.add(halo);

  // Outer halo (bigger, very soft)
  const outerHalo = new THREE.Mesh(
    new THREE.RingGeometry(REG.gateRadius + 1, REG.gateRadius + 9, 64, 1),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.09,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  outerHalo.rotation.y = Math.PI / 2;
  grp.add(outerHalo);

  // Volumetric throat: a short open cone funneling into the ring.
  // Two cones point inward from each side, like a soft mouth.
  for (const sign of [-1, 1]) {
    const cone = new THREE.Mesh(
      new THREE.CylinderGeometry(REG.gateRadius * 0.55, REG.gateRadius * 0.92, 14, 32, 1, true),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.07,
        side: THREE.DoubleSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    cone.rotation.z = Math.PI / 2;
    cone.position.x = sign * 7;
    grp.add(cone);
  }

  // Text sprite label
  grp.add(makeLabel(label, color, REG.gateRadius + 6.5));

  return grp;
}

// ── Canvas text sprite ────────────────────────────────────────────────────────
function makeLabel(text: string, color: number, yOffset: number): THREE.Sprite {
  const W = 256, H = 72;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d')!;
  const hex = '#' + color.toString(16).padStart(6, '0');
  g.clearRect(0, 0, W, H);
  g.font = 'bold 38px "Helvetica Neue", Arial, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  // Glow pass
  g.shadowColor = hex;
  g.shadowBlur = 18;
  g.fillStyle = hex;
  g.fillText(text, W / 2, H / 2);
  // Second bright pass
  g.shadowBlur = 4;
  g.fillStyle = '#ffffff';
  g.fillText(text, W / 2, H / 2);

  const tex = new THREE.CanvasTexture(c);
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set((W / H) * 9, 9, 1);
  sprite.position.set(0, yOffset, 0);
  return sprite;
}

// ── Gate divider ring (first/deep/mouth) — faint orientation marker ───────────
function makeGateDivider(x: number): THREE.Group {
  const grp = new THREE.Group();
  grp.position.x = x;

  // Outer faint ring
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(REG.R * 0.90, 0.28, 6, 64),
    new THREE.MeshBasicMaterial({
      color: C_DIM,
      transparent: true,
      opacity: 0.18,
      depthWrite: false,
    }),
  );
  ring.rotation.y = Math.PI / 2;
  grp.add(ring);

  // Small axis marker dot
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(0.55, 8, 6),
    new THREE.MeshBasicMaterial({ color: C_DIM, transparent: true, opacity: 0.45 }),
  );
  grp.add(dot);

  return grp;
}

// ── Deepline (axis spine) — thin dashed-look line ────────────────────────────
function makeDeepline(): THREE.Line {
  const pts: THREE.Vector3[] = [];
  const N = 64;
  for (let i = 0; i <= N; i++) {
    // "dashed" by skipping every other segment via degenerate points
    const x = -GATE_X + (i / N) * REG.L;
    if (i % 2 === 0) {
      pts.push(new THREE.Vector3(x, 0, 0));
    } else {
      pts.push(new THREE.Vector3(x, 0, 0));
    }
  }
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: C_DIM, transparent: true, opacity: 0.22 }),
  );
}

// ── Instanced spars — 16 rings × 3 = 48 clip-points ────────────────────────
function makeSpars(): THREE.InstancedMesh {
  const geo = new THREE.CapsuleGeometry(0.45, 5.5, 4, 8);
  const mat = new THREE.MeshStandardMaterial({
    color: 0x3a8fa0,
    emissive: 0x1aa6b7,
    emissiveIntensity: 0.55,
    roughness: 0.45,
    metalness: 0.7,
  });
  const RINGS = 16, AROUND = 3;
  const mesh = new THREE.InstancedMesh(geo, mat, RINGS * AROUND);
  mesh.castShadow = false;
  const mtx = new THREE.Matrix4();
  let idx = 0;
  for (let i = 0; i < RINGS; i++) {
    const x = -GATE_X + ((i + 0.5) / RINGS) * REG.L;
    for (let a = 0; a < AROUND; a++) {
      const ang = (a / AROUND) * Math.PI * 2;
      const r   = REG.R * 0.62;
      const cy  = Math.cos(ang), sy = Math.sin(ang);
      mtx.makeBasis(
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, cy, sy),
        new THREE.Vector3(0, -sy, cy),
      );
      mtx.setPosition(x, cy * r, sy * r);
      mesh.setMatrixAt(idx++, mtx);
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
  return mesh;
}

// ── Radial gradient band particles (gravity gradient) ─────────────────────────
// A thin ring of point-sprites at mid-radius that hint at "dense/warm" gravity.
function makeGravityHazeBand(): THREE.Points {
  const COUNT = 600;
  const pos = new Float32Array(COUNT * 3);
  const radMin = REG.R * 0.55, radMax = REG.R * 0.92;
  for (let i = 0; i < COUNT; i++) {
    const ang = Math.random() * Math.PI * 2;
    const r   = radMin + Math.random() * (radMax - radMin);
    const x   = -GATE_X + Math.random() * REG.L;
    pos[i*3]   = x;
    pos[i*3+1] = Math.cos(ang) * r;
    pos[i*3+2] = Math.sin(ang) * r;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({
    color: 0xc06030,    // warm orange dust
    size: 0.55,
    transparent: true,
    opacity: 0.13,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
  });
  return new THREE.Points(geo, mat);
}

// ── Axis light-mote particles (cool/free core) ────────────────────────────────
function makeAxisMotes(): THREE.Points {
  const COUNT = 300;
  const pos = new Float32Array(COUNT * 3);
  for (let i = 0; i < COUNT; i++) {
    const ang = Math.random() * Math.PI * 2;
    const r   = Math.random() * REG.R * 0.28;
    const x   = -GATE_X + Math.random() * REG.L;
    pos[i*3]   = x;
    pos[i*3+1] = Math.cos(ang) * r;
    pos[i*3+2] = Math.sin(ang) * r;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({
    color: 0x7de8f8,
    size: 0.4,
    transparent: true,
    opacity: 0.25,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
  });
  return new THREE.Points(geo, mat);
}

// ─────────────────────────────────────────────────────────────────────────────
export class Calm {
  readonly group = new THREE.Group();

  private skinTex: THREE.CanvasTexture;
  private faithGoal: THREE.Group;
  private freeGoal: THREE.Group;

  /** World-space centres of the two goal rings [faith(+x), free(−x)]. */
  goalCenters(): [THREE.Vector3, THREE.Vector3] {
    return [
      new THREE.Vector3( GATE_X, 0, 0),
      new THREE.Vector3(-GATE_X, 0, 0),
    ];
  }

  constructor(scene: THREE.Scene) {
    // Fog: was Fog(C_BG, L*0.38=243m, L*1.3=832m) which fogged the PLAY
    // VOLUME itself — riggers ~hundreds of m from the spectate cam dissolved
    // into near-black haze (the top legibility bug). Push the NEAR plane far
    // past the play volume (L*1.05 = 672m) and the FAR out (L*2.4 = 1536m)
    // and warm/lighten the fog colour so only the truly distant vista fades —
    // the playspace stays crisp and fully lit.
    // Fog colour: the canon bg lifted toward a soft slate so the distant fade
    // stays on-palette but isn't a black curtain.
    const fogCol = new THREE.Color(C_BG).lerp(new THREE.Color(0x3a4660), 0.55);
    scene.fog = new THREE.Fog(fogCol.getHex(), REG.L * 1.05, REG.L * 2.4);

    // ── Skin (open cylinder, viewed from inside) ────────────────────────────
    this.skinTex = makeSkinTexture();
    const skin = new THREE.Mesh(
      new THREE.CylinderGeometry(REG.R, REG.R, REG.L, 96, 64, true),
      new THREE.MeshStandardMaterial({
        map: this.skinTex,
        side: THREE.BackSide,
        roughness: 0.88,
        metalness: 0.0,
        emissive: 0x0d1520,
        emissiveIntensity: 1,
        // tint the skin a notch warmer at the rim — baked into the texture
      }),
    );
    skin.rotation.z = Math.PI / 2;
    this.group.add(skin);

    // ── Gravity gradient: radial haze bands ─────────────────────────────────
    // Three additive rings at different radii: inner cool, outer warm.
    // Placed perpendicular to X (in YZ) and extended by stacking copies.
    // We use a few long cylinder shells instead of per-ring geometry so the
    // gradient feels volumetric along the whole length.

    // Inner shell — cool axial region (free / weightless)
    const innerShell = new THREE.Mesh(
      new THREE.CylinderGeometry(REG.R * 0.22, REG.R * 0.22, REG.L * 0.92, 24, 1, true),
      new THREE.MeshBasicMaterial({
        color: 0x44d4ee,
        transparent: true,
        opacity: 0.055,
        side: THREE.BackSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    innerShell.rotation.z = Math.PI / 2;
    this.group.add(innerShell);

    // Mid shell — transition zone
    const midShell = new THREE.Mesh(
      new THREE.CylinderGeometry(REG.R * 0.58, REG.R * 0.58, REG.L * 0.90, 24, 1, true),
      new THREE.MeshBasicMaterial({
        color: 0x9060a0,
        transparent: true,
        opacity: 0.03,
        side: THREE.BackSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    midShell.rotation.z = Math.PI / 2;
    this.group.add(midShell);

    // Outer rim haze — warm/dense gravity zone
    const outerHaze = new THREE.Mesh(
      new THREE.CylinderGeometry(REG.R * 0.88, REG.R * 0.88, REG.L * 0.88, 24, 1, true),
      new THREE.MeshBasicMaterial({
        color: 0xc04010,
        transparent: true,
        opacity: 0.045,
        side: THREE.BackSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    outerHaze.rotation.z = Math.PI / 2;
    this.group.add(outerHaze);

    // ── Axis sunline glow ────────────────────────────────────────────────────
    this.group.add(makeAxisGlow());

    // ── Gravity-haze particles ───────────────────────────────────────────────
    this.group.add(makeGravityHazeBand());
    this.group.add(makeAxisMotes());

    // ── Deepline (axis spine) ────────────────────────────────────────────────
    this.group.add(makeDeepline());

    // ── Gate dividers (first water / deep water / mouth at ±GATE_X/2, 0) ────
    // canon: three gates divide the 640 m field into thirds.
    for (const gx of [-GATE_X * 0.5, 0, GATE_X * 0.5]) {
      this.group.add(makeGateDivider(gx));
    }

    // ── Spars ────────────────────────────────────────────────────────────────
    this.group.add(makeSpars());

    // ── Goal assemblies ──────────────────────────────────────────────────────
    this.faithGoal = makeGoalAssembly( GATE_X, C_CYAN,   'FAITH');
    this.freeGoal  = makeGoalAssembly(-GATE_X, C_ORANGE, 'FREE');
    this.group.add(this.faithGoal);
    this.group.add(this.freeGoal);

    // ── Lighting ─────────────────────────────────────────────────────────────
    // Cool axial key — simulates the habitat's sunline along the axis.
    // Raised 1.6 → 2.2 so the riggers' lit side reads as a clear bright body.
    const axisKey = new THREE.DirectionalLight(0xd2ecf6, 2.2);
    axisKey.position.set(REG.L * 0.4, REG.R * 0.15, 0);
    scene.add(axisKey);

    // Warm rim fill — suggests the land/ground below. Raised 0.55 → 1.15 so
    // the SHADOW side of each rigger is still a readable body, not a void.
    const rimFill = new THREE.DirectionalLight(0xf3bd72, 1.15);
    rimFill.position.set(0, -REG.R, REG.R * 0.5);
    scene.add(rimFill);

    // Second fill from camera-ish side so figures never present a black face
    // to the default spectate cam (it sits off the −Z / +radial side).
    const camFill = new THREE.DirectionalLight(0xbcd2e6, 0.7);
    camFill.position.set(-REG.L * 0.15, REG.R * 0.4, -REG.R);
    scene.add(camFill);

    // Ambient: was 0x1a2535 @ 1.1 — near-black, so unlit riggers were
    // silhouettes (the top bug). Substantially raised + warmed to a soft
    // skylight so every figure has a legible base luminance everywhere.
    scene.add(new THREE.AmbientLight(0x4a5872, 2.4));

    // Faith-end point light (cyan) — makes that side glow distinctly
    const faithLight = new THREE.PointLight(C_CYAN, 60, REG.L * 0.55, 1.4);
    faithLight.position.set(GATE_X, 0, 0);
    scene.add(faithLight);

    // Free-end point light (orange)
    const freeLight = new THREE.PointLight(C_ORANGE, 60, REG.L * 0.55, 1.4);
    freeLight.position.set(-GATE_X, 0, 0);
    scene.add(freeLight);

    scene.add(this.group);
  }

  /** Scroll the skin texture to sell spin; pulse goal halos gently. */
  update(dt: number): void {
    // Skin scroll: V offset increases at angular rate × a visual scale.
    // At omega=0.32 r/s and scale 0.12, one full revolution scrolls 0.038
    // texture units — very gentle, clearly directional.
    this.skinTex.offset.y = (this.skinTex.offset.y + dt * REG.omega * 0.12) % 1;

    // Gentle breathing pulse on goal halos (additive materials via scale).
    const t = performance.now() * 0.001;
    const pulse = 0.93 + 0.07 * Math.sin(t * 1.4);
    this.faithGoal.scale.setScalar(pulse);
    this.freeGoal.scale.setScalar(pulse);
  }
}

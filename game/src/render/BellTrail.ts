// The money shot: the bell's Coriolis path drawn as a glowing ribbon whose
// colour/intensity tracks spin-trueness. A clean (ringing) bell lays a
// saturated cyan stripe; a clatter desaturates the fresh segments to grey —
// you SEE the moment it goes off true, exactly when you hear it. On a Loop
// the trail blazes and holds the whole closed arc.
//
// Ribbon technique: pairs of offset vertices (camera-facing quad strip) give
// the trail real thickness the bloom pass can grab onto. A LINE is invisible;
// a ribbon is a stroke.
//
// bellGlow helper (for main.ts / orchestrator):
//   import { bellGlow } from './render/BellTrail';
//   (bellMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = bellGlow(chime);
// Maps chime → emissive intensity 0.25 (clatter) … 3.2 (true ring).

import * as THREE from 'three';

export function bellGlow(chime: number): number {
  // Exponential ramp: a true ring blazes; a clatter barely glows.
  // Tamed (was 0.25 + 2.95·…): at close/goal camera angles the old peak
  // blew the whole frame to white through bloom.
  return 0.12 + 0.5 * Math.pow(chime, 2.2);
}

// ── geometry constants ────────────────────────────────────────────────────────
const NORMAL_LEN = 300;   // segments to hold Coriolis arc
const LOOP_LEN   = 600;   // hold full closed-orbit history in loop mode
const HALF_W_BASE = 0.10; // half-width of ribbon at newest (metres — tune to ring scale)
const HALF_W_LOOP = 0.18; // wider in loop mode for extra drama

// Palette (linear sRGB, not gamma — three.js colour math is linear)
//   chime=1  → brilliant cyan-white  #1aa6b7 saturated + blown toward white
//   chime=0  → muted grey            desaturated, dimmed
const TRUE_R = 0.18, TRUE_G = 0.85, TRUE_B = 1.00; // cyan-white
const FLAT_R = 0.28, FLAT_G = 0.30, FLAT_B = 0.32; // cool grey

// ── material ─────────────────────────────────────────────────────────────────
// Vertex-coloured, additive-blended mesh.  No lighting — colours ARE the light.
function makeMat(): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.78, // visible ribbon (bloom is tamed so this no longer whites out)
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
}

export class BellTrail {
  // expose so callers can adjust render order / layers if needed
  readonly mesh: THREE.Mesh;

  private geo: THREE.BufferGeometry;
  private buf: { x: number; y: number; z: number; c: number }[] = [];
  private limit = NORMAL_LEN;
  private loopMode = false;

  // pre-allocated typed arrays: 2 verts per segment × 3 floats
  private readonly CAP = LOOP_LEN;
  private readonly verts:  Float32Array;  // position  — 2*CAP*3
  private readonly colors: Float32Array;  // color     — 2*CAP*3
  private readonly uvs:    Float32Array;  // uv        — 2*CAP*2 (optional, for future fx)
  private readonly idx:    Uint32Array;   // triangles — (2*CAP-2)*6 / 2 quads per segment

  // camera ref for billboard orientation
  private cam: THREE.Camera | null = null;

  constructor(scene: THREE.Scene, camera?: THREE.Camera) {
    this.cam = camera ?? null;

    const maxVerts = this.CAP * 2;
    const maxTris  = (this.CAP - 1) * 2; // 2 triangles per quad between consecutive point-pairs

    this.verts  = new Float32Array(maxVerts * 3);
    this.colors = new Float32Array(maxVerts * 3);
    this.uvs    = new Float32Array(maxVerts * 2);
    this.idx    = new Uint32Array(maxTris * 3);

    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.verts,  3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('color',    new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('uv',       new THREE.BufferAttribute(this.uvs,    2).setUsage(THREE.DynamicDrawUsage));
    this.geo.setIndex(new THREE.BufferAttribute(this.idx, 1).setUsage(THREE.DynamicDrawUsage));

    this.mesh = new THREE.Mesh(this.geo, makeMat());
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2; // draw over opaque geometry
    scene.add(this.mesh);
  }

  setLoopMode(on: boolean): void {
    this.loopMode = on;
    this.limit = on ? LOOP_LEN : NORMAL_LEN;
  }

  // Attach a camera so the ribbon can billboard toward the viewer.
  // If not attached the ribbon uses a fixed Y-up offset (still visible, less ideal).
  setCamera(cam: THREE.Camera): void {
    this.cam = cam;
  }

  clear(): void {
    this.buf.length = 0;
    this.geo.setDrawRange(0, 0);
  }

  // chime ∈ [0,1]: 1 = ringing true (brilliant cyan-white), 0 = clattered (dim grey)
  push(x: number, y: number, z: number, chime: number): void {
    this.buf.push({ x, y, z, c: chime });
    if (this.buf.length > this.limit) this.buf.shift();
    this._rebuild();
  }

  private _rebuild(): void {
    const n = this.buf.length;
    if (n < 2) { this.geo.setDrawRange(0, 0); return; }

    const halfW = this.loopMode ? HALF_W_LOOP : HALF_W_BASE;

    // We need a "right" vector perpendicular to the trail direction at each point.
    // Strategy: compute tangent at each sample, cross with view direction.
    const camDir = new THREE.Vector3(0, 0, 1); // fallback
    if (this.cam) {
      this.cam.getWorldDirection(camDir);
    }

    const tan  = new THREE.Vector3();
    const perp = new THREE.Vector3();
    const a    = new THREE.Vector3();
    const b    = new THREE.Vector3();

    let quadCount = 0;

    for (let i = 0; i < n; i++) {
      const p = this.buf[i];
      const age = i / (n - 1);  // 0 = oldest, 1 = newest

      // Tangent: central difference where possible
      if (i === 0) {
        const q = this.buf[1];
        tan.set(q.x - p.x, q.y - p.y, q.z - p.z);
      } else if (i === n - 1) {
        const q = this.buf[n - 2];
        tan.set(p.x - q.x, p.y - q.y, p.z - q.z);
      } else {
        const prev = this.buf[i - 1];
        const next = this.buf[i + 1];
        tan.set(next.x - prev.x, next.y - prev.y, next.z - prev.z);
      }
      tan.normalize();

      // Perpendicular: tan × camDir, then normalise
      perp.crossVectors(tan, camDir).normalize();
      if (perp.lengthSq() < 1e-6) {
        // degenerate — fall back to world-Y cross
        perp.crossVectors(tan, THREE.Object3D.DEFAULT_UP).normalize();
      }

      // Width taper: fat at newest (age=1), whisper-thin at oldest (age=0)
      // Use a gentle curve so middle of arc is still readable
      const widthFactor = 0.15 + 0.85 * Math.pow(age, 0.6);
      const hw = halfW * widthFactor;

      // Vertex positions
      const vi = i * 2; // vertex pair index
      const pi = vi * 3;

      a.set(p.x + perp.x * hw, p.y + perp.y * hw, p.z + perp.z * hw);
      b.set(p.x - perp.x * hw, p.y - perp.y * hw, p.z - perp.z * hw);

      this.verts[pi + 0] = a.x; this.verts[pi + 1] = a.y; this.verts[pi + 2] = a.z;
      this.verts[pi + 3] = b.x; this.verts[pi + 4] = b.y; this.verts[pi + 5] = b.z;

      // Colour: blend between grey (c=0) and cyan-white (c=1)
      const t = p.c; // chime
      const cr = FLAT_R + (TRUE_R - FLAT_R) * t;
      const cg = FLAT_G + (TRUE_G - FLAT_G) * t;
      const cb = FLAT_B + (TRUE_B - FLAT_B) * t;

      // Brightness: oldest=dim, newest=bright; in loop mode blast it
      const loopBoost = this.loopMode ? 1.6 : 1.0;
      // Fade floor: 0.08 so the old tail is faint but not gone
      const bright = (0.08 + 0.92 * Math.pow(age, 0.8)) * loopBoost;

      this.colors[pi + 0] = cr * bright;  this.colors[pi + 1] = cg * bright;  this.colors[pi + 2] = cb * bright;
      this.colors[pi + 3] = cr * bright;  this.colors[pi + 4] = cg * bright;  this.colors[pi + 5] = cb * bright;

      // UVs
      const uvi = vi * 2;
      this.uvs[uvi + 0] = 0; this.uvs[uvi + 1] = age;
      this.uvs[uvi + 2] = 1; this.uvs[uvi + 3] = age;

      // Emit two triangles for each quad between (i-1) and (i)
      if (i > 0) {
        const q = quadCount * 6;
        const base = (i - 1) * 2;
        // quad: (base, base+1, base+2), (base+1, base+3, base+2)
        this.idx[q + 0] = base + 0;
        this.idx[q + 1] = base + 1;
        this.idx[q + 2] = base + 2;
        this.idx[q + 3] = base + 1;
        this.idx[q + 4] = base + 3;
        this.idx[q + 5] = base + 2;
        quadCount++;
      }
    }

    const triCount = quadCount * 2;
    this.geo.setDrawRange(0, triCount * 3);
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.color.needsUpdate    = true;
    this.geo.index!.needsUpdate              = true;
  }
}

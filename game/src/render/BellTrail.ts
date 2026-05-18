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
// Widened so the bell path is unmistakably the boldest stroke in the
// chamber: now that rig tethers are thin (~0.13 m radius) the trail must
// clearly out-mass them. The bell is the one object everyone plays for.
const HALF_W_BASE = 0.20; // half-width of ribbon at newest (metres — tune to ring scale)
const HALF_W_LOOP = 0.34; // wider in loop mode for extra drama
// Speed (m/s, derived from sample spacing × assumed push cadence) at which
// the trail reaches its full blaze width/heat. Tunable — lower = blazes
// sooner. The push() cadence is the frame rate (~60 Hz) so we treat the
// per-sample step distance directly as a speed proxy.
const BLAZE_FULL_STEP = 0.55; // metres between consecutive samples = "fast"
const HALF_W_SPEED_GAIN = 1.9; // taut-fast ribbon up to ~1.9× wide

// Palette (linear sRGB, not gamma — three.js colour math is linear)
//   chime=1  → brilliant cyan-white  #1aa6b7 saturated + blown toward white
//   chime=0  → muted grey            desaturated, dimmed
const TRUE_R = 0.18, TRUE_G = 0.85, TRUE_B = 1.00; // cyan-white
const FLAT_R = 0.28, FLAT_G = 0.30, FLAT_B = 0.32; // cool grey

// Hot-path scratch (push()/_rebuild() runs every frame — zero per-call alloc).
const _camDir = new THREE.Vector3();
const _tan    = new THREE.Vector3();
const _perp   = new THREE.Vector3();
const _va     = new THREE.Vector3();
const _vb     = new THREE.Vector3();

// ── material ─────────────────────────────────────────────────────────────────
// Vertex-coloured, additive-blended mesh.  No lighting — colours ARE the light.
function makeMat(): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({
    vertexColors: true,
    transparent: true,
    // Higher than the rig tethers (which are now thin & ≤0.62 opacity, non-
    // additive) so the bell trail is always the dominant glow on screen.
    opacity: 0.9, // visible ribbon (bloom is tamed so this no longer whites out)
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
    const camDir = _camDir.set(0, 0, 1); // fallback
    if (this.cam) {
      this.cam.getWorldDirection(camDir);
    }

    const tan  = _tan;
    const perp = _perp;
    const a    = _va;
    const b    = _vb;

    let quadCount = 0;

    for (let i = 0; i < n; i++) {
      const p = this.buf[i];
      const age = i / (n - 1);  // 0 = oldest, 1 = newest

      // Per-sample SPEED proxy: distance to the neighbouring sample. push()
      // is called at frame cadence, so a bigger step = a faster bell. This
      // makes the ribbon visibly BLAZE wider/hotter exactly where it's fast
      // (a hard swing or a developing Loop) and stay slim where it dawdles —
      // pure geometry, zero sim/determinism coupling.
      let step: number;
      if (i === n - 1) {
        const q = this.buf[n - 2];
        step = Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z);
      } else {
        const q = this.buf[i + 1];
        step = Math.hypot(q.x - p.x, q.y - p.y, q.z - p.z);
      }
      const spd = Math.min(1, step / BLAZE_FULL_STEP); // 0..1 blaze factor

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

      // Width taper: fat at newest (age=1), whisper-thin at oldest (age=0).
      // Speed swells the ribbon so a fast bell carves a bold sweeping arc.
      const widthFactor = 0.15 + 0.85 * Math.pow(age, 0.6);
      const speedSwell = 1 + spd * (HALF_W_SPEED_GAIN - 1) * (0.4 + 0.6 * age);
      const hw = halfW * widthFactor * speedSwell;

      // Vertex positions
      const vi = i * 2; // vertex pair index
      const pi = vi * 3;

      a.set(p.x + perp.x * hw, p.y + perp.y * hw, p.z + perp.z * hw);
      b.set(p.x - perp.x * hw, p.y - perp.y * hw, p.z - perp.z * hw);

      this.verts[pi + 0] = a.x; this.verts[pi + 1] = a.y; this.verts[pi + 2] = a.z;
      this.verts[pi + 3] = b.x; this.verts[pi + 4] = b.y; this.verts[pi + 5] = b.z;

      // Colour: blend between grey (c=0) and cyan-white (c=1). Speed then
      // shifts the HOT (recent) part of the trail toward incandescent white
      // so a fast bell reads as a blazing streak — this carries the drama via
      // HUE/whiteness, NOT raw magnitude, which keeps additive bloom in check.
      const t = p.c; // chime
      let cr = FLAT_R + (TRUE_R - FLAT_R) * t;
      let cg = FLAT_G + (TRUE_G - FLAT_G) * t;
      let cb = FLAT_B + (TRUE_B - FLAT_B) * t;
      const heat = spd * Math.pow(age, 0.7);     // only the recent fast part
      cr += (1.0 - cr) * heat * 0.55;
      cg += (1.0 - cg) * heat * 0.35;            // bias toward cyan-white
      cb += (1.0 - cb) * heat * 0.20;

      // Brightness: oldest=dim, newest=bright; loop blasts it. A modest
      // speed lift sharpens the leading edge. Hard-capped well under a
      // white-out: max ≈ 1.0·1.6(loop)·1.18(speed) ≈ 1.9 on a partly-
      // desaturated colour through opacity-0.78 additive — bloom-safe at 46 m
      // (no increase vs the pre-existing 1.6 loop ceiling territory).
      const loopBoost = this.loopMode ? 1.6 : 1.0;
      const speedLift = 1 + spd * 0.18 * Math.pow(age, 1.2);
      // In loop mode the WHOLE closed orbit should glow evenly (a held ring,
      // not a fading comet): lift the floor and flatten the age curve so the
      // arc reads as one continuous gorgeous loop. Normal mode keeps the
      // comet taper (fade floor 0.08) for the developing-arc money shot.
      const floor = this.loopMode ? 0.45 : 0.08;
      const ageCurve = this.loopMode ? Math.pow(age, 0.35) : Math.pow(age, 0.8);
      const bright = Math.min(
        1.9,
        (floor + (1 - floor) * ageCurve) * loopBoost * speedLift,
      );

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

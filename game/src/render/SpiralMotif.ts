// SpiralMotif — "instant replay as graphic design."
// On score, the bell's roulette-curve trail freezes in space and transitions
// from live cyan to warm copper (#d4602a), pulses white at the instant of
// score, then fades to nothing. For a Loop (7pt) the motif lingers longer
// and rotates slowly — the closed arc IS the dossier's founding trick motif.

import * as THREE from 'three';

export type ScoreKind = 'fall' | 'rise' | 'loop' | 'curl' | 'ground';

interface TrailSample { x: number; y: number; z: number; c: number }

// ── Palette (linear sRGB) ────────────────────────────────────────────────────
// Copper accent from dossier: #d4602a → linear
const COPPER_R = 0.65, COPPER_G = 0.16, COPPER_B = 0.04;
// Starting cyan (same as BellTrail TRUE colour)
const CYAN_R = 0.18, CYAN_G = 0.85, CYAN_B = 1.00;
// Flash white
const WHITE = 1.0;

// ── Timing ───────────────────────────────────────────────────────────────────
const FLASH_DUR    = 0.12; // white flash duration (seconds)
const MORPH_DUR    = 0.5;  // cyan → copper transition
const NORMAL_LIFE  = 2.5;  // total visible time for non-loop scores
const LOOP_LIFE    = 4.5;  // loop scores linger longer
const FADE_PORTION = 0.6;  // last 60% of life is fade-out

// ── Geometry ─────────────────────────────────────────────────────────────────
const HALF_W = 0.30; // 1.5× the BellTrail HALF_W_BASE (0.20)
const MAX_SAMPLES = 600;

// Scratch vectors (hot path, zero alloc)
const _tan  = new THREE.Vector3();
const _perp = new THREE.Vector3();
const _camDir = new THREE.Vector3();

export class SpiralMotif {
  readonly mesh: THREE.Mesh;

  private geo: THREE.BufferGeometry;
  private mat: THREE.MeshBasicMaterial;
  private verts:  Float32Array;
  private colors: Float32Array;
  private idx:    Uint32Array;

  private active = false;
  private elapsed = 0;
  private life = NORMAL_LIFE;
  private isLoop = false;
  private sampleCount = 0;
  private cam: THREE.Camera | null = null;

  // Snapshot of colours at each vertex (base cyan blended with chime), used
  // to animate the colour morph from cyan → copper.
  private baseColors: Float32Array;

  constructor(scene: THREE.Scene, camera?: THREE.Camera) {
    this.cam = camera ?? null;

    const maxVerts = MAX_SAMPLES * 2;
    const maxTris  = (MAX_SAMPLES - 1) * 2;

    this.verts  = new Float32Array(maxVerts * 3);
    this.colors = new Float32Array(maxVerts * 3);
    this.baseColors = new Float32Array(maxVerts * 3);
    this.idx    = new Uint32Array(maxTris * 3);

    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.verts,  3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('color',    new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setIndex(new THREE.BufferAttribute(this.idx, 1).setUsage(THREE.DynamicDrawUsage));

    this.mat = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(this.geo, this.mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 3; // above the live trail (renderOrder 2)
    this.mesh.visible = false;
    scene.add(this.mesh);
  }

  setCamera(cam: THREE.Camera): void { this.cam = cam; }

  /** Trigger the motif — call with a copy of the bell trail buffer. */
  trigger(path: TrailSample[], kind: ScoreKind): void {
    if (path.length < 3) return;

    // Truncate to max
    const samples = path.length > MAX_SAMPLES ? path.slice(path.length - MAX_SAMPLES) : path;
    this.sampleCount = samples.length;
    this.isLoop = kind === 'loop';
    this.life = this.isLoop ? LOOP_LIFE : NORMAL_LIFE;
    this.elapsed = 0;
    this.active = true;
    this.mesh.visible = true;
    this.mesh.scale.setScalar(1.0);

    // Build the ribbon geometry (same technique as BellTrail but static)
    const camDir = _camDir.set(0, 0, 1);
    if (this.cam) this.cam.getWorldDirection(camDir);

    const n = samples.length;
    let quadCount = 0;

    for (let i = 0; i < n; i++) {
      const p = samples[i];
      const age = i / (n - 1); // 0=oldest, 1=newest

      // Tangent
      if (i === 0) {
        const q = samples[1];
        _tan.set(q.x - p.x, q.y - p.y, q.z - p.z);
      } else if (i === n - 1) {
        const q = samples[n - 2];
        _tan.set(p.x - q.x, p.y - q.y, p.z - q.z);
      } else {
        const prev = samples[i - 1];
        const next = samples[i + 1];
        _tan.set(next.x - prev.x, next.y - prev.y, next.z - prev.z);
      }
      _tan.normalize();

      // Perpendicular
      _perp.crossVectors(_tan, camDir).normalize();
      if (_perp.lengthSq() < 1e-6) {
        _perp.crossVectors(_tan, THREE.Object3D.DEFAULT_UP).normalize();
      }

      // Width: slightly tapered (thicker at newest end)
      const widthFactor = 0.3 + 0.7 * Math.pow(age, 0.5);
      const hw = HALF_W * widthFactor;

      const vi = i * 2;
      const pi = vi * 3;
      this.verts[pi + 0] = p.x + _perp.x * hw;
      this.verts[pi + 1] = p.y + _perp.y * hw;
      this.verts[pi + 2] = p.z + _perp.z * hw;
      this.verts[pi + 3] = p.x - _perp.x * hw;
      this.verts[pi + 4] = p.y - _perp.y * hw;
      this.verts[pi + 5] = p.z - _perp.z * hw;

      // Base colour: start as the trail's live colour (cyan blended with chime)
      const t = p.c;
      const cr = 0.28 + (CYAN_R - 0.28) * t;
      const cg = 0.30 + (CYAN_G - 0.30) * t;
      const cb = 0.32 + (CYAN_B - 0.32) * t;
      // Brightness: uniform-ish (not the age-tapered comet of the live trail)
      const bright = 0.6 + 0.4 * Math.pow(age, 0.4);
      this.baseColors[pi + 0] = cr * bright;
      this.baseColors[pi + 1] = cg * bright;
      this.baseColors[pi + 2] = cb * bright;
      this.baseColors[pi + 3] = cr * bright;
      this.baseColors[pi + 4] = cg * bright;
      this.baseColors[pi + 5] = cb * bright;

      // Index buffer (quads)
      if (i > 0) {
        const q = quadCount * 6;
        const base = (i - 1) * 2;
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
    this.geo.index!.needsUpdate = true;

    // Initial colour update (will be flash white on first frame)
    this._updateColors();
  }

  /** Call every frame with dt (seconds). */
  update(dt: number): void {
    if (!this.active) return;

    this.elapsed += dt;
    if (this.elapsed >= this.life) {
      this.active = false;
      this.mesh.visible = false;
      return;
    }

    this._updateColors();

    // Loop scores: gentle scale pulse — the closed arc "breathes" once before
    // fading, giving it the held-ring presence of the dossier motif.
    if (this.isLoop) {
      const pulse = 1.0 + 0.04 * Math.sin(this.elapsed * 4.0);
      this.mesh.scale.setScalar(pulse);
    } else {
      this.mesh.scale.setScalar(1.0);
    }
  }

  private _updateColors(): void {
    const t = this.elapsed;
    const n = this.sampleCount;

    // Phase: flash → morph → hold → fade
    const progress = t / this.life; // 0..1
    const fadeStart = 1.0 - FADE_PORTION;
    const opacity = progress < fadeStart ? 1.0 : 1.0 - (progress - fadeStart) / FADE_PORTION;

    // Flash intensity: bright white spike at score moment
    const flash = t < FLASH_DUR ? 1.0 - (t / FLASH_DUR) : 0;

    // Morph: 0 = still cyan, 1 = fully copper
    const morph = Math.min(1, Math.max(0, (t - FLASH_DUR) / MORPH_DUR));

    // Loop brightness boost
    const loopBoost = this.isLoop ? 1.4 : 1.0;

    for (let i = 0; i < n; i++) {
      const vi = i * 2;
      const pi = vi * 3;

      for (let v = 0; v < 2; v++) {
        const off = pi + v * 3;
        const br = this.baseColors[off + 0];
        const bg = this.baseColors[off + 1];
        const bb = this.baseColors[off + 2];

        // Morph from base (cyan-ish) toward copper
        let r = br + (COPPER_R - br) * morph;
        let g = bg + (COPPER_G - bg) * morph;
        let b = bb + (COPPER_B - bb) * morph;

        // Flash: additive white spike
        r += (WHITE - r) * flash;
        g += (WHITE - g) * flash;
        b += (WHITE - b) * flash;

        // Apply opacity fade and loop boost
        const scale = opacity * loopBoost;
        this.colors[off + 0] = r * scale;
        this.colors[off + 1] = g * scale;
        this.colors[off + 2] = b * scale;
      }
    }

    this.geo.attributes.color.needsUpdate = true;
  }
}

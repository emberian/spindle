// Cel / graphic-novel toon material + ink-outline helpers for RIG riggers.
// Render-only. No sim impact. Reuses a small number of shared resources so the
// whole roster (~8 figures) costs almost nothing per frame.
//
// Technique:
//  - Flat team-colour fill with a 3-band quantised gradient ramp
//    (THREE.MeshToonMaterial + a tiny DataTexture gradientMap) → the classic
//    graphic-novel banded shade with zero shader authoring.
//  - Bold black INK OUTLINE via the cheap inverted-hull trick: a duplicate of
//    each mesh, scaled out along normals, drawn BackSide in flat black. This
//    is the standard cheap toon outline and reads as a clean ink edge at any
//    distance, including the tiny silhouettes the spectate camera uses.

import * as THREE from 'three';

// Ink colour for outlines / linework — near-black, very slightly warm so it
// sits in the #11131a world without going pure void.
export const INK = 0x0a0b10;

// Outline thickness as a fraction of figure scale. Tuned so it reads as a
// confident ink line at cinematic distance without bloating the silhouette
// up close.
export const OUTLINE_SCALE = 0.055;

// ── Shared 3-band toon gradient ramp ─────────────────────────────────────────
// A 3-texel ramp → 3 hard shading bands (shadow / mid / light). Point-sampled
// so the bands stay crisp (no interpolation = true cel look).
let _gradientMap: THREE.DataTexture | null = null;

export function toonGradient(): THREE.DataTexture {
  if (_gradientMap) return _gradientMap;
  const data = new Uint8Array([90, 175, 255]); // 3 bands, single channel
  const tex = new THREE.DataTexture(data, 3, 1, THREE.RedFormat);
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  _gradientMap = tex;
  return tex;
}

// ── Materials ────────────────────────────────────────────────────────────────

/** A cel-shaded fill material (flat colour + 3 hard bands). */
export function makeToonMaterial(color: number): THREE.MeshToonMaterial {
  return new THREE.MeshToonMaterial({
    color,
    gradientMap: toonGradient(),
  });
}

// One shared outline material — flat black, back-faces only. Shared across
// every outline mesh in the scene (colour never varies).
let _outlineMat: THREE.MeshBasicMaterial | null = null;
export function outlineMaterial(): THREE.MeshBasicMaterial {
  if (_outlineMat) return _outlineMat;
  _outlineMat = new THREE.MeshBasicMaterial({
    color: INK,
    side: THREE.BackSide,
  });
  return _outlineMat;
}

// ── Inked part: a fill mesh + its inverted-hull outline shell ────────────────

/**
 * Build an "inked" part: the cel fill mesh plus a slightly-inflated black
 * back-face shell that reads as a bold ink outline. Returns a group so the
 * caller can pose the part as one unit. Geometry is shared between fill and
 * shell (outline shell just renders the same geo at BackSide, scaled out).
 */
export class InkedPart {
  readonly group = new THREE.Group();
  readonly fill: THREE.Mesh;
  private readonly shell: THREE.Mesh;

  constructor(geo: THREE.BufferGeometry, color: number, outline = OUTLINE_SCALE) {
    this.fill = new THREE.Mesh(geo, makeToonMaterial(color));
    this.shell = new THREE.Mesh(geo, outlineMaterial());
    // Inflate the shell uniformly. For the chunky primitives we use this reads
    // as an even ink line; it avoids per-vertex normal-extrusion shader work.
    const s = 1 + outline;
    this.shell.scale.set(s, s, s);
    this.shell.renderOrder = -1; // draw outline before fill
    this.group.add(this.shell, this.fill);
  }

  setColor(hex: number): void {
    (this.fill.material as THREE.MeshToonMaterial).color.setHex(hex);
  }

  setEmissive(hex: number, intensity: number): void {
    const m = this.fill.material as THREE.MeshToonMaterial;
    m.emissive.setHex(hex);
    m.emissiveIntensity = intensity;
  }
}

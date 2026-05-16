// The money shot: the bell's Coriolis path drawn as a glowing ribbon whose
// colour/intensity tracks spin-trueness. A clean (ringing) bell lays a
// saturated cyan stripe; a clatter desaturates the fresh segments to grey —
// you SEE the moment it goes off true, exactly when you hear it. On a Loop
// the trail extends to hold the whole closed arc and blooms.

import * as THREE from 'three';

const NORMAL = 240;
const LOOP = 420;

export class BellTrail {
  readonly line: THREE.Line;
  private geo = new THREE.BufferGeometry();
  private pos: Float32Array;
  private col: Float32Array;
  private cap = LOOP;
  private buf: { x: number; y: number; z: number; c: number }[] = [];
  private limit = NORMAL;

  constructor(scene: THREE.Scene) {
    this.pos = new Float32Array(this.cap * 3);
    this.col = new Float32Array(this.cap * 3);
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    this.line = new THREE.Line(
      this.geo,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    this.line.frustumCulled = false;
    scene.add(this.line);
  }

  setLoopMode(on: boolean): void {
    this.limit = on ? LOOP : NORMAL;
  }

  clear(): void {
    this.buf.length = 0;
    this.geo.setDrawRange(0, 0);
  }

  // chime ∈ [0,1]: 1 = ringing true (bright cyan), 0 = clattered (dim grey)
  push(x: number, y: number, z: number, chime: number): void {
    this.buf.push({ x, y, z, c: chime });
    if (this.buf.length > this.limit) this.buf.shift();
    const n = this.buf.length;
    for (let i = 0; i < n; i++) {
      const p = this.buf[i];
      const age = i / n; // 0 oldest → 1 newest
      const j = i * 3;
      this.pos[j] = p.x;
      this.pos[j + 1] = p.y;
      this.pos[j + 2] = p.z;
      // true → (0.10,0.65,0.72) cyan ; clattered → (0.30,0.31,0.34) grey
      const t = p.c;
      const r = (0.34 - 0.24 * t) * 1.7;
      const g = (0.34 + 0.42 * t) * 1.7;
      const b = (0.38 + 0.50 * t) * 1.7;
      const fade = 0.45 + 0.55 * age; // brighter floor so the whole arc reads
      this.col[j] = r * fade;
      this.col[j + 1] = g * fade;
      this.col[j + 2] = b * fade;
    }
    this.geo.setDrawRange(0, n);
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.color.needsUpdate = true;
  }
}

// RiggerTrail — fading trajectory lines behind each player in the 3D view.
// Lightweight: one THREE.Line per player with a ring buffer of positions.
// Team-colored, fading from bright (recent) to transparent (old).

import * as THREE from 'three';
import type { PlayerSim } from '../sim/types';

const TRAIL_LEN = 90; // samples (~1.5s at 60fps push rate)
const MAX_PLAYERS = 8;

// Home = blue, Away = orange-red
const HOME_COLOR = new THREE.Color(0x3388ff);
const AWAY_COLOR = new THREE.Color(0xff5533);

class PlayerTrail {
  readonly line: THREE.Line;
  private positions: Float32Array;
  private colors: Float32Array;
  private geo: THREE.BufferGeometry;
  private head = 0;
  private count = 0;
  private team: 'home' | 'away' = 'home';

  constructor(group: THREE.Group) {
    this.positions = new Float32Array(TRAIL_LEN * 3);
    this.colors = new Float32Array(TRAIL_LEN * 3);
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('color', new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage));

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      depthWrite: false,
    });
    this.line = new THREE.Line(this.geo, mat);
    this.line.frustumCulled = false;
    this.line.visible = false;
    this.line.renderOrder = 1;
    group.add(this.line);
  }

  push(x: number, y: number, z: number, team: 'home' | 'away') {
    this.team = team;
    const i = this.head * 3;
    this.positions[i] = x;
    this.positions[i + 1] = y;
    this.positions[i + 2] = z;
    this.head = (this.head + 1) % TRAIL_LEN;
    if (this.count < TRAIL_LEN) this.count++;
    this.rebuild();
  }

  private rebuild() {
    if (this.count < 2) {
      this.line.visible = false;
      return;
    }
    this.line.visible = true;
    const baseColor = this.team === 'home' ? HOME_COLOR : AWAY_COLOR;

    // Write positions and colors in order from oldest to newest
    const posAttr = this.geo.getAttribute('position') as THREE.BufferAttribute;
    const colAttr = this.geo.getAttribute('color') as THREE.BufferAttribute;
    const pos = posAttr.array as Float32Array;
    const col = colAttr.array as Float32Array;

    for (let k = 0; k < this.count; k++) {
      // oldest is at (head - count + TRAIL_LEN) % TRAIL_LEN
      const srcIdx = ((this.head - this.count + k + TRAIL_LEN) % TRAIL_LEN) * 3;
      const dstIdx = k * 3;
      pos[dstIdx] = this.positions[srcIdx];
      pos[dstIdx + 1] = this.positions[srcIdx + 1];
      pos[dstIdx + 2] = this.positions[srcIdx + 2];

      // Fade: 0 (oldest) → 1 (newest)
      const alpha = k / (this.count - 1);
      col[dstIdx] = baseColor.r * alpha;
      col[dstIdx + 1] = baseColor.g * alpha;
      col[dstIdx + 2] = baseColor.b * alpha;
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    this.geo.setDrawRange(0, this.count);
  }

  hide() {
    this.line.visible = false;
    this.count = 0;
    this.head = 0;
  }
}

export class RiggerTrails {
  private group = new THREE.Group();
  private trails: PlayerTrail[] = [];

  constructor(scene: THREE.Scene) {
    for (let i = 0; i < MAX_PLAYERS; i++) {
      this.trails.push(new PlayerTrail(this.group));
    }
    scene.add(this.group);
  }

  sync(players: PlayerSim[]) {
    for (let i = 0; i < MAX_PLAYERS; i++) {
      if (i < players.length) {
        const p = players[i];
        this.trails[i].push(p.p.x, p.p.y, p.p.z, p.team);
      } else {
        this.trails[i].hide();
      }
    }
  }
}

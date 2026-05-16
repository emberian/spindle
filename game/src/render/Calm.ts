// The calm, rendered: the interior of the rotating cylinder. World +X is the
// spin axis / 640 m length; (y,z) is the rotating cross-section. We render in
// the CO-ROTATING frame (geometry static) and sell the spin with a scrolling
// skin texture — no nausea, and it matches the sim frame exactly.

import * as THREE from 'three';
import { REG, GATE_X } from '../sim/RegConstants';

const CYAN = 0x1aa6b7;
const ORANGE = 0xd4602a;

function skinTexture(): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 512;
  const g = c.getContext('2d')!;
  g.fillStyle = '#161922';
  g.fillRect(0, 0, 64, 512);
  // six longitudinal stripes: alternating "land" and faint "window" glow
  for (let i = 0; i < 6; i++) {
    const x = (i / 6) * 64;
    g.fillStyle = i % 2 ? 'rgba(26,166,183,0.10)' : '#1f2433';
    g.fillRect(x, 0, 64 / 6, 512);
  }
  // faint lane ticks down the length
  g.strokeStyle = 'rgba(107,112,121,0.18)';
  for (let y = 0; y < 512; y += 32) {
    g.beginPath();
    g.moveTo(0, y);
    g.lineTo(64, y);
    g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(8, 1);
  return t;
}

export class Calm {
  readonly group = new THREE.Group();
  private skinTex = skinTexture();
  private skin: THREE.Mesh;

  constructor(scene: THREE.Scene) {
    scene.fog = new THREE.Fog(0x11131a, REG.L * 0.45, REG.L * 1.25);

    // Skin: open cylinder, length along world X, viewed from inside.
    this.skin = new THREE.Mesh(
      new THREE.CylinderGeometry(REG.R, REG.R, REG.L, 96, 48, true),
      new THREE.MeshStandardMaterial({
        map: this.skinTex,
        side: THREE.BackSide,
        roughness: 1,
        metalness: 0,
        emissive: 0x0a0c12,
      }),
    );
    this.skin.rotation.z = Math.PI / 2; // cylinder axis -> world X
    this.group.add(this.skin);

    // Spars: clip-points down the spine (one instanced draw call).
    const sparGeo = new THREE.CapsuleGeometry(0.5, 6, 4, 8);
    const sparMat = new THREE.MeshStandardMaterial({
      color: 0x3a4150,
      emissive: 0x141a24,
    });
    const rings = 16;
    const around = 3;
    const spars = new THREE.InstancedMesh(sparGeo, sparMat, rings * around);
    const m = new THREE.Matrix4();
    let idx = 0;
    for (let i = 0; i < rings; i++) {
      const x = -GATE_X + ((i + 0.5) / rings) * REG.L;
      for (let a = 0; a < around; a++) {
        const ang = (a / around) * Math.PI * 2;
        const r = REG.R * 0.62;
        m.makeBasis(
          new THREE.Vector3(1, 0, 0),
          new THREE.Vector3(0, Math.cos(ang), Math.sin(ang)),
          new THREE.Vector3(0, -Math.sin(ang), Math.cos(ang)),
        );
        m.setPosition(x, Math.cos(ang) * r, Math.sin(ang) * r);
        spars.setMatrixAt(idx++, m);
      }
    }
    this.group.add(spars);

    // Goal rings + faint gate discs.
    this.group.add(this.ring(GATE_X, CYAN)); // Faith (+x)
    this.group.add(this.ring(-GATE_X, ORANGE)); // Free (−x)
    for (const gx of [-GATE_X * 0.5, 0, GATE_X * 0.5]) {
      const gate = new THREE.Mesh(
        new THREE.TorusGeometry(REG.R * 0.92, 0.4, 6, 64),
        new THREE.MeshBasicMaterial({ color: 0x6b7079, transparent: true, opacity: 0.12 }),
      );
      gate.position.x = gx;
      gate.rotation.y = Math.PI / 2;
      this.group.add(gate);
    }

    // Lighting: a cool axial key + dim ambient (the "sunline").
    const key = new THREE.DirectionalLight(0xbfd6e0, 1.1);
    key.position.set(0, REG.R, REG.L * 0.3);
    scene.add(key);
    scene.add(new THREE.AmbientLight(0x223040, 0.6));

    scene.add(this.group);
  }

  private ring(x: number, color: number): THREE.Mesh {
    const mesh = new THREE.Mesh(
      new THREE.TorusGeometry(REG.gateRadius, 0.55, 16, 64),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.4 }),
    );
    mesh.position.x = x;
    mesh.rotation.y = Math.PI / 2;
    return mesh;
  }

  // Sell the spin: scroll the skin texture at the habitat rate.
  update(dt: number): void {
    this.skinTex.offset.y = (this.skinTex.offset.y + dt * REG.omega * 0.16) % 1;
  }
}

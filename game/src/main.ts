// P0 boot: prove the full pipeline (Vite + TS + three.js + the loop) with a
// minimal but genuinely-3D placeholder of the calm. P1 replaces this with the
// real Scene/Calm. Sim modules are intentionally NOT imported by render yet.

import * as THREE from 'three';
import { GameRuntime } from './core/GameRuntime';
import { EventBus } from './core/EventBus';
import { REG } from './sim/RegConstants';

const app = document.getElementById('app')!;
const boot = document.getElementById('boot');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x11131a, 1);
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x11131a, REG.L * 0.4, REG.L * 1.2);

const camera = new THREE.PerspectiveCamera(62, 1, 0.1, 4000);
camera.position.set(REG.R * 0.9, REG.R * 0.5, REG.L * 0.55);
camera.lookAt(0, 0, 0);

// Placeholder calm: a wireframe cylinder along +X (the spin axis / 640 m),
// two ring glows at the Faith (+x, cyan) and Free (−x, orange) ends.
const skin = new THREE.Mesh(
  new THREE.CylinderGeometry(REG.R, REG.R, REG.L, 64, 1, true),
  new THREE.MeshBasicMaterial({ color: 0x2a2d38, wireframe: true, side: THREE.BackSide }),
);
skin.rotation.z = Math.PI / 2; // cylinder length -> world X
scene.add(skin);

const ring = (x: number, color: number) => {
  const m = new THREE.Mesh(
    new THREE.TorusGeometry(REG.gateRadius, 0.6, 12, 48),
    new THREE.MeshBasicMaterial({ color }),
  );
  m.position.x = x;
  m.rotation.y = Math.PI / 2;
  scene.add(m);
  return m;
};
ring(REG.L / 2, 0x1aa6b7); // Faith
ring(-REG.L / 2, 0xd4602a); // Free

const events = new EventBus();
window.addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  events.emit('resize', { width: innerWidth, height: innerHeight });
});
renderer.setSize(innerWidth, innerHeight);
camera.aspect = innerWidth / innerHeight;
camera.updateProjectionMatrix();

let t = 0;
const runtime = new GameRuntime(
  (dt) => {
    t += dt;
    // slow orbit so the 3D-ness is unmistakable in the P0 placeholder
    camera.position.x = Math.cos(t * 0.15) * REG.R * 1.1;
    camera.position.z = Math.sin(t * 0.15) * REG.L * 0.55 + REG.L * 0.2;
    camera.position.y = REG.R * 0.45;
    camera.lookAt(0, 0, 0);
    skin.rotation.x += dt * REG.omega; // sells the spin (P1: scrolling texture)
  },
  () => renderer.render(scene, camera),
);
runtime.start();

if (boot) boot.remove();
console.info('RIG v2 P0 — pipeline up. ω=%s R=%s L=%s', REG.omega, REG.R, REG.L);

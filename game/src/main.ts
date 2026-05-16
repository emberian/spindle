// P2 boot: the deterministic SimWorld driving the calm, the chime-driven
// money-shot trail, bloom, and the Loop moment — slow-mo + scene-dim + a
// down-the-spine loop-cam + the crowd hush. A scripted rigger fires a
// canonical Loop on a cycle so the signature play is always on screen.

import * as THREE from 'three';
import { GameRuntime } from './core/GameRuntime';
import { FixedStepDriver, SIM_H } from './core/FixedStepDriver';
import { Calm } from './render/Calm';
import { BellTrail } from './render/BellTrail';
import { PostFX } from './render/PostFX';
import { AudioEngine } from './audio/AudioEngine';
import { SimWorld } from './sim/SimWorld';
import { REG } from './sim/RegConstants';
import type { InputFrame } from './sim/types';

const app = document.getElementById('app')!;
document.getElementById('boot')?.remove();

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x11131a, 1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const calm = new Calm(scene);
const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 4000);
const trail = new BellTrail(scene);
let post = new PostFX(renderer, scene, camera);
const audio = new AudioEngine();
addEventListener('pointerdown', () => audio.start(), { once: true });
addEventListener('keydown', () => audio.start(), { once: true });

const bellMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.95, 24, 16),
  new THREE.MeshStandardMaterial({ color: 0xe8e0c8, emissive: 0x1aa6b7, emissiveIntensity: 1 }),
);
scene.add(bellMesh);

function resize(): void {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  post.setSize(innerWidth, innerHeight);
}
addEventListener('resize', resize);
resize();

// ── Scripted demo: P1 holds, fires a canonical Loop, recycle ──────────────
const REL = { x: -50, y: 2, z: 0 };
function makeWorld(): SimWorld {
  const w = new SimWorld(1);
  w.addPlayer('P1', 'home', 'spinner', { ...REL });
  w.addPlayer('A1', 'away', 'reach', { x: 305, y: 0, z: 0 });
  w.bellHeldBy = 'P1';
  return w;
}
let world = makeWorld();
let simTick = 0;
const THROW_TICK = 60; // ~0.25 s after spawn
function frameFor(tick: number): InputFrame {
  return {
    tick,
    players: [
      {
        id: 'P1',
        aim: { x: 0.30, y: 0.62, z: -0.72 }, // up-and-antispinward Coriolis sweep
        fireLineAt: null,
        reel: 0,
        release: false,
        pushoff: false,
        throwCharge: 0.55,
        throwReleased: tick === THROW_TICK,
        throwSpin: 0,
        thrumbler: { x: 0, y: 0, z: 0 },
      },
    ],
  };
}

const driver = new FixedStepDriver(
  () => {
    world.step(frameFor(simTick), SIM_H);
    simTick++;
  },
  () => world.snapshot(),
);

let loopGlow = 0;
let scoreFlash = 0;
const runtime = new GameRuntime(
  (dt) => {
    calm.update(dt);
    driver.advance(dt);

    const a = driver.alpha;
    const b0 = driver.prev.bell;
    const b1 = driver.cur.bell;
    const bx = b0.p.x + (b1.p.x - b0.p.x) * a;
    const by = b0.p.y + (b1.p.y - b0.p.y) * a;
    const bz = b0.p.z + (b1.p.z - b0.p.z) * a;
    const ch = b1.chime;
    bellMesh.position.set(bx, by, bz);
    (bellMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.3 + ch * 1.6;
    trail.push(bx, by, bz, ch);

    // Loop state → glow / slow-mo / hush / cam.
    const li = world.loopInfo();
    const looping = li.free && li.untouched && li.turn > 0.6;
    const target = looping ? Math.min(1, (li.turn - 0.6) / (Math.PI * 0.9)) : 0;
    loopGlow += (target - loopGlow) * Math.min(1, dt * 6);
    trail.setLoopMode(loopGlow > 0.15);
    post.setLoopGlow(loopGlow);
    runtime.timeScale = 1 - 0.28 * loopGlow; // gentle slow-mo (don't stall the arc)
    audio.setHush(Math.max(loopGlow, 0));
    const sp = Math.hypot(b1.w.x, b1.w.y, b1.w.z);
    audio.setBell(ch, sp, Math.max(-1, Math.min(1, bz / REG.R)), li.free);

    if (scoreFlash > 0) scoreFlash = Math.max(0, scoreFlash - dt);

    // Camera: normally a broadside down the calm; during a Loop, swing to a
    // down-the-spine shot so the closed Coriolis arc reads as a loop.
    // Side-on chase: track the bell from inside the calm so the Coriolis
    // curve fills the frame (loopGlow adds a touch of push-in).
    camera.position.set(bx - 34, by * 0.35 + 26, 72 - loopGlow * 14);
    camera.lookAt(bx + 8, by * 0.45, 0);
    camera.fov = 56 + loopGlow * 8;
    camera.updateProjectionMatrix();

    // Recycle: bell left the field or long flight done.
    if (Math.abs(b1.p.x) > REG.L / 2 + 5 || simTick > 240 * 17) {
      if (Math.abs(b1.p.x) > REG.L / 2) {
        audio.erupt(0.8);
        scoreFlash = 1;
      }
      world = makeWorld();
      simTick = 0;
      trail.clear();
      loopGlow = 0;
    }
  },
  () => post.render(),
);
runtime.start();
console.info('RIG v2 P2 — deterministic sim + money-shot Loop. ω=%s', REG.omega);

// P1 boot: the real calm + a live bell tracing the Coriolis roulette so the
// curve/loop is visible in 3D. Render-side demo loop uses the verified sim
// (stepBell + skinBounce); the deterministic fixed-step SimWorld lands in P2.

import * as THREE from 'three';
import { GameRuntime } from './core/GameRuntime';
import { EventBus } from './core/EventBus';
import { REG, GATE_X } from './sim/RegConstants';
import { Calm } from './render/Calm';
import { stepBell, chime, type BellBody } from './sim/Bell';
import { skinBounce } from './sim/Collision';

const app = document.getElementById('app')!;
document.getElementById('boot')?.remove();

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x11131a, 1);
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const calm = new Calm(scene);
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 4000);
const events = new EventBus();

function resize(): void {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  events.emit('resize', { width: innerWidth, height: innerHeight });
}
addEventListener('resize', resize);
resize();

// The bell.
const bellMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.9, 24, 16),
  new THREE.MeshStandardMaterial({ color: 0xe8e0c8, emissive: 0x1aa6b7, emissiveIntensity: 1 }),
);
scene.add(bellMesh);

// P1 trail: a coloured polyline (the gorgeous ribbon is P2's money shot).
const TRAIL = 600;
const trailPos = new Float32Array(TRAIL * 3);
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
const trail = new THREE.Line(
  trailGeo,
  new THREE.LineBasicMaterial({ color: 0x1aa6b7, transparent: true, opacity: 0.85 }),
);
trail.frustumCulled = false;
scene.add(trail);

function freshBell(): BellBody {
  // Launch near the axis, up-and-antispinward → a long Coriolis roulette.
  return {
    p: { x: -GATE_X * 0.55, y: 3, z: 0 },
    v: { x: 14, y: 7.5, z: -9 },
    q: { x: 0, y: 0, z: 0, w: 1 },
    w: { x: 26, y: 0, z: 0 },
  };
}
let bell = freshBell();
let trailN = 0;
function pushTrail(): void {
  const i = (trailN % TRAIL) * 3;
  trailPos[i] = bell.p.x;
  trailPos[i + 1] = bell.p.y;
  trailPos[i + 2] = bell.p.z;
  trailN++;
  trailGeo.setDrawRange(0, Math.min(trailN, TRAIL));
  trailGeo.attributes.position.needsUpdate = true;
}

const H = 1 / 240;
let acc = 0;
let life = 0;

const runtime = new GameRuntime(
  (dt) => {
    calm.update(dt);
    acc += dt;
    life += dt;
    while (acc >= H) {
      bell = stepBell(bell, REG.omega, H);
      skinBounce(bell);
      acc -= H;
    }
    pushTrail();
    if (Math.abs(bell.p.x) > GATE_X || life > 26) {
      bell = freshBell();
      trailN = 0;
      life = 0;
    }
    // Broadcast-ish view: off to the side, down the spine, framing the arc.
    bellMesh.position.set(bell.p.x, bell.p.y, bell.p.z);
    const ch = chime(bell.w);
    (bellMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.3 + ch * 1.4;
    camera.position.set(REG.R * 1.15, REG.R * 0.35, bell.p.x * 0.4 + REG.L * 0.42);
    camera.lookAt(bell.p.x * 0.5, 0, 0);
  },
  () => renderer.render(scene, camera),
);
runtime.start();
console.info('RIG v2 P1 — the calm + live Coriolis bell. ω=%s', REG.omega);

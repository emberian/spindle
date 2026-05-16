// P3 — the first genuinely playable assembly. The Rust/WASM core drives a
// real match: you (P1) play a home rigger via mouse+keys; the other riggers
// (both teams) are the AI; MatchStateMachine consumes the Rust events;
// Calm/Riggers/RigLines/BellTrail/PostFX/HUD render it. No clock; 9 innings.

import * as THREE from 'three';
import { GameRuntime } from './core/GameRuntime';
import { SIM_H } from './core/FixedStepDriver';
import { Calm } from './render/Calm';
import { BellTrail } from './render/BellTrail';
import { PostFX } from './render/PostFX';
import { Riggers } from './render/Rigger';
import { RigLines } from './render/RigLine';
import { AudioEngine } from './audio/AudioEngine';
import { HUD } from './ui/HUD';
import { InputManager } from './input/InputManager';
import { AiSystem, type TeamConfig } from './ai/index';
import { MatchStateMachine } from './match/MatchStateMachine';
import { createWasmSim, type WasmSim } from './sim/wasm';
import { REG, GATE_X } from './sim/RegConstants';
import { TEAMS, styleToProfile } from './league/teams';
import type { InputFrame, TeamSide, RiggerRole } from './sim/types';

const app = document.getElementById('app')!;
const boot = document.getElementById('boot');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x11131a, 1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const calm = new Calm(scene);
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 4000);
let post = new PostFX(renderer, scene, camera);
const trail = new BellTrail(scene);
const riggers = new Riggers(scene);
const riglines = new RigLines(scene);
const audio = new AudioEngine();
const hud = new HUD(app);

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

// Spar anchor points — mirror Calm.ts's 16×3 layout for the aim raycast.
const SPARS: { x: number; y: number; z: number }[] = [];
for (let i = 0; i < 16; i++) {
  const x = -GATE_X + ((i + 0.5) / 16) * REG.L;
  for (let a = 0; a < 3; a++) {
    const ang = (a / 3) * Math.PI * 2;
    SPARS.push({ x, y: Math.cos(ang) * REG.R * 0.62, z: Math.sin(ang) * REG.R * 0.62 });
  }
}

const SEED = 20260516;
const input = new InputManager(renderer.domElement, camera);
input.setSpars(SPARS);
const ai = new AiSystem();

// Two canon franchises for flavour + AI identity.
const homeFr = TEAMS.find((t) => t.id === 'tuebor-spin-detroiters') ?? TEAMS[0];
const awayFr = TEAMS.find((t) => t.id === 'toliman-gradient') ?? TEAMS[1];
const teamConfigs: TeamConfig[] = [
  { side: 'home', profile: styleToProfile(homeFr.styleTag, homeFr.cylinderClass), difficulty: 'pro' },
  { side: 'away', profile: styleToProfile(awayFr.styleTag, awayFr.cylinderClass), difficulty: 'pro' },
];

let sim: WasmSim;
let match: MatchStateMachine;

// Roster: P1 (the human, home spinner) + 3 home AI + 4 away AI.
const ROSTER: { id: string; team: TeamSide; role: RiggerRole; x: number }[] = [
  { id: 'P1', team: 'home', role: 'spinner', x: -40 },
  { id: 'H2', team: 'home', role: 'anchor', x: -90 },
  { id: 'H3', team: 'home', role: 'faithwing', x: -20 },
  { id: 'H4', team: 'home', role: 'reach', x: -260 },
  { id: 'A1', team: 'away', role: 'spinner', x: 40 },
  { id: 'A2', team: 'away', role: 'anchor', x: 90 },
  { id: 'A3', team: 'away', role: 'freewing', x: 20 },
  { id: 'A4', team: 'away', role: 'reach', x: 260 },
];

function seatBell(): void {
  // Hand the bell to a player of the team in possession to (re)start a cast.
  const poss: TeamSide = match.state.possession;
  const holder = ROSTER.find((r) => r.team === poss) ?? ROSTER[0];
  sim.setBellHeld(holder.id);
}

function reArm(): void {
  // Return the match to 'live' after a dead ball / inning break and re-seat
  // the bell — the same toLive trick the league sim uses.
  if (match.state.winner !== null) return;
  if (match.state.phase !== 'live') {
    seatBell();
    match.consume([{ type: 'foul_garrote', by: '__resume__' }], sim.snapshot());
  }
}

let acc = 0;
let runtime: GameRuntime;

async function start(): Promise<void> {
  sim = await createWasmSim(SEED);
  for (const r of ROSTER) {
    const ang = (ROSTER.indexOf(r) / ROSTER.length) * Math.PI * 2;
    sim.addPlayer(r.id, r.team, r.role, {
      x: r.x,
      y: Math.cos(ang) * 8,
      z: Math.sin(ang) * 8,
    });
  }
  sim.setBellHeld('P1');
  match = new MatchStateMachine('+x', 'home');
  match.consume([{ type: 'foul_garrote', by: '__start__' }], sim.snapshot());

  boot?.remove();
  runtime = new GameRuntime(
    (dt) => {
      calm.update(dt);
      acc += dt;
      let steps = 0;
      while (acc >= SIM_H && steps < 8 && match.state.winner === null) {
        const snap = sim.snapshot();
        const p1 = snap.players.find((p) => p.id === 'P1');
        if (p1) input.setPlayerState(p1.p, p1.v, snap.bell.heldBy === 'P1');
        const p1in = input.get(SIM_H);
        const aiFrame = ai.tick(snap, match.state as never, teamConfigs, SEED);
        const frame: InputFrame = { tick: snap.tick, players: [p1in, ...aiFrame.players] };
        const events = sim.step(frame);
        match.consume(events, sim.snapshot() as never);
        reArm();
        acc -= SIM_H;
        steps++;
      }

      const s = sim.snapshot();
      bellMesh.position.set(s.bell.p.x, s.bell.p.y, s.bell.p.z);
      const ch = s.bell.chime;
      (bellMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.3 + ch * 1.6;
      trail.push(s.bell.p.x, s.bell.p.y, s.bell.p.z, ch);
      trail.setLoopMode(s.loopTier === 'loop');
      post.setLoopGlow(s.loopTier === 'loop' ? 1 : s.loopTier === 'curl' ? 0.4 : 0);
      riggers.sync(s.players, 'P1');
      riglines.sync(s.players);
      audio.setBell(ch, Math.hypot(s.bell.w.x, s.bell.w.y, s.bell.w.z),
        Math.max(-1, Math.min(1, s.bell.p.z / REG.R)), s.bell.heldBy === null);
      audio.setHush(s.loopTier === 'loop' ? 1 : 0);

      // Chase cam: track the bell from inside the calm, side-on.
      const bp = s.bell.p;
      camera.position.set(bp.x - 34, bp.y * 0.35 + 26, 72);
      camera.lookAt(bp.x + 8, bp.y * 0.45, 0);
      camera.updateProjectionMatrix();

      hud.render(s as never, match.state as never, input.view);
    },
    () => post.render(),
  );
  runtime.start();
  addEventListener('pointerdown', () => { audio.start(); input.requestPointerLock(); });
  console.info('RIG v2 P3 — playable. %s vs %s', homeFr.name, awayFr.name);
}

start().catch((e) => {
  console.error('RIG boot failed', e);
  if (boot) boot.textContent = 'RIG — boot failed (see console)';
});

// P5 — the full game shell. Title → pick a franchise → The Jump bracket →
// play YOUR games (the Rust/WASM playable match), auto-sim the rest →
// champion. Persistence remembers your franchise + master seed.

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
import { TitleScreen } from './ui/TitleScreen';
import { BracketScreen } from './ui/BracketScreen';
import { InputManager } from './input/InputManager';
import { AiSystem, type TeamConfig } from './ai/index';
import { MatchStateMachine } from './match/MatchStateMachine';
import { createWasmSim, type WasmSim } from './sim/wasm';
import { REG, GATE_X } from './sim/RegConstants';
import { styleToProfile, type Franchise } from './league/teams';
import { Bracket } from './league/Bracket';
import type { MatchResult, BoxScore } from './league/SimMatch';
import * as Persist from './league/Persistence';
import type { InputFrame, TeamSide, RiggerRole } from './sim/types';

const app = document.getElementById('app')!;
document.getElementById('boot')?.remove();

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x11131a, 1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const calm = new Calm(scene);
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 4000);
const post = new PostFX(renderer, scene, camera);
const trail = new BellTrail(scene);
const riggers = new Riggers(scene);
const riglines = new RigLines(scene);
const audio = new AudioEngine();
const hud = new HUD(app);
const title = new TitleScreen(app);
const bracketUI = new BracketScreen(app);
const input = new InputManager(renderer.domElement, camera);
const ai = new AiSystem();

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

const SPARS: { x: number; y: number; z: number }[] = [];
for (let i = 0; i < 16; i++) {
  const x = -GATE_X + ((i + 0.5) / 16) * REG.L;
  for (let a = 0; a < 3; a++) {
    const ang = (a / 3) * Math.PI * 2;
    SPARS.push({ x, y: Math.cos(ang) * REG.R * 0.62, z: Math.sin(ang) * REG.R * 0.62 });
  }
}
input.setSpars(SPARS);

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
const ZBOX = (): BoxScore => ({ falls: 0, rises: 0, loops: 0, curls: 0, grounds: 0 });

let save = Persist.load();
let masterSeed = save.seed >>> 0;
let chosen: Franchise | null = null;
let bracket: Bracket | null = null;
let runtime: GameRuntime | null = null;

// ── The interactive match (player = home P1) ──────────────────────────────────
async function runMatch(
  homeFr: Franchise,
  awayFr: Franchise,
  gameSeed: number,
  onEnd: (winner: TeamSide, scoreHome: number, scoreAway: number) => void,
): Promise<void> {
  const sim: WasmSim = await createWasmSim(gameSeed);
  for (const r of ROSTER) {
    const k = ROSTER.indexOf(r);
    const ang = (k / ROSTER.length) * Math.PI * 2;
    sim.addPlayer(r.id, r.team, r.role, { x: r.x, y: Math.cos(ang) * 8, z: Math.sin(ang) * 8 });
  }
  sim.setBellHeld('P1');
  const match = new MatchStateMachine('+x', 'home');
  match.consume([{ type: 'foul_garrote', by: '__start__' }], sim.snapshot());
  const cfgs: TeamConfig[] = [
    { side: 'home', profile: styleToProfile(homeFr.styleTag, homeFr.cylinderClass), difficulty: 'pro' },
    { side: 'away', profile: styleToProfile(awayFr.styleTag, awayFr.cylinderClass), difficulty: 'pro' },
  ];
  trail.clear();
  let acc = 0;
  let ended = false;

  const reArm = (): void => {
    if (match.state.winner !== null || match.state.phase === 'live') return;
    const poss = match.state.possession;
    sim.setBellHeld(ROSTER.find((r) => r.team === poss)!.id);
    match.consume([{ type: 'foul_garrote', by: '__resume__' }], sim.snapshot());
  };

  runtime?.stop();
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
        const aiFrame = ai.tick(snap, match.state as never, cfgs, gameSeed);
        const frame: InputFrame = { tick: snap.tick, players: [p1in, ...aiFrame.players] };
        match.consume(sim.step(frame), sim.snapshot() as never);
        reArm();
        acc -= SIM_H;
        steps++;
      }
      const s = sim.snapshot();
      bellMesh.position.set(s.bell.p.x, s.bell.p.y, s.bell.p.z);
      (bellMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.3 + s.bell.chime * 1.6;
      trail.push(s.bell.p.x, s.bell.p.y, s.bell.p.z, s.bell.chime);
      trail.setLoopMode(s.loopTier === 'loop');
      post.setLoopGlow(s.loopTier === 'loop' ? 1 : s.loopTier === 'curl' ? 0.4 : 0);
      riggers.sync(s.players, 'P1');
      riglines.sync(s.players);
      audio.setBell(s.bell.chime, Math.hypot(s.bell.w.x, s.bell.w.y, s.bell.w.z),
        Math.max(-1, Math.min(1, s.bell.p.z / REG.R)), s.bell.heldBy === null);
      audio.setHush(s.loopTier === 'loop' ? 1 : 0);
      const bp = s.bell.p;
      camera.position.set(bp.x - 34, bp.y * 0.35 + 26, 72);
      camera.lookAt(bp.x + 8, bp.y * 0.45, 0);
      camera.updateProjectionMatrix();
      hud.render(s as never, match.state as never, input.view);

      if (!ended && match.state.winner !== null) {
        ended = true;
        runtime?.stop();
        onEnd(match.state.winner, match.state.scoreHome, match.state.scoreAway);
      }
    },
    () => post.render(),
  );
  runtime.start();
}

// ── Screen FSM ────────────────────────────────────────────────────────────────
function showChampion(): void {
  bracketUI.show(bracket!.state as never, chosen!.id, () => {
    Persist.save({ ...Persist.freshSave(), seed: (Math.random() * 0xffffffff) >>> 0 });
    location.reload();
  });
}

function proceed(): void {
  if (!bracket || !chosen) return;
  // Auto-sim every game that doesn't involve the chosen franchise.
  let g = bracket.nextGame();
  while (g && g.home && g.away && g.home.id !== chosen.id && g.away.id !== chosen.id) {
    bracket.advance();
    bracketUI.update(bracket.state as never);
    g = bracket.nextGame();
  }
  if (!g) {
    showChampion();
    return;
  }
  // The chosen franchise plays this one interactively. Player is always
  // home P1; orient the bracket result to the seeded home/away.
  const chosenIsBracketHome = g.home!.id === chosen.id;
  const playHome = chosen;
  const playAway = chosenIsBracketHome ? g.away! : g.home!;
  bracketUI.hide();
  void runMatch(playHome, playAway, g.gameSeed, (w, sh, sa) => {
    const chosenWon = w === 'home';
    const result: MatchResult = chosenIsBracketHome
      ? { home: g!.home!, away: g!.away!, scoreHome: sh, scoreAway: sa,
          winner: chosenWon ? 'home' : 'away', box: { home: ZBOX(), away: ZBOX() }, events: [] }
      : { home: g!.home!, away: g!.away!, scoreHome: sa, scoreAway: sh,
          winner: chosenWon ? 'away' : 'home', box: { home: ZBOX(), away: ZBOX() }, events: [] };
    bracket!.playNext(result);
    save = { ...save, bracket: bracket!.state as never, updatedAt: new Date().toISOString() };
    Persist.save(save);
    if (!chosenWon) {
      bracket!.simulateAll();
      showChampion();
    } else {
      bracketUI.show(bracket!.state as never, chosen!.id, proceed);
    }
  });
}

function enterJump(f: Franchise): void {
  chosen = f;
  bracket = new Bracket(masterSeed);
  save = { ...Persist.freshSave(masterSeed), chosenFranchiseId: f.id, seed: masterSeed };
  Persist.save(save);
  title.hide();
  bracketUI.show(bracket.state as never, f.id, proceed);
}

title.show(enterJump);
console.info('RIG v2 P5 — full shell. master seed %s', masterSeed);

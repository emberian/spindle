// P5 — the full game shell. Title → pick a franchise → The Jump bracket →
// play YOUR games (the Rust/WASM playable match), auto-sim the rest →
// champion. Persistence remembers your franchise + master seed.

import * as THREE from 'three';
import { GameRuntime } from './core/GameRuntime';
import { FixedStepDriver, SIM_H } from './core/FixedStepDriver';
import { renderState, type RenderView } from './render/RenderState';
import { Calm } from './render/Calm';
import { DistantHabitat } from './render/DistantHabitat';
import { BellTrail, bellGlow } from './render/BellTrail';
import { PostFX } from './render/PostFX';
import { GameCamera } from './render/Camera';
import { initDebugViz, debugViz } from './render/DebugViz';
import { Riggers } from './render/Rigger';
import { RigLines } from './render/RigLine';
import { AudioEngine } from './audio/AudioEngine';
import { HUD } from './ui/HUD';
import { Onboarding } from './ui/Onboarding';
import { LandingScreen } from './ui/LandingScreen';
import { TitleScreen } from './ui/TitleScreen';
import { BracketScreen } from './ui/BracketScreen';
import { SpectateScreen } from './ui/SpectateScreen';
import { SpectateControls } from './ui/SpectateControls';
import { ReplayScreen } from './ui/ReplayScreen';
import { ReplayRecorder, ReplayStore, type ReplayData } from './league/Replay';
import { solveGateThrow, type GateSolution } from './ai/decide/GateSolve';
import { predictPath } from './sim/trajectory';
import { InputManager } from './input/InputManager';
import { AiSystem, type TeamConfig } from './ai/index';
import { MatchStateMachine } from './match/MatchStateMachine';
import { createWasmSim, type WasmSim } from './sim/wasm';
import { REG, GATE_X } from './sim/RegConstants';
import { styleToProfile, TEAMS, type Franchise } from './league/teams';
import { Bracket } from './league/Bracket';
import type { MatchResult, BoxScore } from './league/SimMatch';
import * as Persist from './league/Persistence';
import type { InputFrame, TeamSide, RiggerRole } from './sim/types';

const app = document.getElementById('app')!;
document.getElementById('boot')?.remove();

// Single debug/viz toggle (backtick key). Defaults OFF — all rich
// developer-only telemetry below (window.__rigai / __rigp) is constructed
// lazily and ONLY when this is on, so it costs nothing in normal play.
initDebugViz(app);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x11131a, 1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
// Pull exposure down: at some cinematic angles the axis sunline + bloom
// blew out to near-white and hurt legibility.
renderer.toneMappingExposure = 0.85;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const calm = new Calm(scene);
// Distant O'Neill-cylinder vista OUTSIDE the play volume. Persistent like
// `calm`. Quality flag defaults ON (it is the atmosphere); pass `false` as
// the 2nd arg to disable and reclaim its (tiny) budget.
const distantHabitat = new DistantHabitat(scene, true);
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 4000);
const post = new PostFX(renderer, scene, camera);
const trail = new BellTrail(scene, camera);
const gcam = new GameCamera(camera);
const riggers = new Riggers(scene);
const riglines = new RigLines(scene);
const audio = new AudioEngine();
const hud = new HUD(app);
const onboarding = new Onboarding(app);

// Explicit-launch "YOUR CAST" prompt: a player must never be surprised by
// the ball moving on its own. Whenever YOU (P1) hold the bell, this says so
// and tells you the action that launches it. Orchestrator-owned overlay so
// HUD's frozen render() contract is untouched. pointer-events:none.
const castPrompt = document.createElement('div');
castPrompt.style.cssText =
  'position:absolute;left:50%;top:16%;transform:translateX(-50%);' +
  'font-family:ui-monospace,"Space Mono",monospace;text-align:center;' +
  'letter-spacing:.16em;font-size:15px;line-height:1.7;' +
  'pointer-events:none;z-index:40;display:none;text-transform:uppercase;';
app.appendChild(castPrompt);
// Three prompt modes for the assisted-shot model:
//  'ready'  — a real Coriolis ring solution exists; the bright arc shows it
//             threading; you just press LEFT to launch the SOLVED shot.
//  'noshot' — no solution from here; grapple (RIGHT) toward the ring.
//  'none'   — you don't have the bell / match over → hidden.
type CastMode = 'none' | 'ready' | 'noshot';
let castMode: CastMode = 'none';
const setCastPrompt = (m: CastMode): void => {
  if (m === castMode) return;
  castMode = m;
  if (m === 'none') { castPrompt.style.display = 'none'; return; }
  castPrompt.style.display = 'block';
  if (m === 'ready') {
    castPrompt.style.color = '#1aa6b7';
    castPrompt.style.textShadow = '0 0 14px #1aa6b7cc';
    castPrompt.innerHTML =
      '▼ SHOT READY — the arc threads the ring<br>' +
      '<span style="color:#f4f1ea;font-size:13px;letter-spacing:.1em">' +
      'press &amp; release <b>LEFT</b> to launch &nbsp;·&nbsp; ' +
      '<b>WASD</b> to fly your rigger</span>';
  } else {
    castPrompt.style.color = '#d4602a';
    castPrompt.style.textShadow = '0 0 14px #d4602aaa';
    castPrompt.innerHTML =
      '▼ YOU HAVE THE BELL — no shot from here<br>' +
      '<span style="color:#f4f1ea;font-size:13px;letter-spacing:.1em">' +
      'fly with <b>WASD</b> toward the ring until a shot opens</span>';
  }
};

// Bold predicted-arc line: the exact Coriolis path of the assisted shot,
// drawn so the human SEES it thread the ring (always-on-top, can't be lost).
const arcGeom = new THREE.BufferGeometry();
arcGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * 64), 3));
const arcLine = new THREE.Line(
  arcGeom,
  new THREE.LineBasicMaterial({
    color: 0x6fe9ff, transparent: true, opacity: 0.95,
    blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
  }),
);
arcLine.renderOrder = 9990;
arcLine.frustumCulled = false;
arcLine.visible = false;
scene.add(arcLine);
const showArc = (pts: { x: number; y: number; z: number }[] | null): void => {
  if (!pts || pts.length < 2) { arcLine.visible = false; return; }
  const n = Math.min(pts.length, 64);
  const a = arcGeom.getAttribute('position') as THREE.BufferAttribute;
  for (let i = 0; i < n; i++) a.setXYZ(i, pts[i].x, pts[i].y, pts[i].z);
  arcGeom.setDrawRange(0, n);
  a.needsUpdate = true;
  arcLine.visible = true;
};
const landing = new LandingScreen(app);
const title = new TitleScreen(app);
const bracketUI = new BracketScreen(app);
const spectate = new SpectateScreen(app);
const spectateControls = new SpectateControls(app);
const replayScreen = new ReplayScreen(app);
const input = new InputManager(renderer.domElement, camera);
const ai = new AiSystem();
let shownOnboarding = false;

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
  const rec = new ReplayRecorder(
    {
      id: `r${Date.now().toString(36)}-${gameSeed >>> 0}`,
      homeId: homeFr.id, awayId: awayFr.id, seed: gameSeed,
      faithEnd: '+x', firstPossession: 'home',
      label: `${homeFr.name} vs ${awayFr.name}`,
    },
    ROSTER.map((r) => ({ id: r.id, team: r.team, role: r.role, x: r.x })),
  );
  trail.clear();
  gcam.reset();
  riggers.setViewScale(2); // close chase-cam → near-human figure scale
  input.setFaithRingX?.(GATE_X); // player (home) attacks the +x Faith ring
  if (!shownOnboarding) {
    shownOnboarding = true;
    onboarding.show();
  }
  let ended = false;
  let prevLoop = false;
  let prevFire = false;
  // SET freeze: the ball must not move until YOU act this cast.
  let armed = false;
  let prevHeld: string | null = null;

  const reArm = (): void => {
    if (match.state.winner !== null || match.state.phase === 'live') return;
    const poss = match.state.possession;
    sim.setBellHeld(ROSTER.find((r) => r.team === poss)!.id);
    match.resumeLive();
  };

  // Latest assisted-shot solution for P1 (computed deterministically in
  // stepOnce; read by the render block to draw the bold arc + prompt).
  let p1Sol: GateSolution | null = null;

  // One deterministic sim tick: exactly the prior per-tick body. The driver
  // calls this once per fixed 1/240 s tick; renderState() never feeds back in.
  const stepOnce = (): void => {
    if (match.state.winner !== null) return;
    const snap = sim.snapshot();
    const p1 = snap.players.find((p) => p.id === 'P1');
    if (p1) input.setPlayerState(p1.p, p1.v, snap.bell.heldBy === 'P1');
    // Catch-commit parity (Gap 2): the bell is grabbable by us only if it's
    // unheld AND we didn't throw it (mirrors wants_catch's early-outs).
    const bellLoose = snap.bell.heldBy === null && snap.bell.thrownBy !== 'P1';
    input.setBellState(snap.bell.p, snap.bell.v, bellLoose);
    // Grapple-latency cue (Gap 3): expose P1's line presence/attached so the
    // reticle can show flight + the re-fire lockout. Render-only.
    input.setLineState(p1?.line != null, p1?.line?.attached === true);
    const p1in = input.get(SIM_H);
    // ── Assisted solved-shot ───────────────────────────────────────────────
    // While YOU hold the bell, the game computes the EXACT Coriolis launch
    // that threads the ring (the same closed-form solver the AI uses). You
    // don't hand-aim an impossible curve — you reposition until a shot
    // exists, then LEFT-release fires the SOLVED velocity. Skill =
    // positioning / reads / timing, not intuiting a rotating-frame ODE.
    p1Sol = (p1 && snap.bell.heldBy === 'P1')
      ? solveGateThrow(p1.p, 'home', snap.omega, 22, 0.3, p1.v)
      : null;
    if (p1Sol && p1in.throwReleased) {
      const tv = p1Sol.throwVec;
      const tl = Math.hypot(tv.x, tv.y, tv.z) || 1;
      p1in.aim = { x: tv.x / tl, y: tv.y / tl, z: tv.z / tl };
      p1in.throwCharge = Math.max(0, Math.min(1, (p1Sol.releaseSpeed - 9) / 25));
      p1in.throwSpin = 0; // solved shot owns the trajectory; no stray spin
    } else if (!p1Sol && p1in.throwReleased) {
      p1in.throwReleased = false; // no solution → don't waste the cast
    }
    // ── Direct rigger drive (WASD) ─────────────────────────────────────────
    // YOU steer your athlete: WASD picks a direction; the orchestrator
    // reuses the AI's planner to grapple the rigger that way (you move by
    // line, canon — but the line is auto-chosen so it's direct & legible).
    // G is kept as an alias for "toward the ring". RMB = manual grapple.
    const md = input.moveDir;
    const drive = (md.x !== 0 || md.z !== 0);
    if (p1 && (drive || input.wantsAssistGrapple)) {
      const dir = drive ? md : { x: 1, y: 0, z: 0 }; // G ⇒ toward +x ring
      // Fire at the best VISIBLE spar in your heading and reel toward it —
      // so the line goes to a structure you can SEE and the movement reads
      // as real physics (the AI planner anchored to the invisible centre
      // axis, which is why the line "always went to the centerpoint").
      let best: { x: number; y: number; z: number } | null = null;
      let bestScore = -Infinity;
      for (const sp of SPARS) {
        const vx = sp.x - p1.p.x, vy = sp.y - p1.p.y, vz = sp.z - p1.p.z;
        const d = Math.hypot(vx, vy, vz);
        if (d < 7 || d > 150) continue;
        const align = (vx * dir.x + vy * dir.y + vz * dir.z) / d; // −1..1
        if (align < 0.1) continue;                 // must be ~in our heading
        const score = align - Math.abs(d - 55) * 0.004; // prefer ~55 m & aligned
        if (score > bestScore) { bestScore = score; best = sp; }
      }
      if (best) {
        if (!p1.line) p1in.fireLineAt = { x: best.x, y: best.y, z: best.z };
        p1in.reel = -1; // haul in toward the spar → you move that way
      }
    }
    const aiFrame = ai.tick(snap, match.state as never, cfgs, gameSeed);
    const frame: InputFrame = { tick: snap.tick, players: [p1in, ...aiFrame.players] };
    rec.push(frame);
    const evs = sim.step(frame);
    const upd = match.consume(evs, sim.snapshot() as never);
    // Soul: drive the one-shot audio off real events.
    for (const e of evs) {
      if (e.type === 'bell_caught') audio.event('catch');
      else if (e.type === 'bell_clatter' || e.type === 'bell_bobble') audio.event('clatter');
    }
    if (p1in.throwReleased) audio.event('throw');
    if (!!p1in.fireLineAt && !prevFire) audio.event('grapple');
    prevFire = !!p1in.fireLineAt;
    if (upd && upd.scored) {
      const k = upd.scored.kind;
      audio.event(
        k === 'loop' ? 'score_loop' : k === 'rise' || k === 'curl' ? 'score_rise'
          : k === 'ground' ? 'score_ground' : 'score_fall',
      );
    } else if (upd && upd.turnover) {
      audio.event('turnover');
    }
    reArm();
  };
  const driver = new FixedStepDriver(stepOnce, () => sim.snapshot());

  runtime?.stop();
  runtime = new GameRuntime(
    (dt) => {
      // SET freeze — ball stands still until you act this cast.
      const pre = sim.snapshot();
      if (pre.bell.heldBy === 'P1' && prevHeld !== 'P1') armed = false;
      prevHeld = pre.bell.heldBy;
      if (input.actedThisFrame) armed = true;
      const frozen = !armed && pre.bell.heldBy === 'P1' && match.state.winner === null;
      calm.update(dt);
      distantHabitat.update(dt);
      if (!frozen) driver.advance(dt);
      const s = renderState(driver.prev as RenderView, driver.cur as RenderView, driver.alpha);
      const lg = s.loopTier === 'loop' ? 1 : s.loopTier === 'curl' ? 0.4 : 0;
      const isLoop = s.loopTier === 'loop';
      bellMesh.position.set(s.bell.p.x, s.bell.p.y, s.bell.p.z);
      (bellMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = bellGlow(s.bell.chime);
      trail.push(s.bell.p.x, s.bell.p.y, s.bell.p.z, s.bell.chime);
      trail.setLoopMode(isLoop);
      post.setLoopGlow(lg);
      riggers.sync(s.players, 'P1');
      for (const pp of s.players) riggers.setPoseEmphasis(pp.id, pp.id === s.bell.heldBy ? 1 : 0);
      riglines.sync(s.players);
      audio.setBell(s.bell.chime, Math.hypot(s.bell.w.x, s.bell.w.y, s.bell.w.z),
        Math.max(-1, Math.min(1, s.bell.p.z / REG.R)), s.bell.heldBy === null);
      audio.setHush(lg);
      if (isLoop && !prevLoop) audio.event('loop_building');
      prevLoop = isLoop;
      const p1r = s.players.find((pp) => pp.id === 'P1');
      gcam.setControl(input.cameraState); // player-controlled orbit
      gcam.update(s.bell.p, p1r ? p1r.p : s.bell.p, GATE_X, lg, REG.R, Math.min(dt, 1 / 30));
      hud.render(s as never, match.state as never, input.view);
      // Assisted-shot feedback: when YOU hold the bell, either show the
      // bold arc threading the ring + "SHOT READY", or "no shot — go
      // closer". The ball never moves on its own; YOU launch it.
      // Recompute the shown solution from the RENDERED P1 every frame so the
      // arc + "SHOT READY" are correct even during the SET freeze (when the
      // deterministic stepOnce that sets p1Sol isn't running).
      const dispSol =
        s.bell.heldBy === 'P1' && p1r && match.state.winner === null
          ? solveGateThrow(p1r.p, 'home', s.omega, 22, 0.3, p1r.v)
          : null;
      if (s.bell.heldBy === 'P1' && match.state.winner === null) {
        if (dispSol) {
          setCastPrompt('ready');
          const ah = Math.max(0.02, dispSol.flightTime / 56);
          showArc(predictPath(p1r ? p1r.p : s.bell.p, dispSol.v0, s.omega, ah, 56));
        } else {
          setCastPrompt('noshot');
          showArc(null);
        }
      } else {
        setCastPrompt('none');
        showArc(null);
      }
      // Rich human-play diagnostic — gated behind the single debugViz flag.
      // Lazily built ONLY when debug viz is on (zero per-frame object/field
      // churn in normal play; the old code allocated this every frame).
      if (debugViz()) {
        const _iv = input.view;
        (window as unknown as { __rigp?: unknown }).__rigp = {
          tick: s.tick,
          p1: p1r ? { x: p1r.p.x, y: p1r.p.y, z: p1r.p.z } : null,
          p1v: p1r ? Math.hypot(p1r.v.x, p1r.v.y, p1r.v.z) : 0,
          p1held: s.bell.heldBy === 'P1',
          bx: s.bell.p.x, bheld: s.bell.heldBy,
          cam: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
          reticle: _iv.reticleState, locked: _iv.pointerLocked, charge: _iv.chargeLevel,
          phase: match.state.phase, sh: match.state.scoreHome, sa: match.state.scoreAway,
        };
      }

      if (!ended && match.state.winner !== null) {
        ended = true;
        setCastPrompt('none');
        showArc(null);
        runtime?.stop();
        try {
          ReplayStore.save(
            rec.finish({
              scoreHome: match.state.scoreHome,
              scoreAway: match.state.scoreAway,
              winner: match.state.winner,
            }),
          );
        } catch { /* replay storage is best-effort */ }
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

// Pointer-lock is owned solely by InputManager now (it requests on a
// deliberate mousedown gesture and swallows refusal); no orchestrator hook.

// ── Spectate: watch a fully AI-vs-AI match (no human P1) ─────────────────────
// Same roster as ROSTER but every rigger is AI-controlled — id 'H1' replaces
// 'P1' so AiSystem (which skips id==='P1') drives all eight.
const WATCH_ROSTER: { id: string; team: TeamSide; role: RiggerRole; x: number }[] =
  ROSTER.map((r) => (r.id === 'P1' ? { ...r, id: 'H1' } : { ...r }));

let watching = false;

async function runWatch(
  homeFr: Franchise,
  awayFr: Franchise,
  gameSeed: number,
  onEnd: (winner: TeamSide, scoreHome: number, scoreAway: number) => void,
): Promise<void> {
  const sim: WasmSim = await createWasmSim(gameSeed);
  for (const r of WATCH_ROSTER) {
    const k = WATCH_ROSTER.indexOf(r);
    const ang = (k / WATCH_ROSTER.length) * Math.PI * 2;
    sim.addPlayer(r.id, r.team, r.role, { x: r.x, y: Math.cos(ang) * 8, z: Math.sin(ang) * 8 });
  }
  sim.setBellHeld('H1');
  const match = new MatchStateMachine('+x', 'home');
  match.consume([{ type: 'foul_garrote', by: '__start__' }], sim.snapshot());
  const cfgs: TeamConfig[] = [
    { side: 'home', profile: styleToProfile(homeFr.styleTag, homeFr.cylinderClass), difficulty: 'pro' },
    { side: 'away', profile: styleToProfile(awayFr.styleTag, awayFr.cylinderClass), difficulty: 'pro' },
  ];
  const rec = new ReplayRecorder(
    {
      id: `r${Date.now().toString(36)}-${gameSeed >>> 0}`,
      homeId: homeFr.id, awayId: awayFr.id, seed: gameSeed,
      faithEnd: '+x', firstPossession: 'home',
      label: `${homeFr.name} vs ${awayFr.name} (watch)`,
    },
    WATCH_ROSTER.map((r) => ({ id: r.id, team: r.team, role: r.role, x: r.x })),
  );
  trail.clear();
  gcam.reset();
  riggers.setViewScale(5); // far cinematic cam → larger-than-life figures
  ai.reset();
  let ended = false;
  let prevLoop = false;
  // AI diagnostics: last sim snapshot + last AI input frame (for __rigai).
  let dbgSnap: ReturnType<typeof sim.snapshot> | null = null;
  let dbgAi: { players: import('./sim/types').PlayerInput[] } | null = null;
  // Cumulative counters updated EVERY sim tick (throw release is a 1-tick
  // event — sampling the flag misses it; count it at the source instead).
  const dbgThrows: Record<string, number> = {};
  const dbgScores: Record<string, number> = {};
  let dbgHeldTicks = 0, dbgTotalTicks = 0;
  // ── Skill signals (the better eval) ──────────────────────────────────────
  // Skillful rig ≠ fast chaos. Track the things that distinguish purpose
  // from flailing: completed PASSES (possession chains), INTERCEPTS (read
  // defense), GATE CLEARS (the cast actually advances), and a chaos proxy:
  // anchor THRASH (how often a rigger's fired anchor jumps — flailing vs
  // committed swings).
  const dbgTeamOf: Record<string, string> = {};
  for (const r of WATCH_ROSTER) dbgTeamOf[r.id] = r.team;
  let dbgPasses = 0, dbgIntercepts = 0, dbgGateClears = 0, dbgThrash = 0;
  // Pass-chain craft: longest run of consecutive completed passes within a
  // single possession (resets on turnover / score). Stringing passes is
  // skill; a one-and-done huck is not.
  let dbgPassStreak = 0, dbgPassStreakMax = 0;
  let dbgPrevHeld: string | null = null;
  let dbgLastThrownBy: string | null = null;
  let dbgPrevGate: string | null = null;
  const dbgPrevFire: Record<string, { x: number; y: number; z: number }> = {};
  const GATE_ORD: Record<string, number> = { first: 0, deep: 1, mouth: 2 };

  const reArm = (): void => {
    if (match.state.winner !== null || match.state.phase === 'live') return;
    const poss = match.state.possession;
    sim.setBellHeld(WATCH_ROSTER.find((r) => r.team === poss)!.id);
    match.resumeLive();
  };

  // One deterministic sim tick (AI-only): exactly the prior per-tick body.
  const stepOnce = (): void => {
    if (match.state.winner !== null) return;
    const snap = sim.snapshot();
    const aiFrame = ai.tick(snap, match.state as never, cfgs, gameSeed);
    // Rich AI telemetry bookkeeping — pure debug, feeds only window.__rigai.
    // Gated behind the single debugViz flag so it is genuinely zero-cost
    // (no snapshot capture, no per-player scans) in normal spectate play.
    if (debugViz()) {
    dbgSnap = snap; dbgAi = aiFrame;
    dbgTotalTicks++;
    if (snap.bell.heldBy) dbgHeldTicks++;
    snap.players.forEach((pl, i) => {
      if (aiFrame.players[i] && aiFrame.players[i].throwReleased) {
        dbgThrows[pl.id] = (dbgThrows[pl.id] ?? 0) + 1;
      }
    });
    // Skill signals (transitions read off the deterministic snapshot).
    {
      const b = snap.bell;
      if (b.heldBy && !dbgPrevHeld && dbgLastThrownBy && dbgLastThrownBy !== b.heldBy) {
        if (dbgTeamOf[dbgLastThrownBy] === dbgTeamOf[b.heldBy]) {
          dbgPasses++;
          dbgPassStreak++;
          if (dbgPassStreak > dbgPassStreakMax) dbgPassStreakMax = dbgPassStreak;
        } else {
          dbgIntercepts++;
          dbgPassStreak = 0; // possession lost — chain broken
        }
      }
      if (!b.heldBy && b.thrownBy) dbgLastThrownBy = b.thrownBy;
      if (b.heldBy) dbgLastThrownBy = null;
      dbgPrevHeld = b.heldBy;
      const g = match.state.cast.gate;
      if (dbgPrevGate !== null && GATE_ORD[g] > GATE_ORD[dbgPrevGate]) dbgGateClears++;
      dbgPrevGate = g;
      snap.players.forEach((pl, i) => {
        const f = aiFrame.players[i] && aiFrame.players[i].fireLineAt;
        if (f) {
          const pf = dbgPrevFire[pl.id];
          if (pf && Math.hypot(f.x - pf.x, f.y - pf.y, f.z - pf.z) > 8) dbgThrash++;
          dbgPrevFire[pl.id] = { x: f.x, y: f.y, z: f.z };
        }
      });
    }
    } // end debugViz() bookkeeping
    const frame: InputFrame = { tick: snap.tick, players: aiFrame.players };
    rec.push(frame);
    const evs = sim.step(frame);
    const upd = match.consume(evs, sim.snapshot() as never);
    for (const e of evs) {
      if (e.type === 'bell_caught') { audio.event('catch'); dbgScores['caught'] = (dbgScores['caught'] ?? 0) + 1; }
      else if (e.type === 'bell_clatter' || e.type === 'bell_bobble') { audio.event('clatter'); dbgScores['clatter'] = (dbgScores['clatter'] ?? 0) + 1; }
    }
    if (upd && upd.scored) {
      const kd = upd.scored.kind;
      dbgScores[kd] = (dbgScores[kd] ?? 0) + 1;
      dbgPassStreak = 0; // possession resolved — chain ends
      audio.event(
        kd === 'loop' ? 'score_loop' : kd === 'rise' || kd === 'curl' ? 'score_rise'
          : kd === 'ground' ? 'score_ground' : 'score_fall',
      );
    } else if (upd && upd.turnover) {
      dbgScores['turnover'] = (dbgScores['turnover'] ?? 0) + 1;
      dbgPassStreak = 0;
      audio.event('turnover');
    }
    reArm();
  };
  const driver = new FixedStepDriver(stepOnce, () => sim.snapshot());

  runtime?.stop();
  watching = true;
  runtime = new GameRuntime(
    (dt) => {
      calm.update(dt);
      distantHabitat.update(dt);
      driver.advance(dt);
      const s = renderState(driver.prev as RenderView, driver.cur as RenderView, driver.alpha);
      const lg = s.loopTier === 'loop' ? 1 : s.loopTier === 'curl' ? 0.4 : 0;
      const isLoop = s.loopTier === 'loop';
      bellMesh.position.set(s.bell.p.x, s.bell.p.y, s.bell.p.z);
      (bellMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = bellGlow(s.bell.chime);
      trail.push(s.bell.p.x, s.bell.p.y, s.bell.p.z, s.bell.chime);
      trail.setLoopMode(isLoop);
      post.setLoopGlow(lg);
      riggers.sync(s.players, '');
      for (const pp of s.players) riggers.setPoseEmphasis(pp.id, pp.id === s.bell.heldBy ? 1 : 0);
      riglines.sync(s.players);
      audio.setBell(s.bell.chime, Math.hypot(s.bell.w.x, s.bell.w.y, s.bell.w.z),
        Math.max(-1, Math.min(1, s.bell.p.z / REG.R)), s.bell.heldBy === null);
      audio.setHush(lg);
      if (isLoop && !prevLoop) audio.event('loop_building');
      prevLoop = isLoop;
      const atkX = match.state.possession === 'home' ? GATE_X : -GATE_X;
      const cinePlayers = s.players.map((pp) => ({ id: pp.id, p: pp.p, team: pp.team }));
      gcam.setControl(input.cameraState);
      gcam.cinematic(s.bell.p, cinePlayers, atkX, lg, REG.R, Math.min(dt, 1 / 30));
      hud.render(s as never, match.state as never, input.view);
      // Diagnostic hook (cheap; lets a harness observe real AI progression).
      (window as unknown as { __rig?: unknown }).__rig = {
        tick: s.tick, bx: s.bell.p.x,
        br: Math.hypot(s.bell.p.y, s.bell.p.z),
        held: s.bell.heldBy,
        poss: match.state.possession,
        sh: match.state.scoreHome, sa: match.state.scoreAway,
        inning: match.state.inning, phase: match.state.phase,
        gate: match.state.cast.gate, throwsLeft: match.state.cast.throwsLeft,
        msg: match.state.message,
      };
      // Rich per-player AI telemetry — gated behind the single debugViz
      // flag. Built lazily and ONLY when debug viz is on: the per-frame
      // object + per-player .map() allocation is zero cost in normal play.
      if (debugViz() && dbgSnap && dbgAi) {
        const bp = dbgSnap.bell.p;
        (window as unknown as { __rigai?: unknown }).__rigai = {
          tick: dbgSnap.tick,
          held: dbgSnap.bell.heldBy,
          bellR: Math.hypot(dbgSnap.bell.p.y, dbgSnap.bell.p.z),
          heldFrac: dbgTotalTicks ? +(dbgHeldTicks / dbgTotalTicks).toFixed(2) : 0,
          throws: { ...dbgThrows },
          scores: { ...dbgScores },
          skill: {
            passes: dbgPasses,
            intercepts: dbgIntercepts,
            gateClears: dbgGateClears,
            thrash: dbgThrash,
            passStreakMax: dbgPassStreakMax,
            ticks: dbgTotalTicks,
          },
          players: dbgSnap.players.map((pl, i) => {
            const inp = dbgAi!.players[i];
            return {
              id: pl.id, role: pl.role, team: pl.team,
              x: pl.p.x, y: pl.p.y, z: pl.p.z,
              spd: Math.hypot(pl.v.x, pl.v.y, pl.v.z),
              line: pl.line ? 1 : 0,
              fire: inp && inp.fireLineAt ? 1 : 0,
              reel: inp ? inp.reel : 0,
              thrown: inp && inp.throwReleased ? 1 : 0,
              dBell: Math.hypot(pl.p.x - bp.x, pl.p.y - bp.y, pl.p.z - bp.z),
              r: Math.hypot(pl.p.y, pl.p.z),
            };
          }),
        };
      }

      if (!ended && match.state.winner !== null) {
        ended = true;
        watching = false;
        runtime?.stop();
        try {
          ReplayStore.save(
            rec.finish({
              scoreHome: match.state.scoreHome,
              scoreAway: match.state.scoreAway,
              winner: match.state.winner,
            }),
          );
        } catch { /* replay storage is best-effort */ }
        onEnd(match.state.winner, match.state.scoreHome, match.state.scoreAway);
      }
    },
    () => post.render(),
  );
  runtime.start();
}

function openSpectate(): void {
  spectate.show(TEAMS, {
    onWatch: (h, a) => startWatch(h, a),
    onWatchBracket: () => startWatchBracket(),
    onBack: () => landing.show(() => title.show(enterJump), openSpectate, openReplay),
  });
}

function exitWatch(): void {
  watching = false;
  runtime?.stop();
  spectateControls.hide();
  openSpectate();
}

function bindWatchControls(): void {
  spectateControls.show({
    onSpeed: (m) => {
      runtime?.setTimeScale(m);
      spectateControls.setSpeed(m);
    },
    onCycleCam: () => gcam.cycleCinematicAngle(),
    onExit: exitWatch,
  });
}

function startWatch(home: Franchise, away: Franchise): void {
  bindWatchControls();
  const seed = (Math.random() * 0xffffffff) >>> 0;
  void runWatch(home, away, seed, () => {
    if (!watching && spectateControls) {
      // single-match: surface the result briefly, then back to the menu
      spectateControls.hide();
      openSpectate();
    }
  });
}

function startWatchBracket(): void {
  const wb = new Bracket((Math.random() * 0xffffffff) >>> 0);
  let aborted = false;
  spectateControls.show({
    onSpeed: (m) => {
      runtime?.setTimeScale(m);
      spectateControls.setSpeed(m);
    },
    onCycleCam: () => gcam.cycleCinematicAngle(),
    onExit: () => {
      aborted = true;
      exitWatch();
    },
  });
  const playNextGame = (): void => {
    if (aborted) return;
    const g = wb.nextGame();
    if (!g || !g.home || !g.away) {
      runtime?.stop();
      spectateControls.hide();
      bracketUI.show(wb.state as never, '', openSpectate);
      return;
    }
    const home = g.home;
    const away = g.away;
    void runWatch(home, away, g.gameSeed, (w, sh, sa) => {
      const result: MatchResult = {
        home, away, scoreHome: sh, scoreAway: sa, winner: w,
        box: { home: ZBOX(), away: ZBOX() }, events: [],
      };
      wb.playNext(result);
      // chain into the next game unless the user bailed out
      if (!aborted) playNextGame();
    });
  };
  playNextGame();
}

// ── The re-call: deterministically replay a recorded match ────────────────────
async function runReplay(
  data: ReplayData,
  onEnd: () => void,
): Promise<void> {
  const sim: WasmSim = await createWasmSim(data.meta.seed);
  const roster = data.roster;
  for (let k = 0; k < roster.length; k++) {
    const r = roster[k];
    const ang = (k / roster.length) * Math.PI * 2;
    sim.addPlayer(r.id, r.team, r.role, { x: r.x, y: Math.cos(ang) * 8, z: Math.sin(ang) * 8 });
  }
  const home0 = roster.find((r) => r.team === 'home') ?? roster[0];
  sim.setBellHeld(home0.id);
  const match = new MatchStateMachine(data.meta.faithEnd, data.meta.firstPossession);
  match.consume([{ type: 'foul_garrote', by: '__start__' }], sim.snapshot());
  trail.clear();
  gcam.reset();
  riggers.setViewScale(5); // far cinematic cam → larger-than-life figures
  let fi = 0;
  let ended = false;
  let prevLoop = false;

  const reArm = (): void => {
    if (match.state.winner !== null || match.state.phase === 'live') return;
    const poss = match.state.possession;
    sim.setBellHeld((roster.find((r) => r.team === poss) ?? roster[0]).id);
    match.resumeLive();
  };

  const finish = (): void => {
    if (ended) return;
    ended = true;
    runtime?.stop();
    onEnd();
  };

  // One deterministic sim tick (replay frame playback): exactly the prior
  // per-tick body, with the frame-exhaustion / winner guards as early-returns.
  const stepOnce = (): void => {
    if (match.state.winner !== null || fi >= data.frames.length) return;
    const frame = data.frames[fi++];
    const evs = sim.step(frame);
    const upd = match.consume(evs, sim.snapshot() as never);
    for (const e of evs) {
      if (e.type === 'bell_caught') audio.event('catch');
      else if (e.type === 'bell_clatter' || e.type === 'bell_bobble') audio.event('clatter');
    }
    if (upd && upd.scored) {
      const kd = upd.scored.kind;
      audio.event(
        kd === 'loop' ? 'score_loop' : kd === 'rise' || kd === 'curl' ? 'score_rise'
          : kd === 'ground' ? 'score_ground' : 'score_fall',
      );
    } else if (upd && upd.turnover) {
      audio.event('turnover');
    }
    reArm();
  };
  const driver = new FixedStepDriver(stepOnce, () => sim.snapshot());

  runtime?.stop();
  runtime = new GameRuntime(
    (dt) => {
      calm.update(dt);
      distantHabitat.update(dt);
      driver.advance(dt);
      const s = renderState(driver.prev as RenderView, driver.cur as RenderView, driver.alpha);
      const lg = s.loopTier === 'loop' ? 1 : s.loopTier === 'curl' ? 0.4 : 0;
      const isLoop = s.loopTier === 'loop';
      bellMesh.position.set(s.bell.p.x, s.bell.p.y, s.bell.p.z);
      (bellMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = bellGlow(s.bell.chime);
      trail.push(s.bell.p.x, s.bell.p.y, s.bell.p.z, s.bell.chime);
      trail.setLoopMode(isLoop);
      post.setLoopGlow(lg);
      riggers.sync(s.players, '');
      for (const pp of s.players) riggers.setPoseEmphasis(pp.id, pp.id === s.bell.heldBy ? 1 : 0);
      riglines.sync(s.players);
      audio.setBell(s.bell.chime, Math.hypot(s.bell.w.x, s.bell.w.y, s.bell.w.z),
        Math.max(-1, Math.min(1, s.bell.p.z / REG.R)), s.bell.heldBy === null);
      audio.setHush(lg);
      if (isLoop && !prevLoop) audio.event('loop_building');
      prevLoop = isLoop;
      const atkX = match.state.possession === 'home' ? GATE_X : -GATE_X;
      const cinePlayers = s.players.map((pp) => ({ id: pp.id, p: pp.p, team: pp.team }));
      gcam.setControl(input.cameraState);
      gcam.cinematic(s.bell.p, cinePlayers, atkX, lg, REG.R, Math.min(dt, 1 / 30));
      hud.render(s as never, match.state as never, input.view);

      if (!ended && (match.state.winner !== null || fi >= data.frames.length)) {
        finish();
      }
    },
    () => post.render(),
  );
  runtime.start();
}

function openReplay(): void {
  replayScreen.show({
    onPlay: (data) => {
      spectateControls.show({
        onSpeed: (m) => {
          runtime?.setTimeScale(m);
          spectateControls.setSpeed(m);
        },
        onCycleCam: () => gcam.cycleCinematicAngle(),
        onExit: () => {
          runtime?.stop();
          spectateControls.hide();
          openReplay();
        },
      });
      void runReplay(data, () => {
        spectateControls.hide();
        openReplay();
      });
    },
    onDelete: () => { /* ReplayScreen removes from store + refreshes itself */ },
    onBack: () => landing.show(() => title.show(enterJump), openSpectate, openReplay),
  });
}

landing.show(() => title.show(enterJump), openSpectate, openReplay);
console.info('RIG v2 P7 — shell + spectate + re-call. master seed %s', masterSeed);

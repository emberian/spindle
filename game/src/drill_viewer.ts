// Drill Viewer — a grid of skill drills shown as lightweight 2D top-down views.
// Each cell runs its own WASM sim + AI (cheap) and renders as simple colored
// dots on a 2D canvas (near-zero GPU cost). The 3D renderer is NOT used here.

import { FixedStepDriver } from './core/FixedStepDriver';
import { createWasmSim, type WasmSim } from './sim/wasm';
import { AiSystem, type TeamConfig } from './ai/index';
import { MatchStateMachine } from './match/MatchStateMachine';
import { styleToProfile } from './league/teams';
import type { TeamSide, RiggerRole } from './sim/types';

// ── Drill Definitions ───────────────────────────────────────────────────────

interface DrillSetup {
  name: string;
  label: string;
  players: { id: string; team: TeamSide; role: RiggerRole; x: number; y: number; z: number }[];
  ballSetup: 'held' | { pos: [number, number, number]; vel: [number, number, number]; thrownBy?: string };
  ballHolder?: string;
  configs: TeamConfig[];
  maxTicks: number;
}

const homePro: TeamConfig = { side: 'home', profile: styleToProfile('fall-dynasty', 'big-slow'), difficulty: 'pro' };
const awayPro: TeamConfig = { side: 'away', profile: styleToProfile('fall-dynasty', 'big-slow'), difficulty: 'pro' };
const awayChaos: TeamConfig = { side: 'away', profile: styleToProfile('rise-chaos', 'small-fast'), difficulty: 'pro' };
const homeRise: TeamConfig = { side: 'home', profile: styleToProfile('rise-power', 'small-fast'), difficulty: 'pro' };
const awayRise: TeamConfig = { side: 'away', profile: styleToProfile('rise-power', 'small-fast'), difficulty: 'pro' };

function fullRoster(): DrillSetup['players'] {
  const roster: [string, TeamSide, RiggerRole, number][] = [
    ['H1', 'home', 'spinner', -40], ['H2', 'home', 'anchor', -90],
    ['H3', 'home', 'faithwing', -20], ['H4', 'home', 'reach', -260],
    ['A1', 'away', 'spinner', 40], ['A2', 'away', 'anchor', 90],
    ['A3', 'away', 'freewing', 20], ['A4', 'away', 'reach', 260],
  ];
  return roster.map(([id, team, role, x], k) => ({
    id, team, role, x,
    y: Math.cos((k / roster.length) * Math.PI * 2) * 8,
    z: Math.sin((k / roster.length) * Math.PI * 2) * 8,
  }));
}

const DRILLS: DrillSetup[] = [
  {
    name: 'Catch', label: 'Catch (30m 20m/s)',
    players: [{ id: 'C1', team: 'home', role: 'spinner', x: 0, y: 10, z: 0 }],
    ballSetup: { pos: [30, 10, 0], vel: [-20, 0, 0] },
    configs: [homePro], maxTicks: 720,
  },
  {
    name: 'Pursuit', label: 'Pursuit (5m 18m/s)',
    players: [{ id: 'H1', team: 'home', role: 'spinner', x: 0, y: 10, z: 0 }],
    ballSetup: { pos: [-40, 15, 0], vel: [18, 0, 0], thrownBy: '__pass__' },
    configs: [homePro], maxTicks: 1800,
  },
  {
    name: 'Pass 30m', label: 'Pass (30m)',
    players: [
      { id: 'C1', team: 'home', role: 'spinner', x: -100, y: 10, z: 0 },
      { id: 'R1', team: 'home', role: 'anchor', x: -70, y: 10, z: 0 },
    ],
    ballSetup: 'held', ballHolder: 'C1',
    configs: [homePro], maxTicks: 2400,
  },
  {
    name: 'Gate Score', label: 'Score (80m)',
    players: [{ id: 'H1', team: 'home', role: 'spinner', x: 240, y: 10, z: 0 }],
    ballSetup: 'held', ballHolder: 'H1',
    configs: [homePro], maxTicks: 2400,
  },
  {
    name: 'Race', label: 'Loose Ball Race',
    players: [
      { id: 'H1', team: 'home', role: 'spinner', x: -30, y: 10, z: 0 },
      { id: 'A1', team: 'away', role: 'spinner', x: 30, y: 10, z: 0 },
    ],
    ballSetup: { pos: [0, 10, 0], vel: [0, 0, 0], thrownBy: '__pass__' },
    configs: [homePro, awayPro], maxTicks: 1200,
  },
  {
    name: 'Dynasty Mirror', label: 'Fall-Dynasty 4v4',
    players: fullRoster(),
    ballSetup: 'held', ballHolder: 'H1',
    configs: [homePro, awayPro], maxTicks: 6000,
  },
  {
    name: 'Cross-Style', label: 'Dynasty vs Chaos',
    players: fullRoster(),
    ballSetup: 'held', ballHolder: 'H1',
    configs: [homePro, awayChaos], maxTicks: 6000,
  },
  {
    name: 'Rise Mirror', label: 'Rise-Power 4v4',
    players: fullRoster(),
    ballSetup: 'held', ballHolder: 'H1',
    configs: [homeRise, awayRise], maxTicks: 6000,
  },
  {
    name: 'Two Pass', label: 'A→B→C Chain',
    players: [
      { id: 'H1', team: 'home', role: 'spinner', x: -120, y: 10, z: 0 },
      { id: 'H2', team: 'home', role: 'anchor', x: -80, y: 10, z: 0 },
      { id: 'H3', team: 'home', role: 'faithwing', x: -40, y: 10, z: 0 },
    ],
    ballSetup: 'held', ballHolder: 'H1',
    configs: [homePro], maxTicks: 4800,
  },
  {
    name: '3v2', label: '3v2 Advance',
    players: [
      { id: 'H1', team: 'home', role: 'spinner', x: -80, y: 10, z: 0 },
      { id: 'H2', team: 'home', role: 'anchor', x: -60, y: 20, z: 0 },
      { id: 'H3', team: 'home', role: 'faithwing', x: -40, y: 5, z: 0 },
      { id: 'A1', team: 'away', role: 'spinner', x: 0, y: 12, z: 0 },
      { id: 'A2', team: 'away', role: 'anchor', x: 20, y: 8, z: 0 },
    ],
    ballSetup: 'held', ballHolder: 'H1',
    configs: [homePro, awayPro], maxTicks: 3600,
  },
  {
    name: 'Fast Break', label: 'Transition',
    players: [
      { id: 'H1', team: 'home', role: 'spinner', x: -20, y: 12, z: 0 },
      { id: 'H2', team: 'home', role: 'anchor', x: 60, y: 8, z: 0 },
      { id: 'A1', team: 'away', role: 'spinner', x: 20, y: 10, z: 5 },
      { id: 'A2', team: 'away', role: 'anchor', x: -60, y: 10, z: -5 },
    ],
    ballSetup: { pos: [0, 10, 0], vel: [5, 3, 0], thrownBy: '__pass__' },
    configs: [homePro, awayPro], maxTicks: 2400,
  },
  {
    name: 'Pass 60m', label: 'Long Pass (60m)',
    players: [
      { id: 'C1', team: 'home', role: 'spinner', x: -100, y: 10, z: 0 },
      { id: 'R1', team: 'home', role: 'anchor', x: -40, y: 10, z: 0 },
    ],
    ballSetup: 'held', ballHolder: 'C1',
    configs: [homePro], maxTicks: 2400,
  },
];

// ── 2D Drill Cell (canvas-based, ultra lightweight) ─────────────────────────

const R = 45; // cylinder radius

class DrillCell {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private sim: WasmSim | null = null;
  private ai: AiSystem | null = null;
  private matchSm: MatchStateMachine | null = null;
  private stepper: FixedStepDriver | null = null;
  private drill: DrillSetup;
  private tick = 0;
  private outcome: 'running' | 'pass' | 'fail' = 'running';
  private restartTimer = 0;
  private statusEl: HTMLElement;
  // Grapple line trail (per player, last N positions)
  private trails: Map<string, { x: number; y: number }[]> = new Map();
  private ballTrail: { x: number; y: number }[] = [];

  constructor(private container: HTMLElement, drill: DrillSetup) {
    this.drill = drill;
    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;

    const labelEl = document.createElement('div');
    labelEl.className = 'label';
    labelEl.textContent = drill.label;
    container.appendChild(labelEl);

    this.statusEl = document.createElement('div');
    this.statusEl.className = 'status running';
    this.statusEl.textContent = '...';
    container.appendChild(this.statusEl);

    this.init();
  }

  private async init() {
    await this.setupSim();
    this.stepper = new FixedStepDriver(
      () => this.simTick(),
      () => this.sim ? this.sim.snapshot() : ({} as any),
    );
  }

  private async setupSim() {
    const seed = (Math.random() * 0xffffffff) >>> 0;
    this.sim = await createWasmSim(seed);
    for (const p of this.drill.players) {
      this.sim.addPlayer(p.id, p.team, p.role, { x: p.x, y: p.y, z: p.z });
    }
    if (this.drill.ballSetup === 'held') {
      this.sim.setBellHeld(this.drill.ballHolder || this.drill.players[0].id);
    } else {
      const bs = this.drill.ballSetup;
      this.sim.launchBell(
        { x: bs.pos[0], y: bs.pos[1], z: bs.pos[2] },
        { x: bs.vel[0], y: bs.vel[1], z: bs.vel[2] },
        { x: 0, y: 0, z: 0 }, bs.thrownBy || '',
      );
    }
    this.ai = new AiSystem();
    this.matchSm = new MatchStateMachine('+x', 'home');
    this.matchSm.consume([{ type: 'foul_garrote', by: '__start__' }], this.sim.snapshot());
    this.tick = 0;
    this.outcome = 'running';
    this.restartTimer = 0;
    this.trails.clear();
    this.ballTrail = [];
    this.statusEl.className = 'status running';
    this.statusEl.textContent = '...';
  }

  private simTick() {
    if (!this.sim || !this.ai || !this.matchSm) return;
    if (this.outcome !== 'running') {
      this.restartTimer++;
      if (this.restartTimer > 480) void this.setupSim();
      return;
    }
    if (this.tick >= this.drill.maxTicks) {
      this.outcome = 'fail';
      this.statusEl.className = 'status fail';
      this.statusEl.textContent = 'TIMEOUT';
      return;
    }

    const snap = this.sim.snapshot();
    if (snap.bell.heldBy && this.tick > 2) {
      const holder = snap.bell.heldBy;
      const isNew = holder !== this.drill.ballHolder;
      if (isNew || this.drill.ballSetup !== 'held') {
        this.outcome = 'pass';
        this.statusEl.className = 'status pass';
        this.statusEl.textContent = `CATCH ${(this.tick / 240).toFixed(1)}s`;
        return;
      }
    }

    const frame = this.ai.tick(snap, this.matchSm.state, this.drill.configs, 1234);
    const events = this.sim.step(frame);
    this.matchSm.consume(events, this.sim.snapshot());
    this.tick++;

    for (const e of events) {
      if (e.type === 'bell_through_ring') {
        this.outcome = 'pass';
        this.statusEl.className = 'status pass';
        this.statusEl.textContent = `SCORE ${(this.tick / 240).toFixed(1)}s`;
        return;
      }
    }

    if (this.matchSm.state.phase !== 'live' && !this.matchSm.state.winner) {
      const poss = this.matchSm.state.possession;
      const p = this.drill.players.find(r => r.team === poss);
      if (p) this.sim.setBellHeld(p.id);
      this.matchSm.resumeLive();
    }
  }

  advance(dt: number) {
    if (!this.stepper) return;
    this.stepper.advance(dt);
  }

  render() {
    if (!this.sim) return;
    const snap = this.sim.snapshot();
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;

    const dpr = Math.min(devicePixelRatio, 2);
    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
    }

    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d0f16';
    ctx.fillRect(0, 0, w, h);

    // Determine view bounds from players + ball
    let minX = -50, maxX = 50;
    for (const p of snap.players) {
      minX = Math.min(minX, p.p.x - 30);
      maxX = Math.max(maxX, p.p.x + 30);
    }
    minX = Math.min(minX, snap.bell.p.x - 20);
    maxX = Math.max(maxX, snap.bell.p.x + 20);
    const rangeX = Math.max(maxX - minX, 80);
    const cx = (minX + maxX) / 2;
    const scale = (w - 20) / rangeX;
    const toScreenX = (x: number) => (x - cx) * scale + w / 2;
    const toScreenY = (y: number) => h / 2 - y * scale;

    // Draw cylinder skin outline (ellipse in side view)
    ctx.strokeStyle = '#1a2030';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(w / 2, h / 2, rangeX / 2 * scale, R * scale * 0.7, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Draw ball trail
    const bp = snap.bell.p;
    this.ballTrail.push({ x: bp.x, y: bp.y });
    if (this.ballTrail.length > 60) this.ballTrail.shift();
    if (this.ballTrail.length > 1) {
      ctx.strokeStyle = '#44ffcc44';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(toScreenX(this.ballTrail[0].x), toScreenY(this.ballTrail[0].y));
      for (let i = 1; i < this.ballTrail.length; i++) {
        ctx.lineTo(toScreenX(this.ballTrail[i].x), toScreenY(this.ballTrail[i].y));
      }
      ctx.stroke();
    }

    // Draw player trails
    for (const p of snap.players) {
      let trail = this.trails.get(p.id);
      if (!trail) { trail = []; this.trails.set(p.id, trail); }
      trail.push({ x: p.p.x, y: p.p.y });
      if (trail.length > 30) trail.shift();
      if (trail.length > 1) {
        const c = p.team === 'home' ? '#3388ff22' : '#ff553322';
        ctx.strokeStyle = c;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(toScreenX(trail[0].x), toScreenY(trail[0].y));
        for (let i = 1; i < trail.length; i++) {
          ctx.lineTo(toScreenX(trail[i].x), toScreenY(trail[i].y));
        }
        ctx.stroke();
      }
    }

    // Draw grapple lines
    for (const p of snap.players) {
      if (p.line) {
        ctx.strokeStyle = p.team === 'home' ? '#3388ff66' : '#ff553366';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(toScreenX(p.p.x), toScreenY(p.p.y));
        ctx.lineTo(toScreenX(p.line.anchorPos.x), toScreenY(p.line.anchorPos.y));
        ctx.stroke();
      }
    }

    // Draw players (big circles with team color)
    for (const p of snap.players) {
      const isHolder = snap.bell.heldBy === p.id;
      const color = p.team === 'home' ? '#3388ff' : '#ff5533';
      const sx = toScreenX(p.p.x);
      const sy = toScreenY(p.p.y);
      const r = isHolder ? 7 : 5;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fill();

      if (isHolder) {
        ctx.strokeStyle = '#44ffcc';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Player id label
      ctx.fillStyle = '#ffffffaa';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(p.id, sx, sy - r - 3);
    }

    // Draw ball (if not held)
    if (!snap.bell.heldBy) {
      const bx = toScreenX(bp.x);
      const by = toScreenY(bp.y);
      ctx.fillStyle = '#44ffcc';
      ctx.beginPath();
      ctx.arc(bx, by, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#44ffcc88';
      ctx.lineWidth = 1;
      // Velocity indicator
      const vScale = 0.15 * scale;
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(bx + snap.bell.v.x * vScale, by - snap.bell.v.y * vScale);
      ctx.stroke();
    }

    // Tick counter
    ctx.fillStyle = '#6b707988';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${(this.tick / 240).toFixed(1)}s`, w - 4, h - 4);
  }

  get result(): 'running' | 'pass' | 'fail' { return this.outcome; }
}

// ── Setup Grid ──────────────────────────────────────────────────────────────

const grid = document.getElementById('grid')!;
const statsEl = document.getElementById('stats')!;
const cells: DrillCell[] = [];

for (const drill of DRILLS) {
  const container = document.createElement('div');
  container.className = 'cell';
  grid.appendChild(container);
  cells.push(new DrillCell(container, drill));
}

// ── Main Loop ───────────────────────────────────────────────────────────────

let lastTime = 0;

function loop(now: number) {
  requestAnimationFrame(loop);
  if (lastTime === 0) { lastTime = now; return; }
  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;

  // Advance all sims + render all (2D canvas is cheap)
  for (const cell of cells) {
    cell.advance(dt * 2);
    cell.render();
  }

  // Update stats
  const pass = cells.filter(c => c.result === 'pass').length;
  const fail = cells.filter(c => c.result === 'fail').length;
  const running = cells.filter(c => c.result === 'running').length;
  statsEl.innerHTML = `<span class="pass">${pass} pass</span> · ${fail} fail · ${running} running`;
}

requestAnimationFrame(loop);

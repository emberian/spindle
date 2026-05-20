// Drill Viewer — grid of skill drills running in Web Workers.
// Each cell spawns a worker that runs WASM sim + AI off the main thread.
// Main thread only renders 2D canvas from posted snapshots. Buttery smooth.

import type { TeamSide, RiggerRole, SimState } from './sim/types';

// ── Drill Definitions ───────────────────────────────────────────────────────

interface DrillDef {
  label: string;
  players: { id: string; team: TeamSide; role: RiggerRole; x: number; y: number; z: number }[];
  ballSetup: 'held' | { pos: [number, number, number]; vel: [number, number, number]; thrownBy?: string };
  ballHolder?: string;
  configs: { side: TeamSide; style: string; cyl: string; difficulty: string }[];
  maxTicks: number;
}

function fullRoster(): DrillDef['players'] {
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

const HP = { side: 'home' as TeamSide, style: 'fall-dynasty', cyl: 'big-slow', difficulty: 'pro' };
const AP = { side: 'away' as TeamSide, style: 'fall-dynasty', cyl: 'big-slow', difficulty: 'pro' };
const AC = { side: 'away' as TeamSide, style: 'rise-chaos', cyl: 'small-fast', difficulty: 'pro' };
const HR = { side: 'home' as TeamSide, style: 'rise-power', cyl: 'small-fast', difficulty: 'pro' };
const AR = { side: 'away' as TeamSide, style: 'rise-power', cyl: 'small-fast', difficulty: 'pro' };

const DRILLS: DrillDef[] = [
  { label: 'Catch (30m 20m/s)', players: [{ id: 'C1', team: 'home', role: 'spinner', x: 0, y: 10, z: 0 }],
    ballSetup: { pos: [30, 10, 0], vel: [-20, 0, 0] }, configs: [HP], maxTicks: 720 },
  { label: 'Pursuit (5m 18m/s)', players: [{ id: 'H1', team: 'home', role: 'spinner', x: 0, y: 10, z: 0 }],
    ballSetup: { pos: [-40, 15, 0], vel: [18, 0, 0], thrownBy: '__pass__' }, configs: [HP], maxTicks: 1800 },
  { label: 'Pass (30m)', players: [
    { id: 'C1', team: 'home', role: 'spinner', x: -100, y: 10, z: 0 },
    { id: 'R1', team: 'home', role: 'anchor', x: -70, y: 10, z: 0 }],
    ballSetup: 'held', ballHolder: 'C1', configs: [HP], maxTicks: 2400 },
  { label: 'Score (80m)', players: [{ id: 'H1', team: 'home', role: 'spinner', x: 240, y: 10, z: 0 }],
    ballSetup: 'held', ballHolder: 'H1', configs: [HP], maxTicks: 2400 },
  { label: 'Loose Ball Race', players: [
    { id: 'H1', team: 'home', role: 'spinner', x: -30, y: 10, z: 0 },
    { id: 'A1', team: 'away', role: 'spinner', x: 30, y: 10, z: 0 }],
    ballSetup: { pos: [0, 10, 0], vel: [0, 0, 0], thrownBy: '__pass__' },
    configs: [HP, AP], maxTicks: 1200 },
  { label: 'Fall-Dynasty 4v4', players: fullRoster(),
    ballSetup: 'held', ballHolder: 'H1', configs: [HP, AP], maxTicks: 6000 },
  { label: 'Dynasty vs Chaos', players: fullRoster(),
    ballSetup: 'held', ballHolder: 'H1', configs: [HP, AC], maxTicks: 6000 },
  { label: 'Rise-Power 4v4', players: fullRoster(),
    ballSetup: 'held', ballHolder: 'H1', configs: [HR, AR], maxTicks: 6000 },
  { label: 'A→B→C Chain', players: [
    { id: 'H1', team: 'home', role: 'spinner', x: -120, y: 10, z: 0 },
    { id: 'H2', team: 'home', role: 'anchor', x: -80, y: 10, z: 0 },
    { id: 'H3', team: 'home', role: 'faithwing', x: -40, y: 10, z: 0 }],
    ballSetup: 'held', ballHolder: 'H1', configs: [HP], maxTicks: 4800 },
  { label: '3v2 Advance', players: [
    { id: 'H1', team: 'home', role: 'spinner', x: -80, y: 10, z: 0 },
    { id: 'H2', team: 'home', role: 'anchor', x: -60, y: 20, z: 0 },
    { id: 'H3', team: 'home', role: 'faithwing', x: -40, y: 5, z: 0 },
    { id: 'A1', team: 'away', role: 'spinner', x: 0, y: 12, z: 0 },
    { id: 'A2', team: 'away', role: 'anchor', x: 20, y: 8, z: 0 }],
    ballSetup: 'held', ballHolder: 'H1', configs: [HP, AP], maxTicks: 3600 },
  { label: 'Fast Break', players: [
    { id: 'H1', team: 'home', role: 'spinner', x: -20, y: 12, z: 0 },
    { id: 'H2', team: 'home', role: 'anchor', x: 60, y: 8, z: 0 },
    { id: 'A1', team: 'away', role: 'spinner', x: 20, y: 10, z: 5 },
    { id: 'A2', team: 'away', role: 'anchor', x: -60, y: 10, z: -5 }],
    ballSetup: { pos: [0, 10, 0], vel: [5, 3, 0], thrownBy: '__pass__' },
    configs: [HP, AP], maxTicks: 2400 },
  { label: 'Long Pass (60m)', players: [
    { id: 'C1', team: 'home', role: 'spinner', x: -100, y: 10, z: 0 },
    { id: 'R1', team: 'home', role: 'anchor', x: -40, y: 10, z: 0 }],
    ballSetup: 'held', ballHolder: 'C1', configs: [HP], maxTicks: 2400 },
];

// ── Cell (renders from worker-posted snapshots) ─────────────────────────────

const R = 45;

class DrillCell {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private worker: Worker;
  private snap: SimState | null = null;
  private tick = 0;
  private outcome: 'running' | 'pass' | 'fail' = 'running';
  private statusEl: HTMLElement;
  private ballTrail: { x: number; y: number }[] = [];
  private playerTrails: Map<string, { x: number; y: number }[]> = new Map();
  private drill: DrillDef;
  private id: number;
  private tickInterval: ReturnType<typeof setInterval> | null = null;

  constructor(container: HTMLElement, drill: DrillDef, id: number) {
    this.drill = drill;
    this.id = id;

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

    this.worker = new Worker(
      new URL('./drill_worker.ts', import.meta.url),
      { type: 'module' },
    );
    this.worker.onmessage = (e) => this.onMessage(e.data);
    this.startDrill();
  }

  private startDrill() {
    this.outcome = 'running';
    this.ballTrail = [];
    this.playerTrails.clear();
    this.tick = 0;
    this.statusEl.className = 'status running';
    this.statusEl.textContent = '...';

    this.worker.postMessage({
      type: 'setup',
      config: {
        id: this.id,
        players: this.drill.players,
        ballSetup: this.drill.ballSetup,
        ballHolder: this.drill.ballHolder,
        configs: this.drill.configs,
        maxTicks: this.drill.maxTicks,
        seed: (Math.random() * 0xffffffff) >>> 0,
      },
    });
  }

  private onMessage(msg: any) {
    if (msg.type === 'ready') {
      // Start ticking at ~30 Hz sim cadence (8 ticks per message at 240Hz = 30 msg/s)
      if (this.tickInterval) clearInterval(this.tickInterval);
      this.tickInterval = setInterval(() => {
        if (this.outcome === 'running') {
          this.worker.postMessage({ type: 'tick', n: 8 });
        }
      }, 33);
    } else if (msg.type === 'snapshot') {
      this.snap = msg.snap;
      this.tick = msg.tick;
      if (msg.outcome !== 'running') {
        this.outcome = msg.outcome;
      }
    } else if (msg.type === 'outcome') {
      this.outcome = msg.outcome;
      this.tick = msg.tick;
      if (msg.outcome === 'pass') {
        this.statusEl.className = 'status pass';
        this.statusEl.textContent = msg.by === 'score'
          ? `SCORE ${(msg.tick / 240).toFixed(1)}s`
          : `CATCH ${(msg.tick / 240).toFixed(1)}s`;
      } else {
        this.statusEl.className = 'status fail';
        this.statusEl.textContent = 'TIMEOUT';
      }
      // Restart after 2s
      setTimeout(() => this.startDrill(), 2000);
    }
  }

  render() {
    if (!this.snap) return;
    const snap = this.snap;
    const container = this.canvas.parentElement!;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;

    const dpr = Math.min(devicePixelRatio, 1.5);
    const cw = Math.round(w * dpr);
    const ch = Math.round(h * dpr);
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw;
      this.canvas.height = ch;
    }

    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d0f16';
    ctx.fillRect(0, 0, w, h);

    // Auto-zoom view bounds
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
    const toX = (x: number) => (x - cx) * scale + w / 2;
    const toY = (y: number) => h / 2 - y * scale;

    // Cylinder skin
    ctx.strokeStyle = '#1a2030';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(w / 2, h / 2, rangeX / 2 * scale, R * scale * 0.7, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Ball trail
    this.ballTrail.push({ x: snap.bell.p.x, y: snap.bell.p.y });
    if (this.ballTrail.length > 40) this.ballTrail.shift();
    if (this.ballTrail.length > 1) {
      ctx.strokeStyle = '#44ffcc55';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(toX(this.ballTrail[0].x), toY(this.ballTrail[0].y));
      for (const pt of this.ballTrail) ctx.lineTo(toX(pt.x), toY(pt.y));
      ctx.stroke();
    }

    // Player trails
    for (const p of snap.players) {
      let trail = this.playerTrails.get(p.id);
      if (!trail) { trail = []; this.playerTrails.set(p.id, trail); }
      trail.push({ x: p.p.x, y: p.p.y });
      if (trail.length > 20) trail.shift();
      if (trail.length > 1) {
        ctx.strokeStyle = p.team === 'home' ? '#3388ff22' : '#ff553322';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(toX(trail[0].x), toY(trail[0].y));
        for (const pt of trail) ctx.lineTo(toX(pt.x), toY(pt.y));
        ctx.stroke();
      }
    }

    // Grapple lines
    for (const p of snap.players) {
      if (p.line) {
        ctx.strokeStyle = p.team === 'home' ? '#3388ff55' : '#ff553355';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(toX(p.p.x), toY(p.p.y));
        ctx.lineTo(toX(p.line.anchorPos.x), toY(p.line.anchorPos.y));
        ctx.stroke();
      }
    }

    // Players
    for (const p of snap.players) {
      const isHolder = snap.bell.heldBy === p.id;
      const color = p.team === 'home' ? '#3388ff' : '#ff5533';
      const sx = toX(p.p.x), sy = toY(p.p.y);
      const r = isHolder ? 6 : 4;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fill();
      if (isHolder) {
        ctx.strokeStyle = '#44ffcc';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.fillStyle = '#ffffffaa';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(p.id, sx, sy - r - 3);
    }

    // Ball (if free)
    if (!snap.bell.heldBy) {
      const bx = toX(snap.bell.p.x), by = toY(snap.bell.p.y);
      ctx.fillStyle = '#44ffcc';
      ctx.beginPath();
      ctx.arc(bx, by, 4, 0, Math.PI * 2);
      ctx.fill();
      const vScale = 0.12 * scale;
      ctx.strokeStyle = '#44ffcc88';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(bx + snap.bell.v.x * vScale, by - snap.bell.v.y * vScale);
      ctx.stroke();
    }

    // Tick/time
    ctx.fillStyle = '#6b707966';
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

for (let i = 0; i < DRILLS.length; i++) {
  const container = document.createElement('div');
  container.className = 'cell';
  grid.appendChild(container);
  cells.push(new DrillCell(container, DRILLS[i], i));
}

// ── Render loop (main thread only renders, no sim work) ─────────────────────

function renderLoop() {
  requestAnimationFrame(renderLoop);
  for (const cell of cells) cell.render();

  const pass = cells.filter(c => c.result === 'pass').length;
  const fail = cells.filter(c => c.result === 'fail').length;
  const running = cells.filter(c => c.result === 'running').length;
  statsEl.innerHTML = `<span class="pass">${pass} pass</span> · ${fail} fail · ${running} running`;
}

requestAnimationFrame(renderLoop);

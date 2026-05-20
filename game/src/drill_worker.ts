// Drill Worker — runs a single WASM sim + AI off the main thread.
// Receives setup commands, ticks autonomously, posts snapshots back.

import { createWasmSim, type WasmSim } from './sim/wasm';
import { AiSystem, type TeamConfig } from './ai/index';
import { MatchStateMachine } from './match/MatchStateMachine';
import { styleToProfile } from './league/teams';
import type { TeamSide, RiggerRole, SimState } from './sim/types';

interface DrillConfig {
  id: number;
  players: { id: string; team: TeamSide; role: RiggerRole; x: number; y: number; z: number }[];
  ballSetup: 'held' | { pos: [number, number, number]; vel: [number, number, number]; thrownBy?: string };
  ballHolder?: string;
  configs: { side: TeamSide; style: string; cyl: string; difficulty: string }[];
  maxTicks: number;
  seed: number;
}

let sim: WasmSim | null = null;
let ai: AiSystem | null = null;
let matchSm: MatchStateMachine | null = null;
let tick = 0;
let maxTicks = 4800;
let drillId = 0;
let ballHolder: string | undefined;
let ballIsHeld = false;
let outcome: 'running' | 'pass' | 'fail' = 'running';
let configs: TeamConfig[] = [];
let players: DrillConfig['players'] = [];

async function setup(cfg: DrillConfig) {
  drillId = cfg.id;
  maxTicks = cfg.maxTicks;
  ballHolder = cfg.ballHolder;
  players = cfg.players;
  tick = 0;
  outcome = 'running';

  sim = await createWasmSim(cfg.seed);
  for (const p of cfg.players) {
    sim.addPlayer(p.id, p.team, p.role, { x: p.x, y: p.y, z: p.z });
  }

  ballIsHeld = cfg.ballSetup === 'held';
  if (cfg.ballSetup === 'held') {
    sim.setBellHeld(cfg.ballHolder || cfg.players[0].id);
  } else {
    const bs = cfg.ballSetup;
    sim.launchBell(
      { x: bs.pos[0], y: bs.pos[1], z: bs.pos[2] },
      { x: bs.vel[0], y: bs.vel[1], z: bs.vel[2] },
      { x: 0, y: 0, z: 0 }, bs.thrownBy || '',
    );
  }

  configs = cfg.configs.map(c => ({
    side: c.side,
    profile: styleToProfile(c.style as any, c.cyl as any),
    difficulty: c.difficulty as any,
  }));

  ai = new AiSystem();
  matchSm = new MatchStateMachine('+x', 'home');
  matchSm.consume([{ type: 'foul_garrote', by: '__start__' }], sim.snapshot());

  postMessage({ type: 'ready', id: drillId });
}

function tickOnce(): SimState | null {
  if (!sim || !ai || !matchSm || outcome !== 'running') return null;

  if (tick >= maxTicks) {
    outcome = 'fail';
    postMessage({ type: 'outcome', id: drillId, outcome: 'fail', tick });
    return sim.snapshot();
  }

  const snap = sim.snapshot();

  if (snap.bell.heldBy && tick > 2) {
    const holder = snap.bell.heldBy;
    const isNew = holder !== ballHolder;
    if (isNew || !ballIsHeld) {
      outcome = 'pass';
      postMessage({ type: 'outcome', id: drillId, outcome: 'pass', tick, by: holder });
      return snap;
    }
  }

  const frame = ai.tick(snap, matchSm.state, configs, 1234);
  const events = sim.step(frame);
  matchSm.consume(events, sim.snapshot());
  tick++;

  for (const e of events) {
    if (e.type === 'bell_through_ring') {
      outcome = 'pass';
      postMessage({ type: 'outcome', id: drillId, outcome: 'pass', tick, by: 'score' });
      return sim.snapshot();
    }
  }

  if (matchSm.state.phase !== 'live' && !matchSm.state.winner) {
    const poss = matchSm.state.possession;
    const p = players.find(r => r.team === poss);
    if (p) sim.setBellHeld(p.id);
    matchSm.resumeLive();
  }

  return sim.snapshot();
}

// Tick N times and post the final snapshot (batch for efficiency)
function tickBatch(n: number) {
  let snap: SimState | null = null;
  for (let i = 0; i < n; i++) {
    snap = tickOnce();
    if (outcome !== 'running') break;
  }
  if (snap) {
    postMessage({ type: 'snapshot', id: drillId, snap, tick, outcome });
  }
}

// Message handler
onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'setup':
      void setup(msg.config);
      break;
    case 'tick':
      tickBatch(msg.n || 8);
      break;
  }
};

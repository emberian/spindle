// headless.ts — the FAST in-process skill harness (the real unlock).
//
// The Playwright eval runs the full rendered game at 4× ≈ 6 min/eval —
// impossible to optimise or train against. The AI (ai/**) and the sim
// TWIN (sim/SimWorld) are both pure TS that already run headless in vitest.
// This wires them into a full AI-vs-AI match loop with the SAME skill
// telemetry as the watch loop, in-process, NO browser/WASM/render → a
// match is milliseconds. This is what makes optimisation / RL tractable.
//
// Determinism: TS twin sim + seeded AI; setting globalThis.__rigtune steers
// the planner exactly as the live knobs do. The twin is the model the AI is
// written against, so relative algorithm/knob comparisons are faithful.
import { SimWorld } from '../sim/SimWorld';
import type { InputFrame, SimState, TeamSide, RiggerRole } from '../sim/types';
import { AiSystem, type TeamConfig } from '../ai/index';
import { MatchStateMachine } from '../match/MatchStateMachine';
import { styleToProfile, TEAMS } from '../league/teams';

const ROSTER: { id: string; team: TeamSide; role: RiggerRole; x: number }[] = [
  { id: 'H1', team: 'home', role: 'spinner',   x: -40 },
  { id: 'H2', team: 'home', role: 'anchor',    x: -90 },
  { id: 'H3', team: 'home', role: 'faithwing', x: -20 },
  { id: 'H4', team: 'home', role: 'reach',     x: -260 },
  { id: 'A1', team: 'away', role: 'spinner',   x: 40 },
  { id: 'A2', team: 'away', role: 'anchor',    x: 90 },
  { id: 'A3', team: 'away', role: 'freewing',  x: 20 },
  { id: 'A4', team: 'away', role: 'reach',     x: 260 },
];
const H = 1 / 240;
const GATE_ORD: Record<string, number> = { first: 0, deep: 1, mouth: 2 };

// Normalisation — kept identical to e2e/skilleval.spec.ts v2.1 so headless
// and browser numbers are directly comparable. (Argue with these once, here
// and there together.)
const CAP = {
  gate: 6, pass: 6, intc: 8, score: 30, poss: 0.3, thrashK: 300, skinPct: 80,
  shotConv: 0.4, chain: 4,
};
const WT = {
  prog: 0.18, pass: 0.15, intc: 0.08, score: 0.16, poss: 0.07,
  calm: 0.12, field: 0.06, shotConv: 0.10, chain: 0.08,
};
const SCORE_PTS: Record<string, number> = { fall: 2, rise: 5, loop: 7, curl: 5, ground: 1 };
const MATCHUPS: [number, number][] = [[0, 0], [1, 4], [2, 7]];
const REG_R = 45;
const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

export interface HeadlessSkill {
  label: string; composite: number;
  parts: Record<string, number>; raw: Record<string, number>;
}

function oneMatch(
  tune: Record<string, number>, hRow: number, aRow: number,
  seed: number, maxTicks: number,
): { sub: Record<string, number>; raw: Record<string, number> } {
  (globalThis as { __rigtune?: unknown }).__rigtune = tune;
  const sim = new SimWorld(seed);
  for (let k = 0; k < ROSTER.length; k++) {
    const r = ROSTER[k];
    const ang = (k / ROSTER.length) * Math.PI * 2;
    sim.addPlayer(r.id, r.team, r.role, { x: r.x, y: Math.cos(ang) * 8, z: Math.sin(ang) * 8 });
  }
  sim.setBellHeld('H1');
  const match = new MatchStateMachine('+x', 'home');
  match.consume([{ type: 'foul_garrote', by: '__start__' } as never], sim.snapshot() as never);
  const hf = TEAMS[hRow % TEAMS.length], af = TEAMS[aRow % TEAMS.length];
  const cfgs: TeamConfig[] = [
    { side: 'home', profile: styleToProfile(hf.styleTag, hf.cylinderClass), difficulty: 'pro' },
    { side: 'away', profile: styleToProfile(af.styleTag, af.cylinderClass), difficulty: 'pro' },
  ];
  const ai = new AiSystem();
  const teamOf: Record<string, string> = {};
  for (const r of ROSTER) teamOf[r.id] = r.team;

  let passes = 0, intercepts = 0, gateClears = 0, thrash = 0;
  let streak = 0, streakMax = 0, heldTicks = 0, ticks = 0, skinTicks = 0;
  const scores: Record<string, number> = {};
  const throws: Record<string, number> = {};
  let prevHeld: string | null = null, lastThrownBy: string | null = null;
  let prevGate: string | null = null;
  const prevFire: Record<string, { x: number; y: number; z: number }> = {};

  for (let t = 0; t < maxTicks && match.state.winner === null; t++) {
    const snap: SimState = sim.snapshot();
    const aiFrame = ai.tick(snap, match.state as never, cfgs, seed);
    ticks++;
    if (snap.bell.heldBy) heldTicks++;
    if (Math.hypot(snap.bell.p.y, snap.bell.p.z) > 31) skinTicks++;
    snap.players.forEach((pl, i) => {
      const inp = aiFrame.players[i];
      if (inp && inp.throwReleased) throws[pl.id] = (throws[pl.id] ?? 0) + 1;
      const f = inp && inp.fireLineAt;
      if (f) {
        const pf = prevFire[pl.id];
        if (pf && Math.hypot(f.x - pf.x, f.y - pf.y, f.z - pf.z) > 8) thrash++;
        prevFire[pl.id] = { x: f.x, y: f.y, z: f.z };
      }
    });
    const b = snap.bell;
    if (b.heldBy && !prevHeld && lastThrownBy && lastThrownBy !== b.heldBy) {
      if (teamOf[lastThrownBy] === teamOf[b.heldBy]) {
        passes++; streak++; if (streak > streakMax) streakMax = streak;
      } else { intercepts++; streak = 0; }
    }
    if (!b.heldBy && b.thrownBy) lastThrownBy = b.thrownBy;
    if (b.heldBy) lastThrownBy = null;
    prevHeld = b.heldBy;
    const g = match.state.cast.gate;
    if (prevGate !== null && GATE_ORD[g] > GATE_ORD[prevGate]) gateClears++;
    prevGate = g;

    const frame: InputFrame = { tick: snap.tick, players: aiFrame.players };
    const evs = sim.step(frame, H);
    const upd = match.consume(evs, sim.snapshot() as never);
    for (const e of evs) {
      if (e.type === 'bell_caught') scores.caught = (scores.caught ?? 0) + 1;
      else if (e.type === 'bell_clatter' || e.type === 'bell_bobble') scores.clatter = (scores.clatter ?? 0) + 1;
    }
    if (upd && upd.scored) { scores[upd.scored.kind] = (scores[upd.scored.kind] ?? 0) + 1; streak = 0; }
    else if (upd && upd.turnover) { scores.turnover = (scores.turnover ?? 0) + 1; streak = 0; }
    // re-arm a dead / inning-break ball
    if (match.state.winner === null && match.state.phase !== 'live') {
      const poss = match.state.possession;
      const r = ROSTER.find((x) => x.team === poss);
      if (r) sim.setBellHeld(r.id);
      match.resumeLive();
    }
  }

  const scorePts = Object.entries(scores).reduce((a, [k, v]) => a + (SCORE_PTS[k] ?? 0) * v, 0);
  const scoreEvents = ['fall', 'rise', 'loop', 'curl', 'ground'].reduce((a, k) => a + (scores[k] ?? 0), 0);
  const totalThrows = Object.values(throws).reduce((a, v) => a + v, 0);
  const shotConv = totalThrows > 0 ? scoreEvents / totalThrows : 0;
  const heldFrac = ticks ? heldTicks / ticks : 0;
  const skinPct = ticks ? (100 * skinTicks) / ticks : 100;
  const thrashK = ticks ? (thrash / ticks) * 1000 : 0;
  void REG_R;
  return {
    sub: {
      prog:     clamp01(gateClears / CAP.gate),
      pass:     clamp01(passes / CAP.pass),
      intc:     clamp01(intercepts / CAP.intc),
      score:    clamp01(scorePts / CAP.score),
      poss:     clamp01(heldFrac / CAP.poss),
      calm:     1 - clamp01(thrashK / CAP.thrashK),
      field:    1 - clamp01(skinPct / CAP.skinPct),
      shotConv: clamp01(shotConv / CAP.shotConv),
      chain:    clamp01(streakMax / CAP.chain),
    },
    raw: {
      gateClears, passes, intercepts, scorePts,
      heldFrac: +heldFrac.toFixed(3), thrashK: +thrashK.toFixed(0),
      skinPct: +skinPct.toFixed(0), shotConv: +shotConv.toFixed(2),
      chainMax: streakMax, ticks,
    },
  };
}

/** Full skill eval averaged over the canonical matchups — milliseconds. */
// ~33 s of sim (8000 ticks @ 1/240) is plenty to accumulate stable skill
// signals for RELATIVE comparison; matches rarely resolve a winner headless
// (twin ≠ runtime), so a fixed budget is the right unit anyway. Bigger only
// adds cost — the per-tick planner rollout is the wall (→ Rust next).
export function evalHeadless(
  label: string, tune: Record<string, number>,
  seed = 1234, maxTicks = 8000,
): HeadlessSkill {
  const subs: Record<string, number>[] = [];
  const raws: Record<string, number>[] = [];
  for (const [h, a] of MATCHUPS) {
    const r = oneMatch(tune, h, a, seed, maxTicks);
    subs.push(r.sub); raws.push(r.raw);
  }
  const avg = (k: string, arr: Record<string, number>[]) =>
    arr.reduce((s, o) => s + (o[k] ?? 0), 0) / arr.length;
  const parts: Record<string, number> = {};
  let composite = 0;
  for (const k of Object.keys(WT) as (keyof typeof WT)[]) {
    const v = avg(k, subs);
    parts[k] = +v.toFixed(3);
    composite += WT[k] * v;
  }
  const raw: Record<string, number> = {};
  for (const k of Object.keys(raws[0])) raw[k] = +avg(k, raws).toFixed(1);
  return { label, composite: +(100 * composite).toFixed(1), parts, raw };
}

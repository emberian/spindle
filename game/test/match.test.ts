// match.test.ts — RIG match-rules layer tests.
// Tests: Scoring table, cast/turnover mechanics, contest count-of-3,
//        9 innings → spine on tie → winner, pure/deterministic invariant.

import { describe, it, expect } from 'vitest';
import { scoreFor, groundFor } from '../src/match/Scoring';
import { resolveContest } from '../src/match/Contest';
import { MatchStateMachine } from '../src/match/MatchStateMachine';
import type { SimState, MatchState, SimEvent, TeamSide } from '../src/sim/types';

// ─── Test helpers ─────────────────────────────────────────────────────────────

const ZERO_V = { x: 0, y: 0, z: 0 };
const ZERO_Q = { x: 0, y: 0, z: 0, w: 1 };

function makeSim(overrides: {
  thrownBy?: string | null;
  passChain?: string[];
  heldBy?: string | null;
  bellX?: number;
  players?: SimState['players'];
} = {}): SimState {
  return {
    tick: 0,
    omega: 0.32,
    bell: {
      p: { x: overrides.bellX ?? 0, y: 0, z: 0 },
      v: ZERO_V,
      q: ZERO_Q,
      w: ZERO_V,
      chime: 1,
      heldBy: overrides.heldBy ?? null,
      thrownBy: overrides.thrownBy ?? null,
      touchedSinceThrow: false,
      releasePos: ZERO_V,
      releaseTick: 0,
      passChain: overrides.passChain ?? [],
    },
    players: overrides.players ?? [],
    rngCursor: {},
  };
}

function makePlayer(id: string, team: TeamSide): SimState['players'][number] {
  return {
    id,
    team,
    role: 'spinner',
    p: ZERO_V,
    v: ZERO_V,
    q: ZERO_Q,
    line: null,
    dvBudget: 10,
    contactRef: null,
    grounded: false,
  };
}

function makeMatch(overrides: Partial<MatchState> = {}): MatchState {
  return {
    inning: 1,
    spine: false,
    possession: 'home',
    faithEnd: '+x',
    cast: { throwsLeft: 3, gate: 'first', spotX: 80 },
    contest: null,
    scoreHome: 0,
    scoreAway: 0,
    phase: 'live',
    message: '',
    winner: null,
    ...overrides,
  };
}

/** Sequence-based RNG for fully deterministic tests. */
function makeSeqRng(values: number[]): () => number {
  let i = 0;
  return () => values[i++ % values.length];
}

/**
 * Advance an MSM from 'set' to 'live' by feeding a foul_garrote event.
 * (foul_garrote is phase-agnostic and sets phase to 'live'.)
 */
function toLive(msm: MatchStateMachine): void {
  msm.consume([{ type: 'foul_garrote', by: '__setup__' }], makeSim());
}

// ─── Scoring table ────────────────────────────────────────────────────────────

describe('Scoring.scoreFor — scoring table', () => {
  // faithEnd='+x' → Faith ring is at +x, Free ring is at -x.

  it('Fall: touched bell through Faith ring → 2 pts to attacker', () => {
    const match = makeMatch({ faithEnd: '+x' });
    const sim = makeSim({ passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    const ev: SimEvent = { type: 'bell_through_ring', end: '+x', touched: true, loopTier: 'none' };
    const result = scoreFor(ev, match, sim);
    expect(result).not.toBeNull();
    expect(result!.kind).toBe('fall');
    expect(result!.points).toBe(2);
    expect(result!.team).toBe('home');
    expect(result!.endsInning).toBe(false);
  });

  it('Rise: bell through Free ring → 5 pts to attacker', () => {
    const match = makeMatch({ faithEnd: '+x' });
    const sim = makeSim({ passChain: ['ap'], players: [makePlayer('ap', 'away')] });
    const ev: SimEvent = { type: 'bell_through_ring', end: '-x', touched: true, loopTier: 'none' };
    const result = scoreFor(ev, match, sim);
    expect(result).not.toBeNull();
    expect(result!.kind).toBe('rise');
    expect(result!.points).toBe(5);
    expect(result!.team).toBe('away');
    expect(result!.endsInning).toBe(false);
  });

  it('Loop: untouched loop tier → 7 pts, endsInning=true', () => {
    const match = makeMatch({ faithEnd: '+x' });
    const sim = makeSim({ passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    const ev: SimEvent = { type: 'bell_through_ring', end: '+x', touched: false, loopTier: 'loop' };
    const result = scoreFor(ev, match, sim);
    expect(result).not.toBeNull();
    expect(result!.kind).toBe('loop');
    expect(result!.points).toBe(7);
    expect(result!.endsInning).toBe(true);
  });

  it('Loop > Curl always (Loop is the maximum score kind)', () => {
    const match = makeMatch({ faithEnd: '+x' });
    const sim = makeSim({ passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    const loopPts = scoreFor({ type: 'bell_through_ring', end: '+x', touched: false, loopTier: 'loop' }, match, sim)!.points;
    const faithCurlPts = scoreFor({ type: 'bell_through_ring', end: '+x', touched: false, loopTier: 'curl' }, match, sim)!.points;
    const freeCurlPts = scoreFor({ type: 'bell_through_ring', end: '-x', touched: false, loopTier: 'curl' }, match, sim)!.points;
    expect(faithCurlPts).toBeLessThan(loopPts);
    expect(freeCurlPts).toBeLessThan(loopPts);
  });

  it('Faith-curl: untouched curl through Faith ring → 4 pts (base 2 + 2)', () => {
    const match = makeMatch({ faithEnd: '+x' });
    const sim = makeSim({ passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    const ev: SimEvent = { type: 'bell_through_ring', end: '+x', touched: false, loopTier: 'curl' };
    const result = scoreFor(ev, match, sim);
    expect(result!.kind).toBe('curl');
    expect(result!.points).toBe(4); // 2 + 2
  });

  it('Free-curl: untouched curl through Free ring → 6 pts (capped below Loop)', () => {
    const match = makeMatch({ faithEnd: '+x' });
    const sim = makeSim({ passChain: ['ap'], players: [makePlayer('ap', 'away')] });
    const ev: SimEvent = { type: 'bell_through_ring', end: '-x', touched: false, loopTier: 'curl' };
    const result = scoreFor(ev, match, sim);
    expect(result!.kind).toBe('curl');
    expect(result!.points).toBe(6); // min(5+2=7, 6) = 6
    expect(result!.points).toBeLessThan(7); // strictly < Loop
  });

  it('Ground: attacker skinned → 1 pt to the defense', () => {
    const sim = makeSim({
      passChain: ['hp'],
      players: [makePlayer('hp', 'home'), makePlayer('ap', 'away')],
    });
    const ev: SimEvent = { type: 'player_skinned', id: 'hp' };
    const result = groundFor(ev, sim);
    expect(result).not.toBeNull();
    expect(result!.kind).toBe('ground');
    expect(result!.points).toBe(1);
    expect(result!.team).toBe('away'); // defense scores
    expect(result!.endsInning).toBe(false);
  });

  it('Ground: defender skinned → null (no point awarded)', () => {
    const sim = makeSim({
      passChain: ['hp'],
      players: [makePlayer('hp', 'home'), makePlayer('ap', 'away')],
    });
    const ev: SimEvent = { type: 'player_skinned', id: 'ap' };
    expect(groundFor(ev, sim)).toBeNull();
  });

  it('Pass-chain attribution: last passChain entry determines scoring team', () => {
    const match = makeMatch({ faithEnd: '+x' });
    // Chain: home-A → away-B → home-C; last is home-C → home scores
    const sim = makeSim({
      passChain: ['hA', 'aB', 'hC'],
      players: [makePlayer('hA', 'home'), makePlayer('aB', 'away'), makePlayer('hC', 'home')],
    });
    const ev: SimEvent = { type: 'bell_through_ring', end: '+x', touched: true, loopTier: 'none' };
    const result = scoreFor(ev, match, sim);
    expect(result!.team).toBe('home');
  });

  it('scoreFor returns null when no attribution chain exists', () => {
    const match = makeMatch({ faithEnd: '+x' });
    const sim = makeSim({ passChain: [], players: [] });
    const ev: SimEvent = { type: 'bell_through_ring', end: '+x', touched: true, loopTier: 'none' };
    expect(scoreFor(ev, match, sim)).toBeNull();
  });
});

// ─── Cast & turnover mechanics ────────────────────────────────────────────────

describe('MatchStateMachine — cast & turnover', () => {
  it('3 unproductive throws → turnover (throwsLeft reaches 0)', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);
    expect(msm.state.phase).toBe('live');

    // Bell at x=5, well short of first gate (80 m for home attacking +x)
    const sim = makeSim({ bellX: 5, passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    const catchEv: SimEvent = { type: 'bell_caught', by: 'hp' };

    msm.consume([catchEv], sim);
    expect(msm.state.cast.throwsLeft).toBe(2);

    msm.consume([catchEv], sim);
    expect(msm.state.cast.throwsLeft).toBe(1);

    const upd = msm.consume([catchEv], sim);
    expect(upd.turnover).toBeDefined();
    expect(upd.turnover!.team).toBe('away'); // possession flips
    expect(upd.inningEnd).toBeDefined();
  });

  it('clearing a gate resets throwsLeft=3 and advances gate', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);

    // Bell at x=90 → past first gate (80 m)
    const sim = makeSim({ bellX: 90, passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    msm.consume([{ type: 'bell_caught', by: 'hp' }], sim);

    expect(msm.state.cast.gate).toBe('deep');
    expect(msm.state.cast.throwsLeft).toBe(3);
  });

  it('gate-cleared at mouth does not re-advance beyond mouth', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);

    // Bell at x=290 → past mouth gate (272 m)
    const sim = makeSim({ bellX: 290, passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    msm.consume([{ type: 'bell_caught', by: 'hp' }], sim);

    // No gate beyond mouth exists; gate stays 'mouth', throws reset
    expect(msm.state.cast.gate).toBe('mouth');
    expect(msm.state.cast.throwsLeft).toBe(3);
  });

  it('bell_bobble and bell_clatter also consume a throw', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);

    const sim = makeSim({ bellX: 5, passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    msm.consume([{ type: 'bell_bobble', by: 'hp' }], sim);
    expect(msm.state.cast.throwsLeft).toBe(2);

    msm.consume([{ type: 'bell_clatter', by: 'hp' }], sim);
    expect(msm.state.cast.throwsLeft).toBe(1);
  });
});

// ─── Scoring events via MatchStateMachine ─────────────────────────────────────

describe('MatchStateMachine — scoring events', () => {
  it('a Fall scores 2 pts for home and ends the inning', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);

    const sim = makeSim({ passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    const upd = msm.consume(
      [{ type: 'bell_through_ring', end: '+x', touched: true, loopTier: 'none' }],
      sim,
    );

    expect(upd.scored!.kind).toBe('fall');
    expect(upd.scored!.points).toBe(2);
    expect(msm.state.scoreHome).toBe(2);
    expect(upd.inningEnd).toBeDefined();
  });

  it('a Loop scores 7 pts and endsInning immediately', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);

    const sim = makeSim({ passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    const upd = msm.consume(
      [{ type: 'bell_through_ring', end: '+x', touched: false, loopTier: 'loop' }],
      sim,
    );

    expect(upd.scored!.kind).toBe('loop');
    expect(upd.scored!.points).toBe(7);
    expect(upd.scored!.endsInning).toBe(true);
    expect(upd.inningEnd).toBeDefined();
  });

  it('Ground +1 to defense does not end inning', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);

    const sim = makeSim({
      passChain: ['hp'],
      players: [makePlayer('hp', 'home'), makePlayer('ap', 'away')],
    });
    const upd = msm.consume([{ type: 'player_skinned', id: 'hp' }], sim);

    expect(msm.state.scoreAway).toBe(1); // defense (away) gets the point
    expect(upd.inningEnd).toBeUndefined();
  });
});

// ─── Contest resolution ───────────────────────────────────────────────────────

describe('Contest.resolveContest — count-of-3, determinism', () => {
  it('same inputs → same result (fully deterministic)', () => {
    const contest = { thrower: 'p1', contester: 'p2', count: 0 as const, radius: 30, direction: 'fair' as const };
    const match = makeMatch();
    const r1 = resolveContest(contest, match, makeSeqRng([0.3, 0.8, 0.3, 0.8]));
    const r2 = resolveContest(contest, match, makeSeqRng([0.3, 0.8, 0.3, 0.8]));
    expect(r1).toEqual(r2);
  });

  it('exchanges is in [2, 3] (best-of-3: first to 2 wins)', () => {
    const contest = { thrower: 'p1', contester: 'p2', count: 0 as const, radius: 20, direction: 'fair' as const };
    const match = makeMatch();
    for (let seed = 0; seed < 50; seed++) {
      let n = seed;
      const rng = () => ((n++ * 1009 + 7) % 97) / 97;
      const result = resolveContest(contest, match, rng);
      expect(result.exchanges).toBeGreaterThanOrEqual(2);
      expect(result.exchanges).toBeLessThanOrEqual(3);
    }
  });

  it('winner is always "thrower" or "contester"', () => {
    const contest = { thrower: 'T', contester: 'C', count: 0 as const, radius: 10, direction: 'cross' as const };
    const match = makeMatch();
    for (let v = 0; v <= 10; v++) {
      const result = resolveContest(contest, match, makeSeqRng([v / 10]));
      expect(['thrower', 'contester']).toContain(result.winner);
    }
  });

  it('deep contest (radius=45) favours thrower over high contest (radius=0)', () => {
    // At radius=45: p_thrower ≈ 0.75. At radius=0: p_thrower = 0.5.
    // With rng always returning 0.5: 0.5 < 0.75 → thrower wins; 0.5 < 0.5 → false → contester wins.
    const match = makeMatch();
    const deepContest = { thrower: 'T', contester: 'C', count: 0 as const, radius: 45, direction: 'fair' as const };
    const highContest = { thrower: 'T', contester: 'C', count: 0 as const, radius: 0, direction: 'fair' as const };
    const deepResult = resolveContest(deepContest, match, () => 0.5);
    const highResult = resolveContest(highContest, match, () => 0.5);
    expect(deepResult.winner).toBe('thrower');
    expect(highResult.winner).toBe('contester');
  });

  it('contest integrated: contest_started → pending → thrower wins → cast continues', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);

    msm.consume([{ type: 'contest_started', thrower: 'p1', contester: 'p2' }], makeSim());
    expect(msm.state.phase).toBe('contest');
    expect(msm.pendingContest).not.toBeNull();

    const sim = makeSim({ bellX: 5, passChain: ['p1'], players: [makePlayer('p1', 'home')] });
    msm.resolveContest({ winner: 'thrower' }, sim);
    expect(msm.state.phase).toBe('live');
    expect(msm.state.contest).toBeNull();
  });

  it('contest integrated: contester wins → turnover, inning ends', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);

    msm.consume([{ type: 'contest_started', thrower: 'p1', contester: 'p2' }], makeSim());
    const sim = makeSim({ bellX: 5, passChain: ['p1'], players: [makePlayer('p1', 'home')] });
    const upd = msm.resolveContest({ winner: 'contester' }, sim);

    expect(upd.turnover).toBeDefined();
    expect(msm.state.possession).toBe('away');
    expect(upd.inningEnd).toBeDefined();
  });
});

// ─── 9 innings, spine, winner ─────────────────────────────────────────────────

describe('MatchStateMachine — innings, spine, winner', () => {
  /**
   * Helper: score one inning for the given team via a Fall (2 pts) at the
   * Faith (+x) ring. Both teams can score at the Faith ring — attribution
   * follows passChain, not which direction they attacked from.
   * Advances to live if needed.
   */
  function scoreInning(msm: MatchStateMachine, team: TeamSide): void {
    if (msm.state.phase !== 'live') toLive(msm);
    const pid = team + '-p';
    const sim = makeSim({ passChain: [pid], players: [makePlayer(pid, team)] });
    // Always score at the Faith ring (+x when faithEnd='+x') for a clean Fall (2 pts).
    msm.consume([{ type: 'bell_through_ring', end: '+x', touched: true, loopTier: 'none' }], sim);
  }

  /**
   * End an inning via turnover (no score): burn 3 throws short of the gate.
   */
  function turnoverInning(msm: MatchStateMachine): void {
    if (msm.state.phase !== 'live') toLive(msm);
    const team = msm.state.possession;
    const pid = team + '-p';
    const sim = makeSim({ bellX: 5, passChain: [pid], players: [makePlayer(pid, team)] });
    const catchEv: SimEvent = { type: 'bell_caught', by: pid };
    msm.consume([catchEv], sim);
    msm.consume([catchEv], sim);
    msm.consume([catchEv], sim);
  }

  it('home scores all 9 innings → home wins, phase final', () => {
    const msm = new MatchStateMachine('+x', 'home');
    for (let i = 0; i < 9; i++) scoreInning(msm, 'home');
    expect(msm.state.winner).toBe('home');
    expect(msm.state.phase).toBe('final');
    expect(msm.state.scoreHome).toBeGreaterThan(msm.state.scoreAway);
  });

  it('after 9 innings tied → spine phase', () => {
    const msm = new MatchStateMachine('+x', 'home');
    // 8 scoring innings: home and away alternate (4 each, 8 pts each)
    for (let i = 0; i < 8; i++) scoreInning(msm, i % 2 === 0 ? 'home' : 'away');
    expect(msm.state.inning).toBe(9);

    // 9th inning: turnover → tie → spine
    turnoverInning(msm);

    expect(msm.state.spine).toBe(true);
    expect(msm.state.phase).toBe('spine');
    expect(msm.state.scoreHome).toBe(msm.state.scoreAway);
  });

  it('spine: first score decides the winner', () => {
    const msm = new MatchStateMachine('+x', 'home');
    for (let i = 0; i < 8; i++) scoreInning(msm, i % 2 === 0 ? 'home' : 'away');
    turnoverInning(msm); // → spine

    expect(msm.state.spine).toBe(true);
    toLive(msm);

    const spinePossession = msm.state.possession;
    const pid = spinePossession + '-sp';
    const spineEnd: '+x' | '-x' = spinePossession === 'home' ? '+x' : '-x';
    const sim = makeSim({ passChain: [pid], players: [makePlayer(pid, spinePossession)] });
    msm.consume([{ type: 'bell_through_ring', end: spineEnd, touched: true, loopTier: 'none' }], sim);

    expect(msm.state.winner).toBe(spinePossession);
    expect(msm.state.phase).toBe('final');
  });

  it('events after final are silently ignored', () => {
    const msm = new MatchStateMachine('+x', 'home');
    for (let i = 0; i < 9; i++) scoreInning(msm, 'home');
    const scoreAfter = msm.state.scoreHome;
    const sim = makeSim({ passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    msm.consume([{ type: 'bell_through_ring', end: '+x', touched: true, loopTier: 'none' }], sim);
    expect(msm.state.scoreHome).toBe(scoreAfter);
  });

  it('winner is determined by score comparison (away wins if away outscores home)', () => {
    const msm = new MatchStateMachine('+x', 'home');
    for (let i = 0; i < 9; i++) scoreInning(msm, 'away');
    expect(msm.state.winner).toBe('away');
  });
});

// ─── Pure / deterministic invariant ──────────────────────────────────────────

describe('MatchStateMachine — pure / deterministic', () => {
  it('same event sequence → identical MatchState (determinism)', () => {
    function runScenario(): MatchState {
      const msm = new MatchStateMachine('+x', 'home');
      toLive(msm);

      const sim = makeSim({ bellX: 5, passChain: ['p1'], players: [makePlayer('p1', 'home')] });
      msm.consume([{ type: 'bell_caught', by: 'p1' }], sim);
      msm.consume([{ type: 'contest_started', thrower: 'p1', contester: 'p2' }], sim);
      msm.resolveContest({ winner: 'thrower' }, sim);

      const scoreSim = makeSim({ passChain: ['p1'], players: [makePlayer('p1', 'home')] });
      msm.consume([{ type: 'bell_through_ring', end: '+x', touched: true, loopTier: 'none' }], scoreSim);
      return msm.state;
    }

    expect(runScenario()).toEqual(runScenario());
  });

  it('does not mutate SimState passed in', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);

    const sim = makeSim({ bellX: 5, passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    const originalBellX = sim.bell.p.x;
    const originalChainLen = sim.bell.passChain.length;

    for (let i = 0; i < 3; i++) {
      msm.consume([{ type: 'bell_caught', by: 'hp' }], sim);
    }

    expect(sim.bell.p.x).toBe(originalBellX);
    expect(sim.bell.passChain.length).toBe(originalChainLen);
  });

  it('Loop immediately ends inning in the same consume() call', () => {
    const msm = new MatchStateMachine('+x', 'home');
    toLive(msm);

    const sim = makeSim({ passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    const upd = msm.consume(
      [{ type: 'bell_through_ring', end: '+x', touched: false, loopTier: 'loop' }],
      sim,
    );

    expect(upd.scored!.kind).toBe('loop');
    expect(upd.scored!.endsInning).toBe(true);
    expect(upd.inningEnd).toBeDefined();
  });

  it('faithEnd=-x flips Faith/Free: bell through -x ring counts as Fall', () => {
    const msm = new MatchStateMachine('-x', 'home');
    toLive(msm);

    // With faithEnd='-x': home attacks -x which is Faith (Fall=2)
    const sim = makeSim({ passChain: ['hp'], players: [makePlayer('hp', 'home')] });
    const upd = msm.consume(
      [{ type: 'bell_through_ring', end: '-x', touched: true, loopTier: 'none' }],
      sim,
    );
    // home is the attacker, -x is faithEnd → Faith ring → Fall
    expect(upd.scored!.kind).toBe('fall');
    expect(upd.scored!.points).toBe(2);
  });
});

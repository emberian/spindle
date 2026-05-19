import { describe, it, expect } from 'vitest';
import { SimWorld, hashSnapshot } from '../src/sim/SimWorld';
import { Recorder, replay } from '../src/sim/Replay';
import { SIM_H } from '../src/core/FixedStepDriver';
import type { InputFrame } from '../src/sim/types';

const ZERO = { x: 0, y: 0, z: 0 };
function build(w: SimWorld): void {
  w.addPlayer('P1', 'home', 'spinner', { x: -150, y: 2, z: 0 });
  w.addPlayer('A1', 'away', 'reach', { x: 280, y: 0, z: 0 });
  w.bellHeldBy = 'P1';
}

// Scripted: P1 holds, throws on tick 5, everyone idle after.
function frame(tick: number): InputFrame {
  const base = {
    aim: { x: 1, y: 0.25, z: -0.35 },
    fireLineAt: null,
    reel: 0 as const,
    release: false,
    pushoff: false,
    throwCharge: 0.9,
    throwReleased: tick === 5,
    throwSpin: 0.2,
    thrumbler: { ...ZERO },
    catchIntent: false,
  };
  return { tick, players: [{ id: 'P1', ...base }, { id: 'A1', ...base, throwReleased: false }] };
}

describe('SimWorld determinism + replay', () => {
  it('two runs with identical inputs produce identical snapshot hashes', () => {
    const a = new SimWorld(42);
    const b = new SimWorld(42);
    build(a);
    build(b);
    for (let t = 0; t < 1500; t++) {
      const f = frame(t);
      a.step(f, SIM_H);
      b.step(f, SIM_H);
      if (t % 250 === 0) expect(hashSnapshot(a.snapshot())).toBe(hashSnapshot(b.snapshot()));
    }
    expect(hashSnapshot(a.snapshot())).toBe(hashSnapshot(b.snapshot()));
  });

  it('replay (seed + inputs) reproduces the live run bit-for-bit', () => {
    const live = new SimWorld(7);
    build(live);
    const rec = new Recorder(7);
    for (let t = 0; t < 1200; t++) {
      const f = frame(t);
      rec.record(f);
      live.step(f, SIM_H);
    }
    const liveHash = hashSnapshot(live.snapshot());
    expect(replay(rec.blob(), build)).toBe(liveHash);
  });

  it('different seeds are allowed to diverge only via RNG, not physics here', () => {
    // (P2 sim has no RNG draws yet; physics is pure → seeds must NOT matter)
    const a = new SimWorld(1);
    const b = new SimWorld(999);
    build(a);
    build(b);
    for (let t = 0; t < 800; t++) {
      const f = frame(t);
      a.step(f, SIM_H);
      b.step(f, SIM_H);
    }
    expect(hashSnapshot(a.snapshot())).toBe(hashSnapshot(b.snapshot()));
  });
});

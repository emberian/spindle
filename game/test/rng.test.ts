import { describe, it, expect } from 'vitest';
import { Rng } from '../src/sim/rng';

describe('rng (sfc32) quality & determinism', () => {
  it('same seed → identical streams (replay determinism)', () => {
    const a = new Rng(12345);
    const b = new Rng(12345);
    const xs = Array.from({ length: 1000 }, () => a.sub('contest').next());
    const ys = Array.from({ length: 1000 }, () => b.sub('contest').next());
    expect(xs).toEqual(ys);
  });

  it('different substreams decorrelate', () => {
    const r = new Rng(99);
    const u = r.sub('aiJitter');
    const v = r.sub('clatter');
    let same = 0;
    for (let i = 0; i < 2000; i++) if (Math.abs(u.next() - v.next()) < 1e-9) same++;
    expect(same).toBe(0);
  });

  it('uniform: mean ≈ 0.5, balanced halves, no degenerate bits', () => {
    const s = new Rng(7).sub('x');
    const N = 200000;
    let sum = 0;
    const bins = new Array(10).fill(0);
    let lowHalf = 0;
    for (let i = 0; i < N; i++) {
      const v = s.next();
      sum += v;
      bins[Math.min(9, (v * 10) | 0)]++;
      if (v < 0.5) lowHalf++;
    }
    expect(sum / N).toBeGreaterThan(0.495);
    expect(sum / N).toBeLessThan(0.505);
    expect(lowHalf / N).toBeGreaterThan(0.49);
    expect(lowHalf / N).toBeLessThan(0.51);
    for (const b of bins) {
      expect(b / N).toBeGreaterThan(0.085); // each decile ~0.1
      expect(b / N).toBeLessThan(0.115);
    }
  });

  it('cursor records draw counts for re-seeding', () => {
    const r = new Rng(1);
    const s = r.sub('contest');
    for (let i = 0; i < 42; i++) s.next();
    expect(r.cursor()).toEqual({ contest: 42 });
  });
});

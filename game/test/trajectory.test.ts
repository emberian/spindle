// The RK4 sim must match the analytic roulette solution, and the canon
// invariant "motion parallel to the spin axis is inertial" must hold exactly.

import { describe, it, expect } from 'vitest';
import { predictPath, rk4Step, analytic, type PointState } from '../src/sim/trajectory';
import { REG } from '../src/sim/RegConstants';

const w = REG.omega;

describe('trajectory: RK4 vs analytic roulette', () => {
  it('axial motion is exactly inertial (x = x0 + vx0·t)', () => {
    const path = predictPath({ x: 0, y: 3, z: 0 }, { x: 25, y: 0, z: 4 }, w, 1 / 240, 240 * 8);
    for (let i = 0; i < path.length; i++) {
      const t = i / 240;
      expect(path[i].x).toBeCloseTo(25 * t, 6);
    }
  });

  it('RK4 tracks the closed-form analytic ζ(t) within tight tolerance', () => {
    const p0 = { x: 0, y: 4, z: -2 };
    const v0 = { x: 18, y: -3, z: 5 };
    let s: PointState = { p: { ...p0 }, v: { ...v0 } };
    const h = 1 / 240;
    for (let i = 1; i <= 240 * 6; i++) {
      s = rk4Step(s, w, h);
      const a = analytic(p0, v0, w, i * h);
      expect(s.p.y).toBeCloseTo(a.y, 3);
      expect(s.p.z).toBeCloseTo(a.z, 3);
    }
  });

  it('a clean axial throw is deterministic & repeatable', () => {
    const a = predictPath({ x: 0, y: 1, z: 0 }, { x: 30, y: 0, z: 2 }, w, 1 / 240, 1000);
    const b = predictPath({ x: 0, y: 1, z: 0 }, { x: 30, y: 0, z: 2 }, w, 1 / 240, 1000);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });
});

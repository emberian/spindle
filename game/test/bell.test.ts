import { describe, it, expect } from 'vitest';
import { stepBell, trueness, chime, clatter, type BellBody } from '../src/sim/Bell';
import { analytic } from '../src/sim/trajectory';
import { REG } from '../src/sim/RegConstants';

const w = REG.omega;
const h = 1 / 240;
const mk = (over: Partial<BellBody> = {}): BellBody => ({
  p: { x: 0, y: 4, z: -2 },
  v: { x: 18, y: -3, z: 5 },
  q: { x: 0, y: 0, z: 0, w: 1 },
  w: { x: 30, y: 0, z: 0 },
  ...over,
});

describe('bell 6-DOF', () => {
  it('COM follows the analytic Coriolis path (sphere = point in vacuum)', () => {
    let s = mk();
    const p0 = { ...s.p },
      v0 = { ...s.v };
    for (let i = 1; i <= 240 * 5; i++) {
      s = stepBell(s, w, h);
      const a = analytic(p0, v0, w, i * h);
      expect(s.p.y).toBeCloseTo(a.y, 2);
      expect(s.p.z).toBeCloseTo(a.z, 2);
      expect(s.p.x).toBeCloseTo(v0.x * i * h, 4); // axial inertial
    }
  });

  it('a true spin stays true (rings)', () => {
    let s = mk({ w: { x: 28, y: 0, z: 0 } });
    for (let i = 0; i < 240 * 4; i++) s = stepBell(s, w, h);
    expect(trueness(s.w)).toBeGreaterThan(0.999);
    expect(chime(s.w)).toBeGreaterThan(0.9);
  });

  it('an off-axis spin tumbles (clatters) and precesses', () => {
    let s = mk({ w: { x: 14, y: 9, z: 6 } });
    const t0 = trueness(s.w);
    let moved = false;
    let prev = s.w.y;
    for (let i = 0; i < 240 * 3; i++) {
      s = stepBell(s, w, h);
      if (Math.abs(s.w.y - prev) > 1e-3) moved = true;
      prev = s.w.y;
    }
    expect(t0).toBeLessThan(0.85);
    expect(chime(s.w)).toBeLessThan(0.9);
    expect(moved).toBe(true); // gyroscopic precession actually moves ω
  });

  it('rigid-body angular momentum magnitude is ~conserved in free flight', () => {
    let s = mk({ w: { x: 14, y: 9, z: 6 } });
    const Lm = (b: BellBody) =>
      Math.hypot(1.0 * b.w.x, 0.5 * b.w.y, 0.5 * b.w.z); // |I·ω| body-frame proxy
    const L0 = Lm(s);
    for (let i = 0; i < 240 * 3; i++) s = stepBell(s, w, h);
    expect(Lm(s)).toBeCloseTo(L0, 1);
  });

  it('clatter knocks a true bell off true', () => {
    const s = mk({ w: { x: 28, y: 0, z: 0 } });
    expect(trueness(s.w)).toBeGreaterThan(0.999);
    clatter(s, { x: 0, y: 12, z: 8 });
    expect(trueness(s.w)).toBeLessThan(0.92);
  });
});

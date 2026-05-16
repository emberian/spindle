import { describe, it, expect } from 'vitest';
import { skinBounce, tryCatch, contestClatter } from '../src/sim/Collision';
import { trueness, type BellBody } from '../src/sim/Bell';
import { REG } from '../src/sim/RegConstants';

describe('collision / catch / clatter', () => {
  it('skin bounce reflects radial velocity and clamps inside R', () => {
    const b = { p: { x: 0, y: REG.R + 2, z: 0 }, v: { x: 5, y: 8, z: 0 } };
    const hit = skinBounce(b);
    expect(hit).toBe(true);
    expect(Math.hypot(b.p.y, b.p.z)).toBeLessThanOrEqual(REG.R);
    expect(b.v.y).toBeLessThan(0); // bounced back inward
    expect(b.v.x).toBeCloseTo(5, 6); // axial untouched
  });

  it('catch requires arm reach (no 5 m homing)', () => {
    expect(tryCatch({ x: 5, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0 })).toBe('miss');
  });

  it('slow in-reach closing bell is caught; fast one bobbles', () => {
    const pos = { x: 0, y: 0, z: 0 };
    const pv = { x: 0, y: 0, z: 0 };
    expect(tryCatch({ x: 0.3, y: 0, z: 0 }, { x: -3, y: 0, z: 0 }, pos, pv)).toBe('caught');
    expect(tryCatch({ x: 0.3, y: 0, z: 0 }, { x: -30, y: 0, z: 0 }, pos, pv)).toBe('bobble');
  });

  it('contest clatter drops trueness', () => {
    const bell: BellBody = {
      p: { x: 0, y: 0, z: 0 },
      v: { x: 20, y: 0, z: 0 },
      q: { x: 0, y: 0, z: 0, w: 1 },
      w: { x: 26, y: 0, z: 0 },
    };
    expect(trueness(bell.w)).toBeGreaterThan(0.999);
    contestClatter(bell, { x: 0, y: 0, z: 0 }, 1);
    expect(trueness(bell.w)).toBeLessThan(0.9);
  });
});

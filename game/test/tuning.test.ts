// THE NUMBER, gated by CI. Retuned (P2.5) for an achievable approximate Loop.

import { describe, it, expect } from 'vitest';
import { REG } from '../src/sim/RegConstants';
import { evaluateTuning } from '../src/sim/tuning';

describe('Reg constants satisfy the retuned gameplay inequalities', () => {
  const r = evaluateTuning({ omega: REG.omega, R: REG.R, L: REG.L, earthG: REG.EARTH_G });

  it('traversable: a strong axial throw crosses >= half the field', () => {
    expect(r.axialReach).toBeGreaterThanOrEqual(0.5 * REG.L);
    expect(r.traversable).toBe(true);
  });

  it('an approximate Loop is achievable (gentle throw winds >= 2.2 rad in-bounds)', () => {
    expect(r.loopWind).toBeGreaterThanOrEqual(2.2);
    expect(r.loopWind).toBeLessThanOrEqual(6.5);
    expect(r.loopFlight).toBeGreaterThan(4); // snappy, not an 18s knife-edge
    expect(r.loopFlight).toBeLessThan(16);
    expect(r.loopAchievable).toBe(true);
  });

  it('gradient is felt but not lethal (0.1g..0.6g at the skin)', () => {
    expect(r.gSkinFraction).toBeGreaterThanOrEqual(0.1);
    expect(r.gSkinFraction).toBeLessThanOrEqual(0.6);
    expect(r.gradientOk).toBe(true);
  });

  it('overall pass', () => {
    expect(r.pass).toBe(true);
  });
});

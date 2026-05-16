// THE NUMBER, gated by CI. If the shipped Reg constants ever violate the
// three gameplay inequalities, this fails — the structural fix for v0.1's
// empirical ω flailing (0.22 -> 0.06).

import { describe, it, expect } from 'vitest';
import { REG } from '../src/sim/RegConstants';
import { evaluateTuning, solveConstants } from '../src/sim/tuning';

describe('Reg constants satisfy the three gameplay inequalities', () => {
  const r = evaluateTuning({ omega: REG.omega, R: REG.R, L: REG.L, earthG: REG.EARTH_G });

  it('is traversable: a strong axial throw crosses >= half the field', () => {
    expect(r.axialReach).toBeGreaterThanOrEqual(0.5 * REG.L);
    expect(r.traversable).toBe(true);
  });

  it('loop is dramatic but controllable (ω·τ in [0.5,1.2])', () => {
    expect(r.omegaTauLoop).toBeGreaterThanOrEqual(0.5);
    expect(r.omegaTauLoop).toBeLessThanOrEqual(1.2);
    expect(r.loopDramatic).toBe(true);
  });

  it('gradient is felt but not lethal (0.1g..0.6g at the skin)', () => {
    expect(r.gSkinFraction).toBeGreaterThanOrEqual(0.1);
    expect(r.gSkinFraction).toBeLessThanOrEqual(0.6);
    expect(r.gradientOk).toBe(true);
  });

  it('overall pass', () => {
    expect(r.pass).toBe(true);
  });

  it('solver round-trips to the shipped constants', () => {
    const { omega, R } = solveConstants(
      REG.omega * 5,
      (REG.omega * REG.omega * REG.R) / REG.EARTH_G,
      REG.EARTH_G,
    );
    expect(omega).toBeCloseTo(REG.omega, 6);
    expect(R).toBeCloseTo(REG.R, 4);
  });
});

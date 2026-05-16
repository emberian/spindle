import { describe, it, expect } from 'vitest';
import { resolveLine, type Body, type Line } from '../src/sim/Grapple';

const h = 1 / 240;

describe('rig-line constraint', () => {
  it('slack line (inside radius) applies no force', () => {
    const p: Body = { p: { x: 5, y: 0, z: 0 }, v: { x: 2, y: 1, z: 0 }, invMass: 1 };
    const line: Line = {
      anchorPos: { x: 0, y: 0, z: 0 },
      anchorBody: null,
      restLen: 20,
      taut: false,
    };
    const v0 = { ...p.v };
    resolveLine(p, line, 0, h);
    expect(p.v).toEqual(v0);
    expect(line.taut).toBe(false);
  });

  it('taut static anchor removes separating radial velocity, keeps tangential', () => {
    // player 10 m out on +x, moving outward (radial) + along +y (tangential)
    const p: Body = { p: { x: 10, y: 0, z: 0 }, v: { x: 6, y: 4, z: 0 }, invMass: 1 };
    const line: Line = {
      anchorPos: { x: 0, y: 0, z: 0 },
      anchorBody: null,
      restLen: 10,
      taut: false,
    };
    resolveLine(p, line, 0, h);
    expect(p.v.x).toBeCloseTo(0, 4); // radial (outward) cancelled
    expect(p.v.y).toBeCloseTo(4, 4); // tangential (the swing) preserved
    expect(line.taut).toBe(true);
  });

  it('player↔player taut slingshot conserves linear momentum', () => {
    const a: Body = { p: { x: 0, y: 0, z: 0 }, v: { x: 0, y: 0, z: 0 }, invMass: 1 / 80 };
    const p: Body = { p: { x: 12, y: 0, z: 0 }, v: { x: 9, y: 0, z: 0 }, invMass: 1 / 75 };
    const line: Line = {
      anchorPos: { x: 0, y: 0, z: 0 },
      anchorBody: a,
      restLen: 12,
      taut: false,
    };
    const mA = 80,
      mP = 75;
    const P0 = mA * a.v.x + mP * p.v.x;
    resolveLine(p, line, 0, h);
    const P1 = mA * a.v.x + mP * p.v.x;
    expect(P1).toBeCloseTo(P0, 6); // momentum conserved → the slingshot is "free"
    expect(a.v.x).toBeGreaterThan(0); // anchor recoils
    expect(p.v.x).toBeLessThan(9); // player decelerated outward
  });

  it('reeling in increases tangential speed (angular momentum conserved)', () => {
    const p: Body = { p: { x: 20, y: 0, z: 0 }, v: { x: 0, y: 5, z: 0 }, invMass: 1 };
    const line: Line = {
      anchorPos: { x: 0, y: 0, z: 0 },
      anchorBody: null,
      restLen: 20,
      taut: false,
    };
    // reel in for ~0.5 s
    for (let i = 0; i < 120; i++) resolveLine(p, line, -1, h);
    const r = Math.hypot(p.p.x, p.p.y, p.p.z);
    const vTan = Math.hypot(p.v.y, p.v.z); // motion is in the x-y plane here
    expect(r).toBeLessThan(20); // pulled inward
    expect(vTan).toBeGreaterThan(5); // sped up — emergent slingshot
    // L = v_t · r should be ~conserved (initial 5·20 = 100)
    expect(vTan * r).toBeCloseTo(100, 0);
  });
});

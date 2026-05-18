import { describe, it, expect } from 'vitest';
import { resolveLine, REEL_RATE, TETHER_MIN, type Body, type Line } from '../src/sim/Grapple';

const h = 1 / 240;

describe('rig-line constraint', () => {
  it('deep-slack line (well inside radius) applies no force', () => {
    const p: Body = { p: { x: 5, y: 0, z: 0 }, v: { x: 2, y: 1, z: 0 }, invMass: 1 };
    const line: Line = {
      anchorPos: { x: 0, y: 0, z: 0 },
      anchorBody: null,
      restLen: 20,
      taut: false,
      attached: true,
      attachTick: 0,
    };
    const v0 = { ...p.v };
    resolveLine(p, line, 0, h);
    expect(p.v).toEqual(v0);
    expect(line.taut).toBe(false);
  });

  it('taut static anchor decelerates outward radial, leaves tangential', () => {
    // player 10.1 m out on +x (stretch +0.1), moving outward + along +y
    const p: Body = { p: { x: 10.1, y: 0, z: 0 }, v: { x: 6, y: 4, z: 0 }, invMass: 1 };
    const line: Line = {
      anchorPos: { x: 0, y: 0, z: 0 },
      anchorBody: null,
      restLen: 10,
      taut: false,
      attached: true,
      attachTick: 0,
    };
    resolveLine(p, line, 0, h);
    expect(p.v.x).toBeLessThan(6); // radial spring force pulls inward
    expect(p.v.y).toBe(4); // tangential untouched in a single radial step
    expect(line.taut).toBe(true);
  });

  it('rope is tension-only: a slack body moving inward gets no outward push', () => {
    // 9.7 m out, restLen 10 ⇒ stretch −0.3 (inside the 0.6 m band)
    const p: Body = { p: { x: 9.7, y: 0, z: 0 }, v: { x: -2, y: 0, z: 0 }, invMass: 1 };
    const line: Line = {
      anchorPos: { x: 0, y: 0, z: 0 },
      anchorBody: null,
      restLen: 10,
      taut: false,
      attached: true,
      attachTick: 0,
    };
    resolveLine(p, line, 0, h);
    expect(p.v.x).toBeLessThanOrEqual(-2); // never pushed outward
    expect(line.taut).toBe(false);
  });

  it('player↔player taut slingshot conserves linear momentum', () => {
    const a: Body = { p: { x: 0, y: 0, z: 0 }, v: { x: 0, y: 0, z: 0 }, invMass: 1 / 80 };
    const p: Body = { p: { x: 12.5, y: 0, z: 0 }, v: { x: 9, y: 0, z: 0 }, invMass: 1 / 75 };
    const line: Line = {
      anchorPos: { x: 0, y: 0, z: 0 },
      anchorBody: a,
      restLen: 12,
      taut: false,
      attached: true,
      attachTick: 0,
    };
    const mA = 80,
      mP = 75;
    const P0 = mA * a.v.x + mP * p.v.x;
    resolveLine(p, line, 0, h);
    const P1 = mA * a.v.x + mP * p.v.x;
    expect(P1).toBeCloseTo(P0, 6); // equal-and-opposite spring ⇒ slingshot is "free"
    expect(a.v.x).toBeGreaterThan(0); // anchor recoils
    expect(p.v.x).toBeLessThan(9); // player decelerated outward
  });

  it('reel only eases restLen toward target — no teleport, no v rescale', () => {
    const p: Body = { p: { x: 20, y: 0, z: 0 }, v: { x: 0, y: 5, z: 0 }, invMass: 1 };
    const line: Line = {
      anchorPos: { x: 0, y: 0, z: 0 },
      anchorBody: null,
      restLen: 20,
      taut: false,
      attached: true,
      attachTick: 0,
    };
    const pBefore = { ...p.p };
    resolveLine(p, line, -1, h);
    expect(line.restLen).toBeCloseTo(20 - REEL_RATE * h, 12);
    expect(p.p).toEqual(pBefore); // never sets position
    for (let i = 0; i < 240; i++) resolveLine(p, line, -1, h);
    expect(line.restLen).toBeLessThan(20);
    expect(line.restLen).toBeGreaterThanOrEqual(TETHER_MIN);
    expect(line.taut).toBe(true); // shrunk restLen ⇒ taut spring pulling
  });
});

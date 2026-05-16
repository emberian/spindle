import { describe, it, expect } from 'vitest';
import { SimWorld } from '../src/sim/SimWorld';
import { SIM_H } from '../src/core/FixedStepDriver';
import type { InputFrame } from '../src/sim/types';
import { LoopTracker } from '../src/sim/LoopDetector';
import { REG } from '../src/sim/RegConstants';
import { predictPath } from '../src/sim/trajectory';

const idle: InputFrame = { tick: 0, players: [] };

describe('Loop detection (the 7-pointer)', () => {
  it('an untouched, hard-curving throw winds a closed arc', () => {
    const w = new SimWorld(1);
    // up-and-antispinward launch near the axis → a long Coriolis roulette
    w.launchBell({ x: 0, y: 2, z: 0 }, { x: 6, y: 10, z: -12 }, { x: 26, y: 0, z: 0 }, 'P1');
    const tr = new LoopTracker();
    tr.onRelease({ x: 0, y: 2, z: 0 });
    for (let t = 0; t < 240 * 14; t++) {
      w.step(idle, SIM_H);
      tr.update(w.bell.v);
    }
    expect(tr.untouched).toBe(true);
    expect(tr.turn).toBeGreaterThan(Math.PI * 0.85);
  });

  it('a near-straight axial throw does NOT register a loop', () => {
    const tr = new LoopTracker();
    tr.onRelease({ x: 0, y: 1, z: 0 });
    const path = predictPath({ x: 0, y: 1, z: 0 }, { x: 30, y: 0, z: 0.5 }, REG.omega, SIM_H, 240 * 4);
    let pv = { x: 30, y: 0, z: 0.5 };
    for (let i = 1; i < path.length; i++) {
      const v = {
        x: (path[i].x - path[i - 1].x) / SIM_H,
        y: (path[i].y - path[i - 1].y) / SIM_H,
        z: (path[i].z - path[i - 1].z) / SIM_H,
      };
      tr.update(v);
      pv = v;
    }
    void pv;
    expect(tr.turn).toBeLessThan(Math.PI * 0.85);
    expect(tr.isLoop(path[path.length - 1])).toBe(false);
  });

  it('a contact mid-flight voids the loop', () => {
    const tr = new LoopTracker();
    tr.onRelease({ x: 0, y: 2, z: 0 });
    for (let i = 0; i < 50; i++) tr.update({ x: 5, y: 8, z: -8 });
    tr.onTouch();
    expect(tr.untouched).toBe(false);
    expect(tr.isLoop({ x: 0, y: 2, z: 0 })).toBe(false);
  });
});

import { describe, it, expect } from 'vitest';
import { SimWorld } from '../src/sim/SimWorld';
import { SIM_H } from '../src/core/FixedStepDriver';
import type { InputFrame } from '../src/sim/types';
import { LoopTracker, LOOP_TURN, CURL_TURN } from '../src/sim/LoopDetector';
import { REG } from '../src/sim/RegConstants';
import { predictPath } from '../src/sim/trajectory';

const idle: InputFrame = { tick: 0, players: [] };

describe('Loop / Curl detection (retuned)', () => {
  it('a gentle near-axis untouched throw winds an approximate Loop', () => {
    const w = new SimWorld(1);
    w.launchBell({ x: 0, y: 1.5, z: 0 }, { x: 7, y: 4, z: -3.5 }, { x: 26, y: 0, z: 0 }, 'P1');
    const tr = new LoopTracker();
    tr.onRelease({ x: 0, y: 1.5, z: 0 });
    for (let t = 0; t < 240 * 12; t++) {
      w.step(idle, SIM_H);
      tr.update(w.bell.v);
    }
    expect(tr.untouched).toBe(true);
    expect(tr.turn).toBeGreaterThanOrEqual(LOOP_TURN);
  });

  it('a moderate curve registers at least a Curl, not nothing', () => {
    const tr = new LoopTracker();
    tr.onRelease({ x: 0, y: 1, z: 0 });
    let s = { p: { x: 0, y: 1, z: 0 }, v: { x: 20, y: 0, z: 6 } };
    const path = predictPath(s.p, s.v, REG.omega, SIM_H, 240 * 3);
    for (let i = 1; i < path.length; i++) {
      tr.update({
        x: (path[i].x - path[i - 1].x) / SIM_H,
        y: (path[i].y - path[i - 1].y) / SIM_H,
        z: (path[i].z - path[i - 1].z) / SIM_H,
      });
    }
    expect(tr.turn).toBeGreaterThanOrEqual(CURL_TURN);
  });

  it('a near-straight axial throw is neither Loop nor Curl', () => {
    const tr = new LoopTracker();
    tr.onRelease({ x: 0, y: 1, z: 0 });
    const path = predictPath({ x: 0, y: 1, z: 0 }, { x: 32, y: 0, z: 0.2 }, REG.omega, SIM_H, 120);
    for (let i = 1; i < path.length; i++) {
      tr.update({
        x: (path[i].x - path[i - 1].x) / SIM_H,
        y: (path[i].y - path[i - 1].y) / SIM_H,
        z: (path[i].z - path[i - 1].z) / SIM_H,
      });
    }
    expect(tr.turn).toBeLessThan(CURL_TURN);
    expect(tr.tier(path[path.length - 1])).toBe('none');
  });

  it('contact mid-flight voids both Loop and Curl', () => {
    const tr = new LoopTracker();
    tr.onRelease({ x: 0, y: 2, z: 0 });
    for (let i = 0; i < 200; i++) tr.update({ x: 5, y: 8, z: -8 });
    tr.onTouch();
    expect(tr.untouched).toBe(false);
    expect(tr.tier({ x: 0, y: 2, z: 0 })).toBe('none');
  });
});

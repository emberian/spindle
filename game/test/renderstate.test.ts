// Render-only interpolation tests. renderState() must:
//  • α=0 ⇒ continuous fields equal prev
//  • α=1 ⇒ continuous fields equal cur
//  • α=0.5 ⇒ arithmetic mean of positions
//  • discrete / identity fields always equal cur
//  • possession change ⇒ bell snapped to cur (no glide-through on re-grip)
//  • teleport (> 8 m move, or non-1-tick gap) ⇒ that entity snapped to cur

import { describe, it, expect } from 'vitest';
import { renderState, type RenderView } from '../src/render/RenderState';
import type { PlayerSim, BellState } from '../src/sim/types';

function mkPlayer(id: string, x: number, vx: number): PlayerSim {
  return {
    id,
    team: 'home',
    role: 'spinner',
    p: { x, y: 0, z: 0 },
    v: { x: vx, y: 0, z: 0 },
    q: { x: 0, y: 0, z: 0, w: 1 },
    line: null,
    dvBudget: 8,
    contactRef: null,
    grounded: false,
  };
}

function mkBell(p: number, heldBy: string | null): BellState {
  return {
    p: { x: p, y: 0, z: 0 },
    v: { x: 1, y: 0, z: 0 },
    q: { x: 0, y: 0, z: 0, w: 1 },
    w: { x: 0, y: 0, z: 0 },
    chime: 0.2,
    heldBy,
    thrownBy: null,
    touchedSinceThrow: false,
    releasePos: { x: 0, y: 0, z: 0 },
    releaseTick: 0,
    passChain: [],
  };
}

function mkView(tick: number, px: number, bellP: number, heldBy: string | null): RenderView {
  return {
    tick,
    omega: 0.32,
    bell: mkBell(bellP, heldBy),
    players: [mkPlayer('P1', px, 2)],
    rngCursor: {},
    loopTurn: tick,
    loopUntouched: true,
    loopTier: 'none',
  };
}

describe('renderState', () => {
  it('α=0 ⇒ continuous fields equal prev', () => {
    const prev = mkView(10, 1, 5, 'P1');
    const cur = mkView(11, 3, 9, 'P1');
    const v = renderState(prev, cur, 0);
    expect(v.players[0].p.x).toBeCloseTo(1, 9);
    expect(v.bell.p.x).toBeCloseTo(5, 9);
    expect(v.bell.chime).toBeCloseTo(0.2, 9);
  });

  it('α=1 ⇒ continuous fields equal cur', () => {
    const prev = mkView(10, 1, 5, 'P1');
    const cur = mkView(11, 3, 9, 'P1');
    const v = renderState(prev, cur, 1);
    expect(v.players[0].p.x).toBeCloseTo(3, 9);
    expect(v.bell.p.x).toBeCloseTo(9, 9);
  });

  it('α=0.5 ⇒ arithmetic mean of positions', () => {
    const prev = mkView(10, 2, 4, 'P1');
    const cur = mkView(11, 6, 10, 'P1');
    const v = renderState(prev, cur, 0.5);
    expect(v.players[0].p.x).toBeCloseTo(4, 9); // (2+6)/2
    expect(v.bell.p.x).toBeCloseTo(7, 9); // (4+10)/2
  });

  it('discrete / identity fields always come from cur', () => {
    const prev = mkView(10, 1, 5, 'P1');
    const cur = mkView(11, 3, 9, 'P1');
    cur.loopTier = 'loop';
    cur.loopUntouched = false;
    const v = renderState(prev, cur, 0);
    expect(v.tick).toBe(11);
    expect(v.loopTier).toBe('loop');
    expect(v.loopUntouched).toBe(false);
    expect(v.loopTurn).toBe(11);
    expect(v.players[0].id).toBe('P1');
  });

  it('possession change ⇒ bell snapped to cur', () => {
    const prev = mkView(10, 1, 5, 'P1');
    const cur = mkView(11, 3, 9, null); // bell released this tick
    const v = renderState(prev, cur, 0.5);
    // bell does NOT glide — it is at cur despite α=0.5
    expect(v.bell.p.x).toBeCloseTo(9, 9);
    expect(v.bell.heldBy).toBeNull();
    // players still glide normally
    expect(v.players[0].p.x).toBeCloseTo(2, 9);
  });

  it('teleport (> 8 m move) ⇒ that entity snapped to cur', () => {
    const prev = mkView(10, 1, 5, 'P1');
    const cur = mkView(11, 50, 9, 'P1'); // player jumped 49 m
    const v = renderState(prev, cur, 0.5);
    expect(v.players[0].p.x).toBeCloseTo(50, 9); // snapped
  });

  it('non-1-tick gap ⇒ everything snapped to cur', () => {
    const prev = mkView(10, 1, 5, 'P1');
    const cur = mkView(20, 3, 9, 'P1'); // 10-tick jump (resync/reset)
    const v = renderState(prev, cur, 0.5);
    expect(v.players[0].p.x).toBeCloseTo(3, 9);
    expect(v.bell.p.x).toBeCloseTo(9, 9);
  });

  it('alpha is clamped to [0,1]', () => {
    // Keep the move under the teleport guard (8 m) so it actually glides.
    const prev = mkView(10, 0, 0, 'P1');
    const cur = mkView(11, 4, 4, 'P1');
    expect(renderState(prev, cur, -1).players[0].p.x).toBeCloseTo(0, 9);
    expect(renderState(prev, cur, 2).players[0].p.x).toBeCloseTo(4, 9);
  });
});

// Replay = the seed + the per-tick input stream (NOT stored snapshots — the
// sim is deterministic, so re-running reproduces it bit-for-bit). This is
// also the lore's "re-call": you don't ship the game, you ship the call.

import type { InputFrame } from './types';
import { SimWorld, hashSnapshot } from './SimWorld';
import { SIM_H } from '../core/FixedStepDriver';

export interface ReplayBlob {
  seed: number;
  codeVersion: string;
  frames: InputFrame[]; // one per sim tick
}

export class Recorder {
  frames: InputFrame[] = [];
  constructor(public seed: number, public codeVersion = 'v2') {}
  record(frame: InputFrame): void {
    this.frames.push({
      tick: frame.tick,
      players: frame.players.map((p) => ({ ...p, aim: { ...p.aim }, thrumbler: { ...p.thrumbler }, fireLineAt: p.fireLineAt ? { ...p.fireLineAt } : null })),
    });
  }
  blob(): ReplayBlob {
    return { seed: this.seed, codeVersion: this.codeVersion, frames: this.frames };
  }
}

/** Re-run a replay through a fresh world. `build` re-creates the same roster.
 *  Returns the final snapshot hash (must equal the live run's). */
export function replay(blob: ReplayBlob, build: (w: SimWorld) => void): string {
  const w = new SimWorld(blob.seed);
  build(w);
  for (const f of blob.frames) w.step(f, SIM_H);
  return hashSnapshot(w.snapshot());
}

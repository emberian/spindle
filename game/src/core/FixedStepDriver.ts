// Fixed-timestep accumulator. The sim ALWAYS advances in 1/240 s ticks
// regardless of frame rate or slow-mo, so it is bit-identical everywhere.
// Render interpolates between the previous and current snapshot by `alpha`.

import type { SimState } from '../sim/types';

export const SIM_H = 1 / 240;
const MAX_SUBSTEPS = 12; // clamp to avoid spiral of death

export class FixedStepDriver {
  private acc = 0;
  prev: SimState;
  cur: SimState;
  alpha = 0;

  constructor(
    private stepOnce: () => void,
    private snapshot: () => SimState,
  ) {
    this.cur = snapshot();
    this.prev = this.cur;
  }

  /** Feed wall dt (already scaled by timeScale). Runs whole sim ticks. */
  advance(dt: number): void {
    this.acc += dt;
    let n = 0;
    while (this.acc >= SIM_H && n < MAX_SUBSTEPS) {
      this.prev = this.cur;
      this.stepOnce();
      this.cur = this.snapshot();
      this.acc -= SIM_H;
      n++;
    }
    if (n === MAX_SUBSTEPS) this.acc = 0; // dropped behind; resync
    this.alpha = this.acc / SIM_H;
  }
}

// AiSystem — thin boundary shim over the Rust AI.
//
// The entire AI (Director, RiggerAI, roles, decide, nav, the sfc32 RNG,
// the director/commit caches) was ported 1:1 to Rust (rig-core/src/ai/**,
// 168 cargo tests) and is the source of truth. This shim preserves the
// exact old public surface — `new AiSystem()`, `.tick(sim, match,
// configs, seed) → InputFrame`, `.reset()` — so main.ts is unchanged,
// but every decision now runs in Rust via the wasm `RigAi` boundary.
// Determinism is unchanged: the Rust `AiSystem` reproduces index.ts
// `makeRng(seed ^ tick ^ teamIndex)` bit-for-bit.

import type { SimState, MatchState, InputFrame } from '../sim/types';
import type { TeamProfile } from '../league/teams';
import { aiTick, aiReset } from '../sim/wasm';

export type Difficulty = 'rookie' | 'pro' | 'legend';

export interface TeamConfig {
  side: 'home' | 'away';
  profile: TeamProfile;
  difficulty: Difficulty;
}

export class AiSystem {
  /**
   * Produce an InputFrame for all AI-controlled players (id !== 'P1').
   * Deterministic: identical (simState, matchState, configs, seed, tick)
   * → identical output (the Rust AiSystem owns the caches + RNG).
   */
  tick(
    simState: SimState,
    matchState: MatchState,
    configs: TeamConfig[],
    seed: number,
  ): InputFrame {
    const out = aiTick(
      JSON.stringify(simState),
      JSON.stringify(matchState),
      JSON.stringify(configs),
      seed,
    );
    return JSON.parse(out) as InputFrame;
  }

  /** Reset director + commitment caches (e.g., on a new inning). */
  reset(): void {
    aiReset();
  }
}

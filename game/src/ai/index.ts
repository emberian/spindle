// AiSystem — the top-level deterministic AI orchestrator.
//
// Given (SimState, MatchState, TeamProfile[], difficulty, seed, tick), produces
// an InputFrame for all AI-controlled players (id !== 'P1').
//
// Deterministic: same (state, seed, tick) → identical InputFrame, guaranteed
// because all randomness flows through the injected sfc32 rng seeded from
// (seed ^ tick).

import type { SimState, MatchState, InputFrame, PlayerInput } from '../sim/types';
import type { TeamProfile } from '../league/teams';
import { runDirector, DIRECTOR_TICK_INTERVAL } from './Director';
import type { DirectorState } from './Director';
import {
  computePlayerInput,
  type Difficulty,
  type PlayerCommitCache,
} from './RiggerAI';

// ── Seeded RNG (sfc32 — same algorithm as sim/rng.ts, inlined to avoid state) ──

function sfc32(a: number, b: number, c: number, d: number): () => number {
  // sfc32 core — bit-identical to rng.ts
  return () => {
    const t = (((a + b) >>> 0) + d) >>> 0;
    d = (d + 1) >>> 0;
    a = (b ^ (b >>> 9)) >>> 0;
    b = (c + (c << 3)) >>> 0;
    c = ((c << 21) | (c >>> 11)) >>> 0;
    c = (c + t) >>> 0;
    return (t >>> 0) / 4294967296;
  };
}

function splitmix32(seed: number): number {
  let s = (seed >>> 0) || 1;
  s = (s + 0x9e3779b9) >>> 0;
  let t = s;
  t = Math.imul(t ^ (t >>> 16), 0x21f0aaad) >>> 0;
  t = Math.imul(t ^ (t >>> 15), 0x735a2d97) >>> 0;
  return (t ^ (t >>> 15)) >>> 0;
}

function makeRng(seed: number, tick: number, teamIndex: number): () => number {
  // Mix seed × tick × team index into four sfc32 words, fully decorrelated.
  const mixed = (seed ^ (tick * 0x9e3779b9) ^ (teamIndex * 0x6c62272e)) >>> 0;
  const a = splitmix32(mixed);
  const b = splitmix32(a);
  const c = splitmix32(b);
  const dd = splitmix32(c);
  const gen = sfc32(a, b, c, dd);
  // Warm-up: discard 16 values (same as Substream in rng.ts).
  for (let i = 0; i < 16; i++) gen();
  return gen;
}

// ── Team configuration ─────────────────────────────────────────────────────────

export interface TeamConfig {
  side: 'home' | 'away';
  profile: TeamProfile;
  difficulty: Difficulty;
}

// ── Director cache: one per team side ─────────────────────────────────────────

interface DirectorCache {
  state: DirectorState;
  lastUpdatedTick: number;
}

// ── AiSystem ──────────────────────────────────────────────────────────────────

export class AiSystem {
  private directorCaches = new Map<string, DirectorCache>();
  /**
   * Per-player decision-commitment cache. This is the heart of visible
   * competence: instead of re-rolling intent every 240 Hz tick (which made
   * players spastic), each player's discrete decisions (which target, throw
   * yes/no, posture) are committed and only re-evaluated at the Director
   * cadence (or when a hard trigger fires — see RiggerAI). Between refreshes
   * the player smoothly *executes* the committed plan.
   */
  private commitCache = new Map<string, PlayerCommitCache>();

  /**
   * Produce an InputFrame for all AI-controlled players for this tick.
   * id !== 'P1' is the convention for AI players.
   * Deterministic: same (simState, matchState, configs, seed, tick) → identical output.
   */
  tick(
    simState: SimState,
    matchState: MatchState,
    configs: TeamConfig[],
    seed: number,
  ): InputFrame {
    const tick = simState.tick;
    const inputs: PlayerInput[] = [];

    // 'P1' is the human-controlled player ONLY when such a player exists in
    // the roster. In a fully-AI (spectate) match there is no P1 and EVERY
    // player must be AI-driven — so we filter by id !== 'P1' which naturally
    // includes all players when P1 is absent.
    for (let ci = 0; ci < configs.length; ci++) {
      const cfg = configs[ci];
      const teamPlayers = simState.players.filter(
        p => p.team === cfg.side && p.id !== 'P1',
      );
      if (teamPlayers.length === 0) continue;

      // Director: update at ~2 Hz.
      const cacheKey = `${cfg.side}-${ci}`;
      let dirCache = this.directorCaches.get(cacheKey);
      const needsDirectorUpdate =
        !dirCache ||
        tick - dirCache.lastUpdatedTick >= DIRECTOR_TICK_INTERVAL;

      if (needsDirectorUpdate) {
        const dirRng = makeRng(seed, tick, ci * 1000);
        const newState = runDirector(simState, matchState, cfg.profile, cfg.side, dirRng);
        dirCache = { state: newState, lastUpdatedTick: tick };
        this.directorCaches.set(cacheKey, dirCache);
      }

      const director = dirCache!.state;

      const directorRefreshed = needsDirectorUpdate;

      // Per-player decisions.
      for (let pi = 0; pi < teamPlayers.length; pi++) {
        const player = teamPlayers[pi];
        // Each player gets its own fully independent rng stream: seed × tick × ci × pi.
        const playerRng = makeRng(seed, tick, ci * 100 + pi);

        const commitKey = `${cfg.side}-${ci}-${player.id}`;
        let commit = this.commitCache.get(commitKey);
        if (!commit) {
          commit = { value: null, decidedTick: -1 };
          this.commitCache.set(commitKey, commit);
        }

        const input = computePlayerInput(
          player,
          simState,
          matchState,
          cfg.profile,
          director,
          cfg.difficulty,
          playerRng,
          commit,
          directorRefreshed,
          DIRECTOR_TICK_INTERVAL,
        );
        inputs.push(input);
      }
    }

    return { tick, players: inputs };
  }

  /** Reset director + commitment caches (e.g., on a new inning). */
  reset(): void {
    this.directorCaches.clear();
    this.commitCache.clear();
  }
}

// Re-export key types for convenience.
export type { Difficulty, PlayerCommit, PlayerCommitCache } from './RiggerAI';
export type { DirectorState, PlayerAssignment } from './Director';

// Bracket.ts — Single-elimination 16-team Jump bracket for the RIG league.
//
// Seed pairing (1v16, 8v9, 5v12, 4v13, 2v15, 7v10, 6v11, 3v14):
//   Round 1 (8 games): 1v16, 8v9, 5v12, 4v13, 2v15, 7v10, 6v11, 3v14
//   Round 2 (4 games), Round 3 (2 games), Final (1 game) — standard bracket.
//
// Higher seed (lower number) is the "home" team.
// advance() plays the next unplayed game deterministically from the master seed.
// Guaranteed to terminate: spine cap + higher-seed-advances fallback in SimMatch.

import { JUMP_SEEDS, type Franchise } from './teams';
import { runHeadless, type MatchResult } from './SimMatch';

// ─── Round-1 matchup ordering ────────────────────────────────────────────────

// Canonical bracket pairings by seed index into JUMP_SEEDS (0-based).
// Pairing: 1v16, 8v9, 5v12, 4v13, 2v15, 7v10, 6v11, 3v14
const ROUND1_PAIRS: Array<[number, number]> = [
  [0, 15], // 1 vs 16
  [7, 8],  // 8 vs  9
  [4, 11], // 5 vs 12
  [3, 12], // 4 vs 13
  [1, 14], // 2 vs 15
  [6, 9],  // 7 vs 10
  [5, 10], // 6 vs 11
  [2, 13], // 3 vs 14
];

// ─── Bracket node ────────────────────────────────────────────────────────────

export interface BracketGame {
  /** Unique game id within the bracket (round-based, 0-indexed within round). */
  id: string;
  round: number;
  /** Home is the higher seed (lower seed number). May be null before prior round resolved. */
  home: Franchise | null;
  away: Franchise | null;
  result: MatchResult | null;
  /** The seed used for SimMatch.runHeadless for this game. */
  gameSeed: number;
}

export interface BracketState {
  games: BracketGame[];
  champion: Franchise | null;
}

// ─── Bracket class ───────────────────────────────────────────────────────────

export class Bracket {
  private _games: BracketGame[];
  private _champion: Franchise | null = null;

  /**
   * @param masterSeed — determines all game seeds deterministically.
   *   Games are seeded as: mulberry32 derived from masterSeed + game index.
   */
  constructor(masterSeed: number) {
    this._games = Bracket._buildGames(masterSeed);
  }

  get state(): Readonly<BracketState> {
    return {
      games: this._games.map((g) => ({ ...g })),
      champion: this._champion,
    };
  }

  /** True if all 15 games have been played. */
  get isComplete(): boolean {
    return this._champion !== null;
  }

  /**
   * Play the next unplayed game that has both participants available.
   * Returns the game that was played, or null if the bracket is already complete.
   */
  advance(): BracketGame | null {
    if (this._champion !== null) return null;

    this._populateParticipants();

    const next = this._games.find((g) => g.result === null && g.home !== null && g.away !== null);
    if (!next) return null;

    const result = runHeadless(next.home!, next.away!, next.gameSeed);
    next.result = result;

    // If this is the final (round 4, game 0), set the champion.
    if (next.round === 4) {
      this._champion = result.winner === 'home' ? next.home! : next.away!;
    }

    return { ...next };
  }

  /** Peek the next playable game (both participants known) WITHOUT playing
   *  it — so the caller can decide to play it interactively vs. simulate. */
  nextGame(): BracketGame | null {
    if (this._champion !== null) return null;
    this._populateParticipants();
    const next = this._games.find(
      (g) => g.result === null && g.home !== null && g.away !== null,
    );
    return next ? { ...next } : null;
  }

  /** Report a player-PLAYED result for the next game (instead of simming
   *  it). Mirrors advance()'s champion bookkeeping. */
  playNext(result: MatchResult): BracketGame | null {
    if (this._champion !== null) return null;
    this._populateParticipants();
    const next = this._games.find(
      (g) => g.result === null && g.home !== null && g.away !== null,
    );
    if (!next) return null;
    next.result = result;
    if (next.round === 4) {
      this._champion = result.winner === 'home' ? next.home! : next.away!;
    }
    return { ...next };
  }

  /**
   * Play all remaining unplayed games to completion.
   * Returns the champion.
   */
  simulateAll(): Franchise {
    while (!this.isComplete) {
      this.advance();
    }
    return this._champion!;
  }

  // ─── Private ───────────────────────────────────────────────────────────────

  /** Build all 15 game slots (8 R1 + 4 R2 + 2 R3 + 1 Final). */
  private static _buildGames(masterSeed: number): BracketGame[] {
    const games: BracketGame[] = [];
    let gameIndex = 0;

    // Round 1: 8 games — participants filled from JUMP_SEEDS
    for (const [aIdx, bIdx] of ROUND1_PAIRS) {
      const home = JUMP_SEEDS[aIdx]; // lower seed number = home
      const away = JUMP_SEEDS[bIdx];
      games.push({
        id: `r1-g${gameIndex}`,
        round: 1,
        home,
        away,
        result: null,
        gameSeed: Bracket._gameSeed(masterSeed, gameIndex),
      });
      gameIndex++;
    }

    // Round 2: 4 games — participants TBD from R1 winners
    for (let g = 0; g < 4; g++) {
      games.push({
        id: `r2-g${g}`,
        round: 2,
        home: null,
        away: null,
        result: null,
        gameSeed: Bracket._gameSeed(masterSeed, gameIndex++),
      });
    }

    // Round 3: 2 games
    for (let g = 0; g < 2; g++) {
      games.push({
        id: `r3-g${g}`,
        round: 3,
        home: null,
        away: null,
        result: null,
        gameSeed: Bracket._gameSeed(masterSeed, gameIndex++),
      });
    }

    // Final: 1 game
    games.push({
      id: `r4-g0`,
      round: 4,
      home: null,
      away: null,
      result: null,
      gameSeed: Bracket._gameSeed(masterSeed, gameIndex++),
    });

    return games;
  }

  /** Propagate winners from completed games into the next-round slots. */
  private _populateParticipants(): void {
    // R1 → R2: 8 R1 winners fill 4 R2 slots in pairs.
    // Bracket structure: R2-g0 gets R1-g0 winner vs R1-g1 winner, etc.
    const r1 = this._games.filter((g) => g.round === 1);
    const r2 = this._games.filter((g) => g.round === 2);
    for (let i = 0; i < 4; i++) {
      const g0 = r1[i * 2];
      const g1 = r1[i * 2 + 1];
      const slot = r2[i];
      if (slot.home === null && g0.result !== null) {
        const w = g0.result.winner === 'home' ? g0.home! : g0.away!;
        // Home is the one with the better seed (lower number)
        slot.home = w;
      }
      if (slot.away === null && g1.result !== null) {
        const w = g1.result.winner === 'home' ? g1.home! : g1.away!;
        slot.away = w;
      }
    }

    // R2 → R3
    const r3 = this._games.filter((g) => g.round === 3);
    for (let i = 0; i < 2; i++) {
      const g0 = r2[i * 2];
      const g1 = r2[i * 2 + 1];
      const slot = r3[i];
      if (slot.home === null && g0.result !== null) {
        slot.home = g0.result.winner === 'home' ? g0.home! : g0.away!;
      }
      if (slot.away === null && g1.result !== null) {
        slot.away = g1.result.winner === 'home' ? g1.home! : g1.away!;
      }
    }

    // R3 → Final
    const final = this._games.find((g) => g.round === 4)!;
    if (final.home === null && r3[0].result !== null) {
      final.home = r3[0].result.winner === 'home' ? r3[0].home! : r3[0].away!;
    }
    if (final.away === null && r3[1].result !== null) {
      final.away = r3[1].result.winner === 'home' ? r3[1].home! : r3[1].away!;
    }
  }

  /** Derive a per-game seed from the master seed and game index. */
  private static _gameSeed(masterSeed: number, gameIndex: number): number {
    // Simple mixing: xor with a shifted index to avoid trivial patterns.
    const mixed = (masterSeed ^ (gameIndex * 0x9e3779b9)) >>> 0;
    return mixed;
  }
}

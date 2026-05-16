// Season.ts — League season container: holds bracket + standings.
//
// simulateRemaining(seed) completes all un-watched games deterministically.
// Exposes champion + full results tree.

import { Bracket, type BracketGame, type BracketState } from './Bracket';
import type { Franchise } from './teams';
import type { MatchResult } from './SimMatch';

// ─── Standing entry ───────────────────────────────────────────────────────────

export interface StandingEntry {
  franchise: Franchise;
  /** Jump round reached (1 = first-round exit, 4 = champion). */
  roundReached: number;
  /** True if this franchise is the champion. */
  champion: boolean;
}

// ─── Results tree (mirrors bracket structure) ─────────────────────────────────

export interface SeasonResultsTree {
  rounds: Array<{
    round: number;
    games: Array<{
      id: string;
      home: Franchise | null;
      away: Franchise | null;
      result: MatchResult | null;
    }>;
  }>;
  champion: Franchise | null;
}

// ─── Season ───────────────────────────────────────────────────────────────────

export class Season {
  private readonly _bracket: Bracket;
  private readonly _seed: number;

  /**
   * @param seed — master seed for all bracket games. Passed to Bracket.
   */
  constructor(seed: number) {
    this._seed = seed;
    this._bracket = new Bracket(seed);
  }

  get seed(): number {
    return this._seed;
  }

  get bracketState(): Readonly<BracketState> {
    return this._bracket.state;
  }

  get champion(): Franchise | null {
    return this._bracket.state.champion;
  }

  /**
   * Simulate all remaining un-watched games to completion.
   * Idempotent: if all games are already played, returns the existing champion.
   *
   * @param _seed — reserved for future per-simulation override; currently unused
   *   (the Season was constructed with a fixed seed and Bracket is already seeded).
   */
  simulateRemaining(_seed?: number): Franchise {
    return this._bracket.simulateAll();
  }

  /**
   * Play the next single unplayed game.
   * Returns the game played, or null if the bracket is complete.
   */
  advance(): BracketGame | null {
    return this._bracket.advance();
  }

  /**
   * Build standings for the Jump — one entry per team ordered by round reached
   * (champion first, then finalists, etc.).
   */
  standings(): StandingEntry[] {
    const state = this._bracket.state;
    const entries: StandingEntry[] = [];
    const seenIds = new Set<string>();

    // Walk all completed games to find each team's exit round.
    for (const game of state.games) {
      if (game.result === null) continue;

      const loser =
        game.result.winner === 'home' ? game.away! : game.home!;

      if (!seenIds.has(loser.id)) {
        seenIds.add(loser.id);
        entries.push({
          franchise: loser,
          roundReached: game.round,
          champion: false,
        });
      }
    }

    if (state.champion && !seenIds.has(state.champion.id)) {
      entries.push({
        franchise: state.champion,
        roundReached: 4,
        champion: true,
      });
    } else if (state.champion) {
      // Upgrade the champion entry if it was already added as a loser somehow
      const entry = entries.find((e) => e.franchise.id === state.champion!.id);
      if (entry) {
        entry.roundReached = 4;
        entry.champion = true;
      }
    }

    // Sort: champion first, then by round desc, then by seed asc.
    entries.sort((a, b) => {
      if (a.champion && !b.champion) return -1;
      if (!a.champion && b.champion) return 1;
      if (b.roundReached !== a.roundReached) return b.roundReached - a.roundReached;
      return (a.franchise.seed ?? 999) - (b.franchise.seed ?? 999);
    });

    return entries;
  }

  /**
   * Build the full results tree, organized by round.
   */
  resultsTree(): SeasonResultsTree {
    const state = this._bracket.state;
    const roundNumbers = [1, 2, 3, 4];

    const rounds = roundNumbers.map((round) => ({
      round,
      games: state.games
        .filter((g) => g.round === round)
        .map(({ id, home, away, result }) => ({ id, home, away, result })),
    }));

    return { rounds, champion: state.champion };
  }
}

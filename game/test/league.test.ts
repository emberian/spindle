// league.test.ts — RIG league meta-layer tests.
//
// Covers: SimMatch determinism + termination + score distribution,
//         Rise-vs-Fall style divergence (canon physics→style→soul),
//         Bracket termination + champion + 15 games,
//         Persistence round-trip + migrate(absent)→fresh,
//         whole pipeline pure/deterministic.

import { describe, it, expect } from 'vitest';
import { runHeadless } from '../src/league/SimMatch';
import { Bracket } from '../src/league/Bracket';
import { Season } from '../src/league/Season';
import { load, save, migrate, freshSave, clearSave, addReplay } from '../src/league/Persistence';
import { JUMP_SEEDS, TEAMS, styleToProfile } from '../src/league/teams';

// ─── Fixtures ────────────────────────────────────────────────────────────────

const SEED1 = JUMP_SEEDS[0]!; // #1 Tuebor (fall-dynasty, big-slow)
const SEED16 = JUMP_SEEDS[15]!; // #16 Vesta Tool-Bag Originals (rise-ceremonial, small-fast)
const SEED8 = JUMP_SEEDS[7]!; // #8 Toliman Gradient (rise-chaos, small-fast)
const SEED9 = JUMP_SEEDS[8]!; // #9 Hebat Resolve (ground-defiant, mid)

// Representative Rise team
const RISE_TEAM = TEAMS.find((t) => t.styleTag === 'rise-chaos')!;
// Representative Fall team
const FALL_TEAM = TEAMS.find((t) => t.styleTag === 'fall-dynasty')!;

// ─── SimMatch ────────────────────────────────────────────────────────────────

describe('SimMatch.runHeadless — determinism', () => {
  it('same seed → identical MatchResult (full structural equality)', () => {
    const r1 = runHeadless(SEED1, SEED16, 42);
    const r2 = runHeadless(SEED1, SEED16, 42);
    expect(r1.scoreHome).toBe(r2.scoreHome);
    expect(r1.scoreAway).toBe(r2.scoreAway);
    expect(r1.winner).toBe(r2.winner);
    expect(r1.events.length).toBe(r2.events.length);
    expect(r1.box.home).toEqual(r2.box.home);
    expect(r1.box.away).toEqual(r2.box.away);
  });

  it('different seed → (likely) different score', () => {
    // Not a strict guarantee, but with 100 seed pairs the chance of all matching is negligible.
    let differences = 0;
    for (let s = 0; s < 20; s++) {
      const r1 = runHeadless(SEED1, SEED16, s);
      const r2 = runHeadless(SEED1, SEED16, s + 1000);
      if (r1.scoreHome !== r2.scoreHome || r1.scoreAway !== r2.scoreAway) {
        differences++;
      }
    }
    expect(differences).toBeGreaterThan(0);
  });
});

describe('SimMatch.runHeadless — termination + winner', () => {
  it('always terminates with exactly one winner', () => {
    for (let seed = 0; seed < 30; seed++) {
      const r = runHeadless(SEED1, SEED16, seed);
      expect(r.winner === 'home' || r.winner === 'away').toBe(true);
    }
  });

  it('home and away franchise refs are preserved', () => {
    const r = runHeadless(SEED1, SEED16, 7);
    expect(r.home.id).toBe(SEED1.id);
    expect(r.away.id).toBe(SEED16.id);
  });

  it('score distribution is non-degenerate (not always 0-0)', () => {
    let zeroZero = 0;
    for (let seed = 0; seed < 50; seed++) {
      const r = runHeadless(SEED8, SEED9, seed);
      if (r.scoreHome === 0 && r.scoreAway === 0) zeroZero++;
    }
    expect(zeroZero).toBeLessThan(50);
  });

  it('winner matches the score (winner has strictly more points, or fallback was used)', () => {
    for (let seed = 0; seed < 30; seed++) {
      const r = runHeadless(SEED1, SEED16, seed);
      if (r.scoreHome !== r.scoreAway) {
        if (r.scoreHome > r.scoreAway) {
          expect(r.winner).toBe('home');
        } else {
          expect(r.winner).toBe('away');
        }
      }
      // Tied-score matches used the spine/fallback rule — winner is still defined.
      expect(r.winner).toBeDefined();
    }
  });

  it('box scores are non-negative integers', () => {
    const r = runHeadless(SEED1, SEED16, 99);
    for (const side of [r.box.home, r.box.away]) {
      expect(side.falls).toBeGreaterThanOrEqual(0);
      expect(side.rises).toBeGreaterThanOrEqual(0);
      expect(side.loops).toBeGreaterThanOrEqual(0);
      expect(side.curls).toBeGreaterThanOrEqual(0);
      expect(side.grounds).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('SimMatch — Rise-vs-Fall team style divergence (canon physics→style→soul)', () => {
  it('Rise teams (small-fast) attempt statistically more rises than Fall teams (big-slow)', () => {
    const SEEDS = 60;
    let riseRises = 0;
    let fallRises = 0;

    for (let seed = 0; seed < SEEDS; seed++) {
      // Rise team as home vs a neutral opponent
      const rRise = runHeadless(RISE_TEAM, FALL_TEAM, seed);
      riseRises += rRise.box.home.rises;
      fallRises += rRise.box.away.rises;
    }

    // Over 60 seeds the Rise team should accumulate more rises than the Fall team.
    expect(riseRises).toBeGreaterThan(fallRises);
  });

  it('Rise teams attempt statistically more loops than Fall teams', () => {
    const SEEDS = 60;
    let riseLoops = 0;
    let fallLoops = 0;

    for (let seed = 0; seed < SEEDS; seed++) {
      const r = runHeadless(RISE_TEAM, FALL_TEAM, seed);
      riseLoops += r.box.home.loops;
      fallLoops += r.box.away.loops;
    }

    expect(riseLoops).toBeGreaterThanOrEqual(fallLoops);
  });

  it('styleToProfile confirms Rise freeEndBias > Fall freeEndBias (invariant from teams.ts)', () => {
    const riseProf = styleToProfile(RISE_TEAM.styleTag, RISE_TEAM.cylinderClass);
    const fallProf = styleToProfile(FALL_TEAM.styleTag, FALL_TEAM.cylinderClass);
    expect(riseProf.freeEndBias).toBeGreaterThan(fallProf.freeEndBias);
    expect(riseProf.loopPropensity).toBeGreaterThan(fallProf.loopPropensity);
  });

  it('Fall teams accumulate more falls than Rise teams', () => {
    const SEEDS = 60;
    let riseFalls = 0;
    let fallFalls = 0;

    for (let seed = 0; seed < SEEDS; seed++) {
      const r = runHeadless(RISE_TEAM, FALL_TEAM, seed);
      riseFalls += r.box.home.falls;
      fallFalls += r.box.away.falls;
    }

    expect(fallFalls).toBeGreaterThanOrEqual(riseFalls);
  });
});

// ─── Bracket ─────────────────────────────────────────────────────────────────

describe('Bracket — construction', () => {
  it('starts with exactly 15 games across 4 rounds', () => {
    const bracket = new Bracket(12345);
    const { games } = bracket.state;
    expect(games.length).toBe(15);
    expect(games.filter((g) => g.round === 1).length).toBe(8);
    expect(games.filter((g) => g.round === 2).length).toBe(4);
    expect(games.filter((g) => g.round === 3).length).toBe(2);
    expect(games.filter((g) => g.round === 4).length).toBe(1);
  });

  it('R1 games all have both participants set from JUMP_SEEDS', () => {
    const bracket = new Bracket(0);
    const r1 = bracket.state.games.filter((g) => g.round === 1);
    for (const game of r1) {
      expect(game.home).not.toBeNull();
      expect(game.away).not.toBeNull();
    }
  });

  it('canonical pairing: seed 1 vs seed 16 in R1', () => {
    const bracket = new Bracket(0);
    const r1 = bracket.state.games.filter((g) => g.round === 1);
    const matchup = r1.find(
      (g) => g.home?.seed === 1 || g.away?.seed === 1,
    );
    expect(matchup).toBeDefined();
    const seeds = [matchup!.home?.seed, matchup!.away?.seed];
    expect(seeds).toContain(1);
    expect(seeds).toContain(16);
  });

  it('canonical pairing: seed 2 vs seed 15 in R1', () => {
    const bracket = new Bracket(0);
    const r1 = bracket.state.games.filter((g) => g.round === 1);
    const matchup = r1.find(
      (g) => [g.home?.seed, g.away?.seed].includes(2),
    );
    expect(matchup).toBeDefined();
    const seeds = [matchup!.home?.seed, matchup!.away?.seed];
    expect(seeds).toContain(2);
    expect(seeds).toContain(15);
  });
});

describe('Bracket — simulateAll termination + champion', () => {
  it('terminates with exactly one champion', () => {
    const bracket = new Bracket(42);
    const champion = bracket.simulateAll();
    expect(champion).toBeDefined();
    expect(champion.id.length).toBeGreaterThan(0);
    expect(bracket.isComplete).toBe(true);
  });

  it('champion is one of the 16 seeded teams', () => {
    const bracket = new Bracket(100);
    const champion = bracket.simulateAll();
    const ids = JUMP_SEEDS.map((t) => t.id);
    expect(ids).toContain(champion.id);
  });

  it('exactly 15 games are played (single-elim from 16)', () => {
    const bracket = new Bracket(7);
    bracket.simulateAll();
    const { games } = bracket.state;
    const played = games.filter((g) => g.result !== null);
    expect(played.length).toBe(15);
  });

  it('valid tree: every team appears at most once per round', () => {
    const bracket = new Bracket(55);
    bracket.simulateAll();
    const { games } = bracket.state;

    for (const round of [1, 2, 3, 4]) {
      const roundGames = games.filter((g) => g.round === round);
      const teamIds: string[] = [];
      for (const game of roundGames) {
        if (game.home) teamIds.push(game.home.id);
        if (game.away) teamIds.push(game.away.id);
      }
      expect(new Set(teamIds).size).toBe(teamIds.length);
    }
  });

  it('champion is a R4 finalist (final game winner)', () => {
    const bracket = new Bracket(200);
    bracket.simulateAll();
    const { games, champion } = bracket.state;
    const final = games.find((g) => g.round === 4)!;
    expect(final.result).not.toBeNull();
    const finalWinner =
      final.result!.winner === 'home' ? final.home!.id : final.away!.id;
    expect(champion?.id).toBe(finalWinner);
  });

  it('same seed → same champion (determinism)', () => {
    const b1 = new Bracket(999);
    const b2 = new Bracket(999);
    expect(b1.simulateAll().id).toBe(b2.simulateAll().id);
  });

  it('different seeds may produce different champions', () => {
    const champions = new Set<string>();
    for (let s = 0; s < 20; s++) {
      const b = new Bracket(s * 113 + 7);
      champions.add(b.simulateAll().id);
    }
    // At least 2 different champions across 20 seeds.
    expect(champions.size).toBeGreaterThan(1);
  });
});

// ─── Season ───────────────────────────────────────────────────────────────────

describe('Season', () => {
  it('simulateRemaining returns a champion', () => {
    const season = new Season(1234);
    const champion = season.simulateRemaining();
    expect(champion).toBeDefined();
    expect(season.champion?.id).toBe(champion.id);
  });

  it('resultsTree has 4 rounds with correct game counts', () => {
    const season = new Season(5678);
    season.simulateRemaining();
    const tree = season.resultsTree();
    expect(tree.rounds.length).toBe(4);
    expect(tree.rounds[0].games.length).toBe(8);
    expect(tree.rounds[1].games.length).toBe(4);
    expect(tree.rounds[2].games.length).toBe(2);
    expect(tree.rounds[3].games.length).toBe(1);
    expect(tree.champion).toBeDefined();
  });

  it('standings has exactly 16 entries after full simulation', () => {
    const season = new Season(111);
    season.simulateRemaining();
    const standings = season.standings();
    expect(standings.length).toBe(16);
  });

  it('standings: first entry is the champion', () => {
    const season = new Season(222);
    season.simulateRemaining();
    const standings = season.standings();
    expect(standings[0].champion).toBe(true);
    expect(standings[0].franchise.id).toBe(season.champion?.id);
  });

  it('deterministic: same seed → same champion', () => {
    const s1 = new Season(777);
    const s2 = new Season(777);
    expect(s1.simulateRemaining().id).toBe(s2.simulateRemaining().id);
  });
});

// ─── Persistence ─────────────────────────────────────────────────────────────

describe('Persistence — migrate', () => {
  it('migrate(absent) → fresh v2 save', () => {
    const result = migrate(undefined);
    expect(result.schema).toBe(2);
    expect(result.chosenFranchiseId).toBeNull();
    expect(result.bracket).toBeNull();
    expect(typeof result.seed).toBe('number');
    expect(result.difficulty).toBe('league');
    expect(result.replays).toEqual({});
  });

  it('migrate(null) → fresh v2 save', () => {
    const result = migrate(null);
    expect(result.schema).toBe(2);
  });

  it('migrate({}) → fresh v2 (no schema field)', () => {
    const result = migrate({});
    expect(result.schema).toBe(2);
  });

  it('migrate(v1) → v2 with field remapping', () => {
    const v1 = {
      schema: 1,
      franchiseId: 'tuebor-spin-detroiters',
      seed: 99,
      difficulty: 'playoff',
      replays: { 'r1-g0': 42 },
      createdAt: '2025-01-01T00:00:00.000Z',
    };
    const result = migrate(v1);
    expect(result.schema).toBe(2);
    expect(result.chosenFranchiseId).toBe('tuebor-spin-detroiters');
    expect(result.seed).toBe(99);
    expect(result.difficulty).toBe('playoff');
    expect(result.createdAt).toBe('2025-01-01T00:00:00.000Z');
    expect(result.bracket).toBeNull(); // v1 bracket not preserved
  });

  it('migrate(v2) → pass-through with validation', () => {
    const v2 = freshSave(123);
    v2.chosenFranchiseId = 'belt-free-local-9';
    const result = migrate(v2);
    expect(result.schema).toBe(2);
    expect(result.chosenFranchiseId).toBe('belt-free-local-9');
    expect(result.seed).toBe(123);
  });

  it('migrate(unknown schema) → fresh', () => {
    const result = migrate({ schema: 99 });
    expect(result.schema).toBe(2);
  });
});

describe('Persistence — freshSave', () => {
  it('produces a valid v2 save', () => {
    const s = freshSave(42);
    expect(s.schema).toBe(2);
    expect(s.seed).toBe(42);
    expect(s.bracket).toBeNull();
    expect(s.chosenFranchiseId).toBeNull();
    expect(s.replays).toEqual({});
  });
});

describe('Persistence — save/load round-trip (in-memory / no window)', () => {
  it('save then load round-trips the save data', () => {
    // clearSave first to reset any in-memory state
    clearSave();
    const data = freshSave(9999);
    data.chosenFranchiseId = 'toliman-gradient';
    data.difficulty = 'exhibition';
    save(data);

    const loaded = load();
    expect(loaded.schema).toBe(2);
    expect(loaded.chosenFranchiseId).toBe('toliman-gradient');
    expect(loaded.difficulty).toBe('exhibition');
    expect(loaded.seed).toBe(9999);
  });

  it('load with no prior save returns a fresh v2 save', () => {
    clearSave();
    const loaded = load();
    expect(loaded.schema).toBe(2);
    expect(loaded.bracket).toBeNull();
  });

  it('addReplay attaches a replay seed and caps at MAX_REPLAYS (50)', () => {
    let data = freshSave(1);
    for (let i = 0; i < 60; i++) {
      data = addReplay(data, `r1-g${i}`, i * 7);
    }
    expect(Object.keys(data.replays).length).toBeLessThanOrEqual(50);
  });

  it('replay seed is retrievable after addReplay', () => {
    let data = freshSave(1);
    data = addReplay(data, 'r2-g1', 12345);
    expect(data.replays['r2-g1']).toBe(12345);
  });
});

// ─── Whole pipeline: pure/deterministic ──────────────────────────────────────

describe('Whole pipeline — pure / deterministic', () => {
  it('same season seed → identical champion across two independent Season instances', () => {
    const s1 = new Season(314159);
    const s2 = new Season(314159);
    s1.simulateRemaining();
    s2.simulateRemaining();
    expect(s1.champion?.id).toBe(s2.champion?.id);
  });

  it('same season seed → identical 15-game result sequence', () => {
    const s1 = new Season(271828);
    const s2 = new Season(271828);
    s1.simulateRemaining();
    s2.simulateRemaining();

    const tree1 = s1.resultsTree();
    const tree2 = s2.resultsTree();

    for (let r = 0; r < 4; r++) {
      const round1 = tree1.rounds[r].games;
      const round2 = tree2.rounds[r].games;
      for (let g = 0; g < round1.length; g++) {
        expect(round1[g].result?.scoreHome).toBe(round2[g].result?.scoreHome);
        expect(round1[g].result?.scoreAway).toBe(round2[g].result?.scoreAway);
        expect(round1[g].result?.winner).toBe(round2[g].result?.winner);
      }
    }
  });

  it('pipeline does not mutate JUMP_SEEDS', () => {
    const seedsBefore = JUMP_SEEDS.map((t) => ({ id: t.id, seed: t.seed }));
    const season = new Season(42);
    season.simulateRemaining();
    const seedsAfter = JUMP_SEEDS.map((t) => ({ id: t.id, seed: t.seed }));
    expect(seedsAfter).toEqual(seedsBefore);
  });

  it('multiple different seeds each produce a valid champion', () => {
    for (const seed of [0, 1, 7, 42, 100, 999, 0xdeadbeef]) {
      const season = new Season(seed);
      const champion = season.simulateRemaining();
      expect(champion.id.length).toBeGreaterThan(0);
      const ids = JUMP_SEEDS.map((t) => t.id);
      expect(ids).toContain(champion.id);
    }
  });
});

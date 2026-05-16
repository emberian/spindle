import { describe, it, expect } from 'vitest';
import {
  TEAMS,
  JUMP_SEEDS,
  styleToProfile,
  type Faction,
  type StyleTag,
  type CylinderClass,
} from '../src/league/teams';

const VALID_FACTIONS: Faction[] = ['FRM', 'CNE', 'CON', 'BLT', 'EUR', 'REF', 'CON/REF', 'REF/CON'];

describe('TEAMS dataset', () => {
  it('has exactly 32 franchises', () => {
    expect(TEAMS.length).toBe(32);
  });

  it('every franchise has a non-empty id, name, logo, and fanbase', () => {
    for (const t of TEAMS) {
      expect(t.id.length).toBeGreaterThan(0);
      expect(t.name.length).toBeGreaterThan(0);
      expect(t.logo.length).toBeGreaterThan(0);
      expect(t.fanbase.length).toBeGreaterThan(0);
    }
  });

  it('ids are unique and kebab-case', () => {
    const ids = TEAMS.map((t) => t.id);
    const unique = new Set(ids);
    expect(unique.size).toBe(32);
    for (const id of ids) {
      expect(id).toMatch(/^[a-z0-9-]+$/);
    }
  });

  it('every faction is a member of the Faction union', () => {
    for (const t of TEAMS) {
      expect(VALID_FACTIONS).toContain(t.faction);
    }
  });

  it('every record has w ≥ 0 and l ≥ 0', () => {
    for (const t of TEAMS) {
      expect(t.record.w).toBeGreaterThanOrEqual(0);
      expect(t.record.l).toBeGreaterThanOrEqual(0);
    }
  });

  it('seed is either null or an integer in 1..16', () => {
    for (const t of TEAMS) {
      if (t.seed !== null) {
        expect(Number.isInteger(t.seed)).toBe(true);
        expect(t.seed).toBeGreaterThanOrEqual(1);
        expect(t.seed).toBeLessThanOrEqual(16);
      }
    }
  });

  it('every conference is sol | near | far', () => {
    const valid = new Set<string>(['sol', 'near', 'far']);
    for (const t of TEAMS) {
      expect(valid.has(t.conference)).toBe(true);
    }
  });

  it('colors are two-element hex tuples', () => {
    const hexRe = /^#[0-9a-fA-F]{6}$/;
    for (const t of TEAMS) {
      expect(t.colors).toHaveLength(2);
      expect(t.colors[0]).toMatch(hexRe);
      expect(t.colors[1]).toMatch(hexRe);
    }
  });
});

describe('JUMP_SEEDS', () => {
  it('has exactly 16 entries', () => {
    expect(JUMP_SEEDS.length).toBe(16);
  });

  it('seeds are contiguous integers 1..16', () => {
    const seeds = JUMP_SEEDS.map((t) => t.seed as number);
    const sorted = [...seeds].sort((a, b) => a - b);
    expect(sorted).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
  });

  it('seed values are unique', () => {
    const seeds = JUMP_SEEDS.map((t) => t.seed as number);
    expect(new Set(seeds).size).toBe(16);
  });

  it('is sorted ascending by seed', () => {
    for (let i = 1; i < JUMP_SEEDS.length; i++) {
      expect(JUMP_SEEDS[i].seed as number).toBeGreaterThan(JUMP_SEEDS[i - 1].seed as number);
    }
  });

  it('#1 seed is Tuebor Spin Detroiters', () => {
    expect(JUMP_SEEDS[0].seed).toBe(1);
    expect(JUMP_SEEDS[0].id).toBe('tuebor-spin-detroiters');
  });

  it('#2 seed is Boston Commons Reach', () => {
    expect(JUMP_SEEDS[1].seed).toBe(2);
    expect(JUMP_SEEDS[1].id).toBe('boston-commons-reach');
  });

  it('#16 seed is Vesta Tool-Bag Originals', () => {
    expect(JUMP_SEEDS[15].seed).toBe(16);
    expect(JUMP_SEEDS[15].id).toBe('vesta-tool-bag-originals');
  });

  it('Jump table matches canon: seeds 1–16 by id', () => {
    const expected: [number, string][] = [
      [1, 'tuebor-spin-detroiters'],
      [2, 'boston-commons-reach'],
      [3, 'roma-aeterna-brokers'],
      [4, 'tau-ceti-mare-nostrum'],
      [5, 'manchester-memorial'],
      [6, 'luhman-orbital-riot'],
      [7, 'kuiper-long-call'],
      [8, 'toliman-gradient'],
      [9, 'hebat-resolve'],
      [10, 'centauri-b-hammerline'],
      [11, 'cascade-works-foremen'],
      [12, 'belt-free-local-9'],
      [13, 'ceres-counterspin'],
      [14, 'lira-free-company'],
      [15, 'wolf-359-migration'],
      [16, 'vesta-tool-bag-originals'],
    ];
    for (const [seed, id] of expected) {
      const team = JUMP_SEEDS.find((t) => t.seed === seed);
      expect(team).toBeDefined();
      expect(team!.id).toBe(id);
    }
  });
});

describe('styleToProfile', () => {
  const riseTeams = TEAMS.filter(
    (t): t is typeof t & { styleTag: StyleTag } =>
      t.styleTag.startsWith('rise') && t.cylinderClass === 'small-fast',
  );
  const fallTeams = TEAMS.filter(
    (t): t is typeof t & { styleTag: StyleTag } =>
      t.styleTag.startsWith('fall') && t.cylinderClass === 'big-slow',
  );

  it('returns a profile where all values are in [0, 1]', () => {
    const cylinderClasses: CylinderClass[] = ['big-slow', 'small-fast', 'mid', 'neutral'];
    for (const t of TEAMS) {
      for (const cc of cylinderClasses) {
        const p = styleToProfile(t.styleTag, cc);
        for (const [key, val] of Object.entries(p)) {
          expect(val, `${t.id} / ${cc} / ${key}`).toBeGreaterThanOrEqual(0);
          expect(val, `${t.id} / ${cc} / ${key}`).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  it('Rise (small-fast) teams have higher freeEndBias than Fall (big-slow) teams — canon physics invariant', () => {
    expect(riseTeams.length).toBeGreaterThan(0);
    expect(fallTeams.length).toBeGreaterThan(0);

    const avgRiseFree =
      riseTeams.reduce((sum, t) => sum + styleToProfile(t.styleTag, t.cylinderClass).freeEndBias, 0) /
      riseTeams.length;
    const avgFallFree =
      fallTeams.reduce((sum, t) => sum + styleToProfile(t.styleTag, t.cylinderClass).freeEndBias, 0) /
      fallTeams.length;

    expect(avgRiseFree).toBeGreaterThan(avgFallFree);
  });

  it('Rise teams have higher variance than Fall teams', () => {
    const avgRiseVar =
      riseTeams.reduce((sum, t) => sum + styleToProfile(t.styleTag, t.cylinderClass).variance, 0) /
      riseTeams.length;
    const avgFallVar =
      fallTeams.reduce((sum, t) => sum + styleToProfile(t.styleTag, t.cylinderClass).variance, 0) /
      fallTeams.length;

    expect(avgRiseVar).toBeGreaterThan(avgFallVar);
  });

  it('Rise teams have higher loopPropensity than Fall teams', () => {
    const avgRiseLoop =
      riseTeams.reduce((sum, t) => sum + styleToProfile(t.styleTag, t.cylinderClass).loopPropensity, 0) /
      riseTeams.length;
    const avgFallLoop =
      fallTeams.reduce((sum, t) => sum + styleToProfile(t.styleTag, t.cylinderClass).loopPropensity, 0) /
      fallTeams.length;

    expect(avgRiseLoop).toBeGreaterThan(avgFallLoop);
  });

  it('mid-cylinder teams have higher awayPointBias than big-slow teams', () => {
    const midTeams = TEAMS.filter((t) => t.cylinderClass === 'mid');
    expect(midTeams.length).toBeGreaterThan(0);

    const avgMidAway =
      midTeams.reduce((sum, t) => sum + styleToProfile(t.styleTag, t.cylinderClass).awayPointBias, 0) /
      midTeams.length;
    const avgFallAway =
      fallTeams.reduce((sum, t) => sum + styleToProfile(t.styleTag, t.cylinderClass).awayPointBias, 0) /
      fallTeams.length;

    expect(avgMidAway).toBeGreaterThan(avgFallAway);
  });

  it('styleToProfile is pure — same inputs return identical objects', () => {
    const a = styleToProfile('rise-chaos', 'small-fast');
    const b = styleToProfile('rise-chaos', 'small-fast');
    expect(a).toEqual(b);

    const c = styleToProfile('fall-dynasty', 'big-slow');
    const d = styleToProfile('fall-dynasty', 'big-slow');
    expect(c).toEqual(d);
  });

  it('TeamProfile has exactly the 8 expected keys', () => {
    const p = styleToProfile('ground-broker', 'mid');
    const keys = Object.keys(p).sort();
    expect(keys).toEqual([
      'aggression',
      'awayPointBias',
      'contestAggression',
      'freeEndBias',
      'grappleRisk',
      'loopPropensity',
      'snatchVsClatter',
      'variance',
    ]);
  });
});

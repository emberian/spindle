// skilleval.spec.ts — the SKILL eval (a tool, NOT a CI gate).
//
// v2: robust + Goodhart-resistant. Lessons from v1:
//  - One deterministic match = one lucky sample → average over MATCHUPS.
//  - An unbounded chaos term swamped the composite (RRT scored −54000;
//    the optimizer would just minimise thrash, ignoring real skill).
//    → EVERY component is normalised to [0,1] and the composite is a
//    weighted average in 0..100, so no single signal can dominate.
//  - Weights are explicit & summed-to-1 here: the teleology, made legible
//    and arguable. Skill = the cast advances (gates), possession chains
//    (passes), defense reads (intercepts), points are actually scored,
//    play stays on-field — and it is calm/committed, not flailing.
//
// Run: npx playwright test e2e/skilleval.spec.ts
// Reuse runSkillEval(page,label,tune) from an optimizer loop.
import { test, expect, type Page } from '@playwright/test';

// Normalisation caps (a component hits 1.0 at "clearly good for one match").
const CAP = {
  gate: 6, pass: 6, intc: 8, score: 30, poss: 0.3, thrashK: 300, skinPct: 80,
  // v2.1 — purpose signals: shots that convert (not heaves) & strung passes.
  shotConv: 0.4, chain: 4,
};
// Weights — sum to 1. Argue with these; this IS the skill model.
const WT = {
  prog: 0.18, pass: 0.15, intc: 0.08, score: 0.16, poss: 0.07,
  calm: 0.12, field: 0.06, shotConv: 0.10, chain: 0.08,
};
const SCORE_PTS: Record<string, number> = { fall: 2, rise: 5, loop: 7, curl: 5, ground: 1 };
// A few distinct matchups (home-row, away-row in the spectate picker) so a
// score is not one team-pairing's fluke.
const MATCHUPS: [number, number][] = [[0, 0], [1, 4], [2, 7]];

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

export interface SkillResult {
  label: string; composite: number;
  parts: Record<string, number>; raw: Record<string, unknown>;
}

async function oneMatch(
  page: Page, tune: Record<string, number>, hRow: number, aRow: number, samples: number,
): Promise<{ sub: Record<string, number>; raw: Record<string, number> }> {
  await page.goto('/spindle/', { waitUntil: 'networkidle' });
  await page.evaluate((t) => { (window as { __rigtune?: unknown }).__rigtune = t; }, tune);
  await page.locator('#rig-watch-btn').click();
  await page.waitForSelector('#rig-spectate-overlay', { state: 'visible' });
  const q = page.locator('.rig-spec-pane');
  await q.first().locator('.rig-spec-team-row').nth(hRow).click();
  await q.last().locator('.rig-spec-team-row').nth(aRow).click();
  await page.locator('#rig-spectate-overlay .rig-spec-btn.primary').click();
  await page.waitForSelector('#rig-spec-controls', { state: 'visible' });
  await page.evaluate((t) => { (window as { __rigtune?: unknown }).__rigtune = t; }, tune);
  await page.locator('#rig-spec-controls .rig-sc-btn:has-text("4×")').click();
  let last: Record<string, unknown> | null = null, n = 0, skin = 0;
  for (let i = 0; i < samples; i++) {
    await page.waitForTimeout(900);
    const s = await page.evaluate(() => (window as { __rigai?: unknown }).__rigai ?? null);
    if (s) { last = s as Record<string, unknown>; n++; if (((s as { bellR?: number }).bellR ?? 0) > 31) skin++; }
  }
  const sk = (last?.skill ?? {}) as Record<string, number>;
  const sc = (last?.scores ?? {}) as Record<string, number>;
  const held = (last?.heldFrac as number) ?? 0;
  const ticks = sk.ticks || 1;
  const scorePts = Object.entries(sc).reduce((a, [k, v]) => a + (SCORE_PTS[k] ?? 0) * v, 0);
  // Scoring EVENTS (not turnover/clatter/caught) — for shot conversion.
  const scoreEvents = ['fall', 'rise', 'loop', 'curl', 'ground']
    .reduce((a, k) => a + (sc[k] ?? 0), 0);
  const tw = (last?.throws ?? {}) as Record<string, number>;
  const totalThrows = Object.values(tw).reduce((a, v) => a + v, 0);
  const shotConv = totalThrows > 0 ? scoreEvents / totalThrows : 0;
  const chainMax = sk.passStreakMax || 0;
  const skinPct = n ? (100 * skin) / n : 100;
  const thrashK = (sk.thrash || 0) / ticks * 1000;
  return {
    sub: {
      prog:     clamp01((sk.gateClears || 0) / CAP.gate),
      pass:     clamp01((sk.passes || 0) / CAP.pass),
      intc:     clamp01((sk.intercepts || 0) / CAP.intc),
      score:    clamp01(scorePts / CAP.score),
      poss:     clamp01(held / CAP.poss),
      calm:     1 - clamp01(thrashK / CAP.thrashK),
      field:    1 - clamp01(skinPct / CAP.skinPct),
      shotConv: clamp01(shotConv / CAP.shotConv),
      chain:    clamp01(chainMax / CAP.chain),
    },
    raw: {
      gateClears: sk.gateClears || 0, passes: sk.passes || 0,
      intercepts: sk.intercepts || 0, scorePts,
      heldFrac: held, thrashK: +thrashK.toFixed(0), skinPct: +skinPct.toFixed(0),
      shotConv: +shotConv.toFixed(2), chainMax,
    },
  };
}

export async function runSkillEval(
  page: Page, label: string, tune: Record<string, number>, samplesPerMatch = 40,
): Promise<SkillResult> {
  const subs: Record<string, number>[] = [];
  const raws: Record<string, number>[] = [];
  for (const [h, a] of MATCHUPS) {
    const r = await oneMatch(page, tune, h, a, samplesPerMatch);
    subs.push(r.sub); raws.push(r.raw);
  }
  const avg = (key: string, arr: Record<string, number>[]) =>
    arr.reduce((s, o) => s + (o[key] ?? 0), 0) / arr.length;
  const parts: Record<string, number> = {};
  let composite = 0;
  for (const k of Object.keys(WT) as (keyof typeof WT)[]) {
    const v = avg(k, subs);
    parts[k] = +v.toFixed(3);
    composite += WT[k] * v;
  }
  const raw: Record<string, unknown> = {};
  for (const k of Object.keys(raws[0])) raw[k] = +avg(k, raws).toFixed(1);
  return { label, composite: +(100 * composite).toFixed(1), parts, raw };
}

test('skill eval v2 — MPC vs RRT vs CEM (multi-matchup, normalised)', async ({ page }) => {
  test.setTimeout(1_200_000);
  const cfgs: { label: string; t: Record<string, number> }[] = [
    { label: 'MPC', t: { planner: 0 } },
    { label: 'RRT', t: { planner: 1 } },
    { label: 'CEM', t: { planner: 2 } },
  ];
  const out: SkillResult[] = [];
  for (const c of cfgs) out.push(await runSkillEval(page, c.label, c.t));
  for (const r of out) console.log('SKILL ' + JSON.stringify(r)); // eslint-disable-line no-console
  expect(out.length).toBe(cfgs.length);
});

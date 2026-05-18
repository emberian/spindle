// skilleval.ts — the SKILL evaluation harness (not a CI gate; a tool).
//
// Why: optimising toward a bad metric is Goodhart. "Skillful rig" is NOT
// "fast" and NOT "high possession" — both correlated with the BORING or
// CHAOTIC regimes this project hit. Skill = purpose: the cast advances
// (gate clears), possession chains via completed passes, defense reads
// (intercepts), real points are scored, and it is NOT flailing (anchor
// thrash) or pinned dead at the skin.
//
// The composite weights are deliberately EXPLICIT and inspectable here so
// the eval can be argued with and tuned — it is the teleology, made legible.
// Run:  npx playwright test e2e/skilleval.ts --config=playwright.config.ts
// (or import runSkillEval from an optimizer loop).
import { test, expect, type Page } from '@playwright/test';

// ── The skill model (argue with these) ───────────────────────────────────────
const W = {
  pass:       6,   // a completed, possession-keeping pass — core craft
  gateClear:  5,   // the cast actually advanced a gate — "playing the game"
  intercept:  3,   // a defensive read that took the bell
  scorePt:    1.5, // per weighted score point actually put up
  possession: 8,   // × heldFrac (modest — high alone is dwelly/boring)
  chaos:      9,   // × thrash-per-1000-ticks  (PENALTY — flailing)
  skinRot:    0.4, // × pctBellNearSkin        (PENALTY — dead float)
};
const SCORE_PTS: Record<string, number> = {
  fall: 2, rise: 5, loop: 7, curl: 5, ground: 1,
};

export interface SkillResult {
  label: string;
  composite: number;
  parts: Record<string, number>;
  raw: Record<string, unknown>;
}

async function startWatch(page: Page, tune: Record<string, number>): Promise<void> {
  await page.goto('/spindle/', { waitUntil: 'networkidle' });
  await page.evaluate((t) => { (window as { __rigtune?: unknown }).__rigtune = t; }, tune);
  await page.locator('#rig-watch-btn').click();
  await page.waitForSelector('#rig-spectate-overlay', { state: 'visible' });
  const q = page.locator('.rig-spec-pane');
  await q.first().locator('.rig-spec-team-row').first().click();
  await q.last().locator('.rig-spec-team-row').first().click();
  await page.locator('#rig-spectate-overlay .rig-spec-btn.primary').click();
  await page.waitForSelector('#rig-spec-controls', { state: 'visible' });
  await page.evaluate((t) => { (window as { __rigtune?: unknown }).__rigtune = t; }, tune);
  await page.locator('#rig-spec-controls .rig-sc-btn:has-text("4×")').click();
}

export async function runSkillEval(
  page: Page, label: string, tune: Record<string, number>, samples = 70,
): Promise<SkillResult> {
  await startWatch(page, tune);
  let last: Record<string, unknown> | null = null;
  let nSamp = 0, skinHits = 0;
  for (let i = 0; i < samples; i++) {
    await page.waitForTimeout(900);
    const s = await page.evaluate(() => (window as { __rigai?: unknown }).__rigai ?? null);
    if (s) {
      last = s as Record<string, unknown>;
      nSamp++;
      if (((s as { bellR?: number }).bellR ?? 0) > 31) skinHits++;
    }
  }
  const sk = (last?.skill ?? {}) as Record<string, number>;
  const scores = (last?.scores ?? {}) as Record<string, number>;
  const heldFrac = (last?.heldFrac as number) ?? 0;
  const ticks = sk.ticks || 1;
  const scorePts = Object.entries(scores)
    .reduce((a, [k, n]) => a + (SCORE_PTS[k] ?? 0) * n, 0);
  const nearSkinPct = nSamp ? (100 * skinHits) / nSamp : 0;
  const thrashPerK = (sk.thrash || 0) / ticks * 1000;

  const parts = {
    pass:       W.pass * (sk.passes || 0),
    gateClear:  W.gateClear * (sk.gateClears || 0),
    intercept:  W.intercept * (sk.intercepts || 0),
    score:      W.scorePt * scorePts,
    possession: W.possession * heldFrac,
    chaos:     -W.chaos * thrashPerK,
    skinRot:   -W.skinRot * nearSkinPct,
  };
  const composite = Object.values(parts).reduce((a, b) => a + b, 0);
  return {
    label,
    composite: +composite.toFixed(1),
    parts: Object.fromEntries(Object.entries(parts).map(([k, v]) => [k, +v.toFixed(1)])),
    raw: {
      passes: sk.passes || 0, intercepts: sk.intercepts || 0,
      gateClears: sk.gateClears || 0, thrashPerK: +thrashPerK.toFixed(1),
      scorePts, heldFrac, nearSkinPct: +nearSkinPct.toFixed(0), scores,
    },
  };
}

// ── Baseline: measure the regimes we have, so "skill" is now a number ────────
test('skill eval — MPC vs RRT vs RRT+space', async ({ page }) => {
  test.setTimeout(900000);
  const cfgs: { label: string; t: Record<string, number> }[] = [
    { label: 'MPC (default)', t: { planner: 0, wSpace: 0 } },
    { label: 'RRT',           t: { planner: 1, wSpace: 0 } },
    { label: 'RRT+space',     t: { planner: 1, wSpace: 4 } },
  ];
  const results: SkillResult[] = [];
  for (const c of cfgs) results.push(await runSkillEval(page, c.label, c.t));
  for (const r of results) {
    // eslint-disable-next-line no-console
    console.log('SKILL ' + JSON.stringify(r));
  }
  expect(results.length).toBe(cfgs.length);
});

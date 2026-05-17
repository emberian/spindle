/**
 * e2e/progression.spec.ts — AI-vs-AI match progression regression gate.
 *
 * Catches the class of bug where the bell flies off-field (bx ≈ 5160 on a
 * ±320 m field) and the match runs as a "hollow" shell with no real game
 * state advancing.  The test exercises the exact same flow as the manual
 * diagnostic: open the spectate picker, choose any home + away pair, start
 * the match at 4× speed, and observe `window.__rig` for ~60 s of wall time.
 *
 * ── Threshold rationale ────────────────────────────────────────────────────
 *
 *   GATE_X = 320         Physical half-length of the field in metres.
 *   BX_MAX = GATE_X + 80 = 400
 *     The bell must stay within a generous 80 m buffer around the field
 *     boundary.  Under the runaway bug bx reached ≈5160 — 16× outside this
 *     limit — so the assertion fails catastrophically while leaving a healthy
 *     game ≈75 m of breathing room beyond the goal face.
 *
 *   BX_ACTIVE_SPAN = 60
 *     The bell's axial position must sweep ≥60 m across the run.  A live
 *     match works the bell up and down the tube (hundreds of m of span); a
 *     hollow/frozen match leaves it ≈constant.  Render-cadence independent
 *     (replaces the old held-fraction proxy, which a competent fast-throwing
 *     AI on slow CI software-GL almost never sampled — bell is in flight,
 *     not held, most render frames; confounded local-vs-CI).
 *
 *   PROGRESSION: at least one of
 *     • possession changes ≥ 1      (teams trade the bell)
 *     • inning ≥ 2                  (a down was scored)
 *     • score (sh+sa) increased     (points were registered)
 *   Under the hollow bug all three were stuck at initial values for the full
 *   run.  A real match at 4× easily satisfies several of these within the
 *   first 10 s of wall time.
 *
 *   SAMPLE_COUNT = 40, SAMPLE_INTERVAL_MS = 1500
 *     40 samples × 1.5 s = 60 s wall time.  At 4× speed this covers 240 s
 *     of simulated time — well into inning 2+ for a healthy game.  The total
 *     test budget (including navigation + setup) fits comfortably under the
 *     120 s Playwright timeout configured in playwright.config.ts.
 */

import { test, expect } from '@playwright/test';

// ── Constants (guards against specific bugs) ───────────────────────────────

/** Half-length of the RIG field in metres (authoritative from RegConstants). */
const GATE_X = 320;

/**
 * Maximum allowed bell x-position (±).
 * The bell-flew-off bug put bx at ~5160; this threshold is 400 m — well
 * inside normal physics but generous enough not to be flaky on corner cases.
 */
const BX_MAX = GATE_X + 80; // 400 m

/**
 * Minimum axial span (m) the bell must sweep across the run. A live match
 * works the bell up and down the tube; a hollow/frozen match does not. This
 * is render-cadence independent (unlike sampling the brief `held` instant,
 * which a competent fast-throwing AI on slow CI software-GL almost never
 * lands on — that proxy was confounded and is replaced by this).
 */
const BX_ACTIVE_SPAN = 60;

/** Number of samples to collect while the match runs. */
const SAMPLE_COUNT = 40;

/** Wall-clock milliseconds between consecutive samples. */
const SAMPLE_INTERVAL_MS = 1500;

// ── Shape of the diagnostic hook ──────────────────────────────────────────

interface RigState {
  tick: number;
  bx: number;
  br: number;
  held: string | null;
  poss: 'home' | 'away';
  sh: number;
  sa: number;
  inning: number;
  phase: string;
  gate: unknown;
  throwsLeft: number;
  msg: string;
}

// ── Test ──────────────────────────────────────────────────────────────────

test('AI-vs-AI match progresses without runaway bell or hollow state', async ({ page }) => {
  // ── Collect page errors and console errors ─────────────────────────────

  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];

  page.on('pageerror', (err) => {
    pageErrors.push(err.message);
  });

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });

  // ── Navigate to the app ────────────────────────────────────────────────

  await page.goto('/spindle/', { waitUntil: 'networkidle' });

  // The landing screen should appear; find and click the Watch button.
  await page.locator('#rig-watch-btn').click();

  // ── Wait for the spectate team-picker overlay ──────────────────────────

  await page.waitForSelector('#rig-spectate-overlay', { state: 'visible' });

  // The overlay has two panes: home (first .rig-spec-pane) and away (last).
  // Click the first team row in the home pane.
  const homePanes = page.locator('.rig-spec-pane');
  await homePanes.first().locator('.rig-spec-team-row').first().click();

  // Click the first team row in the away pane (last pane).
  await homePanes.last().locator('.rig-spec-team-row').first().click();

  // ── Start the match ────────────────────────────────────────────────────

  // The primary "WATCH MATCH" button becomes enabled once both teams are chosen.
  await page.locator('#rig-spectate-overlay .rig-spec-btn.primary').click();

  // Wait for the in-match controls bar (confirms the match loop started).
  await page.waitForSelector('#rig-spec-controls', { state: 'visible' });

  // ── Set 4× speed ──────────────────────────────────────────────────────

  await page.locator('#rig-spec-controls .rig-sc-btn:has-text("4×")').click();

  // ── Sample window.__rig over time ─────────────────────────────────────

  const samples: RigState[] = [];

  for (let i = 0; i < SAMPLE_COUNT; i++) {
    // Wait between samples (first sample also lets the match warm up a tick).
    await page.waitForTimeout(SAMPLE_INTERVAL_MS);

    const snap = await page.evaluate(() => {
      return (window as unknown as { __rig?: unknown }).__rig ?? null;
    });

    if (snap !== null) {
      samples.push(snap as RigState);
    }
  }

  // ── Assertions ────────────────────────────────────────────────────────

  // 0. We must have collected enough samples to draw conclusions.
  expect(samples.length).toBeGreaterThanOrEqual(
    Math.floor(SAMPLE_COUNT * 0.75), // tolerate up to 25 % missed frames
    `Expected at least ${Math.floor(SAMPLE_COUNT * 0.75)} __rig samples but got ${samples.length}. ` +
    'Is window.__rig being set during the watch loop?',
  );

  // 1. Zero page errors / console errors.
  expect(pageErrors).toEqual(
    [],
    `Page errors detected during AI-vs-AI run:\n${pageErrors.join('\n')}`,
  );
  expect(consoleErrors).toEqual(
    [],
    `Console errors detected during AI-vs-AI run:\n${consoleErrors.join('\n')}`,
  );

  // 2. Bell stays within the field.
  //    BUG GUARD: runaway bell (bx ≈ 5160 under the old bug — 16× over BX_MAX).
  for (const s of samples) {
    expect(
      Math.abs(s.bx),
      `Bell escaped the field: bx=${s.bx.toFixed(1)} at tick=${s.tick} (limit ±${BX_MAX} m). ` +
      'This matches the runaway-bell bug where bx reached ~5160 on a ±320 m field.',
    ).toBeLessThanOrEqual(BX_MAX);
  }

  // 3. The bell is actively in play: its axial position must sweep a
  //    meaningful span across the run (render-cadence independent).
  //    BUG GUARD: hollow/frozen-match (bx ≈ constant). Runaway (bx huge)
  //    is already caught by #2.
  const bxs = samples.map((s) => s.bx);
  const bxMin = Math.min(...bxs);
  const bxMax = Math.max(...bxs);
  const bxSpan = bxMax - bxMin;
  expect(
    bxSpan,
    `Bell barely moved: x spanned only ${bxSpan.toFixed(1)} m across ${samples.length} ` +
    `samples (min ${bxMin.toFixed(1)}, max ${bxMax.toFixed(1)}; threshold ${BX_ACTIVE_SPAN} m). ` +
    'A live match works the bell up and down the tube; a hollow/frozen match does not.',
  ).toBeGreaterThanOrEqual(BX_ACTIVE_SPAN);

  // 4. Match progresses: at least one of possession changes / inning advance / score increase.
  //    BUG GUARD: hollow-match bug (all state stuck at initial values throughout).
  const firstPoss = samples[0]?.poss;
  const possessionChanged = samples.some((s) => s.poss !== firstPoss);

  const maxInning = Math.max(...samples.map((s) => s.inning));
  const inningAdvanced = maxInning >= 2;

  const firstScore = (samples[0]?.sh ?? 0) + (samples[0]?.sa ?? 0);
  const maxScore = Math.max(...samples.map((s) => s.sh + s.sa));
  const scoreIncreased = maxScore > firstScore;

  const progressed = possessionChanged || inningAdvanced || scoreIncreased;
  expect(
    progressed,
    `Match showed no progression over ${samples.length} samples spanning ~${
      Math.round((samples.length * SAMPLE_INTERVAL_MS) / 1000)
    } s of wall time:\n` +
    `  possessionChanged=${possessionChanged} (first=${firstPoss})\n` +
    `  inningAdvanced=${inningAdvanced} (maxInning=${maxInning})\n` +
    `  scoreIncreased=${scoreIncreased} (first=${firstScore}, max=${maxScore})\n` +
    'Under the hollow-match bug all three were permanently stuck at initial values.',
  ).toBe(true);
});

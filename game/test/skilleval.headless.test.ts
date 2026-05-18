// Headless skill eval runner. Env-gated so the CI vitest gate SKIPS it
// (instant no-op) — run on demand:
//   SKILLEVAL=1 npx vitest run test/skilleval.headless.test.ts
// It plays full AI-vs-AI matches in-process (no browser) → milliseconds.
import { describe, it, expect } from 'vitest';
import { evalHeadless } from '../src/eval/headless';

const ON = !!process.env.SKILLEVAL;

describe.skipIf(!ON)('headless skill eval — MPC vs RRT vs CEM', () => {
  it('ranks the algorithm classes (in-process)', { timeout: 600_000 }, () => {
    const t0 = Date.now();
    const out = [
      evalHeadless('MPC', { planner: 0 }),
      evalHeadless('RRT', { planner: 1 }),
      evalHeadless('CEM', { planner: 2 }),
    ];
    for (const r of out) console.log('SKILL ' + JSON.stringify(r)); // eslint-disable-line no-console
    console.log(`HEADLESS_MS ${Date.now() - t0}`); // eslint-disable-line no-console
    expect(out.length).toBe(3);
  });
});

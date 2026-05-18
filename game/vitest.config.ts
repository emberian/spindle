import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
    // The AI difficulty/planner property tests run millions of deterministic
    // sim ops (the off-axis lattice ~4×'d the planner candidate set). They
    // pass fast locally but the default 5 s vitest timeout is marginal on
    // slow CI runners → spurious timeouts. 30 s gives real headroom; a true
    // hang still fails well within CI's job budget.
    testTimeout: 30_000,
    hookTimeout: 30_000,
  },
});

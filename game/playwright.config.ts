import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config scoped to e2e/ regression tests only.
 *
 * webServer: assumes `dist/` already exists (CI runs `npm run build` first).
 * `npm run preview` (vite preview) serves at http://localhost:4173/spindle/.
 *
 * The single browser target is headed-less Chromium with SwiftShader so the
 * WebGL canvas renders in CI without a GPU.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,       // generous per-test budget; the match sampling takes ~60 s at 4×
  expect: { timeout: 15_000 },
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://localhost:4173',
    launchOptions: {
      // SwiftShader gives us a software WebGL renderer in headless CI.
      args: [
        '--use-gl=angle',
        '--use-angle=swiftshader',
        '--enable-unsafe-swiftshader',
        '--disable-web-security',    // avoids same-origin noise for WASM fetches
        '--no-sandbox',              // required in most CI containers
        '--disable-setuid-sandbox',
      ],
    },
    headless: true,
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium-swiftshader',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    // `npm run build` must have already produced dist/ before Playwright starts.
    command: 'npm run preview',
    url: 'http://localhost:4173/spindle/',
    reuseExistingServer: !process.env['CI'],
    timeout: 60_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});

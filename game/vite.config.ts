import { defineConfig } from 'vite';

// GitHub Pages serves this project under https://emberian.github.io/spindle/
// so base must be the absolute project subpath for hashed assets to resolve.
// The Rust core is built (wasm-pack) into rig-core/pkg before `vite build`;
// the wasm-pack "web" target loads its .wasm via `new URL(..., import.meta.url)`
// which Vite fingerprints as an asset (assetsInclude).
export default defineConfig({
  base: '/spindle/',
  build: { target: 'esnext', outDir: 'dist' },
  assetsInclude: ['**/*.wasm'],
  server: { fs: { allow: ['..'] } },
});

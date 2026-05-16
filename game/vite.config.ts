import { defineConfig } from 'vite';

// GitHub Pages serves this project under https://emberian.github.io/spindle/
// so the base path must be the absolute project subpath for hashed assets
// to resolve. public/.nojekyll is copied verbatim into dist/.
export default defineConfig({
  base: '/spindle/',
  build: {
    target: 'esnext',
    outDir: 'dist',
  },
});

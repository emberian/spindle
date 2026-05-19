# RIG explainer site

Static, self-contained explainer/marketing site for RIG. **No build step, no
dependencies, no framework** — plain HTML + one shared CSS file + inline SVG +
a tiny bit of CSS animation. Safe to open straight from disk.

## Where it deploys

Vite serves `game/public/` at base `/spindle/`, so these files land at:

    https://emberian.github.io/spindle/explainer/

Internal links are all **relative** (`sport.html`, `explainer.css`, …) and the
"Watch it live" CTAs point at `../` (the game at `/spindle/`).

## Pages

| File             | Purpose                                                      |
|------------------|--------------------------------------------------------------|
| `index.html`     | Hero / "What is RIG" — the hook + nav to subpages             |
| `sport.html`     | The Sport — canon: the calm, the bell, the Loop, the league  |
| `physics.html`   | The Physics — Coriolis, the inertial axis, determinism       |
| `swarm.html`     | The Swarm — Director + per-rigger controllers, coordination  |
| `learning.html`  | Watching It Learn — Gym, neural swarm, ES, game-as-visualizer |
| `tech.html`      | How It's Built — Rust → wasm, determinism gate, build style   |
| `vision.html`    | Why — the thesis, where it's heading, the outro CTA           |
| `explainer.css`  | Shared styles (palette, layout, motion, components)           |
| `og.svg`         | OpenGraph / Twitter-card cover image (1200×630)               |

## Design system

- Palette mirrors the game's `src/ui/palette.ts`: bg `#11131a`, cyan `#1aa6b7`,
  orange `#d4602a`, paper `#f4f1ea`, dim `#6b7079`.
- System font stack only — no web font fetch.
- Motion: CSS-only drifting starfield, a bell tracing a Coriolis arc, subtle
  hovers. All motion is disabled under `prefers-reduced-motion: reduce`.
- Diagrams are inline `<svg>` (Coriolis Loop, bell chime vs clatter, Director
  → roles, Gym loop, core→wasm→render architecture).
- Every page has OpenGraph + Twitter-card meta referencing `og.svg`.

## Extending

Add a new page by copying any subpage, swapping the `<section>` content, the
`<title>`/meta, and the `aria-current="page"` on the nav. The header/footer nav
is inlined per page (no JS) — if you add a page, add its link to the `nav.site`
block and both footer rows on every page. Keep it dependency-free and keep the
relative links relative.

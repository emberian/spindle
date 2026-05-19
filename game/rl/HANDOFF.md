# RIG RL — Handoff Notes

Notes for anyone picking this project up (human or AI).

## The repo

- **Canonical:** `/Users/ember/dev/spindle` (git root)
- **The game:** `game/` subdirectory (the monorepo houses design docs, league data, etc. at the top level)
- **Branch:** `dev` (deploy-on-push to GitHub Pages via CI)
- **CI:** `.github/workflows/ci.yml` runs: cargo test (release), wasm-pack build, vite build, Playwright progression gate, deploy
- **Commits:** `git -c commit.gpgsign=false` (1Password GPG signer fails non-interactively)

## Working conventions

- **Determinism is sacred.** The sim is bit-exact given a seed. No `thread_rng()`, no `HashMap` iteration order, no wall-clock in any sim/gym/rl path. Every new feature must preserve the named determinism guards (see ARCHITECTURE.md).
- **Parallel agent pattern:** Clone the repo to an isolated path (e.g. `/Users/ember/dev/spindle-exp-foo`), work there, commit (don't push), then the orchestrator merges + resolves conflicts + verifies gauntlet + pushes. This prevents file-corruption from concurrent edits.
- **No training in always-on tests.** Anything that calls `train_policy`/`train_mappo`/`run_self_play` is `#[ignore]`. Always-on coverage = cheap pure forward/determinism/shape checks. `cargo test` should stay under ~90s.
- **wasm cdylib must always build.** The `rl::policy` and `rl::attention` modules compile into wasm (pure f64, no rand/rayon). The `rl::train`, `rl::value`, `rl::self_play`, `rl::mappo` modules are native-only (gated `cfg(not(target_arch = "wasm32"))`).
- **The five JSON hops.** Any new field in `PlayerInput`/`InputFrame` must be plumbed through: ai_wasm.rs emit, wasm.rs parse, TS types, WasmSim serialization, InputManager literal, Replay pack/unpack. tsc catches missing TS fields. See `rl/` memory note on the wasm boundary.

## What's working well (don't break)

- Catches, grapple latency, player collision, contest detection, active defense — the sim is honest
- The legibility overlay — makes the swarm's coordination visible without debug mode
- The Python FFI bridge — the training-at-scale path (PyTorch/JAX driving the Rust gym)
- The live dashboard — the monitoring path for training runs
- The explainer site — the sharing/communication path
- The entity-attention actor forward pass deploys to wasm — trained weights go directly to the browser

## What needs work (in priority order)

1. **Analytical backprop** (or use PyTorch via the FFI bridge) — the ES gradient is the scaling bottleneck
2. **Escape the keep-away basin** — reweight Phi, curriculum, or simply more compute
3. **Checkpoint/resume** — long runs shouldn't lose progress to crashes
4. **The offline judge** is interesting but never meaningfully separated "learned something good" from "learned keep-away" — the judge signals are informative but the training reward doesn't respond to them (by design — they're offline only). Consider whether a curriculum staged on the judge would be appropriate.

## Gotchas

- `cargo test` takes ~80s because skill_eval runs real 8000-tick matches (rayon-parallel, but still real sim time). Don't try to speed this up by shortening episodes — those tests are the behavioral-fidelity gate.
- The `committed_artifact_reproduces_and_beats_baseline` test is `#[ignore]` and takes ~200s. Run it explicitly after changing sim dynamics (the artifact was trained against the current sim — if physics change, regenerate: `cargo test regenerate_artifact -- --ignored`).
- `npx tsc --noEmit` after a fresh clone fails until `npm run build:wasm` generates `rig-core/pkg/`. Always build wasm first, or use `npm run build` (which does both).
- The game's three `main.ts` loops (runMatch/runWatch/runReplay) are near-duplicates with subtle divergences. Changes to one often need mirroring in the others.
- `EfeParams::default()` and `RewardConfig::default()` are byte-verbatim production contracts. Changing them changes every existing test baseline and the live game behavior. Only change intentionally.

## The big picture

RIG is a fictional zero-g sport that doubles as a deterministic multi-agent coordination lab.
The web game is the visualizer for learned swarms. The sport has real canon, a league, culture.
The learning stack is real MARL (entity-attention CTDE, self-play, centralized credit assignment).
The gap between "prototype" and "massive arena" is the gradient computation method, not the architecture.
Everything else (gym, policy, population, dashboard, browser deploy) is in place and proven.

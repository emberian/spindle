# SPINDLE · RIG

> *The game played in the calm at the heart of a spinning world.*
>
> **▶ Play: https://emberian.github.io/spindle/** — classic 2D: `/spindle/classic/`

A zero-gravity sport, derived (not decorated) from the physics of an
O'Neill cylinder, canonized into the *Vivere Astra* science-fiction
universe, and shipped as both a typeset sourcebook and a deterministic,
playable simulation.

---

## For the mathematician

Rig is a sport whose entire rulebook falls out of one differential
equation. Play happens in **the calm** — the weightless volume on the
spin axis of a rotating habitat. Work in the co-rotating frame; let the
cross-section be the complex plane `ζ = y + iz` (the spin axis `x` is
ignorable — motion along it is inertial). A free body obeys

$$\ddot\zeta \;=\; \omega^{2}\,\zeta \;-\; 2 i\,\omega\,\dot\zeta$$

centrifugal + Coriolis, nothing else. The characteristic polynomial
`r² + 2iω r − ω² = 0` has the **double root `r = −iω`**, so every
trajectory is a degenerate two-term solution

$$\zeta(t) \;=\; \bigl(\zeta_0 + (\dot\zeta_0 + i\omega\,\zeta_0)\,t\bigr)\,e^{-i\omega t}.$$

These are **roulettes** — involutes of a circle when "dropped",
Archimedean spirals when the path crosses the axis, hybrids otherwise.
The game's marquee score, the **Loop**, is exactly the closed-orbit /
self-returning member of this family: a boundary-value problem
`ζ(0) ≈ ζ(τ)` with a winding constraint `∮ dθ ≳ 2π`, subject to a
hard obstacle (the skin, `|ζ| < R`) and a transversality condition
(threading an 8-metre ring 320 m downfield). It is provably *rare* —
which is why, in the fiction, most players never score one.

Two pieces a mathematician may enjoy:

- **Tuning as constrained optimization.** The free parameter `ω` (the
  habitat's reference spin) is *chosen*, not guessed: it is a feasible
  point of three coupled inequalities — field traversability, a
  "dramatic-but-controllable" winding band, and a livable gravity
  gradient `g(R)=ω²R`. `game/rig-core/src/tuning.rs` ships the solver;
  CI fails if the shipped constants leave the feasible region. The
  earlier hand-tuning (ω: 0.22 → 0.06 → 0.15 → 0.32) is exactly the
  flailing this removes.
- **A dual-use solver.** `game/rig-core/src/analysis.rs` does regime
  characterization and a deterministic search of throw space for
  loop-feasible initial conditions. It found a scoring Loop a human
  search missed (near-axial, `v_⊥ ≈ 0.4 m/s`, cancelling the
  centrifugal sweep) — *the same routine is the AI's loop weapon.*

The simulation is a pure function `(state, inputs) ↦ (state, events)`
at a fixed `1/240 s` step, RK4, all randomness via named, counter-seeded
`sfc32` substreams — so a match is reproducible from `(seed, inputs)`
alone (the in-fiction "re-call"). The Rust core and a TypeScript oracle
are held to **bit-identical** RNG output by a known-answer test.

## For the science-fiction author

Rig is canon in *Vivere Astra* (a 25th-century, post-collapse,
interstellar campaign — GM: Kanzokax). The pitch, in one line: **rig is
what a scattered, frightened, faster-than-light humanity has instead of
a shared sky.**

It is the bastard child of the two American sports religions — football's
territory and set-pieces, baseball's clockless, stat-haunted,
one-on-one soul — reinvented for vacuum by the construction crews who
*built* the cylinders. There is **no clock, ever**. The ball is a
**bell**: it rings when it spins true and clatters when it tumbles, so
the sport is read by ear — which is *why* its economy is audio-first,
its broadcasts are jump-delayed "re-calls", and its in-play betting is
denominated in the army-less Concordat's Lira. Mobility is grapple-only
(you cannot fly; you push off structure or throw a line and haul).
Whether a habitat's culture prefers the honest cheap **Fall** or the
vain, against-the-spin **Rise** is a genuine read on that society.

The worldbuilding lives in **`pdf/rig.pdf`** — a 17-page typeset
dossier: the sport, its founding → codification → interstellar arc, its
economics, a human-readable ruleset with deliberately blunt jargon
("cross", "clatter", "snatch", "skinned", "spine"), and a 32-franchise
major league across the canon's settled systems with a 16-seed playoff.
`COMPARISON.md` documents the eerie convergence with an independent AI
pass at the same brief — evidence the design space has a true shape.

---

## What's in this repo

| Path | What |
|---|---|
| `pdf/`, `design/`, `league/`, `research/` | the dossier, the canon, the 32 teams, the source sweep |
| `web/` | **v0.1** — the shipped 2D arcade (vanilla JS), live at `/spindle/classic/` |
| `game/` | **v2** — TypeScript + Three.js + a Rust/WASM physics core (in progress) |
| `game/rig-core/` | the deterministic Rust physics crate (`cargo test`) |
| `COMPARISON.md` | RIG vs. the parallel AI pass; the convergence finding |

**v2 status:** deep-sim physics (6-DOF bell, momentum-correct rig-line,
the calm rendered in 3D), deterministic `SimWorld` + replay, the
retuned Loop/Curl scoring, the Rust port + the dynamical-systems
analysis/solver, the match-rules layer, and the 32-team dataset are all
implemented and tested (Rust + TS suites green). Remaining: the WASM
facade + binding, the playable match, AI, the league/Jump UX, and the
polish pass.

## Build & test

```sh
cd game
npm install
npm test                       # typecheck + vitest (TS oracle + match + league)
( cd rig-core && cargo test )  # the Rust physics core, incl. the Number gate
npm run dev                    # play it locally
```

Static deploy: `vite build` → GitHub Pages (Actions); v0.1 staged at
`/classic`. The sim never touches the DOM; render/audio/UI are
read-only consumers of immutable snapshots — which is what makes the
determinism, the replays, and the headless test harness possible.

---

*No clock. The chime is still ringing.*

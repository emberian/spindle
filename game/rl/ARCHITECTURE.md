# RIG RL — Architecture

## System diagram

```
                    ┌─────────────────────────────────────────────┐
                    │         MAPPO Training Loop (Rust)           │
                    │  ┌─────────┐  ┌───────────┐  ┌──────────┐  │
                    │  │ Rollout │  │ PPO Obj + │  │ Self-Play│  │
                    │  │ Collect │  │ ES Gradient│  │Population│  │
                    │  └────┬────┘  └─────┬─────┘  └────┬─────┘  │
                    │       │             │              │         │
                    └───────┼─────────────┼──────────────┼─────────┘
                            │             │              │
              ┌─────────────▼─────────────▼──────────────▼────────────┐
              │              SelfPlayEnv (both teams controlled)        │
              │    ┌──────────────┐              ┌──────────────┐      │
              │    │  Home (4)    │              │  Away (4)    │      │
              │    │  current     │              │  sampled     │      │
              │    │  policy      │              │  from pop    │      │
              │    └──────┬───────┘              └──────┬───────┘      │
              │           │                            │               │
              │           ▼                            ▼               │
              │    ┌─────────────────────────────────────────┐        │
              │    │        Deterministic Gym (RigEnv)        │        │
              │    │   SimWorld + MatchSM + AiSystem triad    │        │
              │    │   Bit-exact. Seeded. 240 Hz physics.     │        │
              │    └─────────────────────────────────────────┘        │
              └────────────────────────────────────────────────────────┘
                            │
                            │ writes training_output/
                            ▼
              ┌──────────────────────────────────────┐
              │    Live Dashboard (localhost:8384)     │
              │  Training curves + match grid + SVG   │
              └──────────────────────────────────────┘

              ┌──────────────────────────────────────┐
              │    Browser Viz (GitHub Pages)          │
              │  Spectate: Baseline vs Learned(RL)    │
              │  Legibility overlay (roles/intent)    │
              │  Same wasm sim + Three.js renderer    │
              └──────────────────────────────────────┘
```

## Component boundaries

### Sim core (`rig-core/src/sim_world.rs`, `bell.rs`, `player.rs`, `grapple.rs`, `collision.rs`)
- Pure deterministic physics: rotating-frame Coriolis + centrifugal, RK4 bell, spring-damper grapple
- Player-player soft collision (momentum-conserving), contest detection, garrote foul
- 240 Hz fixed timestep, symplectic Euler for springs, no wall-clock/rng in physics

### AI baseline (`rig-core/src/ai/`)
- Director @2Hz assigns roles (primary/shadow/outlet/mark/support)
- Per-rigger EFE + w-maximization controller
- Active defense: lane-interception, strip contests
- The "opponent" the learner trains against (frozen in self-play snapshots)

### Gym (`rig-core/src/gym.rs`)
- `Env` trait over SimWorld + MatchSM + AiSystem
- `Observation` / `Action` (= `ai::PlayerInput`) / `Reward` / `Step`
- `snapshot()/restore()` deep-clones the FULL triad (bit-exact restore)
- Agent-control selector: `controlled_ids` bypass AiSystem; the rest get baseline

### RL policy layer (`rig-core/src/rl/`)
- `policy.rs` — MLP policy (legacy, 11,350 params, FEAT_W=89, OUT_W=22)
- `attention.rs` — entity-attention actor (24,598 params, wasm-safe)
- `value.rs` — centralized value (34,433 params, native-only)
- `self_play.rs` — SelfPlayEnv + Population
- `mappo.rs` — the training loop (ES-shaped PPO)
- `train.rs` — the older ES-only trainer (kept for comparison/legacy)

### FFI bridge (`game/rig-gym-py/`)
- PyO3 cdylib wrapping RigEnv for Python/JAX training if desired
- GIL-released step/reset; VecRigGym(N) for parallel rollout

### Browser deployment
- `policy_wasm.rs` — `RigPolicy` wasm-bindgen class, auto-detects MLP vs attention
- Artifact: `src/rl/policy-v1.json` (bundled by vite, loaded on spectate toggle)
- Legibility overlay: per-rigger intent/role/who-controls surfaced via `ai_debug_json`

## Data flow during training

1. `train_mappo` creates N `SelfPlayEnv`s with seeded rng
2. Each gen: sample opponent from `Population`
3. Rollout: policy forward → actions for Home (4 riggers); opponent policy → Away (4)
4. Collect (obs, action, log_prob, value, reward, done) per agent per timestep
5. Compute GAE advantages from centralized value
6. ES-shaped PPO: evaluate PPO clipped-surrogate + entropy as shaped fitness
7. Update actor weights (rank-normalized ES gradient); update critic (MSE minimization)
8. Every N gens: checkpoint best into Population
9. Write metrics + match replays to `training_output/`

## Key type contracts

```rust
// What the policy sees (per agent, egocentric):
pub struct Observation { tick, bell_p, bell_v, bell_held_by, possessed,
    possession, gate, score_home, score_away, phase, players: Vec<ObsPlayer>,
    controlled_ids }

// What the policy emits (per agent):
pub type Action = ai::PlayerInput; // aim, fire_line_at, reel, release,
    // pushoff, throw_charge, throw_released, throw_spin, thrumbler, catch_intent

// What comes back:
pub struct Step { obs, reward: Reward, done, truncated, info: StepInfo }
pub struct Reward { possession_held, gate_advanced, scored, contest_won,
    bell_out, terminal, total }
```

## Determinism contract

- ALL rng is seeded ChaCha8; no wall-clock, no HashMap-iteration folds
- Parallel rollout: rayon fixed-index collect (scheduling cannot perturb results)
- Entity ordering: stable sort by (distance.to_bits(), id) — no permutation variance
- `hash_snapshot` folds tick + bell p/v/w + player p/v — covers all physics-relevant state
- Named guards (must pass run==run, bit-identical): `determinism_via_rigsim`, `parallel_eval_is_bit_identical_and_stable`, `hash_snapshot_is_stable`, `identical_inputs_produce_identical_hashes`, `ga_is_deterministic`, `learner_is_deterministic_same_seed`, gym determinism tests, rl forward determinism tests

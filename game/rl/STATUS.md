# RIG RL — Current Status

**Last updated:** 2026-05-19 (one-day sprint session)  
**Branch:** `dev`  
**Live game:** https://emberian.github.io/spindle/  
**Explainer:** https://emberian.github.io/spindle/explainer/

## What exists and works

The full MARL training stack is built end-to-end in native Rust, with a
browser visualizer, a live training dashboard, and a Python FFI bridge.
Everything is deterministic (bit-exact given a seed), CI-gated, and
deployed.

### The deterministic gym (`rig-core/src/gym.rs`)
- `Env` trait: `reset(seed, Scenario) → Observation`, `step(actions) → Step`, `snapshot()/restore()`
- Agent-control selector: any subset of the 8 riggers can be externally driven; the rest run the baseline AI
- Reward is the game's own intrinsic outcome (possession/score/contest/gate/terminal) — NOT a composite ranker
- Policy-invariant potential-based shaping (gate-progress + pass-completion) available via `RewardConfig::shaping_gamma`
- Full-control mode: `SelfPlayEnv` wraps RigEnv with all 8 in controlled_ids (both teams policy-driven)

### Entity-attention actor (`rig-core/src/rl/attention.rs`)
- 8 entities (self, 3 teammates, 3 opponents, bell), each 32-dim raw features
- Entity encoder: Linear(32→48) + tanh, shared
- 2-layer multi-head self-attention (4 heads, d_model=48)
- Action head: Linear(48→22) — aim, reel, fire, pass-to-k, throw, thrumbler, catch
- **ATTN_PARAM_W = 24,598**
- **WASM-SAFE** (pure f64 matmuls + stable softmax — deploys to browser)
- Entity ordering deterministic: self, teammates sorted by (distance_bits, id), opponents same, bell last

### Centralized value function (`rig-core/src/rl/value.rs`)
- Sees ALL 8 players' absolute states + bell + match context (137 features)
- 2-layer MLP (128 hidden, tanh) → scalar V(s)
- **VALUE_PARAM_W = 34,433**
- Native-only (training aid, not deployed)
- `compute_gae(rewards, values, dones, gamma, lambda)` implemented and tested

### Self-play population (`rig-core/src/rl/self_play.rs`)
- `Population`: capped at 50 entries, oldest non-best pruned
- Prioritized fictitious self-play sampling: 50% best win-rate, 30% uniform random past, 20% most recent
- Deterministic sampling (seeded ChaCha8)
- `SelfPlayEnv`: full-control, reward from both perspectives

### MAPPO trainer (`rig-core/src/rl/mappo.rs` + `src/bin/train_mappo.rs`)
- ES-shaped PPO: uses the PPO clipped surrogate + GAE + entropy as shaped fitness for ES
- Vectorized rollout: n_envs parallel SelfPlayEnvs (rayon)
- `act_with_logprob` for mixed discrete/continuous action space
- Self-play integration: checkpoint every N gens into population
- Binary: `cargo run --release --bin train_mappo -- --envs 16 --gens 100`

### Python FFI bridge (`game/rig-gym-py/`)
- PyO3/maturin native extension
- `RigGym` class: reset/step/snapshot/restore (GIL-released)
- `VecRigGym(N)`: parallel envs via ThreadPoolExecutor
- ~8250 transitions/sec @ 32 envs

### Live training dashboard (`game/dashboard/`)
- File-based protocol: trainer writes `training_output/metrics.jsonl` + `training_output/matches/`
- Vanilla JS dashboard: SVG training curves + 2D match grid (click-to-expand with scrub)
- `python3 dashboard/serve.py` → http://localhost:8384/

### Browser visualization
- Spectate toggle: Baseline AI vs Learned (RL) per team
- `RigPolicy` wasm binding auto-detects MLP vs attention artifact
- Legibility overlay: role/job labels, intent lines, who's-controlled rings, decision pulses
- Default ON in watch/spectate; `L` cycles detail (full → labels → off)

## What we proved works

- The gym is bit-exact (determinism tests, run==run)
- The attention policy produces valid actions and drives the gym without panics
- The centralized value produces finite scalars; GAE is mathematically correct (unit-tested)
- Self-play both-sides-controlled is deterministic
- The MAPPO trainer runs and shows entropy decline + positive rewards before population difficulty ramps
- The Python bridge preserves bit-exact determinism across the FFI boundary
- The old ES learner (with richer obs + shaping) finds "elegant keep-away" (spatial_control=1.0, pass_chain=1499) but NOT gate-clearing scoring play

## Training run to date

Short proof-of-concept (8 gens, 16 envs, 256 horizon):
```
Gen   Reward   Entropy  WinRate  Pop
0     0.000    13.646    0.000    1
2     0.640    12.948    0.667    1
4     0.030    10.899    0.800    2
7     0.000    10.296    0.500    3
```
Entropy declining = policy becoming more deterministic. Population difficulty increasing = win-rate regressing (correct behavior — self-play prevents degenerate convergence).

No large-scale run has been done yet. The infrastructure is proven; the compute hasn't been spent.

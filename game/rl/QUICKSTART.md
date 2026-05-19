# RIG RL — Quickstart

## Prerequisites

- Rust nightly (for the `rig-core` workspace)
- `wasm-pack` (for browser-deployable policy builds)
- Node 18+ (for the game frontend / vitest / vite)
- Python 3.10+ with `maturin` (for the FFI bridge, optional)

## Run the MAPPO trainer

```bash
cd game
cargo run --release --bin train_mappo --manifest-path rig-core/Cargo.toml -- \
  --envs 16 --horizon 256 --gens 100 --seed 42
```

This trains an entity-attention policy via ES-shaped PPO with self-play.
Output: `policy-mappo.json` (the trained weights artifact).

Key flags:
- `--envs N` — parallel environments (default 16; more = smoother gradient)
- `--horizon N` — steps per rollout (default 256)
- `--gens N` — training generations (default 100)
- `--es-pop N` — ES population size (default 20; more = better gradient estimate)
- `--seed N` — master seed (deterministic: same seed = identical training)

## Watch training live (dashboard)

```bash
# Terminal 1: train (writes training_output/)
cargo run --release --bin train_mappo --manifest-path rig-core/Cargo.toml

# Terminal 2: serve dashboard
cd game && python3 dashboard/serve.py
# Open http://localhost:8384/
```

The dashboard shows:
- Training curves (reward, win-rate, entropy, value loss) — auto-updating
- Match grid with 2D SVG mini-replays of completed evaluation matches
- Live generation indicator + throughput

## Watch the trained swarm in the browser

```bash
cd game && npm run build   # builds wasm + frontend
npx vite preview           # serves at localhost:4173/spindle/
```

In-browser: click Spectate → pick a team → toggle `Learned (RL)`.
The learned policy drives that team; the other is baseline AI.
Press `L` to cycle the legibility overlay (roles/intent/who's-controlled).

## Deploy a new trained policy to the browser

1. Train: `cargo run --release --bin train_mappo ... > policy-mappo.json`
2. Copy: `cp policy-mappo.json game/src/rl/policy-v1.json`
3. Build: `cd game && npm run build`
4. Push: the wasm `RigPolicy` auto-detects MLP vs attention from the artifact JSON

The artifact format:
```json
{"type":"attention","seed":42,"config":"...","dims":{"feat":89,"out":22,"entities":8,"d_model":48,"param":24598},"baseline":9.0,"fitness":9.13,"weights":[...24598 f64s...]}
```

## Run tests

```bash
cd game
cargo test --manifest-path rig-core/Cargo.toml --lib   # ~60-80s, 254 pass
npx tsc --noEmit                                        # type-check frontend
npx vitest run                                          # 139 TS tests
npm run test:progression                                # AI-vs-AI e2e smoke
```

On-demand (slow training tests, NOT in the always-on suite):
```bash
cargo test --manifest-path rig-core/Cargo.toml --lib learn_real -- --ignored --nocapture
cargo test --manifest-path rig-core/Cargo.toml --lib self_play_smoke -- --ignored --nocapture
cargo test --manifest-path rig-core/Cargo.toml --lib committed_artifact_reproduces -- --ignored
```

## Use the Python FFI bridge

```bash
cd game/rig-gym-py
pip install maturin
maturin develop --release
python test_gym.py  # smoke tests

# In Python:
from rig_gym_py import RigGym
from vec_env import VecRigGym

env = RigGym()
obs = env.reset(seed=42, scenario={
    "home_style": "fall-dynasty", "home_cyl": "big-slow",
    "away_style": "rise-power", "away_cyl": "small-fast",
    "controlled_ids": ["H1","H2","H3","H4"],
    "max_ticks": 8000,
})
# obs is a dict with bell_p, players, etc.

step = env.step([
    {"id":"H1","aim":[1,0,0],"fire_line_at":None,"reel":0,
     "release":False,"pushoff":False,"throw_charge":0,
     "throw_released":False,"throw_spin":0,"thrumbler":[0,0,0],
     "catch_intent":False},
    # ... H2, H3, H4
])
# step["reward"]["total"], step["done"], step["obs"]

# Vectorized (for PyTorch/JAX training):
vec = VecRigGym(n_envs=32, base_seed=0)
obs_batch = vec.reset_all(scenario)
step_batch = vec.step_all(actions_batch)
```

## Key files

| What | Where |
|------|-------|
| Gym | `rig-core/src/gym.rs` |
| Attention actor | `rig-core/src/rl/attention.rs` |
| Centralized value | `rig-core/src/rl/value.rs` |
| Self-play | `rig-core/src/rl/self_play.rs` |
| MAPPO trainer | `rig-core/src/rl/mappo.rs` |
| Train binary | `rig-core/src/bin/train_mappo.rs` |
| Python bridge | `rig-gym-py/src/lib.rs` |
| Dashboard | `dashboard/index.html` + `dashboard.js` |
| Browser policy | `src/rl/policy-v1.json` (artifact) |
| Wasm binding | `rig-core/src/policy_wasm.rs` |
| Explainer site | `public/explainer/` |

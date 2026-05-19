# rig-gym-py

PyO3 native extension exposing the RIG deterministic gym to Python for MAPPO training.

## Install

```bash
cd game/rig-gym-py
pip install maturin
maturin develop --release
```

## Quick start

```python
import rig_gym_py
from vec_env import VecRigGym

# Single environment
gym = rig_gym_py.RigGym()
obs = gym.reset(seed=42, scenario={
    "home_style": "fall-dynasty",
    "home_cyl": "big-slow",
    "away_style": "rise-power",
    "away_cyl": "small-fast",
    "controlled_ids": ["H1", "H2"],
    "max_ticks": 2000,
    "reward_config": {
        "w_possession": 0.01,
        "w_gate": 0.1,
        "w_score": 1.0,
        "w_contest": 0.2,
        "w_bell_out": 0.05,
        "w_terminal": 1.0,
        "shaping_gamma": 0.997,  # or None to disable
    },
})

# Step with actions for each controlled rigger
step = gym.step([
    {"id": "H1", "aim": [1,0,0], "pushoff": True, ...},
    {"id": "H2", "aim": [0,1,0], "reel": -1, ...},
])
print(step["reward"]["total"], step["done"])

# Vectorized (32 parallel envs for MAPPO)
vec = VecRigGym(n_envs=32, base_seed=0)
obs_batch = vec.reset_all(scenario)
steps = vec.step_all(actions_batch)  # list of 32 action-lists
```

## API

### `RigGym` class

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `(seed: int, scenario: dict) -> obs_dict` | Reset env, return initial observation |
| `step` | `(actions: list[dict]) -> step_dict` | Advance one tick |
| `snapshot` | `() -> int` | Save state, return opaque handle |
| `restore` | `(handle: int) -> None` | Restore to a saved state |
| `drop_snapshot` | `(handle: int) -> None` | Free a snapshot's memory |
| `observation_space` | `() -> dict` | Schema of the observation dict |
| `action_space` | `() -> dict` | Schema of an action dict |

### Observation dict

```python
{
    "tick": int,
    "bell_p": [float, float, float],
    "bell_v": [float, float, float],
    "bell_held_by": str | None,
    "possessed": bool,
    "possession": int,        # 0=Home, 1=Away
    "gate": int,              # 0=First, 1=Deep, 2=Mouth
    "score_home": int,
    "score_away": int,
    "phase": int,             # 0..6
    "attack_sign": float,     # +1 or -1
    "gate_plane_x": float,
    "players": [
        {
            "id": str,
            "team": int,      # 0=Home, 1=Away
            "role": int,      # 0=Anchor..4=Reach
            "p": [float, float, float],
            "v": [float, float, float],
            "line_anchor": [float, float, float] | None,
            "line_rest_len": float | None,
            "assignment": {"job": int, "mark_id": str | None} | None,
        },
        ...  # 8 players
    ],
    "controlled_ids": [str, ...],
}
```

### Action dict

```python
{
    "id": str,                         # rigger id (e.g. "H1")
    "aim": [float, float, float],      # unit direction
    "fire_line_at": [float, float, float] | None,
    "reel": int,                       # -1 (in), 0, or 1 (out)
    "release": bool,
    "pushoff": bool,
    "throw_charge": float,             # [0, 1]
    "throw_released": bool,
    "throw_spin": float,               # [-1, 1]
    "thrumbler": [float, float, float],# [-1,1] per axis
    "catch_intent": bool,
}
```

### Step dict

```python
{
    "obs": obs_dict,
    "reward": {
        "possession_held": float,
        "gate_advanced": float,
        "scored": float,
        "contest_won": float,
        "bell_out": float,
        "terminal": float,
        "total": float,    # the scalar the learner optimizes
    },
    "done": bool,
    "truncated": bool,
    "info": {
        "step_score_home": int,
        "step_score_away": int,
        "turnover": bool,
        "unmatched_actions": int,
    },
}
```

### Scenario dict

```python
{
    "home_style": str,
    "home_cyl": str,
    "away_style": str,
    "away_cyl": str,
    "controlled_ids": [str, ...],
    "max_ticks": int,
    "reward_config": {               # optional; defaults apply if absent
        "w_possession": float,       # default 0.01
        "w_gate": float,             # default 0.1
        "w_score": float,            # default 1.0
        "w_contest": float,          # default 0.2
        "w_bell_out": float,         # default 0.05
        "w_terminal": float,         # default 1.0
        "reward_side": "home"|"away" | None,  # inferred if absent
        "shaping_gamma": float | None,        # None = shaping OFF
    },
}
```

### `VecRigGym` class (Python-side vectorized wrapper)

```python
from vec_env import VecRigGym

vec = VecRigGym(n_envs=32, base_seed=0)
obs_batch = vec.reset_all(scenario)           # list of 32 obs dicts
steps = vec.step_all(actions_batch)           # list of 32 step dicts
steps = vec.step_uniform(actions)             # same actions for all
handles = vec.snapshot_all()
vec.restore_all(handles)
vec.close()
```

The vectorized wrapper uses `ThreadPoolExecutor`. Since the Rust gym
releases the GIL during `reset`/`step`, all 32 envs truly run in
parallel on separate OS threads.

## Determinism

Same `seed + scenario + actions` through the Python bridge produces
bit-identical results as calling the Rust gym directly. The bridge is a
pure pass-through (dict parsing + type conversion), never transforms
game state.

## Tests

```bash
cd game/rig-gym-py
maturin develop --release
python test_gym.py
```

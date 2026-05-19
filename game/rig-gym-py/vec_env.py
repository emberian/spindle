"""Vectorized RigGym wrapper for MAPPO training.

Holds `n_envs` independent RigGym instances. Since the Rust gym releases
the GIL during `reset`/`step`, we drive them from a ThreadPoolExecutor
for true parallelism (no multiprocessing serialization overhead — each
env is thread-safe once the GIL is released in native code).

Usage:
    from vec_env import VecRigGym

    vec = VecRigGym(n_envs=32, base_seed=0)
    scenario = { ... }
    obs_batch = vec.reset_all(scenario)
    # obs_batch: list of n_envs obs dicts

    # actions_batch: list of n_envs action-lists
    steps_batch = vec.step_all(actions_batch)
    # steps_batch: list of n_envs step dicts
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import rig_gym_py


class VecRigGym:
    """A vectorized collection of independent RigGym environments.

    Each env gets a deterministic seed derived from `base_seed + env_index`.
    The ThreadPoolExecutor drives all envs in parallel (GIL released in
    native Rust during reset/step).
    """

    def __init__(self, n_envs: int, base_seed: int = 0, max_workers: int | None = None):
        """
        Args:
            n_envs: Number of parallel environments.
            base_seed: Base seed; env i gets seed = base_seed + i on reset.
            max_workers: Thread pool size. Defaults to n_envs (each env
                         gets its own thread; since the GIL is released,
                         they truly run in parallel).
        """
        self.n_envs = n_envs
        self.base_seed = base_seed
        self.envs = [rig_gym_py.RigGym() for _ in range(n_envs)]
        self._pool = ThreadPoolExecutor(max_workers=max_workers or n_envs)

    def reset_all(self, scenario: dict) -> list[dict]:
        """Reset all envs with the given scenario. Env i gets seed = base_seed + i.

        Returns a list of n_envs observation dicts.
        """
        def reset_one(i):
            return self.envs[i].reset(self.base_seed + i, scenario)

        futures = [self._pool.submit(reset_one, i) for i in range(self.n_envs)]
        return [f.result() for f in futures]

    def reset_idx(self, idx: int, scenario: dict, seed: int | None = None) -> dict:
        """Reset a single env by index. Returns its observation dict."""
        s = seed if seed is not None else (self.base_seed + idx)
        return self.envs[idx].reset(s, scenario)

    def step_all(self, actions_batch: list[list[dict]]) -> list[dict]:
        """Step all envs in parallel.

        Args:
            actions_batch: A list of n_envs action-lists. Each action-list
                           is a list of action dicts for that env's
                           controlled riggers.

        Returns:
            A list of n_envs step dicts (each with obs, reward, done,
            truncated, info).
        """
        assert len(actions_batch) == self.n_envs, (
            f"expected {self.n_envs} action lists, got {len(actions_batch)}"
        )

        def step_one(i):
            return self.envs[i].step(actions_batch[i])

        futures = [self._pool.submit(step_one, i) for i in range(self.n_envs)]
        return [f.result() for f in futures]

    def step_uniform(self, actions: list[dict]) -> list[dict]:
        """Step all envs with the SAME action list (useful for baseline-only
        or uniform-policy testing).

        Returns a list of n_envs step dicts.
        """
        def step_one(i):
            return self.envs[i].step(actions)

        futures = [self._pool.submit(step_one, i) for i in range(self.n_envs)]
        return [f.result() for f in futures]

    def snapshot_all(self) -> list[int]:
        """Take a snapshot of every env. Returns a list of handles."""
        return [env.snapshot() for env in self.envs]

    def restore_all(self, handles: list[int]) -> None:
        """Restore all envs from previously taken snapshots."""
        assert len(handles) == self.n_envs
        for i, h in enumerate(handles):
            self.envs[i].restore(h)

    def close(self):
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=False)

    def __del__(self):
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass

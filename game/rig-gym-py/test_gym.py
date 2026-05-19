"""Smoke tests for the rig_gym_py native extension.

Verifies: import, reset, step, obs/reward shapes, determinism, and
snapshot/restore round-trip.
"""

import rig_gym_py


def make_scenario(controlled=None, max_ticks=200):
    return {
        "home_style": "fall-dynasty",
        "home_cyl": "big-slow",
        "away_style": "rise-power",
        "away_cyl": "small-fast",
        "controlled_ids": controlled or [],
        "max_ticks": max_ticks,
    }


def idle_action(player_id):
    return {
        "id": player_id,
        "aim": [0.0, 0.0, 0.0],
        "fire_line_at": None,
        "reel": 0,
        "release": False,
        "pushoff": False,
        "throw_charge": 0.0,
        "throw_released": False,
        "throw_spin": 0.0,
        "thrumbler": [0.0, 0.0, 0.0],
        "catch_intent": False,
    }


def test_reset_and_obs_shape():
    gym = rig_gym_py.RigGym()
    obs = gym.reset(42, make_scenario())

    assert isinstance(obs, dict)
    assert "tick" in obs
    assert obs["tick"] == 0
    assert len(obs["bell_p"]) == 3
    assert len(obs["bell_v"]) == 3
    assert len(obs["players"]) == 8
    assert isinstance(obs["controlled_ids"], list)
    assert obs["possession"] in (0, 1)
    assert obs["gate"] in (0, 1, 2)
    assert obs["phase"] in range(7)

    # Player dict shape
    p = obs["players"][0]
    assert "id" in p
    assert "team" in p
    assert "role" in p
    assert len(p["p"]) == 3
    assert len(p["v"]) == 3
    print("  reset + obs shape: OK")


def test_step_and_reward_shape():
    gym = rig_gym_py.RigGym()
    gym.reset(123, make_scenario())

    step = gym.step([])
    assert isinstance(step, dict)
    assert "obs" in step
    assert "reward" in step
    assert "done" in step
    assert "truncated" in step
    assert "info" in step

    r = step["reward"]
    for key in ("possession_held", "gate_advanced", "scored",
                "contest_won", "bell_out", "terminal", "total"):
        assert key in r, f"missing reward key: {key}"
        assert isinstance(r[key], float), f"reward[{key}] not float"
        assert r[key] == r[key], f"reward[{key}] is NaN"  # finite check

    info = step["info"]
    assert "step_score_home" in info
    assert "step_score_away" in info
    assert "turnover" in info
    assert "unmatched_actions" in info
    print("  step + reward shape: OK")


def test_controlled_step():
    controlled = ["H1", "H2"]
    gym = rig_gym_py.RigGym()
    gym.reset(7, make_scenario(controlled=controlled, max_ticks=100))

    actions = [
        {
            "id": "H1",
            "aim": [1.0, 0.0, 0.0],
            "fire_line_at": None,
            "reel": -1,
            "release": False,
            "pushoff": True,
            "throw_charge": 0.0,
            "throw_released": False,
            "throw_spin": 0.0,
            "thrumbler": [0.0, 0.0, 0.0],
            "catch_intent": False,
        },
        idle_action("H2"),
    ]

    for _ in range(10):
        step = gym.step(actions)
        assert not step["done"] or step["obs"]["tick"] > 0
    print("  controlled step (10 ticks): OK")


def test_determinism():
    """Same seed + scenario + actions => bit-identical observations."""
    sc = make_scenario(controlled=["H1"], max_ticks=100)
    actions = [{
        "id": "H1",
        "aim": [0.0, 1.0, 0.0],
        "fire_line_at": None,
        "reel": 0,
        "release": False,
        "pushoff": True,
        "throw_charge": 0.0,
        "throw_released": False,
        "throw_spin": 0.0,
        "thrumbler": [0.0, 0.0, 0.0],
        "catch_intent": False,
    }]

    def run_episode(seed):
        gym = rig_gym_py.RigGym()
        gym.reset(seed, sc)
        results = []
        for _ in range(50):
            s = gym.step(actions)
            results.append((
                s["obs"]["bell_p"],
                s["reward"]["total"],
                s["obs"]["tick"],
            ))
            if s["done"]:
                break
        return results

    r1 = run_episode(999)
    r2 = run_episode(999)
    assert len(r1) == len(r2), "different episode lengths"
    for i, (a, b) in enumerate(zip(r1, r2)):
        assert a == b, f"determinism violation at step {i}: {a} != {b}"
    print("  determinism: OK")


def test_snapshot_restore():
    """Snapshot at tick N, continue both original and restored, assert same."""
    sc = make_scenario(controlled=["H1"], max_ticks=500)
    action = [{
        "id": "H1",
        "aim": [1.0, 0.0, 0.0],
        "fire_line_at": None,
        "reel": -1,
        "release": False,
        "pushoff": False,
        "throw_charge": 0.0,
        "throw_released": False,
        "throw_spin": 0.0,
        "thrumbler": [0.0, 0.0, 0.0],
        "catch_intent": False,
    }]

    gym = rig_gym_py.RigGym()
    gym.reset(77, sc)

    # Run 50 steps
    for _ in range(50):
        gym.step(action)

    # Take snapshot
    handle = gym.snapshot()

    # Continue original for 30 steps, record trajectory
    orig_traj = []
    for _ in range(30):
        s = gym.step(action)
        orig_traj.append(s["obs"]["bell_p"])

    # Restore and replay
    gym.restore(handle)
    for i in range(30):
        s = gym.step(action)
        assert s["obs"]["bell_p"] == orig_traj[i], \
            f"snapshot/restore diverged at step {i}"

    print("  snapshot/restore: OK")


def test_observation_action_space():
    gym = rig_gym_py.RigGym()
    obs_space = gym.observation_space()
    act_space = gym.action_space()
    assert isinstance(obs_space, dict)
    assert isinstance(act_space, dict)
    assert "tick" in obs_space
    assert "id" in act_space
    print("  space introspection: OK")


def test_reward_finite_extended():
    """Run a longer episode and assert all rewards stay finite."""
    gym = rig_gym_py.RigGym()
    gym.reset(1234, make_scenario(max_ticks=600))
    for _ in range(600):
        s = gym.step([])
        r = s["reward"]
        for k, v in r.items():
            assert v == v, f"NaN in reward.{k}"
            assert abs(v) < 1e10, f"reward.{k} exploded: {v}"
        if s["done"] or s["truncated"]:
            break
    print("  reward finite (600 ticks): OK")


if __name__ == "__main__":
    print("Running rig_gym_py tests...")
    test_reset_and_obs_shape()
    test_step_and_reward_shape()
    test_controlled_step()
    test_determinism()
    test_snapshot_restore()
    test_observation_action_space()
    test_reward_finite_extended()
    print("\nAll tests passed.")

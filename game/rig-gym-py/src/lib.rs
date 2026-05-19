//! PyO3 bridge exposing the RIG deterministic gym to Python.
//!
//! Wraps `rig_core::gym::RigEnv` behind a Python class `RigGym` that
//! accepts/returns plain dicts + lists (numpy-friendly). Releases the
//! GIL during `reset`/`step` so a vectorized Python wrapper can drive
//! multiple envs in parallel from threads.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use rig_core::gym::{
    Action, Env, Observation, ObsPlayer, RewardConfig, RigEnv, Scenario, Snapshot,
    Step, TeamSide,
};
use rig_core::math::Vec3;

// ── Conversion helpers ──────────────────────────────────────────────────────

fn vec3_to_list<'py>(py: Python<'py>, v: Vec3) -> Bound<'py, PyList> {
    PyList::new_bound(py, &[v.x, v.y, v.z])
}

fn list_to_vec3(obj: &Bound<'_, PyAny>) -> PyResult<Vec3> {
    let seq = obj.downcast::<PyList>()?;
    if seq.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected a list of 3 floats for Vec3",
        ));
    }
    Ok(Vec3::new(
        seq.get_item(0)?.extract::<f64>()?,
        seq.get_item(1)?.extract::<f64>()?,
        seq.get_item(2)?.extract::<f64>()?,
    ))
}

fn opt_vec3_to_py<'py>(py: Python<'py>, v: Option<Vec3>) -> PyObject {
    match v {
        Some(v) => vec3_to_list(py, v).into(),
        None => py.None(),
    }
}

fn obs_player_to_dict<'py>(py: Python<'py>, p: &ObsPlayer) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("id", &p.id)?;
    d.set_item("team", p.team)?;
    d.set_item("role", p.role)?;
    d.set_item("p", vec3_to_list(py, p.p))?;
    d.set_item("v", vec3_to_list(py, p.v))?;
    d.set_item("line_anchor", opt_vec3_to_py(py, p.line_anchor))?;
    d.set_item("line_rest_len", p.line_rest_len)?;
    // Assignment (Director hints)
    match &p.assignment {
        Some(asg) => {
            let ad = PyDict::new_bound(py);
            ad.set_item("job", asg.job)?;
            ad.set_item("mark_id", asg.mark_id.as_deref())?;
            d.set_item("assignment", ad)?;
        }
        None => {
            d.set_item("assignment", py.None())?;
        }
    }
    Ok(d)
}

fn obs_to_dict<'py>(py: Python<'py>, obs: &Observation) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("tick", obs.tick)?;
    d.set_item("bell_p", vec3_to_list(py, obs.bell_p))?;
    d.set_item("bell_v", vec3_to_list(py, obs.bell_v))?;
    d.set_item("bell_held_by", obs.bell_held_by.as_deref())?;
    d.set_item("possessed", obs.possessed)?;
    d.set_item("possession", obs.possession)?;
    d.set_item("gate", obs.gate)?;
    d.set_item("score_home", obs.score_home)?;
    d.set_item("score_away", obs.score_away)?;
    d.set_item("phase", obs.phase)?;
    d.set_item("attack_sign", obs.attack_sign)?;
    d.set_item("gate_plane_x", obs.gate_plane_x)?;

    let players = PyList::empty_bound(py);
    for p in &obs.players {
        players.append(obs_player_to_dict(py, p)?)?;
    }
    d.set_item("players", players)?;

    let ctrl: Vec<&str> = obs.controlled_ids.iter().map(|s| s.as_str()).collect();
    let ctrl_list = PyList::new_bound(py, &ctrl);
    d.set_item("controlled_ids", ctrl_list)?;

    Ok(d)
}

fn reward_to_dict<'py>(py: Python<'py>, r: &rig_core::gym::Reward) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("possession_held", r.possession_held)?;
    d.set_item("gate_advanced", r.gate_advanced)?;
    d.set_item("scored", r.scored)?;
    d.set_item("contest_won", r.contest_won)?;
    d.set_item("bell_out", r.bell_out)?;
    d.set_item("terminal", r.terminal)?;
    d.set_item("total", r.total)?;
    Ok(d)
}

fn step_to_dict<'py>(py: Python<'py>, s: &Step) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("obs", obs_to_dict(py, &s.obs)?)?;
    d.set_item("reward", reward_to_dict(py, &s.reward)?)?;
    d.set_item("done", s.done)?;
    d.set_item("truncated", s.truncated)?;

    let info = PyDict::new_bound(py);
    info.set_item("step_score_home", s.info.step_score_home)?;
    info.set_item("step_score_away", s.info.step_score_away)?;
    info.set_item("turnover", s.info.turnover)?;
    info.set_item("unmatched_actions", s.info.unmatched_actions)?;
    d.set_item("info", info)?;

    Ok(d)
}

fn parse_scenario(dict: &Bound<'_, PyDict>) -> PyResult<Scenario> {
    let home_style: String = dict
        .get_item("home_style")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("home_style"))?
        .extract()?;
    let home_cyl: String = dict
        .get_item("home_cyl")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("home_cyl"))?
        .extract()?;
    let away_style: String = dict
        .get_item("away_style")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("away_style"))?
        .extract()?;
    let away_cyl: String = dict
        .get_item("away_cyl")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("away_cyl"))?
        .extract()?;
    let controlled_ids: Vec<String> = dict
        .get_item("controlled_ids")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("controlled_ids"))?
        .extract()?;
    let max_ticks: u64 = dict
        .get_item("max_ticks")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("max_ticks"))?
        .extract()?;

    let reward_config = if let Some(rc) = dict.get_item("reward_config")? {
        let rc = rc.downcast::<PyDict>()?;
        let get_f = |key: &str, default: f64| -> PyResult<f64> {
            match rc.get_item(key)? {
                Some(v) => v.extract(),
                None => Ok(default),
            }
        };
        let defaults = RewardConfig::default();
        let reward_side = match rc.get_item("reward_side")? {
            Some(v) => {
                if v.is_none() {
                    None
                } else {
                    let s: String = v.extract()?;
                    match s.as_str() {
                        "home" | "Home" => Some(TeamSide::Home),
                        "away" | "Away" => Some(TeamSide::Away),
                        _ => None,
                    }
                }
            }
            None => None,
        };
        let shaping_gamma = match rc.get_item("shaping_gamma")? {
            Some(v) => {
                if v.is_none() {
                    None
                } else {
                    Some(v.extract::<f64>()?)
                }
            }
            None => None,
        };
        RewardConfig {
            w_possession: get_f("w_possession", defaults.w_possession)?,
            w_gate: get_f("w_gate", defaults.w_gate)?,
            w_score: get_f("w_score", defaults.w_score)?,
            w_contest: get_f("w_contest", defaults.w_contest)?,
            w_bell_out: get_f("w_bell_out", defaults.w_bell_out)?,
            w_terminal: get_f("w_terminal", defaults.w_terminal)?,
            reward_side,
            shaping_gamma,
        }
    } else {
        RewardConfig::default()
    };

    Ok(Scenario {
        home_style,
        home_cyl,
        away_style,
        away_cyl,
        controlled_ids,
        max_ticks,
        reward_config,
    })
}

fn parse_action(dict: &Bound<'_, PyAny>) -> PyResult<Action> {
    let dict = dict.downcast::<PyDict>()?;

    let id: String = dict
        .get_item("id")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("id"))?
        .extract()?;

    let aim = match dict.get_item("aim")? {
        Some(v) => list_to_vec3(&v)?,
        None => Vec3::new(0.0, 0.0, 0.0),
    };

    let fire_line_at = match dict.get_item("fire_line_at")? {
        Some(v) => {
            if v.is_none() {
                None
            } else {
                Some(list_to_vec3(&v)?)
            }
        }
        None => None,
    };

    let reel: i32 = dict
        .get_item("reel")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0);

    let release: bool = dict
        .get_item("release")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(false);

    let pushoff: bool = dict
        .get_item("pushoff")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(false);

    let throw_charge: f64 = dict
        .get_item("throw_charge")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.0);

    let throw_released: bool = dict
        .get_item("throw_released")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(false);

    let throw_spin: f64 = dict
        .get_item("throw_spin")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.0);

    let thrumbler = match dict.get_item("thrumbler")? {
        Some(v) => {
            if v.is_none() {
                Vec3::new(0.0, 0.0, 0.0)
            } else {
                list_to_vec3(&v)?
            }
        }
        None => Vec3::new(0.0, 0.0, 0.0),
    };

    let catch_intent: bool = dict
        .get_item("catch_intent")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(false);

    Ok(Action {
        id,
        aim,
        fire_line_at,
        reel,
        release,
        pushoff,
        throw_charge,
        throw_released,
        throw_spin,
        thrumbler,
        catch_intent,
    })
}

// ── The Python class ────────────────────────────────────────────────────────

/// In-process snapshot handle (avoids serialization — Snapshot derives Clone
/// but contains private fields from rig-core we cannot serialize without
/// modifying rig-core). We hold a Vec of snapshots and return integer
/// handles to Python.
#[pyclass]
struct RigGym {
    env: RigEnv,
    snapshots: Vec<Snapshot>,
}

#[pymethods]
impl RigGym {
    #[new]
    fn new() -> Self {
        RigGym {
            env: RigEnv::new(),
            snapshots: Vec::new(),
        }
    }

    /// Reset the environment. Returns the initial observation dict.
    /// Releases the GIL during the Rust reset.
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: u32,
        scenario: &Bound<'py, PyDict>,
    ) -> PyResult<PyObject> {
        let sc = parse_scenario(scenario)?;
        let obs = py.allow_threads(|| self.env.reset(seed, &sc));
        Ok(obs_to_dict(py, &obs)?.into())
    }

    /// Step the environment with a list of action dicts. Returns a step dict.
    /// Releases the GIL during the Rust step.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyList>,
    ) -> PyResult<PyObject> {
        let acts: Vec<Action> = actions
            .iter()
            .map(|item| parse_action(&item))
            .collect::<PyResult<Vec<_>>>()?;
        let step = py.allow_threads(|| self.env.step(&acts));
        Ok(step_to_dict(py, &step)?.into())
    }

    /// Take a snapshot of the current env state. Returns an opaque integer
    /// handle (index into an internal Vec). Use `restore(handle)` to
    /// restore later.
    fn snapshot(&mut self) -> usize {
        let s = self.env.snapshot();
        let idx = self.snapshots.len();
        self.snapshots.push(s);
        idx
    }

    /// Restore the env to a previously taken snapshot (by handle).
    fn restore(&mut self, py: Python<'_>, handle: usize) -> PyResult<()> {
        let s = self
            .snapshots
            .get(handle)
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "invalid snapshot handle: {}",
                    handle
                ))
            })?
            .clone();
        py.allow_threads(|| self.env.restore(&s));
        Ok(())
    }

    /// Drop a snapshot to free memory. After this the handle is invalid.
    fn drop_snapshot(&mut self, handle: usize) -> PyResult<()> {
        if handle >= self.snapshots.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "invalid snapshot handle: {}",
                handle
            )));
        }
        // Replace with a dummy to avoid shifting indices. The memory of
        // the old snapshot is freed.
        self.snapshots[handle] = self.env.snapshot();
        Ok(())
    }

    /// Return the observation space schema as a dict (for documentation /
    /// introspection from Python).
    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("tick", "int")?;
        d.set_item("bell_p", "[3] float")?;
        d.set_item("bell_v", "[3] float")?;
        d.set_item("bell_held_by", "str | None")?;
        d.set_item("possessed", "bool")?;
        d.set_item("possession", "int (0=Home, 1=Away)")?;
        d.set_item("gate", "int (0=First, 1=Deep, 2=Mouth)")?;
        d.set_item("score_home", "int")?;
        d.set_item("score_away", "int")?;
        d.set_item("phase", "int (0..6)")?;
        d.set_item("attack_sign", "float (+1 or -1)")?;
        d.set_item("gate_plane_x", "float")?;
        d.set_item(
            "players",
            "[{id, team, role, p:[3], v:[3], line_anchor:[3]|None, line_rest_len:float|None, assignment:{job:int,mark_id:str|None}|None}]",
        )?;
        d.set_item("controlled_ids", "[str]")?;
        Ok(d.into())
    }

    /// Return the action space schema as a dict.
    fn action_space<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("id", "str (rigger id, e.g. 'H1')")?;
        d.set_item("aim", "[3] float (unit direction)")?;
        d.set_item("fire_line_at", "[3] float | None")?;
        d.set_item("reel", "int (-1, 0, or 1)")?;
        d.set_item("release", "bool")?;
        d.set_item("pushoff", "bool")?;
        d.set_item("throw_charge", "float [0,1]")?;
        d.set_item("throw_released", "bool")?;
        d.set_item("throw_spin", "float [-1,1]")?;
        d.set_item("thrumbler", "[3] float [-1,1] per axis")?;
        d.set_item("catch_intent", "bool")?;
        Ok(d.into())
    }

    /// Number of snapshots currently held.
    fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }
}

// ── Module registration ─────────────────────────────────────────────────────

#[pymodule]
fn rig_gym_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RigGym>()?;
    Ok(())
}

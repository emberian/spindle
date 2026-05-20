//! rl/loco_env.rs — Single-player reaching environment for locomotion training.
//!
//! A stripped-down SimWorld with one player and no ball. The agent must reach
//! a target position using grapple mechanics (fire, reel, release, pushoff,
//! thrumbler). Trained with a dense reaching reward.
//!
//! NATIVE-ONLY: uses rand for random resets.

use crate::math::Vec3;
use crate::planner::spar_positions;
use crate::rl::loco::{loco_decode, loco_featurize, LocoAction, LocoObs, LocoPlayerState, LocoPolicy, LOCO_OUT_W};
use crate::sim_world::{InputFrame, PlayerInput, SimWorld, TeamSide, RiggerRole};
use crate::tuning;

use rand::Rng;
use rand_chacha::ChaCha8Rng;

const H: f64 = 1.0 / 240.0;
const ARRIVAL_RADIUS: f64 = 8.0;

/// Configuration for the locomotion training environment.
#[derive(Clone, Debug)]
pub struct LocoEnvConfig {
    /// Minimum distance to target on reset.
    pub min_dist: f64,
    /// Maximum distance to target on reset.
    pub max_dist: f64,
    /// Maximum episode length in ticks.
    pub max_ticks: u64,
    /// Maximum initial speed (randomized up to this).
    pub max_init_speed: f64,
}

impl Default for LocoEnvConfig {
    fn default() -> Self {
        LocoEnvConfig {
            min_dist: 30.0,
            max_dist: 120.0,
            max_ticks: 960,
            max_init_speed: 0.0,
        }
    }
}

/// Reward components for one step.
#[derive(Clone, Debug, Default)]
pub struct LocoReward {
    pub approach: f64,
    pub speed_toward: f64,
    pub arrival: f64,
    pub skin_penalty: f64,
    pub total: f64,
}

/// Step result from the environment.
pub struct LocoStep {
    pub obs: LocoObs,
    pub reward: LocoReward,
    pub done: bool,
    pub arrived: bool,
    pub ticks_elapsed: u64,
}

/// The locomotion training environment.
pub struct LocoEnv {
    sim: SimWorld,
    target: Vec3,
    prev_dist: f64,
    elapsed: u64,
    config: LocoEnvConfig,
    spars: Vec<Vec3>,
}

impl LocoEnv {
    pub fn new(config: LocoEnvConfig) -> Self {
        let sim = SimWorld::new(0);
        LocoEnv {
            sim,
            target: Vec3::new(0.0, 0.0, 0.0),
            prev_dist: 0.0,
            elapsed: 0,
            config,
            spars: spar_positions(),
        }
    }

    /// Reset the environment with a new random starting position and target.
    pub fn reset(&mut self, rng: &mut ChaCha8Rng) -> LocoObs {
        // Fresh sim
        self.sim = SimWorld::new(rng.gen::<u32>());
        self.elapsed = 0;

        // Random starting position within the cylinder
        let r_max = tuning::R - 5.0; // stay away from skin
        let half_l = tuning::L / 2.0 - 20.0;

        let x = rng.gen_range(-half_l..half_l);
        let ang: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let r: f64 = rng.gen_range(0.0..r_max);
        let start = Vec3::new(x, r * ang.cos(), r * ang.sin());

        self.sim.add_player("P", TeamSide::Home, RiggerRole::Spinner, start);

        // Random initial velocity
        if self.config.max_init_speed > 0.0 {
            let speed = rng.gen_range(0.0..self.config.max_init_speed);
            let vang: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            let vphi: f64 = rng.gen_range(-1.0_f64..1.0).acos();
            let vx = speed * vphi.sin() * vang.cos();
            let vy = speed * vphi.sin() * vang.sin();
            let vz = speed * vphi.cos();
            self.sim.players[0].body.v = Vec3::new(vx, vy, vz);
        }

        // Random target at configured distance
        let dist = rng.gen_range(self.config.min_dist..self.config.max_dist);
        let tang: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let tphi: f64 = rng.gen_range(0.3..2.8); // avoid poles
        let dir = Vec3::new(
            tphi.sin() * tang.cos(),
            tphi.sin() * tang.sin(),
            tphi.cos(),
        );
        let raw_target = Vec3::new(
            start.x + dir.x * dist,
            start.y + dir.y * dist,
            start.z + dir.z * dist,
        );
        // Clamp target inside cylinder
        let tx = raw_target.x.clamp(-half_l, half_l);
        let tr = raw_target.y.hypot(raw_target.z);
        let (ty, tz) = if tr > r_max && tr > 1e-6 {
            let s = r_max / tr;
            (raw_target.y * s, raw_target.z * s)
        } else {
            (raw_target.y, raw_target.z)
        };
        self.target = Vec3::new(tx, ty, tz);

        self.prev_dist = self.dist_to_target();
        self.observe()
    }

    /// Step the environment with a policy output.
    pub fn step_action(&mut self, out: &[f64; LOCO_OUT_W]) -> LocoStep {
        let player_p = self.sim.players[0].body.p;
        let action = loco_decode(out, player_p, self.target);

        // Convert to SimWorld PlayerInput
        let inp = PlayerInput {
            id: "P".to_string(),
            aim: action.aim,
            fire_line_at: action.fire_line_at,
            reel: action.reel,
            release: action.release,
            pushoff: action.pushoff,
            throw_charge: 0.0,
            throw_released: false,
            throw_spin: 0.0,
            thrumbler: action.thrumbler,
            catch_intent: false,
        };

        let frame = InputFrame {
            tick: self.elapsed,
            players: vec![inp],
        };
        self.sim.step(&frame, H);
        self.elapsed += 1;

        // Compute reward
        let new_dist = self.dist_to_target();
        let approach = (self.prev_dist - new_dist) / tuning::REEL_PULL_SPEED * 240.0;
        // Normalized: approaching at reel speed gives ~1.0 per tick

        let player_v = self.sim.players[0].body.v;
        let to_target = Vec3::new(
            self.target.x - self.sim.players[0].body.p.x,
            self.target.y - self.sim.players[0].body.p.y,
            self.target.z - self.sim.players[0].body.p.z,
        );
        let d = to_target.len();
        let speed_toward = if d > 1e-6 {
            let dir = Vec3::new(to_target.x / d, to_target.y / d, to_target.z / d);
            let vt = player_v.x * dir.x + player_v.y * dir.y + player_v.z * dir.z;
            (vt / tuning::REEL_PULL_SPEED).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let arrived = new_dist < ARRIVAL_RADIUS;
        let arrival = if arrived { 10.0 } else { 0.0 };

        let grounded = self.sim.players[0].body.grounded;
        let skin_penalty = if grounded { -0.5 } else { 0.0 };

        let total = approach + 0.3 * speed_toward + arrival + skin_penalty;

        self.prev_dist = new_dist;

        let done = arrived || self.elapsed >= self.config.max_ticks;

        LocoStep {
            obs: self.observe(),
            reward: LocoReward {
                approach,
                speed_toward,
                arrival,
                skin_penalty,
                total,
            },
            done,
            arrived,
            ticks_elapsed: self.elapsed,
        }
    }

    fn dist_to_target(&self) -> f64 {
        let p = self.sim.players[0].body.p;
        let dx = self.target.x - p.x;
        let dy = self.target.y - p.y;
        let dz = self.target.z - p.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn observe(&self) -> LocoObs {
        let body = &self.sim.players[0].body;
        let line_anchor = body.line.as_ref().and_then(|l| {
            if l.attached { Some(l.anchor_pos) } else { None }
        });
        let line_in_flight = body.line.as_ref().map(|l| !l.attached).unwrap_or(false);
        let line_taut = body.line.as_ref().map(|l| l.attached).unwrap_or(false);
        let line_rest_len = body.line.as_ref().map(|l| l.rest_len);

        let state = LocoPlayerState {
            p: body.p,
            v: body.v,
            line_anchor,
            line_rest_len,
            line_taut,
            line_in_flight,
            grounded: body.grounded,
            omega: tuning::OMEGA,
        };
        loco_featurize(&state, self.target, &self.spars)
    }
}

/// Run a full episode with the given policy and return total reward + arrived flag.
pub fn evaluate_episode(
    policy: &LocoPolicy,
    config: &LocoEnvConfig,
    rng: &mut ChaCha8Rng,
) -> (f64, bool, u64) {
    let mut env = LocoEnv::new(config.clone());
    let mut obs = env.reset(rng);
    let mut total_reward = 0.0_f64;

    loop {
        let out = policy.forward(&obs);
        let step = env.step_action(&out);
        total_reward += step.reward.total;
        if step.done {
            return (total_reward, step.arrived, step.ticks_elapsed);
        }
        obs = step.obs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn env_resets_and_steps() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = LocoEnvConfig::default();
        let mut env = LocoEnv::new(config);
        let obs = env.reset(&mut rng);
        assert!(obs.feat[9] > 0.0); // target distance > 0

        let zero_action = [0.0_f64; LOCO_OUT_W];
        let step = env.step_action(&zero_action);
        assert!(!step.done); // shouldn't be done after 1 tick
        assert!(step.reward.total.is_finite());
    }

    #[test]
    fn episode_is_deterministic() {
        let config = LocoEnvConfig {
            min_dist: 40.0,
            max_dist: 60.0,
            max_ticks: 200,
            max_init_speed: 5.0,
        };
        let policy = LocoPolicy::new_zeros();
        let mut rng_a = ChaCha8Rng::seed_from_u64(123);
        let mut rng_b = ChaCha8Rng::seed_from_u64(123);
        let (ra, aa, ta) = evaluate_episode(&policy, &config, &mut rng_a);
        let (rb, ab, tb) = evaluate_episode(&policy, &config, &mut rng_b);
        assert_eq!(ra.to_bits(), rb.to_bits());
        assert_eq!(aa, ab);
        assert_eq!(ta, tb);
    }
}

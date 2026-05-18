//! AiSystem — the top-level deterministic AI orchestrator. 1:1 port of
//! src/ai/index.ts. Given (SimState, MatchState, TeamConfig[], seed),
//! produces an InputFrame for all AI players (id != "P1").
//!
//! Determinism: identical (sim_state, match_state, configs, seed, tick)
//! → identical InputFrame, because all randomness flows through
//! `AiRng::make(seed, tick, team_index)` (the bit-exact `makeRng` port).
//! Director runs at ~2 Hz (cached per team); per-player commitments are
//! cached and only re-evaluated at the Director cadence / on hard
//! triggers (the heart of visible competence — see rigger_ai).

use super::decision_types::{
    DirectorState, PlayerCommitCache, TeamConfig, DIRECTOR_TICK_INTERVAL,
};
use super::director::run_director;
use super::rigger_ai::compute_player_input;
use super::rng::AiRng;
use super::types::{InputFrame, MatchState, SimState, TeamSide};
use std::collections::{HashMap, HashSet};

/// Director cache: one per team side (index.ts DirectorCache).
#[derive(Clone)]
struct DirectorCache {
    state: DirectorState,
    last_updated_tick: f64,
}

/// Side discriminant used as part of cache keys. Mirrors the two
/// `TeamSide` variants 1:1 (Home→0, Away→1) so cache identity is
/// exactly what `format!("{:?}", side)` produced before, only without
/// the per-tick String allocation.
#[inline]
fn side_tag(side: TeamSide) -> u8 {
    match side {
        TeamSide::Home => 0,
        TeamSide::Away => 1,
    }
}

/// Stable per-player commitment key. Replaces the per-tick
/// `format!("{:?}-{}-{}", side, ci, id)` String allocation with a tuple
/// that hashes/compares to the exact same logical identity, so which
/// cache entry maps where is byte-for-byte unchanged.
#[derive(Clone, PartialEq, Eq, Hash)]
struct CommitKey {
    side: u8,
    ci: usize,
    player_id: String,
}

#[derive(Default, Clone)]
pub struct AiSystem {
    director_caches: HashMap<(u8, usize), DirectorCache>,
    commit_cache: HashMap<CommitKey, PlayerCommitCache>,
    /// Rigger ids whose Action is supplied externally (the gym `Env`
    /// caller drives them). EMPTY by default ⇒ the baseline AI controls
    /// everyone exactly as before (production / skill_eval path unchanged).
    /// When non-empty, `tick` SKIPS these ids — it never emits an input
    /// for them and never advances their commit/director caches off their
    /// state, so the external policy fully owns them.
    controlled: HashSet<String>,
}

impl AiSystem {
    pub fn new() -> Self {
        AiSystem::default()
    }

    /// Install the externally-controlled rigger id set (the gym
    /// agent-control selector). Deterministic: membership is a pure id
    /// lookup, no iteration over the set in the tick hot path.
    pub fn set_controlled<I: IntoIterator<Item = String>>(&mut self, ids: I) {
        self.controlled = ids.into_iter().collect();
    }

    /// Produce an InputFrame for all AI-controlled players (id != "P1")
    /// for this tick. index.ts:91-159 verbatim.
    pub fn tick(
        &mut self,
        sim_state: &SimState,
        match_state: &MatchState,
        configs: &[TeamConfig],
        seed: u32,
    ) -> InputFrame {
        let tick = sim_state.tick;
        let tick_u = tick as u32;
        let mut inputs = Vec::new();

        for (ci, cfg) in configs.iter().enumerate() {
            // 'P1' is human-controlled only when present; filtering by
            // id != "P1" naturally includes everyone in a full-AI match.
            let team_players: Vec<usize> = sim_state
                .players
                .iter()
                .enumerate()
                // Agent-control selector: a rigger in `controlled` is
                // driven by the external Env caller, so the baseline AI
                // skips it entirely (no input emitted, caches untouched).
                .filter(|(_, p)| {
                    p.team == cfg.side
                        && p.id != "P1"
                        && !self.controlled.contains(&p.id)
                })
                .map(|(i, _)| i)
                .collect();
            if team_players.is_empty() {
                continue;
            }

            // Director: update at ~2 Hz.
            let dir_key = (side_tag(cfg.side), ci);
            let needs_director_update = match self.director_caches.get(&dir_key) {
                None => true,
                Some(dc) => tick - dc.last_updated_tick >= DIRECTOR_TICK_INTERVAL,
            };
            if needs_director_update {
                let mut dir_rng = AiRng::make(seed, tick_u, (ci as u32) * 1000);
                let new_state = run_director(
                    sim_state,
                    match_state,
                    &cfg.profile,
                    cfg.side,
                    &mut dir_rng,
                );
                self.director_caches.insert(
                    dir_key,
                    DirectorCache {
                        state: new_state,
                        last_updated_tick: tick,
                    },
                );
            }
            let director_refreshed = needs_director_update;
            // Borrow the cached director plan in-place. `director_caches`
            // and `commit_cache` are disjoint fields, so an immutable
            // borrow of the former coexists with the &mut entry borrow of
            // the latter — no per-tick DirectorState clone needed.
            let director = &self.director_caches[&dir_key].state;
            let commit_cache = &mut self.commit_cache;

            for (pi, &p_idx) in team_players.iter().enumerate() {
                let player = &sim_state.players[p_idx];
                // Each player gets a fully independent rng stream.
                let mut player_rng =
                    AiRng::make(seed, tick_u, (ci as u32) * 100 + pi as u32);

                let commit_key = CommitKey {
                    side: side_tag(cfg.side),
                    ci,
                    player_id: player.id.clone(),
                };
                let commit = commit_cache
                    .entry(commit_key)
                    .or_insert_with(|| PlayerCommitCache {
                        value: None,
                        decided_tick: -1.0,
                    });

                let input = compute_player_input(
                    player,
                    sim_state,
                    match_state,
                    &cfg.profile,
                    director,
                    cfg.difficulty,
                    &mut player_rng,
                    commit,
                    director_refreshed,
                    DIRECTOR_TICK_INTERVAL,
                );
                inputs.push(input);
            }
        }

        InputFrame { tick, players: inputs }
    }

    /// Reset director + commitment caches (e.g., on a new inning).
    pub fn reset(&mut self) {
        self.director_caches.clear();
        self.commit_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::profile::{style_to_profile, Difficulty};
    use crate::ai::types::{
        BellState, FaithEnd, MatchPhase, PlayerSim, RiggerRole, TeamSide,
    };
    use crate::ai::types::{Cast, Gate, MatchState as Ms};
    use crate::math::{Quat, Vec3};

    fn vz() -> Vec3 {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    fn mk_player(id: &str, team: TeamSide, role: RiggerRole, x: f64) -> PlayerSim {
        PlayerSim {
            id: id.to_string(),
            team,
            role,
            p: Vec3 { x, y: 4.0, z: 0.0 },
            v: vz(),
            q: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
            line: None,
            dv_budget: 6.0,
            contact_ref: None,
            grounded: false,
        }
    }

    fn fixture() -> (SimState, Ms, Vec<TeamConfig>) {
        let players = vec![
            mk_player("H1", TeamSide::Home, RiggerRole::Spinner, -40.0),
            mk_player("H2", TeamSide::Home, RiggerRole::Anchor, -90.0),
            mk_player("A1", TeamSide::Away, RiggerRole::Spinner, 40.0),
            mk_player("A2", TeamSide::Away, RiggerRole::Reach, 90.0),
        ];
        let st = SimState {
            tick: 0.0,
            omega: crate::tuning::OMEGA,
            bell: BellState {
                p: vz(),
                v: vz(),
                q: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                w: vz(),
                chime: 1.0,
                held_by: Some("H1".to_string()),
                thrown_by: None,
                touched_since_throw: false,
                release_pos: vz(),
                release_tick: 0.0,
                pass_chain: vec![],
            },
            players,
        };
        let m = Ms {
            inning: 1.0,
            spine: false,
            possession: TeamSide::Home,
            faith_end: FaithEnd::PlusX,
            cast: Cast { throws_left: 3, gate: Gate::First, spot_x: 0.0 },
            contest: None,
            score_home: 0.0,
            score_away: 0.0,
            phase: MatchPhase::Live,
            message: String::new(),
            winner: None,
        };
        let cfgs = vec![
            TeamConfig {
                side: TeamSide::Home,
                profile: style_to_profile("fall-dynasty", "big-slow"),
                difficulty: Difficulty::Pro,
            },
            TeamConfig {
                side: TeamSide::Away,
                profile: style_to_profile("rise-chaos", "small-fast"),
                difficulty: Difficulty::Pro,
            },
        ];
        (st, m, cfgs)
    }

    #[test]
    fn deterministic_same_seed_same_frame() {
        let (st, m, cfgs) = fixture();
        let mut a = AiSystem::new();
        let mut b = AiSystem::new();
        let fa = a.tick(&st, &m, &cfgs, 1234);
        let fb = b.tick(&st, &m, &cfgs, 1234);
        assert_eq!(fa.players.len(), fb.players.len());
        assert_eq!(fa.players.len(), 4, "all 4 AI players controlled");
        for (x, y) in fa.players.iter().zip(fb.players.iter()) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.aim.x.to_bits(), y.aim.x.to_bits());
            assert_eq!(x.reel, y.reel);
            assert_eq!(x.throw_released, y.throw_released);
        }
    }

    #[test]
    fn director_cached_across_window() {
        let (mut st, m, cfgs) = fixture();
        let mut sys = AiSystem::new();
        // tick 0 builds the director; ticks 1..29 reuse it (no panic, stable)
        for t in 0..5 {
            st.tick = t as f64;
            let f = sys.tick(&st, &m, &cfgs, 99);
            assert_eq!(f.players.len(), 4);
            for p in &f.players {
                assert!(p.aim.x.is_finite());
                assert!(p.reel == -1 || p.reel == 0 || p.reel == 1);
                assert!((0.0..=1.0).contains(&p.throw_charge));
            }
        }
        sys.reset();
        assert!(sys.director_caches.is_empty());
        assert!(sys.commit_cache.is_empty());
    }
}

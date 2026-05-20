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
    AiDebugRec, DirectorState, Job, PlayerAssignment, PlayerCommitCache, TeamConfig,
    DIRECTOR_TICK_INTERVAL,
};
use super::director::run_director;
use super::rigger_ai::{
    bell_intercept, compute_player_input, is_contest_committer, is_dive_committer, job_str,
    role_str, wants_catch,
};
use super::rng::AiRng;
use super::types::{InputFrame, MatchState, SimState, TeamSide};
use crate::math::Vec3;
use std::collections::{HashMap, HashSet};

/// Director cache: one per team side (index.ts DirectorCache).
#[derive(Clone)]
struct DirectorCache {
    state: DirectorState,
    last_updated_tick: f64,
    /// Tracks the ball holder at the last Director evaluation so we can
    /// detect possession changes and force an immediate re-run (no 0.5s
    /// stale window after catches/throws/turnovers).
    prev_bell_held_by: Option<String>,
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
    /// RENDER-ONLY legibility side channel — one record per AI-controlled
    /// rigger this tick, parallel to the emitted frame. Filled by `tick`
    /// from the SAME committed director/commit state the input came from
    /// (pure deterministic re-reads, no rng), serialized over a SEPARATE
    /// wasm method. NEVER folded into the InputFrame, `Snapshot`, or
    /// `hash_snapshot` — it cannot perturb the sim or determinism.
    last_debug: Vec<AiDebugRec>,
    /// AI-side pass contract: when a carrier releases a teammate-targeted
    /// pass, remember the intended catcher for that team's in-flight ball so
    /// non-target teammates keep their spacing instead of all swarming the
    /// same friendly pass.
    active_pass_targets: HashMap<(u8, usize), String>,
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
        self.last_debug.clear();

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
            let own_throw_in_flight = sim_state.bell.held_by.is_none()
                && sim_state
                    .bell
                    .thrown_by
                    .as_deref()
                    .and_then(|tid| sim_state.players.iter().find(|p| p.id == tid))
                    .map(|p| p.team == cfg.side)
                    .unwrap_or(false);
            if !own_throw_in_flight {
                self.active_pass_targets.remove(&dir_key);
            }
            let active_pass_target = self.active_pass_targets.get(&dir_key).cloned();
            let possession_changed = self
                .director_caches
                .get(&dir_key)
                .map(|dc| dc.prev_bell_held_by.as_deref() != sim_state.bell.held_by.as_deref())
                .unwrap_or(false);
            let needs_director_update = match self.director_caches.get(&dir_key) {
                None => true,
                Some(dc) => {
                    possession_changed
                        || tick - dc.last_updated_tick >= DIRECTOR_TICK_INTERVAL
                }
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
                        prev_bell_held_by: sim_state.bell.held_by.clone(),
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
            let mut released_pass_target: Option<String> = None;

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
                    active_pass_target.as_deref(),
                    commit,
                    director_refreshed,
                    DIRECTOR_TICK_INTERVAL,
                );
                if input.throw_released {
                    if let Some(target_id) = commit
                        .value
                        .as_ref()
                        .and_then(|c| c.throw_target_id.clone())
                    {
                        released_pass_target = Some(target_id);
                    }
                }
                inputs.push(input);

                // ── RENDER-ONLY legibility record ────────────────────────
                // Re-read the SAME committed director/commit state the
                // input above came from to label what this rigger is doing
                // and why. All predicates are pure deterministic reads (no
                // rng, no cache mutation) so building this never perturbs
                // the emitted frame, the sim, or the determinism hash.
                let assignment: PlayerAssignment = director
                    .assignments
                    .get(&player.id)
                    .cloned()
                    .unwrap_or(PlayerAssignment {
                        job: Job::Support,
                        mark_id: None,
                        depth_slot: 0.4,
                        radius_slot: 0.45,
                        pressure: 0.0,
                    });
                let is_primary =
                    director.recover_id.as_deref() == Some(player.id.as_str());
                let is_shadow = assignment.is_shadow();
                let is_outlet = assignment.job == Job::Receive;
                let is_diver = is_dive_committer(
                    player,
                    sim_state,
                    director,
                    &assignment,
                    active_pass_target.as_deref(),
                );
                let is_contester = !is_diver
                    && is_contest_committer(player, sim_state, match_state, director);
                let catching = wants_catch(
                    player,
                    sim_state,
                    &assignment,
                    active_pass_target.as_deref(),
                );
                let have_bell =
                    sim_state.bell.held_by.as_deref() == Some(player.id.as_str());
                // Committed point this rigger is acting on, in priority
                // order matching the executor in `compute_player_input`.
                let intent_target: Option<Vec3> = if have_bell {
                    // Throwing/holding in place — the bell IS the rigger;
                    // no separate target line reads cleaner than a stub.
                    None
                } else if is_diver {
                    Some(sim_state.bell.p)
                } else if is_contester {
                    Some(sim_state.bell.p)
                } else if catching {
                    Some(bell_intercept(player, sim_state))
                } else {
                    commit_cache
                        .get(&CommitKey {
                            side: side_tag(cfg.side),
                            ci,
                            player_id: player.id.clone(),
                        })
                        .and_then(|c| c.value.as_ref())
                        .and_then(|v| v.nav_target)
                };
                self.last_debug.push(AiDebugRec {
                    id: player.id.clone(),
                    role: role_str(player.role).to_string(),
                    job: job_str(assignment.job).to_string(),
                    intent_target,
                    is_diver,
                    is_contester,
                    is_primary,
                    is_shadow,
                    is_outlet,
                    controlled_by: "baseline",
                });
            }
            if let Some(target_id) = released_pass_target {
                self.active_pass_targets.insert(dir_key, target_id);
            }
        }

        InputFrame { tick, players: inputs }
    }

    /// Reset director + commitment caches (e.g., on a new inning).
    pub fn reset(&mut self) {
        self.director_caches.clear();
        self.commit_cache.clear();
        self.last_debug.clear();
        self.active_pass_targets.clear();
    }

    /// RENDER-ONLY: the legibility records produced by the last `tick`,
    /// in emit order (parallel to the InputFrame players). Read by the
    /// wasm `ai_debug_json` seam for the spectate overlay only.
    pub fn last_debug(&self) -> &[AiDebugRec] {
        &self.last_debug
    }
}

/// Serialize the legibility records to the render-only JSON the overlay
/// consumes. Hand-rolled to match the `wasm`/`ai_wasm` JSON dialect
/// exactly (one seam dialect). Pure; no rng. NOT a sim/determinism
/// surface — it is the human-watchable channel only.
pub fn ai_debug_to_json(recs: &[AiDebugRec]) -> String {
    fn jstr(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }
    fn jnum(x: f64) -> String {
        if x.is_finite() {
            format!("{:?}", x)
        } else {
            "0.0".to_string()
        }
    }
    fn jvec(v: &Option<Vec3>) -> String {
        match v {
            Some(v) => format!(
                "{{\"x\":{},\"y\":{},\"z\":{}}}",
                jnum(v.x),
                jnum(v.y),
                jnum(v.z)
            ),
            None => "null".to_string(),
        }
    }
    let parts: Vec<String> = recs
        .iter()
        .map(|r| {
            format!(
                "{{\"id\":{},\"role\":{},\"job\":{},\"intentTargetPos\":{},\"isDiver\":{},\"isContester\":{},\"isPrimary\":{},\"isShadow\":{},\"isOutlet\":{},\"controlledBy\":{}}}",
                jstr(&r.id),
                jstr(&r.role),
                jstr(&r.job),
                jvec(&r.intent_target),
                if r.is_diver { "true" } else { "false" },
                if r.is_contester { "true" } else { "false" },
                if r.is_primary { "true" } else { "false" },
                if r.is_shadow { "true" } else { "false" },
                if r.is_outlet { "true" } else { "false" },
                jstr(r.controlled_by),
            )
        })
        .collect();
    format!("[{}]", parts.join(","))
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

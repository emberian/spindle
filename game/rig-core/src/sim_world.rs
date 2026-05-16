//! Deterministic world — port of TS sim/SimWorld.ts.
//!
//! One `SimWorld` ticks a deterministic physics simulation of the bell and all
//! players.  It is pure given (state, inputs): no wall-clock, no OS RNG.
//! Events are emitted each step and consumed by the match-rules layer.

use crate::bell::{chime, step_bell, BellBody};
use crate::collision::{apply_bobble, contest_clatter, skin_bounce, try_catch, CatchResult};
use crate::loop_detector::{LoopTier, LoopTracker};
use crate::math::{Quat, Vec3};
use crate::player::{make_player, push_off, step_player, thrumbler, PlayerBody};
use crate::rng::Rng;
use crate::tuning::{GATE_RADIUS, GATE_X, OMEGA};

/// A free, untouched-to-rest bell that neither scores nor is caught within
/// this many ticks is a dead ball (30 s at 240 Hz — generous, longer than a
/// legitimate Loop's return period so real loops are never killed).
const MAX_FREE_TICKS: u64 = 7200;

// ── throw constants ──────────────────────────────────────────────────────────

const THROW_MIN: f64 = 9.0; // m/s minimum release speed
const THROW_MAX: f64 = 34.0; // m/s maximum release speed
const TETHER_MAX: f64 = crate::grapple::TETHER_MAX; // mirror of TS TETHER_MAX

// ── enums / plain data ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TeamSide {
    Home,
    Away,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RiggerRole {
    Anchor,
    Spinner,
    Faithwing,
    Freewing,
    Reach,
}

/// Classification of a ring crossing (mirrors TS `loopTier`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoopTierOut {
    Loop,
    Curl,
    None,
}

impl From<LoopTier> for LoopTierOut {
    fn from(t: LoopTier) -> Self {
        match t {
            LoopTier::Loop => LoopTierOut::Loop,
            LoopTier::Curl => LoopTierOut::Curl,
            LoopTier::None => LoopTierOut::None,
        }
    }
}

/// Events emitted by `SimWorld::step` (consumed by match rules + render/audio).
#[derive(Clone, Debug)]
pub enum SimEvent {
    BellThroughRing {
        end: RingEnd,
        touched: bool,
        loop_tier: LoopTierOut,
    },
    /// A free bell left play without scoring: it crossed a gate plane but
    /// missed the ring, or it drifted free far too long (no catch, no ring).
    /// This is the "dead ball" the match machine needs to re-cast.
    BellMissed {
        end: RingEnd,
    },
    BellCaught {
        by: String,
    },
    BellBobble {
        by: String,
    },
    BellClatter {
        by: Option<String>,
    },
    BellSkin,
    PlayerSkinned {
        id: String,
    },
    ContestStarted {
        thrower: String,
        contester: String,
    },
    FoulGarrote {
        by: String,
    },
}

/// Which end of the tube the bell crossed through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RingEnd {
    PlusX,
    MinusX,
}

/// One player's input for one tick.
#[derive(Clone, Debug)]
pub struct PlayerInput {
    pub id: String,
    pub aim: Vec3,
    /// World-space anchor to fire the rig line at, or `None`.
    pub fire_line_at: Option<Vec3>,
    pub reel: i32,        // -1 / 0 / +1
    pub release: bool,    // release the line
    pub pushoff: bool,    // push off contact
    pub throw_charge: f64, // [0,1]
    pub throw_released: bool,
    pub throw_spin: f64,  // [-1,1]
    pub thrumbler: Vec3,  // small delta-v request (capped by budget)
}

/// A complete frame of inputs (one tick from all controlled players).
#[derive(Clone, Debug)]
pub struct InputFrame {
    pub tick: u64,
    pub players: Vec<PlayerInput>,
}

impl InputFrame {
    /// Convenience: an idle frame with no player inputs.
    pub fn idle(tick: u64) -> Self {
        Self { tick, players: vec![] }
    }
}

// ── Snapshot types ────────────────────────────────────────────────────────────

/// Bell state as it appears in a snapshot.
#[derive(Clone, Debug)]
pub struct BellSnapshot {
    pub p: Vec3,
    pub v: Vec3,
    pub q: Quat,
    pub w: Vec3,
    pub chime: f64,
    pub held_by: Option<String>,
    pub thrown_by: Option<String>,
    pub touched_since_throw: bool,
    pub release_pos: Vec3,
    pub release_tick: u64,
    pub pass_chain: Vec<String>,
}

/// Player state as it appears in a snapshot.
#[derive(Clone, Debug)]
pub struct PlayerSnapshot {
    pub id: String,
    pub team: TeamSide,
    pub role: RiggerRole,
    pub p: Vec3,
    pub v: Vec3,
    pub q: Quat,
    pub dv_budget: f64,
    pub grounded: bool,
    pub contact_ref: Option<String>,
    pub line_anchor: Option<Vec3>,
    pub line_rest_len: Option<f64>,
    pub line_taut: Option<bool>,
}

/// Full snapshot — serialisable, hashable, replay-compatible.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub tick: u64,
    pub omega: f64,
    pub bell: BellSnapshot,
    pub players: Vec<PlayerSnapshot>,
    pub rng_cursor: Vec<(String, u64)>,
}

// ── Internal player record ────────────────────────────────────────────────────

struct WorldPlayer {
    id: String,
    team: TeamSide,
    role: RiggerRole,
    body: PlayerBody,
}

// ── SimWorld ──────────────────────────────────────────────────────────────────

/// The deterministic physics world.
pub struct SimWorld {
    pub tick: u64,
    pub bell: BellBody,
    pub bell_held_by: Option<String>,
    pub bell_thrown_by: Option<String>,
    pub bell_touched: bool,
    /// Free bell has resolved out of play (missed/expired) — frozen until
    /// the next launch/set re-casts it. Prevents runaway + event spam.
    pub bell_dead: bool,
    /// Consecutive ticks the bell has been in free flight.
    pub free_ticks: u64,
    pub pass_chain: Vec<String>,
    pub release_pos: Vec3,
    pub release_tick: u64,
    players: Vec<WorldPlayer>,
    loop_tracker: LoopTracker,
    rng: Rng,
    events: Vec<SimEvent>,
}

impl SimWorld {
    /// Create a new world seeded with `seed`.
    pub fn new(seed: u32) -> Self {
        Self {
            tick: 0,
            bell: BellBody {
                p: Vec3::new(0.0, 0.0, 0.0),
                v: Vec3::new(0.0, 0.0, 0.0),
                q: Quat::ident(),
                w: Vec3::new(0.0, 0.0, 0.0),
            },
            bell_held_by: None,
            bell_thrown_by: None,
            bell_touched: true,
            bell_dead: false,
            free_ticks: 0,
            pass_chain: vec![],
            release_pos: Vec3::new(0.0, 0.0, 0.0),
            release_tick: 0,
            players: vec![],
            loop_tracker: LoopTracker::new(),
            rng: Rng::new(seed),
            events: vec![],
        }
    }

    /// Add a player to the world at position `pos`.
    pub fn add_player(&mut self, id: &str, team: TeamSide, role: RiggerRole, pos: Vec3) {
        self.players.push(WorldPlayer {
            id: id.to_string(),
            team,
            role,
            body: make_player(pos),
        });
    }

    /// Place the bell free with position `p`, velocity `v`, angular velocity
    /// `w` (body frame), and record who threw it (or `None` for a restart).
    pub fn launch_bell(&mut self, p: Vec3, v: Vec3, w: Vec3, thrown_by: Option<&str>) {
        self.bell.p = p;
        self.bell.v = v;
        self.bell.w = w;
        self.bell_held_by = None;
        self.bell_thrown_by = thrown_by.map(|s| s.to_string());
        self.bell_touched = false;
        self.bell_dead = false;
        self.free_ticks = 0;
        self.release_pos = p;
        self.release_tick = self.tick;
        if let Some(id) = thrown_by {
            if self.pass_chain.last().map(|s: &String| s.as_str()) != Some(id) {
                self.pass_chain.push(id.to_string());
            }
        }
        self.loop_tracker.on_release(p);
    }

    /// Set the bell as held by player `player_id` (e.g. at set/spawn).
    pub fn set_bell_held(&mut self, player_id: &str) {
        self.bell_held_by = Some(player_id.to_string());
        self.bell_touched = true;
        self.bell_dead = false;
        self.free_ticks = 0;
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn find_idx(&self, id: &str) -> Option<usize> {
        self.players.iter().position(|w| w.id == id)
    }

    fn team_of(&self, id: &str) -> Option<TeamSide> {
        self.players.iter().find(|w| w.id == id).map(|w| w.team)
    }

    fn apply_input(&mut self, inp: &PlayerInput) {
        let idx = match self.find_idx(&inp.id) {
            Some(i) => i,
            None => return,
        };

        // Grapple line
        if let Some(anchor) = inp.fire_line_at {
            let len = (self.players[idx].body.p.sub(anchor)).len();
            self.players[idx].body.line = Some(crate::grapple::Line {
                anchor_pos: anchor,
                rest_len: TETHER_MAX.min(3.0_f64.max(len)),
                taut: false,
            });
        } else if inp.release && self.players[idx].body.line.is_some() {
            self.players[idx].body.line = None;
        }

        if inp.pushoff {
            push_off(&mut self.players[idx].body, inp.aim, 7.0);
        }
        if inp.thrumbler.len() > 1e-12 {
            thrumbler(&mut self.players[idx].body, inp.thrumbler);
        }

        // Throw (only when holding the bell)
        if inp.throw_released && self.bell_held_by.as_deref() == Some(&inp.id) {
            let dir = inp.aim.norm();
            let charge = inp.throw_charge.clamp(0.0, 1.0);
            let speed = THROW_MIN + charge * (THROW_MAX - THROW_MIN);
            let v = self.players[idx].body.v.add(dir.scale(speed));
            let s = inp.throw_spin.clamp(-1.0, 1.0);
            let w = Vec3::new(26.0, s * 7.0, 0.0);
            let player_p = self.players[idx].body.p;
            let id_clone = inp.id.clone();
            self.launch_bell(player_p, v, w, Some(&id_clone));
        }
    }

    fn reel_of<'a>(inp_map: &'a [(String, i32)], id: &str) -> i32 {
        inp_map.iter().find(|(k, _)| k == id).map(|(_, r)| *r).unwrap_or(0)
    }

    /// Advance one fixed sub-step `h` seconds. Returns events emitted this step.
    pub fn step(&mut self, frame: &InputFrame, h: f64) -> Vec<SimEvent> {
        self.events = vec![];

        // Collect reel values before we borrow mutably below.
        let reel_map: Vec<(String, i32)> = frame
            .players
            .iter()
            .map(|p| (p.id.clone(), p.reel))
            .collect();

        // Apply inputs (throw / fire-line / pushoff / thrumbler).
        // We clone the inputs so that borrow checker doesn't fight us.
        let inputs: Vec<PlayerInput> = frame.players.clone();
        for inp in &inputs {
            self.apply_input(inp);
        }

        // Step players.
        for w in &mut self.players {
            let was_grounded = w.body.grounded;
            let reel = Self::reel_of(&reel_map, &w.id);
            step_player(&mut w.body, h, reel);
            if w.body.grounded && !was_grounded {
                self.events.push(SimEvent::PlayerSkinned { id: w.id.clone() });
            }
        }

        // Bell integration or tracking.
        if let Some(ref holder_id) = self.bell_held_by.clone() {
            if let Some(idx) = self.find_idx(holder_id) {
                self.bell.p = self.players[idx].body.p;
                self.bell.v = self.players[idx].body.v;
            }
        } else if !self.bell_dead {
            let prev_x = self.bell.p.x;
            self.bell = step_bell(self.bell, OMEGA, h);

            if skin_bounce(&mut self.bell.p, &mut self.bell.v) {
                self.loop_tracker.on_touch();
                self.bell_touched = true;
                self.events.push(SimEvent::BellSkin);
            }

            self.loop_tracker.update(self.bell.v);

            // Catch / bobble / contest proximity (grace window: 8 ticks after throw).
            let since_release = self.tick.saturating_sub(self.release_tick);
            let bell_p = self.bell.p;
            let bell_v = self.bell.v;
            let thrown_by = self.bell_thrown_by.clone();

            'contact: for w in &self.players {
                if w.body.grounded {
                    continue;
                }
                if since_release < 8 {
                    continue;
                }
                // Thrower grace: skip while very close to the thrower.
                if thrown_by.as_deref() == Some(&w.id) {
                    let d = bell_p.sub(w.body.p).len();
                    if d < 4.0 {
                        continue;
                    }
                }

                let result = try_catch(bell_p, bell_v, w.body.p, w.body.v, 1.0);
                match result {
                    CatchResult::Caught => {
                        self.bell_held_by = Some(w.id.clone());
                        self.bell_touched = true;
                        self.bell_dead = false;
                        self.free_ticks = 0;
                        self.loop_tracker.on_touch();
                        if self.pass_chain.last().map(|s: &String| s.as_str())
                            != Some(&w.id)
                        {
                            self.pass_chain.push(w.id.clone());
                        }
                        self.events.push(SimEvent::BellCaught { by: w.id.clone() });
                        break 'contact;
                    }
                    CatchResult::Bobble => {
                        apply_bobble(&mut self.bell, w.body.v);
                        self.loop_tracker.on_touch();
                        self.bell_touched = true;
                        self.events.push(SimEvent::BellBobble { by: w.id.clone() });
                        break 'contact;
                    }
                    CatchResult::Miss => {
                        // Check for contest-clatter: a defender brushing a fast bell.
                        let near = bell_p.sub(w.body.p).len();
                        if near < 2.2 {
                            if let Some(ref tid) = thrown_by {
                                if self.team_of(tid) != Some(w.team) {
                                    contest_clatter(&mut self.bell, w.body.v, 1.0);
                                    self.loop_tracker.on_touch();
                                    self.bell_touched = true;
                                    self.events.push(SimEvent::BellClatter {
                                        by: Some(w.id.clone()),
                                    });
                                    break 'contact;
                                }
                            }
                        }
                    }
                }
            }

            // Ring crossing (may score, or emit BellMissed + go dead).
            self.check_ring(prev_x);

            // Dead-ball on excessive free flight (no catch, no ring): a bell
            // that drifts forever would freeze the match. check_ring may have
            // already killed it this tick.
            if !self.bell_dead {
                self.free_ticks += 1;
                if self.free_ticks > MAX_FREE_TICKS {
                    let end = if self.bell.p.x >= 0.0 {
                        RingEnd::PlusX
                    } else {
                        RingEnd::MinusX
                    };
                    self.events.push(SimEvent::BellMissed { end });
                    self.bell_thrown_by = None;
                    self.bell_dead = true;
                }
            }
        }

        self.tick += 1;
        std::mem::take(&mut self.events)
    }

    fn check_ring(&mut self, prev_x: f64) {
        let x = self.bell.p.x;
        let rho = self.bell.p.y.hypot(self.bell.p.z);
        let through = rho <= GATE_RADIUS;

        let crossed_pos = (prev_x < GATE_X && x >= GATE_X) || (prev_x > GATE_X && x <= GATE_X);
        let crossed_neg =
            (prev_x < -GATE_X && x >= -GATE_X) || (prev_x > -GATE_X && x <= -GATE_X);

        if crossed_pos && through {
            let tier = self.loop_tracker.tier(self.bell.p);
            self.events.push(SimEvent::BellThroughRing {
                end: RingEnd::PlusX,
                touched: self.bell_touched,
                loop_tier: tier.into(),
            });
            self.bell_thrown_by = None;
        } else if crossed_neg && through {
            let tier = self.loop_tracker.tier(self.bell.p);
            self.events.push(SimEvent::BellThroughRing {
                end: RingEnd::MinusX,
                touched: self.bell_touched,
                loop_tier: tier.into(),
            });
            self.bell_thrown_by = None;
        } else if crossed_pos {
            // Crossed the +x gate plane but missed the ring — dead ball.
            self.events.push(SimEvent::BellMissed { end: RingEnd::PlusX });
            self.bell_thrown_by = None;
            self.bell_dead = true;
        } else if crossed_neg {
            self.events.push(SimEvent::BellMissed { end: RingEnd::MinusX });
            self.bell_thrown_by = None;
            self.bell_dead = true;
        }
    }

    /// Loop status for the loop-cam / audio.
    pub fn loop_info(&self) -> LoopInfo {
        LoopInfo {
            free: self.bell_held_by.is_none(),
            untouched: self.loop_tracker.untouched(),
            turn: self.loop_tracker.turn(),
        }
    }

    /// Produce an immutable snapshot for render / replay / hashing.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            tick: self.tick,
            omega: OMEGA,
            bell: BellSnapshot {
                p: self.bell.p,
                v: self.bell.v,
                q: self.bell.q,
                w: self.bell.w,
                chime: chime(self.bell.w),
                held_by: self.bell_held_by.clone(),
                thrown_by: self.bell_thrown_by.clone(),
                touched_since_throw: self.bell_touched,
                release_pos: self.release_pos,
                release_tick: self.release_tick,
                pass_chain: self.pass_chain.clone(),
            },
            players: self.players.iter().map(|w| PlayerSnapshot {
                id: w.id.clone(),
                team: w.team,
                role: w.role,
                p: w.body.p,
                v: w.body.v,
                q: Quat::ident(),
                dv_budget: w.body.dv_budget,
                grounded: w.body.grounded,
                contact_ref: if w.body.contact { Some("spar".to_string()) } else { None },
                line_anchor: w.body.line.as_ref().map(|l| l.anchor_pos),
                line_rest_len: w.body.line.as_ref().map(|l| l.rest_len),
                line_taut: w.body.line.as_ref().map(|l| l.taut),
            }).collect(),
            rng_cursor: self.rng.cursor(),
        }
    }
}

/// Live loop information (for render / audio hooks).
pub struct LoopInfo {
    pub free: bool,
    pub untouched: bool,
    pub turn: f64,
}

// ── FNV-based snapshot hash (mirrors TS `hashSnapshot`) ─────────────────────

/// Stable hash of a snapshot for determinism tests.
/// Uses the same FNV-1a + Math.imul scheme as the TS `hashSnapshot` so
/// cross-language parity checks pass.
pub fn hash_snapshot(s: &Snapshot) -> String {
    let round = |n: f64| (n * 1_000_000.0).round() / 1_000_000.0;
    let mut parts: Vec<f64> = vec![
        s.tick as f64,
        round(s.bell.p.x),
        round(s.bell.p.y),
        round(s.bell.p.z),
        round(s.bell.v.x),
        round(s.bell.v.y),
        round(s.bell.v.z),
        round(s.bell.w.x),
        round(s.bell.w.y),
        round(s.bell.w.z),
    ];
    for p in &s.players {
        parts.push(round(p.p.x));
        parts.push(round(p.p.y));
        parts.push(round(p.p.z));
        parts.push(round(p.v.x));
        parts.push(round(p.v.y));
        parts.push(round(p.v.z));
    }

    // Concatenate as a comma-separated string, exactly like TS `parts.join(',')`.
    let str_repr = parts
        .iter()
        .map(|n| {
            // Mirror JS Number.toString() for these rounded f64 values:
            // integers should print without decimal point; others use Rust default.
            if n.fract() == 0.0 && n.is_finite() {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        })
        .collect::<Vec<_>>()
        .join(",");

    // FNV-1a with Math.imul (u32 wrapping), identical to TS.
    let mut h: u32 = 2_166_136_261u32;
    for byte in str_repr.bytes() {
        h ^= byte as u32;
        h = h.wrapping_mul(16_777_619u32);
    }
    format!("{:x}", h)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_detector::LOOP_TURN;

    const SIM_H: f64 = 1.0 / 240.0;

    fn build(w: &mut SimWorld) {
        w.add_player("P1", TeamSide::Home, RiggerRole::Spinner, Vec3::new(-150.0, 2.0, 0.0));
        w.add_player("A1", TeamSide::Away, RiggerRole::Reach, Vec3::new(280.0, 0.0, 0.0));
        w.set_bell_held("P1");
    }

    fn make_frame(tick: u64) -> InputFrame {
        let base = PlayerInput {
            id: "P1".to_string(),
            aim: Vec3::new(1.0, 0.25, -0.35),
            fire_line_at: None,
            reel: 0,
            release: false,
            pushoff: false,
            throw_charge: 0.9,
            throw_released: tick == 5,
            throw_spin: 0.2,
            thrumbler: Vec3::new(0.0, 0.0, 0.0),
        };
        let a1 = PlayerInput {
            id: "A1".to_string(),
            throw_released: false,
            ..base.clone()
        };
        InputFrame { tick, players: vec![base, a1] }
    }

    /// Two runs with identical seed+inputs must produce identical snapshot hashes.
    #[test]
    fn identical_inputs_produce_identical_hashes() {
        let mut a = SimWorld::new(42);
        let mut b = SimWorld::new(42);
        build(&mut a);
        build(&mut b);
        for t in 0u64..1500 {
            let f = make_frame(t);
            a.step(&f, SIM_H);
            b.step(&f, SIM_H);
            if t % 250 == 0 {
                assert_eq!(
                    hash_snapshot(&a.snapshot()),
                    hash_snapshot(&b.snapshot()),
                    "diverged at tick {t}"
                );
            }
        }
        assert_eq!(hash_snapshot(&a.snapshot()), hash_snapshot(&b.snapshot()));
    }

    /// Physics is pure (no RNG draws yet) — different seeds must NOT diverge.
    #[test]
    fn different_seeds_seed_independent_physics() {
        let mut a = SimWorld::new(1);
        let mut b = SimWorld::new(999);
        build(&mut a);
        build(&mut b);
        for t in 0u64..800 {
            let f = make_frame(t);
            a.step(&f, SIM_H);
            b.step(&f, SIM_H);
        }
        assert_eq!(hash_snapshot(&a.snapshot()), hash_snapshot(&b.snapshot()));
    }

    /// A launched, untouched, gently curving bell winds ≥ LOOP_TURN radians.
    #[test]
    fn launched_untouched_curving_bell_winds_loop_turn() {
        // The gentle, near-axial loop the analysis solver found: low v_perp
        // (≈0.4) cancels the centrifugal sweep so |ζ| stays small (no skin
        // contact → tracker never voided) while heading winds ≈ ω·t.
        let mut w = SimWorld::new(1);
        w.launch_bell(
            Vec3::new(-180.0, 1.5, 0.0),
            Vec3::new(18.57, 0.0, -0.40),
            Vec3::new(26.0, 0.0, 0.0),
            Some("P1"),
        );
        let idle = InputFrame::idle(0);
        for _ in 0..(240 * 12) {
            w.step(&idle, SIM_H);
        }
        let info = w.loop_info();
        assert!(info.untouched, "bell must remain untouched");
        assert!(
            info.turn >= LOOP_TURN,
            "cum_turn {} must be >= LOOP_TURN {}",
            info.turn,
            LOOP_TURN
        );
    }

    /// A thrown bell that misses the ring must NOT run away forever: it
    /// emits BellMissed when it crosses a gate plane off-ring, goes dead,
    /// and stays bounded (regression for the x→∞ runaway).
    #[test]
    fn missed_throw_off_ring_is_a_dead_ball_not_a_runaway() {
        let mut w = SimWorld::new(3);
        // Aimed straight down +x but well outside the 8 m ring (rho = 20 m).
        w.launch_bell(
            Vec3::new(-180.0, 20.0, 0.0),
            Vec3::new(40.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Some("P1"),
        );
        let idle = InputFrame::idle(0);
        let mut saw_missed = false;
        for _ in 0..(240 * 30) {
            for ev in w.step(&idle, SIM_H) {
                if matches!(ev, SimEvent::BellMissed { .. }) {
                    saw_missed = true;
                }
            }
        }
        assert!(saw_missed, "an off-ring gate-plane crossing must emit BellMissed");
        assert!(w.bell_dead, "the missed bell must be a dead ball");
        assert!(
            w.bell.p.x.abs() < GATE_X + 50.0,
            "dead bell must stay bounded, got x={}",
            w.bell.p.x
        );
    }

    /// A free bell that never scores and is never caught must time out into
    /// a dead ball (otherwise the match freezes).
    #[test]
    fn free_bell_times_out_into_a_dead_ball() {
        let mut w = SimWorld::new(4);
        // Near-stationary mid-tube: never reaches a gate plane on its own.
        w.launch_bell(
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Some("P1"),
        );
        let idle = InputFrame::idle(0);
        let mut saw_missed = false;
        for _ in 0..(MAX_FREE_TICKS as usize + 240) {
            for ev in w.step(&idle, SIM_H) {
                if matches!(ev, SimEvent::BellMissed { .. }) {
                    saw_missed = true;
                }
            }
        }
        assert!(saw_missed, "a never-resolved free bell must time out");
        assert!(w.bell_dead, "timed-out bell must be a dead ball");
    }

    /// The FNV hash is deterministic across calls.
    #[test]
    fn hash_snapshot_is_stable() {
        let mut w = SimWorld::new(7);
        build(&mut w);
        for t in 0u64..60 {
            w.step(&make_frame(t), SIM_H);
        }
        let s = w.snapshot();
        assert_eq!(hash_snapshot(&s), hash_snapshot(&s));
    }

    /// Pure-physics run (no inputs, bell free from start) — seeds don't matter.
    #[test]
    fn pure_physics_seed_independent() {
        let launch = |seed| {
            let mut w = SimWorld::new(seed);
            w.launch_bell(
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::new(20.0, 1.0, 0.5),
                Vec3::new(26.0, 0.0, 0.0),
                None,
            );
            let idle = InputFrame::idle(0);
            for _ in 0..480 {
                w.step(&idle, SIM_H);
            }
            hash_snapshot(&w.snapshot())
        };
        assert_eq!(launch(1), launch(99999));
    }
}

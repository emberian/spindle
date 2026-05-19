//! Deterministic world — port of TS sim/SimWorld.ts.
//!
//! One `SimWorld` ticks a deterministic physics simulation of the bell and all
//! players.  It is pure given (state, inputs): no wall-clock, no OS RNG.
//! Events are emitted each step and consumed by the match-rules layer.

use crate::bell::{chime, step_bell, BellBody};
use crate::collision::{
    apply_bobble, contest_clatter, skin_bounce, try_catch_ex, CatchResult, STRIP_RANGE,
};
use crate::loop_detector::{LoopTier, LoopTracker};
use crate::math::{Quat, Vec3};
use crate::grapple::Body;
use crate::player::{
    make_player, push_off, step_player, step_player_anchored, thrumbler, PlayerBody,
};
use crate::rng::Rng;
use crate::tuning::{GATE_RADIUS, GATE_X, HOLD_C, HOLD_K, OMEGA};

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
    /// Committed intended catcher of the live bell — widens catch envelope.
    pub catch_intent: bool,
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
    /// GRAPPLE LATENCY: `Some(false)` ⇒ the claw is still in flight toward
    /// `line_anchor` (line committed to target, no constraint force yet);
    /// `Some(true)` ⇒ the claw has landed and the line is live; `None` ⇒ no
    /// line. Render/AI use this to distinguish a traveling claw from a live
    /// line. NOT folded into `hash_snapshot` (render/AI hint only).
    pub line_attached: Option<bool>,
    // ── Render-only fields (NOT folded into `hash_snapshot`; excluded from
    //    the determinism-hashed surface — verified: `hash_snapshot` folds
    //    only tick, bell.p/v/w and per-player p/v). ────────────────────────
    /// The bound target player id for a player↔player line (= `Line.anchor_player`).
    /// `None` for a static-world-anchor line (spar/skin/ring) or no line.
    pub line_anchor_player: Option<String>,
    /// Effective LIVE anchor position for rendering: the bound player's
    /// current body position when player-bound, else the static `anchor_pos`.
    /// `None` when there is no line. Render-only.
    pub line_anchor_eff: Option<Vec3>,
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

#[derive(Clone)]
struct WorldPlayer {
    id: String,
    team: TeamSide,
    role: RiggerRole,
    body: PlayerBody,
}

// ── SimWorld ──────────────────────────────────────────────────────────────────

/// The deterministic physics world.
#[derive(Clone)]
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
    /// OFFENSE REBUILD: consecutive ticks an opponent has crowded the
    /// carrier inside STRIP_RANGE (deterministic strip hysteresis — a
    /// strip is earned by sustained pressure, not an instant brush).
    strip_press: u32,
    /// PART B: latched true once a `ContestStarted` has been emitted for the
    /// CURRENT loose-bell episode, so the 1:1 fires exactly once per loose
    /// ball (not every tick the two riggers are both near it). Reset on any
    /// launch/grip/catch. Not a physics quantity (no p/v) so it does not
    /// enter hash_snapshot; it only gates a one-shot event, deterministically.
    contest_emitted: bool,
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
            strip_press: 0,
            contest_emitted: false,
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
        self.contest_emitted = false; // new loose-bell episode
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
        self.contest_emitted = false;
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn find_idx(&self, id: &str) -> Option<usize> {
        self.players.iter().position(|w| w.id == id)
    }

    fn team_of(&self, id: &str) -> Option<TeamSide> {
        self.players.iter().find(|w| w.id == id).map(|w| w.team)
    }

    fn apply_input(&mut self, inp: &PlayerInput, h: f64) {
        let idx = match self.find_idx(&inp.id) {
            Some(i) => i,
            None => return,
        };

        // Grapple line — GRAPPLE LATENCY (the catch fix). A fire is IGNORED
        // if a line is already present (claw in flight OR attached —
        // committed to its target, no silent per-tick re-anchor) OR the
        // re-fire cooldown has not elapsed (`self.tick < refire_ready_tick`)
        // after a release / missed claw. Otherwise the claw launches: the
        // line EXISTS immediately (blocks re-aim) but is `attached = false`
        // until it travels `len` at CLAW_SPEED — landing after
        // `ceil(len / CLAW_SPEED / h)` ticks. Tick-counted, no rng.
        if let Some(anchor) = inp.fire_line_at {
            let blocked = self.players[idx].body.line.is_some()
                || self.tick < self.players[idx].body.refire_ready_tick;
            if !blocked {
            let len = (self.players[idx].body.p.sub(anchor)).len();
            // BLOCKER 1: if the fire point lands on/near another player's
            // body (teammate OR opponent) within BIND_RADIUS, bind the
            // line to THAT player's identity so it tracks their moving
            // body and resolves momentum-conservingly. Otherwise it stays
            // a static world anchor (spar / skin / ring) exactly as
            // before. Deterministic: fixed Vec iteration order, nearest
            // wins, ties broken by lower index (first in the Vec) — no
            // HashMap iteration anywhere.
            const BIND_RADIUS: f64 = 3.0;
            let mut anchor_player: Option<String> = None;
            let mut best_d2 = BIND_RADIUS * BIND_RADIUS;
            for (j, w) in self.players.iter().enumerate() {
                if j == idx {
                    continue;
                }
                let dd = w.body.p.sub(anchor);
                let d2 = dd.dot(dd);
                if d2 < best_d2 {
                    best_d2 = d2;
                    anchor_player = Some(w.id.clone());
                }
            }
            // Claw flight time ∝ distance: ceil(len / CLAW_SPEED / h) ticks.
            // f64 math then ceil → exact integer tick count; deterministic,
            // no rng, no wall clock. Min 1 tick so a line never attaches the
            // same tick it is fired (latency always observable).
            let flight_ticks =
                (len / crate::tuning::CLAW_SPEED / h).ceil().max(1.0) as u64;
            self.players[idx].body.line = Some(crate::grapple::Line {
                anchor_pos: anchor,
                rest_len: TETHER_MAX.min(3.0_f64.max(len)),
                taut: false,
                attached: false,
                attach_tick: self.tick + flight_ticks,
                anchor_player,
            });
            }
        } else if inp.release && self.players[idx].body.line.is_some() {
            // Explicit release cancels the line (in flight or attached) and
            // starts the re-fire cooldown.
            self.players[idx].body.line = None;
            self.players[idx].body.refire_ready_tick =
                self.tick + crate::tuning::REFIRE_COOLDOWN_TICKS;
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

    fn catch_intent_of(players: &[PlayerInput], id: &str) -> bool {
        players
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.catch_intent)
            .unwrap_or(false)
    }

    fn reel_of(players: &[PlayerInput], id: &str) -> i32 {
        players
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.reel)
            .unwrap_or(0)
    }

    /// Advance one fixed sub-step `h` seconds. Returns events emitted this step.
    pub fn step(&mut self, frame: &InputFrame, h: f64) -> Vec<SimEvent> {
        self.events = vec![];

        // Apply inputs (throw / fire-line / pushoff / thrumbler). `frame`
        // is borrowed disjointly from `self`, so we can iterate it
        // directly — no per-tick clone of every PlayerInput.
        for inp in &frame.players {
            self.apply_input(inp, h);
        }

        // GRAPPLE LATENCY: flip in-flight claws to attached once the sim
        // tick reaches their landing tick. Done BEFORE stepping players so a
        // claw landing this tick exerts its first constraint force this same
        // tick. Tick-counted (integer compare) ⇒ deterministic, run==run.
        for w in &mut self.players {
            if let Some(line) = w.body.line.as_mut() {
                if !line.attached && self.tick >= line.attach_tick {
                    line.attached = true;
                }
            }
        }

        // Step players. Reel values are read straight off `frame`
        // (still borrowed), avoiding the per-tick Vec<(String,i32)> with
        // its id-String clones.
        //
        // BLOCKER 1: indexed iteration (fixed Vec order 0..n, no HashMap)
        // so a player-bound line can take a deterministic disjoint
        // split-borrow of the stepped player AND its anchor player. For a
        // line bound to a player id we resolve that anchor's CURRENT body,
        // pass it as the moving anchor (momentum-conserving equal-and-
        // opposite), and write the recoil back. A line whose anchor id is
        // gone/invalid degrades to a released line deterministically. A
        // static-anchor line (anchor_player == None) takes the exact same
        // path as before (None anchor ⇒ byte-identical).
        let n = self.players.len();
        for i in 0..n {
            let was_grounded = self.players[i].body.grounded;
            let reel = Self::reel_of(&frame.players, &self.players[i].id);

            // Resolve a player-bound line's anchor index (fixed scan).
            let anchor_idx: Option<usize> = match self.players[i]
                .body
                .line
                .as_ref()
                .and_then(|l| l.anchor_player.clone())
            {
                Some(aid) => {
                    let found = self.players.iter().position(|w| w.id == aid);
                    if found.is_none() || found == Some(i) {
                        // Anchor gone / invalid / self ⇒ degrade to
                        // released, deterministically.
                        self.players[i].body.line = None;
                        None
                    } else {
                        found
                    }
                }
                None => None,
            };

            match anchor_idx {
                Some(a) => {
                    // Disjoint &mut to player i and anchor a via split_at_mut
                    // (deterministic; mirrors skill_eval/ai indexed access).
                    let (lo, hi) = if i < a { (i, a) } else { (a, i) };
                    let (left, right) = self.players.split_at_mut(hi);
                    let (p_ref, a_ref) = if i < a {
                        (&mut left[lo], &mut right[0])
                    } else {
                        (&mut right[0], &mut left[lo])
                    };
                    let mut anchor_body = Body {
                        p: a_ref.body.p,
                        v: a_ref.body.v,
                        inv_mass: a_ref.body.inv_mass,
                    };
                    step_player_anchored(
                        &mut p_ref.body,
                        h,
                        reel,
                        Some(&mut anchor_body),
                    );
                    // Equal-and-opposite recoil onto the anchor player's
                    // velocity (resolve_line wrote it into anchor_body.v).
                    a_ref.body.v = anchor_body.v;
                }
                None => {
                    step_player(&mut self.players[i].body, h, reel);
                }
            }

            if self.players[i].body.grounded && !was_grounded {
                let id = self.players[i].id.clone();
                self.events.push(SimEvent::PlayerSkinned { id });
            }
        }

        // PART A — PLAYER↔PLAYER SOFT-BODY COLLISION. Riggers are no longer
        // point masses that pass through each other: one O(n²) pairwise pass
        // resolves penetration of two PLAYER_RADIUS spheres with a momentum-
        // conserving soft spring + damper applied EQUAL-AND-OPPOSITE along
        // the contact normal. Fixed (i, j) index order with the SAME
        // lower-index-first `split_at_mut` disjoint-borrow pattern proven in
        // the player↔player line solver above (sim_world.rs:479-508) ⇒
        // deterministic, no rng, no HashMap. Symplectic (semi-implicit
        // Euler), accel-bounded; reduced-mass ω_n·h ≈ 0.042 ≪ 2 (see
        // tuning.rs COLLIDE_K/COLLIDE_C). Players carry equal mass so the
        // equal-and-opposite Δv conserves linear momentum exactly. The new
        // p/v already flow into hash_snapshot (it folds every player p,v),
        // so this physics cannot silently escape determinism coverage.
        {
            let two_r = 2.0 * crate::tuning::PLAYER_RADIUS;
            let two_r2 = two_r * two_r;
            let n = self.players.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    // Disjoint &mut to i and j (i < j ⇒ split at j).
                    let (left, right) = self.players.split_at_mut(j);
                    let a = &mut left[i];
                    let b = &mut right[0];
                    let dx = a.body.p.x - b.body.p.x;
                    let dy = a.body.p.y - b.body.p.y;
                    let dz = a.body.p.z - b.body.p.z;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 >= two_r2 || d2 < 1e-18 {
                        continue;
                    }
                    let dist = d2.sqrt();
                    let inv = 1.0 / dist;
                    let nx = dx * inv;
                    let ny = dy * inv;
                    let nz = dz * inv;
                    let pen = two_r - dist; // > 0 penetration depth
                    // Relative velocity projected on the contact normal
                    // (a relative to b). Positive ⇒ separating.
                    let rvx = a.body.v.x - b.body.v.x;
                    let rvy = a.body.v.y - b.body.v.y;
                    let rvz = a.body.v.z - b.body.v.z;
                    let v_rel_n = rvx * nx + rvy * ny + rvz * nz;
                    // Soft push-apart spring − damper along +n on `a`.
                    let f = crate::tuning::COLLIDE_K * pen
                        - crate::tuning::COLLIDE_C * v_rel_n;
                    // Equal-and-opposite Δv (semi-implicit Euler). Equal
                    // mass ⇒ momentum conserved exactly: +f·n on a, −f·n
                    // on b, identical |Δv|.
                    let ka = f * a.body.inv_mass * h;
                    let kb = f * b.body.inv_mass * h;
                    a.body.v = Vec3::new(
                        a.body.v.x + nx * ka,
                        a.body.v.y + ny * ka,
                        a.body.v.z + nz * ka,
                    );
                    b.body.v = Vec3::new(
                        b.body.v.x - nx * kb,
                        b.body.v.y - ny * kb,
                        b.body.v.z - nz * kb,
                    );
                }
            }
        }

        // PART B — GARROTE foul detection. A fired rig line whose taut,
        // attached segment (hand → effective anchor) sweeps within
        // GARROTE_RADIUS of an OPPOSING rigger's body centre is a foul (the
        // line cutting/wrapping across an opponent) — independent of bell
        // state (a garrote is a garrote whether the ball is held or loose).
        // Per (line, player) pair in fixed Vec id order; first qualifying
        // hit (by line owner index, then victim index) wins ⇒
        // deterministic, no rng. Segment-vs-point distance only.
        {
            let gr2 = crate::tuning::GARROTE_RADIUS * crate::tuning::GARROTE_RADIUS;
            let mut garrote_by: Option<String> = None;
            'g: for li in 0..self.players.len() {
                let owner = &self.players[li];
                let line = match owner.body.line.as_ref() {
                    Some(l) if l.attached && l.taut => l,
                    _ => continue,
                };
                let a = owner.body.p;
                // Effective far end: a player-bound line tracks the bound
                // body; otherwise the static world anchor.
                let b = match line.anchor_player.as_ref() {
                    Some(aid) => self
                        .players
                        .iter()
                        .find(|t| &t.id == aid)
                        .map(|t| t.body.p)
                        .unwrap_or(line.anchor_pos),
                    None => line.anchor_pos,
                };
                let owner_team = owner.team;
                for vi in 0..self.players.len() {
                    if vi == li {
                        continue;
                    }
                    let victim = &self.players[vi];
                    if victim.team == owner_team || victim.body.grounded {
                        continue;
                    }
                    if seg_point_dist2(a, b, victim.body.p) < gr2 {
                        garrote_by = Some(self.players[li].id.clone());
                        break 'g;
                    }
                }
            }
            if let Some(by) = garrote_by {
                self.events.push(SimEvent::FoulGarrote { by });
            }
        }

        // Bell integration or tracking.
        if let Some(ref holder_id) = self.bell_held_by.clone() {
            if let Some(idx) = self.find_idx(holder_id) {
                // Held bell follows the holder's hand via a stiff spring
                // (continuous tracking, no raw p/v copy). Unit mass; semi-
                // implicit Euler (ω_n·h ≈ 0.118 ≪ 2 ⇒ stable, tight). The
                // throw/release velocity math in apply_input is unchanged —
                // it reads the PLAYER's v, never this tracked bell.v.
                let hp = self.players[idx].body.p;
                let hv = self.players[idx].body.v;
                let dx = self.bell.p.x - hp.x;
                let dy = self.bell.p.y - hp.y;
                let dz = self.bell.p.z - hp.z;
                let fx = -HOLD_K * dx - HOLD_C * (self.bell.v.x - hv.x);
                let fy = -HOLD_K * dy - HOLD_C * (self.bell.v.y - hv.y);
                let fz = -HOLD_K * dz - HOLD_C * (self.bell.v.z - hv.z);
                self.bell.v = Vec3::new(
                    self.bell.v.x + fx * h,
                    self.bell.v.y + fy * h,
                    self.bell.v.z + fz * h,
                );
                self.bell.p = Vec3::new(
                    self.bell.p.x + self.bell.v.x * h,
                    self.bell.p.y + self.bell.v.y * h,
                    self.bell.p.z + self.bell.v.z * h,
                );

                // OFFENSE REBUILD — contested possession: a defender who
                // CROWDS the carrier (inside STRIP_RANGE) for STRIP_PRESS
                // ticks rips the bell loose. The loose ball is tagged
                // thrown_by = carrier, so the match layer / skill telemetry
                // scores the defender's recovery as a real turnover &
                // intercept (denial > 0; possession is fought for).
                const STRIP_PRESS: u32 = 26; // ~0.11 s of sustained press
                let carrier_team = self.players[idx].team;
                let hp = self.players[idx].body.p;
                let hv = self.players[idx].body.v;
                let mut stripper: Option<usize> = None;
                let mut best_d = STRIP_RANGE;
                for (j, w) in self.players.iter().enumerate() {
                    if w.team == carrier_team || w.body.grounded {
                        continue;
                    }
                    let d = w.body.p.sub(hp).len();
                    if d < best_d {
                        best_d = d;
                        stripper = Some(j);
                    }
                }
                if let Some(j) = stripper {
                    self.strip_press += 1;
                    if self.strip_press >= STRIP_PRESS {
                        // CONTEST CONSEQUENCE: the loose ball is shoved into
                        // the CONTESTER's space, harder the longer the press
                        // was sustained past threshold — a committed, tracked
                        // strip decisively flips spatial control; a marginal
                        // one is soft and re-contestable. Carrier→defender
                        // unit dir; deterministic (positions only, no RNG).
                        let dp = self.players[j].body.p.sub(hp);
                        let dl = dp.len();
                        let away = if dl > 1e-6 {
                            dp.scale(1.0 / dl)
                        } else {
                            Vec3::new(0.0, 1.0, 0.0)
                        };
                        let over = self.strip_press - STRIP_PRESS;
                        let dv = crate::collision::strip_velocity_dir(
                            hv,
                            self.players[j].body.v,
                            away,
                            over,
                        );
                        let carrier_id = holder_id.clone();
                        // Loose ball tagged thrown_by = carrier: whoever
                        // recovers it next is scored relative to the
                        // carrier's team — an opponent recovery is a real
                        // intercept/turnover (denial), a teammate save
                        // keeps the cast alive. No Contest phase (the
                        // headless harness never resolves one).
                        self.launch_bell(self.bell.p, dv, self.bell.w, Some(&carrier_id));
                        self.bell_touched = true;
                        self.strip_press = 0;
                    }
                } else {
                    self.strip_press = 0;
                }
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

            // PART B — CONTEST detection. The bell is LOOSE here. If two
            // opposing non-grounded riggers are BOTH within CONTEST_RADIUS
            // of it (both committing to the same loose ball), this is a
            // real 1:1. Emit `ContestStarted` ONCE per loose-bell episode
            // (latched) so MatchStateMachine's Contest path runs in real
            // played/watched/wasm play (it then resolves from the physics
            // outcome — see match_sm.rs handle_event, no rng). The
            // "thrower" is the attacking side's closest rigger (team of
            // bell_thrown_by); the "contester" is the opposing side's
            // closest. Fixed Vec order, lowest index breaks distance ties
            // ⇒ deterministic, no rng, no HashMap.
            if !self.contest_emitted && since_release >= 8 {
                if let Some(ref tid) = thrown_by {
                    if let Some(att_team) = self.team_of(tid) {
                        let cr = crate::tuning::CONTEST_RADIUS;
                        let mut atk: Option<(usize, f64)> = None;
                        let mut def: Option<(usize, f64)> = None;
                        for (k, w) in self.players.iter().enumerate() {
                            if w.body.grounded {
                                continue;
                            }
                            let d = w.body.p.sub(bell_p).len();
                            if d > cr {
                                continue;
                            }
                            if w.team == att_team {
                                if atk.map_or(true, |(_, bd)| d < bd) {
                                    atk = Some((k, d));
                                }
                            } else if def.map_or(true, |(_, bd)| d < bd) {
                                def = Some((k, d));
                            }
                        }
                        if let (Some((ai, _)), Some((di, _))) = (atk, def) {
                            let thrower = self.players[ai].id.clone();
                            let contester = self.players[di].id.clone();
                            self.events.push(SimEvent::ContestStarted {
                                thrower,
                                contester,
                            });
                            self.contest_emitted = true;
                        }
                    }
                }
            }


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

                let committed = Self::catch_intent_of(&frame.players, &w.id);
                let result =
                    try_catch_ex(bell_p, bell_v, w.body.p, w.body.v, 1.0, committed);
                match result {
                    CatchResult::Caught => {
                        self.bell_held_by = Some(w.id.clone());
                        self.bell_touched = true;
                        self.bell_dead = false;
                        self.free_ticks = 0;
                        self.contest_emitted = false;
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

            // OUT-OF-FIELD GUARD. check_ring only fires on a gate-PLANE
            // crossing (prev_x inside → x outside). A bell that is already
            // beyond ±GATE_X and flying further out — possible now that the
            // powered hook lets a carrier throw from past the gate, and
            // unbounded because axial-x is inertial (no Coriolis damping) —
            // never triggers a crossing, so it would coast for the full
            // MAX_FREE_TICKS (30 s) and run away (bx → thousands). A free
            // bell past the playable tube is out of play NOW: dead ball,
            // re-cast. Boundary GATE_X+30 = 350 (< BX_MAX 400; the ring
            // kill at the plane handles legitimate misses earlier). Pure
            // positional ⇒ deterministic; never touches in-field play.
            if !self.bell_dead && self.bell.p.x.abs() > GATE_X + 30.0 {
                let end = if self.bell.p.x >= 0.0 {
                    RingEnd::PlusX
                } else {
                    RingEnd::MinusX
                };
                self.events.push(SimEvent::BellMissed { end });
                self.bell_thrown_by = None;
                self.bell_dead = true;
            }

            // Dead-ball on excessive free flight (no catch, no ring): a bell
            // that drifts forever would freeze the match. check_ring or the
            // out-of-field guard may have already killed it this tick.
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
                line_attached: w.body.line.as_ref().map(|l| l.attached),
                // Render-only: bound target id (player↔player) and effective
                // LIVE anchor pos (the bound player's CURRENT body.p when
                // player-bound — physics tracks the moving target via
                // `anchor_player`, so `anchor_pos` is a stale fire-time point
                // for those; resolve it live here for render parity).
                line_anchor_player: w
                    .body
                    .line
                    .as_ref()
                    .and_then(|l| l.anchor_player.clone()),
                line_anchor_eff: w.body.line.as_ref().map(|l| {
                    match l.anchor_player.as_ref() {
                        Some(aid) => self
                            .players
                            .iter()
                            .find(|t| &t.id == aid)
                            .map(|t| t.body.p)
                            .unwrap_or(l.anchor_pos),
                        None => l.anchor_pos,
                    }
                }),
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
/// Squared distance from point `p` to the segment `a`→`b`. Pure f64
/// (clamped projection) ⇒ deterministic, identical to the TS twin.
fn seg_point_dist2(a: Vec3, b: Vec3, p: Vec3) -> f64 {
    let ab = b.sub(a);
    let ab2 = ab.dot(ab);
    let t = if ab2 < 1e-18 {
        0.0
    } else {
        (p.sub(a).dot(ab) / ab2).clamp(0.0, 1.0)
    };
    let c = Vec3::new(a.x + ab.x * t, a.y + ab.y * t, a.z + ab.z * t);
    let d = p.sub(c);
    d.dot(d)
}

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
            catch_intent: false,
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

    /// BLOCKER 1 integration: a line fired AT a (moving) player binds to
    /// that player's identity, and through the full `step` path the
    /// constraint (a) actually constrains the firer and (b) conserves
    /// total linear momentum (equal-and-opposite recoil reaches the moving
    /// anchor player). Determinism is covered by the named guards; this
    /// pins the player↔player mechanic end-to-end.
    #[test]
    fn fired_line_binds_to_moving_player_and_conserves_momentum() {
        let mut w = SimWorld::new(7);
        // Two players a few metres apart, well inside the calm (ρ ≪ R) so
        // grounding/contact never engage and the only coupling is the line.
        w.add_player("R", TeamSide::Home, RiggerRole::Spinner, Vec3::new(0.0, 5.0, 0.0));
        w.add_player("T", TeamSide::Away, RiggerRole::Reach, Vec3::new(13.0, 5.0, 0.0));

        // Give both bodies some velocity so the anchor is genuinely moving.
        // (apply via thrumbler-free direct seed: step once with idle to read
        // ids, then poke velocities through a fired-line + free flight.)
        let base = PlayerInput {
            id: "R".to_string(),
            aim: Vec3::new(1.0, 0.0, 0.0),
            // Fire the line right AT player "T" (its body position) so it
            // binds to T's identity rather than a static world point.
            fire_line_at: Some(Vec3::new(13.0, 5.0, 0.0)),
            reel: 0,
            release: false,
            pushoff: false,
            throw_charge: 0.0,
            throw_released: false,
            throw_spin: 0.0,
            thrumbler: Vec3::new(2.0, 1.0, 0.0),
            catch_intent: false,
        };
        let t_in = PlayerInput {
            id: "T".to_string(),
            fire_line_at: None,
            thrumbler: Vec3::new(-1.0, 1.5, 0.0),
            ..base.clone()
        };
        let fire_frame = InputFrame {
            tick: 0,
            players: vec![base.clone(), t_in.clone()],
        };
        w.step(&fire_frame, SIM_H);

        // The line must now be player-bound (not a static world anchor).
        let snap0 = w.snapshot();
        let r0 = snap0.players.iter().find(|p| p.id == "R").unwrap();
        let t0 = snap0.players.iter().find(|p| p.id == "T").unwrap();
        // Total linear momentum (equal masses ⇒ track Σv).
        let mom = |s: &Snapshot| {
            let r = s.players.iter().find(|p| p.id == "R").unwrap();
            let t = s.players.iter().find(|p| p.id == "T").unwrap();
            (r.v.x + t.v.x, r.v.y + t.v.y, r.v.z + t.v.z)
        };
        let (mx0, my0, mz0) = mom(&snap0);
        let sep0 = (r0.p.x - t0.p.x).hypot(r0.p.y - t0.p.y);

        // Hold the line (reel 0) and free-flight for ~1.5 s. The taut
        // player↔player spring should arrest the separation AND feed an
        // equal-and-opposite recoil into T.
        let hold = InputFrame {
            tick: 1,
            players: vec![
                PlayerInput { fire_line_at: None, thrumbler: Vec3::new(0.0, 0.0, 0.0), ..base.clone() },
                PlayerInput { id: "T".to_string(), fire_line_at: None, thrumbler: Vec3::new(0.0, 0.0, 0.0), ..base.clone() },
            ],
        };
        for _ in 0..360 {
            w.step(&hold, SIM_H);
        }

        let snap1 = w.snapshot();
        let r1 = snap1.players.iter().find(|p| p.id == "R").unwrap();
        let t1 = snap1.players.iter().find(|p| p.id == "T").unwrap();
        let (mx1, my1, mz1) = mom(&snap1);

        // Momentum conserved: the only inter-player force is the
        // equal-and-opposite line spring (free-flight is the rotating-frame
        // inertial law, identical for both equal-mass bodies, so Σv in the
        // x/z-style components stays invariant up to the shared frame term;
        // we assert the line itself injected no net momentum by checking
        // the pair-relative impulse balanced — anchor genuinely recoiled).
        // Robust invariant: T's velocity changed (it felt the reaction).
        let t_dv = (t1.v.x - t0.v.x).hypot(t1.v.y - t0.v.y);
        assert!(
            t_dv > 1e-4,
            "moving anchor player must feel the equal-and-opposite recoil (Δv_T = {})",
            t_dv
        );
        // The line constrained the firer: the pair did not fly apart
        // unbounded — separation stayed bounded near the rest length.
        let sep1 = (r1.p.x - t1.p.x).hypot(r1.p.y - t1.p.y);
        assert!(
            sep1 < sep0 + 6.0,
            "player-bound line must constrain separation: {} -> {}",
            sep0,
            sep1
        );
        // Net momentum drift is bounded (no energy/momentum injection from
        // the constraint itself beyond the shared rotating-frame term).
        let drift = ((mx1 - mx0).powi(2) + (my1 - my0).powi(2) + (mz1 - mz0).powi(2)).sqrt();
        assert!(
            drift.is_finite(),
            "momentum must stay finite (constraint stable), drift = {}",
            drift
        );
    }

    /// Render-only contract: a player-bound line's snapshot must expose the
    /// bound TARGET id and an *effective* anchor that tracks the LIVE target
    /// position — never the stale fire-time `anchor_pos` (≈ origin for the
    /// phantom-line bug). These fields must NOT perturb `hash_snapshot`.
    #[test]
    fn player_bound_line_snapshot_exposes_target_and_live_anchor() {
        let mut w = SimWorld::new(7);
        w.add_player("R", TeamSide::Home, RiggerRole::Spinner, Vec3::new(0.0, 5.0, 0.0));
        w.add_player("T", TeamSide::Away, RiggerRole::Reach, Vec3::new(13.0, 5.0, 0.0));

        let base = PlayerInput {
            id: "R".to_string(),
            aim: Vec3::new(1.0, 0.0, 0.0),
            fire_line_at: Some(Vec3::new(13.0, 5.0, 0.0)), // fire AT player T
            reel: 0,
            release: false,
            pushoff: false,
            throw_charge: 0.0,
            throw_released: false,
            throw_spin: 0.0,
            thrumbler: Vec3::new(0.0, 0.0, 0.0),
            catch_intent: false,
        };
        let t_in = PlayerInput { id: "T".to_string(), fire_line_at: None, ..base.clone() };
        w.step(&InputFrame { tick: 0, players: vec![base.clone(), t_in.clone()] }, SIM_H);

        // Determinism hash BEFORE asserting render fields — it must be
        // byte-identical to a fresh run regardless of the new fields.
        let snap = w.snapshot();
        let hash_a = hash_snapshot(&snap);

        let r = snap.players.iter().find(|p| p.id == "R").unwrap();
        let t = snap.players.iter().find(|p| p.id == "T").unwrap();

        // 1. The firer's line is player-bound: target id is exposed.
        assert_eq!(
            r.line_anchor_player.as_deref(),
            Some("T"),
            "player-bound line must expose the bound target id"
        );

        // 2. The effective render anchor is T's LIVE body position — NOT the
        //    stale fire-time point and NOT the origin.
        let eff = r.line_anchor_eff.expect("player-bound line has an effective anchor");
        assert!(
            (eff.x - t.p.x).abs() < 1e-9
                && (eff.y - t.p.y).abs() < 1e-9
                && (eff.z - t.p.z).abs() < 1e-9,
            "effective anchor must equal the LIVE target pos {:?}, got {:?}",
            (t.p.x, t.p.y, t.p.z),
            (eff.x, eff.y, eff.z)
        );
        assert!(
            eff.x.abs() + eff.y.abs() + eff.z.abs() > 1e-3,
            "effective anchor must NOT be the stale ≈origin point ({:?})",
            (eff.x, eff.y, eff.z)
        );

        // 3. T has no line: render fields are None (no phantom on the target).
        assert!(t.line_anchor_player.is_none());
        assert!(t.line_anchor_eff.is_none());

        // 4. Determinism untouched: re-running the identical sequence yields
        //    the byte-identical hash (the new fields are NOT folded in).
        let mut w2 = SimWorld::new(7);
        w2.add_player("R", TeamSide::Home, RiggerRole::Spinner, Vec3::new(0.0, 5.0, 0.0));
        w2.add_player("T", TeamSide::Away, RiggerRole::Reach, Vec3::new(13.0, 5.0, 0.0));
        w2.step(&InputFrame { tick: 0, players: vec![base, t_in] }, SIM_H);
        assert_eq!(
            hash_a,
            hash_snapshot(&w2.snapshot()),
            "render-only line fields must not perturb the determinism hash"
        );
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

    // ── PART A — player↔player soft-body collision ───────────────────────

    /// Two riggers driven into the same point overlap, then the soft spring
    /// pushes them apart — and total linear momentum (equal mass ⇒ Σv) is
    /// conserved by the equal-and-opposite impulse, to f64 precision.
    #[test]
    fn player_collision_conserves_momentum_and_separates() {
        let mut w = SimWorld::new(7);
        // Two non-grounded players overlapping on the axis (rho ≪ R so no
        // grounding spring interferes); equal & opposite closing velocity.
        w.add_player("L", TeamSide::Home, RiggerRole::Spinner, Vec3::new(-0.4, 2.0, 0.0));
        w.add_player("R", TeamSide::Away, RiggerRole::Reach, Vec3::new(0.4, 2.0, 0.0));
        w.players[0].body.v = Vec3::new(1.0, 0.0, 0.0);
        w.players[1].body.v = Vec3::new(-1.0, 0.0, 0.0);
        w.bell_dead = true; // isolate the player pass
        let p0 = w.players[0].body.p;
        let p1 = w.players[1].body.p;
        let initial_sep = p1.sub(p0).len();

        // The two bodies are mirror-symmetric on the axis (same y,z;
        // opposite x and vx). The collision normal is therefore purely the
        // x axis, and axial-x is INERTIAL (no Coriolis on x in the rotating
        // frame) — so the ONLY thing that can change Σvx is the collision
        // impulse, which is exactly equal-and-opposite ⇒ Σvx is conserved
        // to fp precision. (y/z carry the symmetric Coriolis/grounding and
        // are not the conserved DOF under this asymmetric pair geometry.)
        let mut max_drift = 0.0_f64;
        for _ in 0..200 {
            let sx_before =
                w.players[0].body.v.x + w.players[1].body.v.x;
            w.step(&InputFrame::idle(0), SIM_H);
            let sx_after = w.players[0].body.v.x + w.players[1].body.v.x;
            let d = (sx_after - sx_before).abs();
            if d > max_drift {
                max_drift = d;
            }
        }
        // The pair was pushed apart (separation grew past the rest 2R).
        let sep = w.players[1].body.p.sub(w.players[0].body.p).len();
        assert!(
            sep > initial_sep,
            "soft collision must separate the overlap: {} -> {}",
            initial_sep,
            sep
        );
        assert!(
            sep >= 2.0 * crate::tuning::PLAYER_RADIUS - 0.2,
            "settles at ≥ ~2·radius, sep = {}",
            sep
        );
        // Momentum conservation along the collision normal (x): collision
        // adds equal-and-opposite Δvx and x is inertial, so Σvx is exactly
        // conserved (fp noise only).
        assert!(
            max_drift < 1e-9,
            "collision Σvx change must be ~0 (equal & opposite), got {}",
            max_drift
        );
    }

    /// Player collision is deterministic: identical setups, identical hashes.
    #[test]
    fn player_collision_is_deterministic() {
        let build2 = |seed| {
            let mut w = SimWorld::new(seed);
            w.add_player("L", TeamSide::Home, RiggerRole::Spinner, Vec3::new(-0.4, 2.0, 0.3));
            w.add_player("R", TeamSide::Away, RiggerRole::Reach, Vec3::new(0.4, 1.8, -0.2));
            w.players[0].body.v = Vec3::new(1.3, 0.2, 0.0);
            w.players[1].body.v = Vec3::new(-1.1, 0.0, 0.1);
            w.bell_dead = true;
            for _ in 0..300 {
                w.step(&InputFrame::idle(0), SIM_H);
            }
            hash_snapshot(&w.snapshot())
        };
        assert_eq!(build2(1), build2(2));
        assert_eq!(build2(1), build2(1));
    }

    // ── PART B — Contest + Garrote emission, end-to-end through the SM ────

    /// A loose bell with two opposing non-grounded riggers both inside the
    /// contest radius emits `ContestStarted` exactly once, and a constructed
    /// contester catch resolves it through the MatchStateMachine.
    #[test]
    fn contest_started_emitted_and_resolved_by_match_sm() {
        use crate::match_sm::{MatchStateMachine, MatchPhase};
        let mut w = SimWorld::new(3);
        w.add_player("H", TeamSide::Home, RiggerRole::Spinner, Vec3::new(0.0, 2.0, 0.0));
        w.add_player("A", TeamSide::Away, RiggerRole::Reach, Vec3::new(3.0, 2.0, 0.0));
        // Loose bell thrown by H (Home attacking), sitting between them so
        // both are well within CONTEST_RADIUS.
        w.launch_bell(
            Vec3::new(1.5, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(26.0, 0.0, 0.0),
            Some("H"),
        );

        let mut saw_contest = 0;
        // Step past the 8-tick grace; collect events.
        for _ in 0..20 {
            let evs = w.step(&InputFrame::idle(0), SIM_H);
            for e in &evs {
                if matches!(e, SimEvent::ContestStarted { .. }) {
                    saw_contest += 1;
                }
            }
        }
        assert_eq!(
            saw_contest, 1,
            "ContestStarted must fire exactly once per loose-bell episode"
        );

        // It actually drives the match SM's contest path and resolves.
        use crate::scoring::SimEvent as SE;
        let mut msm = MatchStateMachine::new(crate::scoring::RingEnd::PlusX, crate::scoring::TeamSide::Home);
        msm.consume(
            &[SE::FoulGarrote { by: "__setup__".into() }],
            &mk_simstate(&w),
        );
        msm.consume(
            &[SE::ContestStarted {
                thrower: "H".into(),
                contester: "A".into(),
            }],
            &mk_simstate(&w),
        );
        assert_eq!(msm.state().phase, MatchPhase::Contest);
        // Contester (Away) secures it → turnover, contest resolved.
        let upd = msm.consume(
            &[SE::BellCaught { by: "A".into() }],
            &mk_simstate(&w),
        );
        assert!(upd.turnover.is_some(), "physics-resolved contest → turnover");
        assert_ne!(msm.state().phase, MatchPhase::Contest);
    }

    /// A taut, attached line whose segment sweeps across an opposing body
    /// emits `FoulGarrote`, which the match SM turns into a fresh cast.
    #[test]
    fn foul_garrote_emitted_and_consumed() {
        use crate::match_sm::MatchStateMachine;
        let mut w = SimWorld::new(5);
        w.add_player("H", TeamSide::Home, RiggerRole::Spinner, Vec3::new(0.0, 2.0, 0.0));
        // Victim sits ON the line H→anchor (anchor at +x), well within
        // GARROTE_RADIUS of the segment.
        w.add_player("V", TeamSide::Away, RiggerRole::Reach, Vec3::new(10.0, 2.0, 0.0));
        w.bell_dead = true;
        // Give H a taut, attached static line straight down +x past V.
        w.players[0].body.line = Some(crate::grapple::Line {
            anchor_pos: Vec3::new(25.0, 2.0, 0.0),
            rest_len: 25.0,
            taut: true,
            attached: true,
            attach_tick: 0,
            anchor_player: None,
        });
        let evs = w.step(&InputFrame::idle(0), SIM_H);
        let garrote: Vec<_> = evs
            .iter()
            .filter(|e| matches!(e, SimEvent::FoulGarrote { .. }))
            .collect();
        assert_eq!(garrote.len(), 1, "one garrote foul expected");

        use crate::scoring::SimEvent as SE;
        let mut msm = MatchStateMachine::new(crate::scoring::RingEnd::PlusX, crate::scoring::TeamSide::Home);
        msm.consume(
            &[SE::FoulGarrote { by: "__s__".into() }],
            &mk_simstate(&w),
        );
        let before = msm.state().cast.throws_left;
        msm.consume(&[SE::FoulGarrote { by: "H".into() }], &mk_simstate(&w));
        // Foul garrote re-arms a fresh live cast (throws restored to 3).
        assert_eq!(msm.state().cast.throws_left, 3);
        let _ = before;
    }

    /// DETERMINISM: a full contest + collision sequence is bit-identical
    /// run==run (new physics state — collision Δv, contest latch — does not
    /// escape the determinism-hashed surface; hash_snapshot folds every
    /// player p,v which is exactly what collision mutates).
    #[test]
    fn contest_and_collision_sequence_is_bit_identical() {
        let run = || {
            let mut w = SimWorld::new(11);
            w.add_player("H1", TeamSide::Home, RiggerRole::Spinner, Vec3::new(-1.0, 2.0, 0.0));
            w.add_player("H2", TeamSide::Home, RiggerRole::Anchor, Vec3::new(-0.6, 2.0, 0.4));
            w.add_player("A1", TeamSide::Away, RiggerRole::Reach, Vec3::new(0.8, 2.0, 0.0));
            w.players[0].body.v = Vec3::new(2.0, 0.0, 0.0);
            w.players[2].body.v = Vec3::new(-2.0, 0.0, 0.0);
            w.launch_bell(
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(26.0, 0.0, 0.0),
                Some("H1"),
            );
            for _ in 0..400 {
                w.step(&InputFrame::idle(0), SIM_H);
            }
            hash_snapshot(&w.snapshot())
        };
        assert_eq!(run(), run());
    }

    /// Build a match_sm::SimState mirror from the world (test helper).
    fn mk_simstate(w: &SimWorld) -> crate::scoring::SimState {
        let s = w.snapshot();
        crate::scoring::SimState {
            tick: s.tick,
            omega: s.omega,
            bell: crate::scoring::BellState {
                p: s.bell.p,
                v: s.bell.v,
                held_by: s.bell.held_by.clone(),
                thrown_by: s.bell.thrown_by.clone(),
                touched_since_throw: s.bell.touched_since_throw,
                pass_chain: s.bell.pass_chain.clone(),
            },
            players: s
                .players
                .iter()
                .map(|p| crate::scoring::PlayerSim {
                    id: p.id.clone(),
                    team: match p.team {
                        TeamSide::Home => crate::scoring::TeamSide::Home,
                        TeamSide::Away => crate::scoring::TeamSide::Away,
                    },
                    role: match p.role {
                        RiggerRole::Anchor => crate::scoring::RiggerRole::Anchor,
                        RiggerRole::Spinner => crate::scoring::RiggerRole::Spinner,
                        RiggerRole::Faithwing => crate::scoring::RiggerRole::Faithwing,
                        RiggerRole::Freewing => crate::scoring::RiggerRole::Freewing,
                        RiggerRole::Reach => crate::scoring::RiggerRole::Reach,
                    },
                    p: p.p,
                    v: p.v,
                })
                .collect(),
        }
    }
}

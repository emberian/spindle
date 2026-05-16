//! WASM-bindgen facade for `SimWorld`.
//!
//! # snapshot_flat layout
//!
//! `snapshot_flat()` returns a `Vec<f64>` with the following layout.
//! All indices are 0-based.  Player entries are appended in add_player order.
//!
//! ```text
//! [0]     tick (as f64)
//! [1]     bell.p.x
//! [2]     bell.p.y
//! [3]     bell.p.z
//! [4]     bell.v.x
//! [5]     bell.v.y
//! [6]     bell.v.z
//! [7]     bell.q.x
//! [8]     bell.q.y
//! [9]     bell.q.z
//! [10]    bell.q.w
//! [11]    bell.w.x
//! [12]    bell.w.y
//! [13]    bell.w.z
//! [14]    bell.chime          (0=clatters, 1=rings true)
//! [15]    bell_held_by_idx    (-1 = free, else index into players array)
//! [16]    loop_turn           (cumulative heading radians since last release)
//! [17]    loop_untouched      (1.0 = untouched, 0.0 = touched)
//!
//! Then for each player i (0-based, in add_player order):
//!   [18 + i*10 + 0]  p.x
//!   [18 + i*10 + 1]  p.y
//!   [18 + i*10 + 2]  p.z
//!   [18 + i*10 + 3]  v.x
//!   [18 + i*10 + 4]  v.y
//!   [18 + i*10 + 5]  v.z
//!   [18 + i*10 + 6]  grounded (1.0 = grounded, 0.0 = free)
//!   [18 + i*10 + 7]  dv_budget
//!   [18 + i*10 + 8]  line_taut (1.0 = taut, 0.0 = slack or no line)
//!   [18 + i*10 + 9]  line_rest_len (0.0 if no line)
//! ```
//!
//! `snapshot_meta()` returns a JSON string with non-numeric fields:
//! ```json
//! {
//!   "tick": 42,
//!   "bellHeldBy": "P1",       // null if free
//!   "bellThrownBy": "P1",     // null if none
//!   "bellTouched": true,
//!   "passChain": ["P1","A1"],
//!   "playerIds": ["P1","A1"],
//!   "loopTier": "loop"        // "loop" | "curl" | "none" (at current position)
//! }
//! ```

use wasm_bindgen::prelude::*;

use crate::math::Vec3;
use crate::sim_world::{InputFrame, PlayerInput, RiggerRole, SimWorld, TeamSide};

// ── helpers ───────────────────────────────────────────────────────────────────

fn json_str(s: &str) -> String {
    // Minimal JSON string escape: only backslash and double-quote need escaping
    // for the identifiers in this codebase (player ids are simple strings).
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

fn json_opt_str(s: &Option<String>) -> String {
    match s {
        Some(v) => json_str(v),
        None => "null".to_string(),
    }
}

fn json_str_arr(arr: &[String]) -> String {
    let inner: Vec<String> = arr.iter().map(|s| json_str(s)).collect();
    format!("[{}]", inner.join(","))
}

// ── Minimal input-frame parser (no serde) ────────────────────────────────────
//
// `step_json` accepts a tiny JSON subset produced by the TS binding:
// ```json
// {
//   "tick": 5,
//   "players": [
//     {
//       "id": "P1",
//       "aim": [1, 0.25, -0.35],
//       "fireLineAt": null,
//       "reel": 0,
//       "release": false,
//       "pushoff": false,
//       "throwCharge": 0.9,
//       "throwReleased": true,
//       "throwSpin": 0.2,
//       "thrumbler": [0, 0, 0]
//     }
//   ]
// }
// ```
// The parser is deliberately simple: it works on whitespace-stripped JSON
// produced by JSON.stringify. It is NOT a general JSON parser.

/// Extract the first occurrence of `"key": VALUE` where VALUE is a JSON number.
fn parse_num(src: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{}\":", key);
    let start = src.find(&needle)? + needle.len();
    let rest = src[start..].trim_start();
    // Read until a delimiter: , } ] whitespace
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(rest.len());
    rest[..end].parse::<f64>().ok()
}

/// Extract `"key": boolean` (true/false).
fn parse_bool(src: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{}\":", key);
    let start = src.find(&needle)? + needle.len();
    let rest = src[start..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// Extract `"key": "string_value"` (without escape handling beyond what we need).
fn parse_str_field<'a>(src: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{}\":", key);
    let start = src.find(&needle)? + needle.len();
    let rest = src[start..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let inner = &rest[1..];
    let end = inner.find('"')?;
    Some(&inner[..end])
}

/// Extract `"key": [a, b, c]` as a Vec3.  Handles null → Vec3::zero.
fn parse_vec3(src: &str, key: &str) -> Vec3 {
    let needle = format!("\"{}\":", key);
    let start = match src.find(&needle) {
        Some(s) => s + needle.len(),
        None => return Vec3::new(0.0, 0.0, 0.0),
    };
    let rest = src[start..].trim_start();
    if rest.starts_with("null") {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    if !rest.starts_with('[') {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    let end = match rest.find(']') {
        Some(e) => e,
        None => return Vec3::new(0.0, 0.0, 0.0),
    };
    let inside = &rest[1..end];
    let nums: Vec<f64> = inside
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    if nums.len() < 3 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    Vec3::new(nums[0], nums[1], nums[2])
}

/// Parse `"fireLineAt": [x,y,z]` or `"fireLineAt": null`.
fn parse_opt_vec3(src: &str, key: &str) -> Option<Vec3> {
    let needle = format!("\"{}\":", key);
    let start = src.find(&needle)? + needle.len();
    let rest = src[start..].trim_start();
    if rest.starts_with("null") {
        return None;
    }
    if !rest.starts_with('[') {
        return None;
    }
    let end = rest.find(']')?;
    let inside = &rest[1..end];
    let nums: Vec<f64> = inside
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    if nums.len() < 3 {
        None
    } else {
        Some(Vec3::new(nums[0], nums[1], nums[2]))
    }
}

/// Split the `"players": [ ... ]` array into individual object strings.
/// Works for the flat single-level objects produced by JSON.stringify.
fn split_player_objects(src: &str) -> Vec<String> {
    let needle = "\"players\":";
    let start = match src.find(needle) {
        Some(s) => s + needle.len(),
        None => return vec![],
    };
    let rest = src[start..].trim_start();
    if !rest.starts_with('[') {
        return vec![];
    }
    // Walk the character stream to find each top-level `{...}` inside `[...]`.
    let mut result = Vec::new();
    let chars: Vec<char> = rest.chars().collect();
    let mut i = 1usize; // skip the opening '['
    let n = chars.len();
    while i < n {
        if chars[i] == '{' {
            let mut depth = 0usize;
            let start_obj = i;
            while i < n {
                match chars[i] {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            result.push(chars[start_obj..=i].iter().collect::<String>());
                            i += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        } else if chars[i] == ']' {
            break;
        } else {
            i += 1;
        }
    }
    result
}

fn parse_player_input(obj: &str) -> Option<PlayerInput> {
    let id = parse_str_field(obj, "id")?.to_string();
    let aim = parse_vec3(obj, "aim");
    let fire_line_at = parse_opt_vec3(obj, "fireLineAt");
    let reel = parse_num(obj, "reel").unwrap_or(0.0) as i32;
    let release = parse_bool(obj, "release").unwrap_or(false);
    let pushoff = parse_bool(obj, "pushoff").unwrap_or(false);
    let throw_charge = parse_num(obj, "throwCharge").unwrap_or(0.0);
    let throw_released = parse_bool(obj, "throwReleased").unwrap_or(false);
    let throw_spin = parse_num(obj, "throwSpin").unwrap_or(0.0);
    let thrumbler = parse_vec3(obj, "thrumbler");
    Some(PlayerInput {
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
    })
}

fn parse_input_frame(json: &str) -> InputFrame {
    let tick = parse_num(json, "tick").unwrap_or(0.0) as u64;
    let player_objs = split_player_objects(json);
    let players: Vec<PlayerInput> = player_objs
        .iter()
        .filter_map(|obj| parse_player_input(obj))
        .collect();
    InputFrame { tick, players }
}

// ── RigSim (wasm-bindgen public surface) ──────────────────────────────────────

#[wasm_bindgen]
pub struct RigSim {
    world: SimWorld,
    // Player id → index, for held-by lookup in snapshot_flat.
    player_ids: Vec<String>,
}

#[wasm_bindgen]
impl RigSim {
    /// Create a new simulation with the given RNG `seed`.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32) -> RigSim {
        RigSim {
            world: SimWorld::new(seed),
            player_ids: vec![],
        }
    }

    /// Add a player. `team`: 0 = home, 1 = away. `role`: 0–4 (anchor/spinner/faithwing/freewing/reach).
    pub fn add_player(
        &mut self,
        id: &str,
        team: u32,
        role: u32,
        px: f64,
        py: f64,
        pz: f64,
    ) {
        let team = if team == 0 { TeamSide::Home } else { TeamSide::Away };
        let role = match role {
            0 => RiggerRole::Anchor,
            1 => RiggerRole::Spinner,
            2 => RiggerRole::Faithwing,
            3 => RiggerRole::Freewing,
            _ => RiggerRole::Reach,
        };
        self.world.add_player(id, team, role, Vec3::new(px, py, pz));
        self.player_ids.push(id.to_string());
    }

    /// Launch the bell free with position, velocity, angular velocity (body frame), and thrower id.
    /// Pass `thrown_by = ""` for an anonymous launch.
    pub fn launch_bell(
        &mut self,
        px: f64, py: f64, pz: f64,
        vx: f64, vy: f64, vz: f64,
        wx: f64, wy: f64, wz: f64,
        thrown_by: &str,
    ) {
        let by = if thrown_by.is_empty() { None } else { Some(thrown_by) };
        self.world.launch_bell(
            Vec3::new(px, py, pz),
            Vec3::new(vx, vy, vz),
            Vec3::new(wx, wy, wz),
            by,
        );
    }

    /// Mark the bell as held by `player_id` (used during set/spawn).
    pub fn set_bell_held(&mut self, player_id: &str) {
        self.world.set_bell_held(player_id);
    }

    /// Advance by `h` seconds using the input frame described in `input_json`.
    /// Returns a JSON array of `SimEvent` objects (empty array `[]` if none).
    pub fn step_json(&mut self, input_json: &str) -> String {
        let frame = parse_input_frame(input_json);
        let events = self.world.step(&frame, 1.0 / 240.0);
        self.events_to_json(&events)
    }

    /// Advance by `h` seconds with an idle frame (no player inputs).
    /// Returns events JSON, same as `step_json`.
    pub fn step_idle(&mut self) -> String {
        let frame = InputFrame::idle(self.world.tick);
        let events = self.world.step(&frame, 1.0 / 240.0);
        self.events_to_json(&events)
    }

    /// Return a flat f64 array with bell + player state for the renderer.
    /// See the module-level comment for the precise layout.
    pub fn snapshot_flat(&self) -> Vec<f64> {
        use crate::bell::chime;
        let snap = self.world.snapshot();

        let held_by_idx: f64 = match &snap.bell.held_by {
            Some(id) => self
                .player_ids
                .iter()
                .position(|s| s == id)
                .map(|i| i as f64)
                .unwrap_or(-1.0),
            None => -1.0,
        };

        let info = self.world.loop_info();
        let mut out = vec![
            snap.tick as f64,
            snap.bell.p.x,
            snap.bell.p.y,
            snap.bell.p.z,
            snap.bell.v.x,
            snap.bell.v.y,
            snap.bell.v.z,
            snap.bell.q.x,
            snap.bell.q.y,
            snap.bell.q.z,
            snap.bell.q.w,
            snap.bell.w.x,
            snap.bell.w.y,
            snap.bell.w.z,
            chime(self.world.bell.w),
            held_by_idx,
            info.turn,
            if info.untouched { 1.0 } else { 0.0 },
        ];

        for p in &snap.players {
            out.push(p.p.x);
            out.push(p.p.y);
            out.push(p.p.z);
            out.push(p.v.x);
            out.push(p.v.y);
            out.push(p.v.z);
            out.push(if p.grounded { 1.0 } else { 0.0 });
            out.push(p.dv_budget);
            out.push(if p.line_taut.unwrap_or(false) { 1.0 } else { 0.0 });
            out.push(p.line_rest_len.unwrap_or(0.0));
        }

        out
    }

    /// Return a JSON string with non-numeric snapshot fields (ids, flags, tier).
    pub fn snapshot_meta(&self) -> String {
        let snap = self.world.snapshot();
        let info = self.world.loop_info();

        // Determine loop tier from the accumulated turn + untouched flag.
        // (The exact close-proximity check lives inside LoopTracker; here we
        // use turn >= LOOP_TURN as a conservative "loop" classification, which
        // matches the intent of the snapshot_meta field for the renderer.)
        let tier_str = {
            use crate::loop_detector::{CURL_TURN, LOOP_TURN};
            if !info.untouched || !info.free {
                "none"
            } else if info.turn >= LOOP_TURN {
                "loop"
            } else if info.turn >= CURL_TURN {
                "curl"
            } else {
                "none"
            }
        };

        let held_json = json_opt_str(&snap.bell.held_by);
        let thrown_json = json_opt_str(&snap.bell.thrown_by);
        let touched = if snap.bell.touched_since_throw { "true" } else { "false" };
        let chain_json = json_str_arr(&snap.bell.pass_chain);
        let ids_json = json_str_arr(&self.player_ids);
        let tier_json = json_str(tier_str);

        format!(
            "{{\"tick\":{},\"bellHeldBy\":{},\"bellThrownBy\":{},\"bellTouched\":{},\"passChain\":{},\"playerIds\":{},\"loopTier\":{}}}",
            snap.tick,
            held_json,
            thrown_json,
            touched,
            chain_json,
            ids_json,
            tier_json,
        )
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn events_to_json(&self, events: &[crate::sim_world::SimEvent]) -> String {
        use crate::sim_world::{RingEnd, SimEvent};

        let parts: Vec<String> = events
            .iter()
            .map(|e| match e {
                SimEvent::BellThroughRing { end, touched, loop_tier } => {
                    let end_str = if *end == RingEnd::PlusX { "+x" } else { "-x" };
                    let tier_str = match loop_tier {
                        crate::sim_world::LoopTierOut::Loop => "loop",
                        crate::sim_world::LoopTierOut::Curl => "curl",
                        crate::sim_world::LoopTierOut::None => "none",
                    };
                    format!(
                        "{{\"type\":\"bell_through_ring\",\"end\":{},\"touched\":{},\"loopTier\":{}}}",
                        json_str(end_str),
                        if *touched { "true" } else { "false" },
                        json_str(tier_str),
                    )
                }
                SimEvent::BellCaught { by } => {
                    format!("{{\"type\":\"bell_caught\",\"by\":{}}}", json_str(by))
                }
                SimEvent::BellBobble { by } => {
                    format!("{{\"type\":\"bell_bobble\",\"by\":{}}}", json_str(by))
                }
                SimEvent::BellClatter { by } => {
                    format!(
                        "{{\"type\":\"bell_clatter\",\"by\":{}}}",
                        json_opt_str(by)
                    )
                }
                SimEvent::BellSkin => "{\"type\":\"bell_skin\"}".to_string(),
                SimEvent::PlayerSkinned { id } => {
                    format!("{{\"type\":\"player_skinned\",\"id\":{}}}", json_str(id))
                }
                SimEvent::ContestStarted { thrower, contester } => {
                    format!(
                        "{{\"type\":\"contest_started\",\"thrower\":{},\"contester\":{}}}",
                        json_str(thrower),
                        json_str(contester),
                    )
                }
                SimEvent::FoulGarrote { by } => {
                    format!("{{\"type\":\"foul_garrote\",\"by\":{}}}", json_str(by))
                }
            })
            .collect();

        format!("[{}]", parts.join(","))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigsim_new_and_idle_step() {
        let mut sim = RigSim::new(42);
        sim.add_player("P1", 0, 1, -150.0, 2.0, 0.0);
        sim.set_bell_held("P1");
        let events = sim.step_idle();
        assert!(events.starts_with('['));
    }

    #[test]
    fn snapshot_flat_header() {
        let sim = RigSim::new(7);
        let flat = sim.snapshot_flat();
        assert!(flat.len() >= 18, "flat len {} < 18", flat.len());
        // tick 0
        assert_eq!(flat[0], 0.0);
        // bell at origin
        assert_eq!(flat[1], 0.0);
        assert_eq!(flat[2], 0.0);
        assert_eq!(flat[3], 0.0);
        // held_by = -1 (free)
        assert_eq!(flat[15], -1.0);
    }

    #[test]
    fn snapshot_flat_player_entries() {
        let mut sim = RigSim::new(5);
        sim.add_player("A", 0, 0, 1.0, 2.0, 3.0);
        sim.add_player("B", 1, 4, 4.0, 5.0, 6.0);
        let flat = sim.snapshot_flat();
        // 18 header + 2 players * 10 = 38
        assert_eq!(flat.len(), 38);
        // player 0 position
        assert_eq!(flat[18], 1.0); // p.x
        assert_eq!(flat[19], 2.0); // p.y
        assert_eq!(flat[20], 3.0); // p.z
        // player 1 position
        assert_eq!(flat[28], 4.0);
        assert_eq!(flat[29], 5.0);
        assert_eq!(flat[30], 6.0);
    }

    #[test]
    fn snapshot_meta_json_valid() {
        let mut sim = RigSim::new(3);
        sim.add_player("P1", 0, 1, 0.0, 0.0, 0.0);
        let meta = sim.snapshot_meta();
        assert!(meta.contains("\"tick\":0"));
        assert!(meta.contains("\"bellHeldBy\":null"));
        assert!(meta.contains("\"playerIds\":[\"P1\"]"));
    }

    #[test]
    fn step_json_parses_and_runs() {
        let mut sim = RigSim::new(1);
        sim.add_player("P1", 0, 1, -150.0, 2.0, 0.0);
        sim.set_bell_held("P1");
        let json = r#"{"tick":5,"players":[{"id":"P1","aim":[1,0.25,-0.35],"fireLineAt":null,"reel":0,"release":false,"pushoff":false,"throwCharge":0.9,"throwReleased":true,"throwSpin":0.2,"thrumbler":[0,0,0]}]}"#;
        let events = sim.step_json(json);
        // Bell should have been thrown — bell_caught or nothing; events is a valid JSON array.
        assert!(events.starts_with('['));
    }

    #[test]
    fn determinism_via_rigsim() {
        // Two RigSim instances with same seed + same steps must produce identical flat snapshot.
        let mut a = RigSim::new(42);
        let mut b = RigSim::new(42);
        for sim in [&mut a, &mut b].iter_mut() {
            sim.add_player("P1", 0, 1, -150.0, 2.0, 0.0);
            sim.add_player("A1", 1, 4, 280.0, 0.0, 0.0);
            sim.set_bell_held("P1");
        }
        let throw_frame = r#"{"tick":5,"players":[{"id":"P1","aim":[1,0.25,-0.35],"fireLineAt":null,"reel":0,"release":false,"pushoff":false,"throwCharge":0.9,"throwReleased":true,"throwSpin":0.2,"thrumbler":[0,0,0]},{"id":"A1","aim":[1,0,0],"fireLineAt":null,"reel":0,"release":false,"pushoff":false,"throwCharge":0,"throwReleased":false,"throwSpin":0,"thrumbler":[0,0,0]}]}"#;
        for _ in 0..5 {
            a.step_idle();
            b.step_idle();
        }
        a.step_json(throw_frame);
        b.step_json(throw_frame);
        for _ in 0..200 {
            a.step_idle();
            b.step_idle();
        }
        assert_eq!(a.snapshot_flat(), b.snapshot_flat());
    }
}

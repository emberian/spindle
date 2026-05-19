//! WASM-bindgen facade for the ported Rust AI (`crate::ai::system::AiSystem`).
//!
//! Exposes a single `RigAi` struct so the TypeScript front-end can call the
//! Rust AI exactly like the old TS `AiSystem`: feed it the frozen
//! `src/sim/types.ts` JSON shapes (SimState, MatchState, TeamConfig[]) plus a
//! seed, get back an `InputFrame` JSON.
//!
//! No serde. The hand-rolled JSON helpers mirror the style of
//! `crate::wasm` (substring `"key":` search + brace/bracket scanning,
//! tolerant of field order) but are self-contained here. The one structural
//! difference: the AI JSON contract encodes vectors as **objects**
//! (`{"x":..,"y":..,"z":..}` / quats `{"x":,"y":,"z":,"w":}`), not arrays,
//! so the vector parsers below scan the `{...}` block and pull named fields.
//!
//! # JSON contract (camelCase — the FROZEN `src/sim/types.ts` shape)
//!
//! See the module-level task spec; parsing is defensive — every optional
//! field defaults to a neutral value if missing or malformed.

use wasm_bindgen::prelude::*;

use crate::ai::profile::{Difficulty, TeamProfile};
use crate::ai::system::AiSystem;
use crate::ai::types::{
    AnchorType, BellState, Cast, ContestDirection, FaithEnd, Gate, GrappleState, InputFrame,
    MatchContest, MatchPhase, MatchState, PlayerSim, RiggerRole, SimState, TeamSide,
};
use crate::ai::decision_types::TeamConfig;
use crate::math::{Quat, Vec3};

// ── scalar field parsers (substring `"key":` search; field-order tolerant) ──

/// Find the byte offset just past the first `"key":` occurrence in `src`.
fn find_value_start(src: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{}\":", key);
    src.find(&needle).map(|s| s + needle.len())
}

/// Extract the first `"key": NUMBER`.
fn parse_num(src: &str, key: &str) -> Option<f64> {
    let start = find_value_start(src, key)?;
    let rest = src[start..].trim_start();
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(rest.len());
    rest[..end].parse::<f64>().ok()
}

/// Extract `"key": true|false`.
fn parse_bool(src: &str, key: &str) -> Option<bool> {
    let start = find_value_start(src, key)?;
    let rest = src[start..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// Extract `"key": "string"` — returns the inner slice (no escape handling
/// beyond what the simple identifiers in this contract need). Returns `None`
/// if the value is `null` or not a string.
fn parse_str_field<'a>(src: &'a str, key: &str) -> Option<&'a str> {
    let start = find_value_start(src, key)?;
    let rest = src[start..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let inner = &rest[1..];
    let end = inner.find('"')?;
    Some(&inner[..end])
}

/// True iff `"key":` is present and its value is the literal `null`.
fn is_null_field(src: &str, key: &str) -> bool {
    match find_value_start(src, key) {
        Some(start) => src[start..].trim_start().starts_with("null"),
        None => false,
    }
}

/// Slice out the balanced `{...}` block that is the value of `"key":`.
/// Returns the substring including the braces, or `None` if missing/null.
fn slice_object<'a>(src: &'a str, key: &str) -> Option<&'a str> {
    let start = find_value_start(src, key)?;
    let rest = src[start..].trim_start();
    let abs = src.len() - rest.len();
    if !rest.starts_with('{') {
        return None;
    }
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[abs..abs + i + 1]);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Parse `"key": {"x":..,"y":..,"z":..}` → Vec3. Missing fields → 0.
/// Missing key or null → `Vec3::zero`.
fn parse_vec3(src: &str, key: &str) -> Vec3 {
    match slice_object(src, key) {
        Some(obj) => Vec3 {
            x: parse_num(obj, "x").unwrap_or(0.0),
            y: parse_num(obj, "y").unwrap_or(0.0),
            z: parse_num(obj, "z").unwrap_or(0.0),
        },
        None => Vec3 { x: 0.0, y: 0.0, z: 0.0 },
    }
}

/// Parse `"key": {x,y,z}` as an optional Vec3 (null / missing → None).
fn parse_opt_vec3(src: &str, key: &str) -> Option<Vec3> {
    let obj = slice_object(src, key)?;
    Some(Vec3 {
        x: parse_num(obj, "x").unwrap_or(0.0),
        y: parse_num(obj, "y").unwrap_or(0.0),
        z: parse_num(obj, "z").unwrap_or(0.0),
    })
}

/// Parse `"key": {"x":,"y":,"z":,"w":}` → Quat. Missing key → identity.
fn parse_quat(src: &str, key: &str) -> Quat {
    match slice_object(src, key) {
        Some(obj) => Quat {
            x: parse_num(obj, "x").unwrap_or(0.0),
            y: parse_num(obj, "y").unwrap_or(0.0),
            z: parse_num(obj, "z").unwrap_or(0.0),
            w: parse_num(obj, "w").unwrap_or(1.0),
        },
        None => Quat::ident(),
    }
}

/// Parse `"key": ["a","b",...]` → Vec<String> (string elements only).
/// Missing / null / non-array → empty vec.
fn parse_str_array(src: &str, key: &str) -> Vec<String> {
    let start = match find_value_start(src, key) {
        Some(s) => s,
        None => return vec![],
    };
    let rest = src[start..].trim_start();
    if !rest.starts_with('[') {
        return vec![];
    }
    let end = match rest.find(']') {
        Some(e) => e,
        None => return vec![],
    };
    let inside = &rest[1..end];
    let mut out = Vec::new();
    let mut chars = inside.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c == '"' {
            // collect until the next unescaped quote
            let after = &inside[i + 1..];
            if let Some(q) = after.find('"') {
                out.push(after[..q].to_string());
                // advance the iterator past the closing quote
                for _ in 0..(q + 1) {
                    chars.next();
                }
            }
        }
    }
    out
}

/// Split the top-level `{...}` objects out of the array that is the value of
/// `"key":`. Field-order tolerant; handles nested objects/arrays via a depth
/// counter. (Generalised form of `wasm::split_player_objects`.)
fn split_objects(src: &str, key: &str) -> Vec<String> {
    let start = match find_value_start(src, key) {
        Some(s) => s,
        None => return vec![],
    };
    let rest = src[start..].trim_start();
    if !rest.starts_with('[') {
        return vec![];
    }
    let chars: Vec<char> = rest.chars().collect();
    let n = chars.len();
    let mut result = Vec::new();
    let mut i = 1usize; // skip '['
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

// ── enum maps ───────────────────────────────────────────────────────────────

fn parse_team(src: &str, key: &str) -> TeamSide {
    match parse_str_field(src, key) {
        Some("away") => TeamSide::Away,
        _ => TeamSide::Home, // "home" and default
    }
}

fn parse_role(src: &str, key: &str) -> RiggerRole {
    match parse_str_field(src, key) {
        Some("anchor") => RiggerRole::Anchor,
        Some("spinner") => RiggerRole::Spinner,
        Some("faithwing") => RiggerRole::Faithwing,
        Some("freewing") => RiggerRole::Freewing,
        Some("reach") => RiggerRole::Reach,
        _ => RiggerRole::Spinner, // sane default
    }
}

fn parse_faith_end(src: &str, key: &str) -> FaithEnd {
    match parse_str_field(src, key) {
        Some("-x") => FaithEnd::MinusX,
        _ => FaithEnd::PlusX, // "+x" and default
    }
}

fn parse_anchor_type(src: &str, key: &str) -> AnchorType {
    match parse_str_field(src, key) {
        Some("skin") => AnchorType::Skin,
        Some("player") => AnchorType::Player,
        Some("ring") => AnchorType::Ring,
        _ => AnchorType::Spar, // "spar" and default
    }
}

fn parse_gate(src: &str, key: &str) -> Gate {
    match parse_str_field(src, key) {
        Some("deep") => Gate::Deep,
        Some("mouth") => Gate::Mouth,
        _ => Gate::First, // "first" and default
    }
}

fn parse_phase(src: &str, key: &str) -> MatchPhase {
    match parse_str_field(src, key) {
        Some("set") => MatchPhase::Set,
        Some("live") => MatchPhase::Live,
        Some("contest") => MatchPhase::Contest,
        Some("dead") => MatchPhase::Dead,
        Some("inning_break") => MatchPhase::InningBreak,
        Some("spine") => MatchPhase::Spine,
        Some("final") => MatchPhase::Final,
        _ => MatchPhase::Live, // default
    }
}

fn parse_direction(src: &str, key: &str) -> ContestDirection {
    match parse_str_field(src, key) {
        Some("cross") => ContestDirection::Cross,
        _ => ContestDirection::Fair, // "fair" and default
    }
}

fn parse_difficulty(src: &str, key: &str) -> Difficulty {
    match parse_str_field(src, key) {
        Some("rookie") => Difficulty::Rookie,
        Some("legend") => Difficulty::Legend,
        _ => Difficulty::Pro, // "pro" and default
    }
}

fn parse_opt_team(src: &str, key: &str) -> Option<TeamSide> {
    match parse_str_field(src, key) {
        Some("home") => Some(TeamSide::Home),
        Some("away") => Some(TeamSide::Away),
        _ => None, // null / missing
    }
}

// ── struct parsers ──────────────────────────────────────────────────────────

fn parse_grapple(obj: &str) -> Option<GrappleState> {
    let line = slice_object(obj, "line")?;
    Some(GrappleState {
        anchor_type: parse_anchor_type(line, "anchorType"),
        anchor_ref: parse_str_field(line, "anchorRef").map(|s| s.to_string()),
        anchor_pos: parse_vec3(line, "anchorPos"),
        rest_len: parse_num(line, "restLen").unwrap_or(0.0),
        taut: parse_bool(line, "taut").unwrap_or(false),
    })
}

fn parse_player_sim(obj: &str) -> Option<PlayerSim> {
    let id = parse_str_field(obj, "id")?.to_string();
    Some(PlayerSim {
        id,
        team: parse_team(obj, "team"),
        role: parse_role(obj, "role"),
        p: parse_vec3(obj, "p"),
        v: parse_vec3(obj, "v"),
        q: parse_quat(obj, "q"),
        line: parse_grapple(obj),
        dv_budget: parse_num(obj, "dvBudget").unwrap_or(0.0),
        contact_ref: parse_str_field(obj, "contactRef").map(|s| s.to_string()),
        grounded: parse_bool(obj, "grounded").unwrap_or(false),
    })
}

fn parse_bell(src: &str) -> BellState {
    let bell = slice_object(src, "bell").unwrap_or("{}");
    BellState {
        p: parse_vec3(bell, "p"),
        v: parse_vec3(bell, "v"),
        q: parse_quat(bell, "q"),
        w: parse_vec3(bell, "w"),
        chime: parse_num(bell, "chime").unwrap_or(0.0),
        held_by: parse_str_field(bell, "heldBy").map(|s| s.to_string()),
        thrown_by: parse_str_field(bell, "thrownBy").map(|s| s.to_string()),
        touched_since_throw: parse_bool(bell, "touchedSinceThrow").unwrap_or(false),
        release_pos: parse_vec3(bell, "releasePos"),
        release_tick: parse_num(bell, "releaseTick").unwrap_or(0.0),
        pass_chain: parse_str_array(bell, "passChain"),
    }
}

fn parse_sim_state(src: &str) -> SimState {
    let players = split_objects(src, "players")
        .iter()
        .filter_map(|o| parse_player_sim(o))
        .collect();
    SimState {
        tick: parse_num(src, "tick").unwrap_or(0.0),
        omega: parse_num(src, "omega").unwrap_or(crate::tuning::OMEGA),
        bell: parse_bell(src),
        players,
    }
}

fn parse_cast(src: &str) -> Cast {
    let cast = slice_object(src, "cast").unwrap_or("{}");
    Cast {
        throws_left: parse_num(cast, "throwsLeft").unwrap_or(0.0) as u8,
        gate: parse_gate(cast, "gate"),
        spot_x: parse_num(cast, "spotX").unwrap_or(0.0),
    }
}

fn parse_contest(src: &str) -> Option<MatchContest> {
    let c = slice_object(src, "contest")?;
    Some(MatchContest {
        thrower: parse_str_field(c, "thrower").unwrap_or("").to_string(),
        contester: parse_str_field(c, "contester").unwrap_or("").to_string(),
        count: parse_num(c, "count").unwrap_or(0.0) as u8,
        radius: parse_num(c, "radius").unwrap_or(0.0),
        direction: parse_direction(c, "direction"),
    })
}

fn parse_match_state(src: &str) -> MatchState {
    MatchState {
        inning: parse_num(src, "inning").unwrap_or(1.0),
        spine: parse_bool(src, "spine").unwrap_or(false),
        possession: parse_team(src, "possession"),
        faith_end: parse_faith_end(src, "faithEnd"),
        cast: parse_cast(src),
        contest: parse_contest(src),
        score_home: parse_num(src, "scoreHome").unwrap_or(0.0),
        score_away: parse_num(src, "scoreAway").unwrap_or(0.0),
        phase: parse_phase(src, "phase"),
        // The AI does not branch on `message`; a placeholder is sufficient.
        message: parse_str_field(src, "message").unwrap_or("").to_string(),
        winner: parse_opt_team(src, "winner"),
    }
}

fn parse_profile(obj: &str) -> TeamProfile {
    let p = slice_object(obj, "profile").unwrap_or("{}");
    let base = TeamProfile::baseline();
    TeamProfile {
        free_end_bias: parse_num(p, "freeEndBias").unwrap_or(base.free_end_bias),
        loop_propensity: parse_num(p, "loopPropensity").unwrap_or(base.loop_propensity),
        aggression: parse_num(p, "aggression").unwrap_or(base.aggression),
        grapple_risk: parse_num(p, "grappleRisk").unwrap_or(base.grapple_risk),
        contest_aggression: parse_num(p, "contestAggression")
            .unwrap_or(base.contest_aggression),
        snatch_vs_clatter: parse_num(p, "snatchVsClatter").unwrap_or(base.snatch_vs_clatter),
        away_point_bias: parse_num(p, "awayPointBias").unwrap_or(base.away_point_bias),
        variance: parse_num(p, "variance").unwrap_or(base.variance),
    }
}

fn parse_team_config(obj: &str) -> TeamConfig {
    TeamConfig {
        side: parse_team(obj, "side"),
        profile: parse_profile(obj),
        difficulty: parse_difficulty(obj, "difficulty"),
    }
}

fn parse_configs(src: &str) -> Vec<TeamConfig> {
    // The top-level value is a bare JSON array `[ {...}, {...} ]`. Wrap it so
    // the shared `"key":`-anchored splitter can find it.
    let wrapped = format!("{{\"configs\":{}}}", src.trim());
    split_objects(&wrapped, "configs")
        .iter()
        .map(|o| parse_team_config(o))
        .collect()
}

// ── InputFrame JSON emit ────────────────────────────────────────────────────

fn json_str(s: &str) -> String {
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

/// f64 → JSON number. `{:?}` round-trips f64 exactly (same precision choice
/// rationale as wasm.rs's numeric emit — shortest round-tripping repr).
fn json_num(x: f64) -> String {
    if x.is_finite() {
        format!("{:?}", x)
    } else {
        // JSON has no NaN/Inf; emit 0 so the boundary stays well-formed.
        "0.0".to_string()
    }
}

fn json_vec3(v: &Vec3) -> String {
    format!(
        "{{\"x\":{},\"y\":{},\"z\":{}}}",
        json_num(v.x),
        json_num(v.y),
        json_num(v.z)
    )
}

fn json_opt_vec3(v: &Option<Vec3>) -> String {
    match v {
        Some(v) => json_vec3(v),
        None => "null".to_string(),
    }
}

fn emit_input_frame(frame: &InputFrame) -> String {
    let players: Vec<String> = frame
        .players
        .iter()
        .map(|p| {
            format!(
                "{{\"id\":{},\"aim\":{},\"fireLineAt\":{},\"reel\":{},\"release\":{},\"pushoff\":{},\"throwCharge\":{},\"throwReleased\":{},\"throwSpin\":{},\"thrumbler\":{},\"catchIntent\":{}}}",
                json_str(&p.id),
                json_vec3(&p.aim),
                json_opt_vec3(&p.fire_line_at),
                p.reel,
                if p.release { "true" } else { "false" },
                if p.pushoff { "true" } else { "false" },
                json_num(p.throw_charge),
                if p.throw_released { "true" } else { "false" },
                json_num(p.throw_spin),
                json_vec3(&p.thrumbler),
                if p.catch_intent { "true" } else { "false" },
            )
        })
        .collect();
    format!(
        "{{\"tick\":{},\"players\":[{}]}}",
        json_num(frame.tick),
        players.join(",")
    )
}

// ── RigAi (wasm-bindgen public surface) ─────────────────────────────────────

#[wasm_bindgen]
pub struct RigAi {
    sys: AiSystem,
}

#[wasm_bindgen]
impl RigAi {
    /// Create a fresh AI orchestrator with empty director / commitment caches.
    #[wasm_bindgen(constructor)]
    pub fn new() -> RigAi {
        RigAi { sys: AiSystem::new() }
    }

    /// Run one AI tick. Parses the three frozen-contract JSON args, calls
    /// `AiSystem::tick`, and returns the resulting `InputFrame` as JSON.
    pub fn tick(
        &mut self,
        sim_json: &str,
        match_json: &str,
        configs_json: &str,
        seed: u32,
    ) -> String {
        let sim = parse_sim_state(sim_json);
        let mat = parse_match_state(match_json);
        let configs = parse_configs(configs_json);
        let frame = self.sys.tick(&sim, &mat, &configs, seed);
        emit_input_frame(&frame)
    }

    /// Reset director + commitment caches (e.g., on a new inning).
    pub fn reset(&mut self) {
        self.sys.reset();
    }
}

impl Default for RigAi {
    fn default() -> Self {
        RigAi::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built representative JSON for the frozen contract.
    fn sim_json() -> String {
        r#"{
          "tick": 0,
          "omega": 0.5,
          "bell": {
            "p": {"x":0.0,"y":0.0,"z":0.0},
            "v": {"x":0.0,"y":0.0,"z":0.0},
            "q": {"x":0.0,"y":0.0,"z":0.0,"w":1.0},
            "w": {"x":0.0,"y":0.0,"z":0.0},
            "chime": 1.0,
            "heldBy": "H1",
            "thrownBy": null,
            "touchedSinceThrow": false,
            "releasePos": {"x":0.0,"y":0.0,"z":0.0},
            "releaseTick": 0,
            "passChain": ["H1","A1"]
          },
          "players": [
            {"id":"H1","team":"home","role":"spinner","p":{"x":-40.0,"y":4.0,"z":0.0},"v":{"x":0.0,"y":0.0,"z":0.0},"q":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"line":null,"dvBudget":6.0,"contactRef":null,"grounded":false},
            {"id":"H2","team":"home","role":"anchor","p":{"x":-90.0,"y":4.0,"z":0.0},"v":{"x":0.0,"y":0.0,"z":0.0},"q":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"line":{"anchorType":"spar","anchorRef":null,"anchorPos":{"x":-90.0,"y":0.0,"z":0.0},"restLen":12.0,"taut":true},"dvBudget":6.0,"contactRef":null,"grounded":false},
            {"id":"A1","team":"away","role":"spinner","p":{"x":40.0,"y":4.0,"z":0.0},"v":{"x":0.0,"y":0.0,"z":0.0},"q":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"line":null,"dvBudget":6.0,"contactRef":null,"grounded":false},
            {"id":"A2","team":"away","role":"reach","p":{"x":90.0,"y":4.0,"z":0.0},"v":{"x":0.0,"y":0.0,"z":0.0},"q":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"line":null,"dvBudget":6.0,"contactRef":null,"grounded":false}
          ]
        }"#
        .to_string()
    }

    fn match_json() -> String {
        r#"{
          "inning": 1,
          "spine": false,
          "possession": "home",
          "faithEnd": "+x",
          "cast": {"throwsLeft": 3, "gate": "first", "spotX": 0.0},
          "contest": null,
          "scoreHome": 0,
          "scoreAway": 0,
          "phase": "live",
          "message": "play ball",
          "winner": null
        }"#
        .to_string()
    }

    fn configs_json() -> String {
        r#"[
          {"side":"home","profile":{"freeEndBias":0.2,"loopPropensity":0.15,"aggression":0.7,"grappleRisk":0.3,"contestAggression":0.75,"snatchVsClatter":0.35,"awayPointBias":0.3,"variance":0.15},"difficulty":"pro"},
          {"side":"away","profile":{"freeEndBias":0.9,"loopPropensity":0.85,"aggression":0.85,"grappleRisk":0.9,"contestAggression":0.75,"snatchVsClatter":0.75,"awayPointBias":0.45,"variance":0.95},"difficulty":"pro"}
        ]"#
        .to_string()
    }

    #[test]
    fn tick_returns_well_shaped_input_frame() {
        let mut ai = RigAi::new();
        let out = ai.tick(&sim_json(), &match_json(), &configs_json(), 1234);
        assert!(out.contains("\"players\""), "missing players: {out}");
        assert!(out.starts_with("{\"tick\":"));
        // All four non-P1 players are AI-controlled.
        assert!(out.contains("\"id\":\"H1\""));
        assert!(out.contains("\"id\":\"A2\""));
        // No NaN/Inf leaked into the JSON.
        assert!(!out.contains("NaN"));
        assert!(!out.contains("inf"));
        // The required InputFrame field keys are present.
        for k in [
            "\"aim\"",
            "\"fireLineAt\"",
            "\"reel\"",
            "\"release\"",
            "\"pushoff\"",
            "\"throwCharge\"",
            "\"throwReleased\"",
            "\"throwSpin\"",
            "\"thrumbler\"",
        ] {
            assert!(out.contains(k), "missing {k} in {out}");
        }
    }

    #[test]
    fn tick_is_deterministic_fresh_instance_same_seed() {
        let a = RigAi::new().tick(&sim_json(), &match_json(), &configs_json(), 99);
        let b = RigAi::new().tick(&sim_json(), &match_json(), &configs_json(), 99);
        assert_eq!(a, b, "same seed + fresh instance must be bit-identical");
    }

    #[test]
    fn reset_clears_then_still_ticks() {
        let mut ai = RigAi::new();
        let _ = ai.tick(&sim_json(), &match_json(), &configs_json(), 7);
        ai.reset();
        let out = ai.tick(&sim_json(), &match_json(), &configs_json(), 7);
        assert!(out.contains("\"players\""));
    }

    #[test]
    fn vec3_and_quat_parsers_round_trip() {
        let src = r#"{"a":{"x":1.5,"y":-2.25,"z":3.75},"q":{"x":0.1,"y":0.2,"z":0.3,"w":0.9}}"#;
        let v = parse_vec3(src, "a");
        assert_eq!(v, Vec3 { x: 1.5, y: -2.25, z: 3.75 });
        let q = parse_quat(src, "q");
        assert_eq!(q, Quat { x: 0.1, y: 0.2, z: 0.3, w: 0.9 });
        // missing key → zero vec / identity quat
        assert_eq!(parse_vec3(src, "nope"), Vec3 { x: 0.0, y: 0.0, z: 0.0 });
        assert_eq!(parse_quat(src, "nope"), Quat::ident());
        // optional vec3: present vs null/missing
        assert_eq!(parse_opt_vec3(src, "a"), Some(Vec3 { x: 1.5, y: -2.25, z: 3.75 }));
        assert_eq!(parse_opt_vec3(r#"{"a":null}"#, "a"), None);
    }

    #[test]
    fn enum_parsers_map_every_variant() {
        let s = r#"{"team":"away","role":"faithwing","fe":"-x","at":"ring","g":"mouth","ph":"inning_break","d":"cross","df":"legend","w":"away"}"#;
        assert_eq!(parse_team(s, "team"), TeamSide::Away);
        assert_eq!(parse_role(s, "role"), RiggerRole::Faithwing);
        assert_eq!(parse_faith_end(s, "fe"), FaithEnd::MinusX);
        assert_eq!(parse_anchor_type(s, "at"), AnchorType::Ring);
        assert_eq!(parse_gate(s, "g"), Gate::Mouth);
        assert_eq!(parse_phase(s, "ph"), MatchPhase::InningBreak);
        assert_eq!(parse_direction(s, "d"), ContestDirection::Cross);
        assert_eq!(parse_difficulty(s, "df"), Difficulty::Legend);
        assert_eq!(parse_opt_team(s, "w"), Some(TeamSide::Away));
        // defaults on missing
        assert_eq!(parse_team("{}", "team"), TeamSide::Home);
        assert_eq!(parse_phase("{}", "ph"), MatchPhase::Live);
        assert_eq!(parse_opt_team("{}", "w"), None);
        // "final" / "spine" phase + null winner
        assert_eq!(parse_phase(r#"{"ph":"final"}"#, "ph"), MatchPhase::Final);
        assert_eq!(parse_phase(r#"{"ph":"spine"}"#, "ph"), MatchPhase::Spine);
        assert_eq!(parse_opt_team(r#"{"w":null}"#, "w"), None);
    }

    #[test]
    fn str_array_and_null_helpers() {
        assert_eq!(
            parse_str_array(r#"{"c":["H1","A1","P1"]}"#, "c"),
            vec!["H1".to_string(), "A1".to_string(), "P1".to_string()]
        );
        assert_eq!(parse_str_array(r#"{"c":[]}"#, "c"), Vec::<String>::new());
        assert!(is_null_field(r#"{"x":null}"#, "x"));
        assert!(!is_null_field(r#"{"x":1}"#, "x"));
        // configs array parses both teams
        let cfgs = parse_configs(&configs_json());
        assert_eq!(cfgs.len(), 2);
        assert_eq!(cfgs[0].side, TeamSide::Home);
        assert_eq!(cfgs[1].side, TeamSide::Away);
    }
}

//! scoring.rs — pure scoring attribution for RIG bell_through_ring events.
//!
//! 1:1 port of `src/match/Scoring.ts` (FROZEN contract; codes only against the
//! `SimEvent` / `SimState` / `MatchState` shapes mirrored here).
//!
//! Canonical point table (rules.md §6 / sport.md §3.5):
//!   Fall  = 2  — bell cleanly through the Faith (spinward) ring, touched OK
//!   Rise  = 5  — bell through the Free (antispinward) ring
//!   Loop  = 7  — untouched closed Coriolis arc (loop_tier == Loop); ends inning
//!   Curl  = base+2, capped below Loop:
//!            Faith-curl = 4 (Fall base 2 + 2), Free-curl = 6 (Rise base 5+2=7 → cap 6)
//!   Ground = 1 — awarded to the DEFENSE when attacker is skinned
//!
//! Faith end identification:
//!   faith_end == PlusX → ring at +x is Faith (spinward), ring at -x is Free.
//!   faith_end == MinusX → ring at -x is Faith, ring at +x is Free.
//!
//! Pass-chain-safe attribution mirrors the TS exactly:
//!   prefer the last `pass_chain` entry's player team; fall back to `thrown_by`,
//!   then `held_by`. `None` if attribution is impossible.

use crate::math::Vec3;

// ─── Shared shapes mirrored from src/sim/types.ts ────────────────────────────

/// Mirrors TS `TeamSide = 'home' | 'away'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeamSide {
    Home,
    Away,
}

impl TeamSide {
    /// The opposing side.
    pub fn other(self) -> TeamSide {
        match self {
            TeamSide::Home => TeamSide::Away,
            TeamSide::Away => TeamSide::Home,
        }
    }
}

/// Mirrors the TS `'+x' | '-x'` ring-end / faith-end literal union.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingEnd {
    PlusX,
    MinusX,
}

/// Mirrors the TS `loopTier: 'loop' | 'curl' | 'none'` literal union.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopTier {
    Loop,
    Curl,
    None,
}

/// Mirrors the TS `RiggerRole` union (only carried for shape fidelity).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiggerRole {
    Anchor,
    Spinner,
    Faithwing,
    Freewing,
    Reach,
}

/// Mirrors the subset of TS `PlayerSim` the match/scoring logic reads.
#[derive(Clone, Debug)]
pub struct PlayerSim {
    pub id: String,
    pub team: TeamSide,
    pub role: RiggerRole,
    pub p: Vec3,
    pub v: Vec3,
}

/// Mirrors the subset of TS `BellState` the match/scoring logic reads.
#[derive(Clone, Debug)]
pub struct BellState {
    pub p: Vec3,
    pub v: Vec3,
    pub held_by: Option<String>,
    pub thrown_by: Option<String>,
    pub touched_since_throw: bool,
    pub pass_chain: Vec<String>,
}

/// Mirrors the subset of TS `SimState` the match/scoring logic reads.
#[derive(Clone, Debug)]
pub struct SimState {
    pub tick: u64,
    pub omega: f64,
    pub bell: BellState,
    pub players: Vec<PlayerSim>,
}

/// Local mirror of `src/sim/types.ts` `SimEvent`. Only the variants the match
/// logic reads carry the fields it consumes; the rest are ported faithfully.
#[derive(Clone, Debug, PartialEq)]
pub enum SimEvent {
    BellThroughRing {
        end: RingEnd,
        touched: bool,
        loop_tier: LoopTier,
    },
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

// ─── ScoreResult (mirror of TS ScoreResult / ScoreKind) ──────────────────────

/// Mirrors TS `ScoreKind = 'fall' | 'rise' | 'loop' | 'curl' | 'ground'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreKind {
    Fall,
    Rise,
    Loop,
    Curl,
    Ground,
}

impl ScoreKind {
    /// `kind.toUpperCase()` parity for the MatchStateMachine message string.
    pub fn upper(self) -> &'static str {
        match self {
            ScoreKind::Fall => "FALL",
            ScoreKind::Rise => "RISE",
            ScoreKind::Loop => "LOOP",
            ScoreKind::Curl => "CURL",
            ScoreKind::Ground => "GROUND",
        }
    }
}

/// Mirrors TS `ScoreResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoreResult {
    pub team: TeamSide,
    pub kind: ScoreKind,
    pub points: i64,
    pub ends_inning: bool,
}

// ─── Internal helpers (1:1 with TS) ──────────────────────────────────────────

/// Is this ring-end the Faith end? (TS `isFaithEnd`)
fn is_faith_end(end: RingEnd, faith_end: RingEnd) -> bool {
    end == faith_end
}

/// Determine the attacking team from the bell state. (TS `attackingTeam`)
///
/// Prefer the last `pass_chain` entry; fall back to `thrown_by`, then
/// `held_by`. Returns `None` if attribution is impossible.
fn attacking_team(sim: &SimState) -> Option<TeamSide> {
    let chain = &sim.bell.pass_chain;
    if let Some(last_id) = chain.last() {
        if let Some(pl) = sim.players.iter().find(|p| &p.id == last_id) {
            return Some(pl.team);
        }
    }
    if let Some(thrown_by) = &sim.bell.thrown_by {
        if let Some(pl) = sim.players.iter().find(|p| &p.id == thrown_by) {
            return Some(pl.team);
        }
    }
    if let Some(held_by) = &sim.bell.held_by {
        if let Some(pl) = sim.players.iter().find(|p| &p.id == held_by) {
            return Some(pl.team);
        }
    }
    None
}

// ─── Public API (1:1 with TS scoreFor / groundFor) ───────────────────────────

/// Map a `bell_through_ring` event to a scoring result.
///
/// Returns `None` if attribution fails (no known thrower) or if `ev` is not a
/// `BellThroughRing` (the TS signature is statically narrowed; here we mirror
/// that by returning `None` for any other variant).
///
/// Port of `scoreFor` (Scoring.ts lines 70–101).
pub fn score_for(ev: &SimEvent, faith_end: RingEnd, sim: &SimState) -> Option<ScoreResult> {
    let (end, touched, loop_tier) = match ev {
        SimEvent::BellThroughRing {
            end,
            touched,
            loop_tier,
        } => (*end, *touched, *loop_tier),
        _ => return None,
    };

    let attacker = attacking_team(sim)?;

    let faith = is_faith_end(end, faith_end);
    let untouched = !touched;

    // Loop: untouched closed Coriolis arc.
    if loop_tier == LoopTier::Loop && untouched {
        return Some(ScoreResult {
            team: attacker,
            kind: ScoreKind::Loop,
            points: 7,
            ends_inning: true,
        });
    }

    // Curl: partial Coriolis arc, untouched.
    if loop_tier == LoopTier::Curl && untouched {
        let base: i64 = if faith { 2 } else { 5 };
        let points = (base + 2).min(6);
        return Some(ScoreResult {
            team: attacker,
            kind: ScoreKind::Curl,
            points,
            ends_inning: false,
        });
    }

    // Fall (Faith ring, any touch state for ordinary score).
    if faith {
        return Some(ScoreResult {
            team: attacker,
            kind: ScoreKind::Fall,
            points: 2,
            ends_inning: false,
        });
    }

    // Rise (Free ring).
    Some(ScoreResult {
        team: attacker,
        kind: ScoreKind::Rise,
        points: 5,
        ends_inning: false,
    })
}

/// Map a `player_skinned` event to a Ground score for the defense.
///
/// Ground = 1 pt to the team that is NOT the attacker, and only when the
/// skinned player is on the attacking team. Returns `None` if attribution
/// fails, the player is unknown, the player is a defender, or `ev` is not a
/// `PlayerSkinned` variant.
///
/// Port of `groundFor` (Scoring.ts lines 110–126).
pub fn ground_for(ev: &SimEvent, sim: &SimState) -> Option<ScoreResult> {
    let id = match ev {
        SimEvent::PlayerSkinned { id } => id,
        _ => return None,
    };

    let attacker = attacking_team(sim)?;

    let skinned = sim.players.iter().find(|p| &p.id == id)?;
    if skinned.team != attacker {
        return None; // defender skinned → no ground
    }

    let defense = attacker.other();
    Some(ScoreResult {
        team: defense,
        kind: ScoreKind::Ground,
        points: 1,
        ends_inning: false,
    })
}

// ─── Tests (mirror test/match.test.ts "Scoring.scoreFor" describe block) ─────

#[cfg(test)]
mod tests {
    use super::*;

    fn zero() -> Vec3 {
        Vec3::new(0.0, 0.0, 0.0)
    }

    fn make_player(id: &str, team: TeamSide) -> PlayerSim {
        PlayerSim {
            id: id.to_string(),
            team,
            role: RiggerRole::Spinner,
            p: zero(),
            v: zero(),
        }
    }

    fn make_sim(
        thrown_by: Option<&str>,
        pass_chain: &[&str],
        held_by: Option<&str>,
        bell_x: f64,
        players: Vec<PlayerSim>,
    ) -> SimState {
        SimState {
            tick: 0,
            omega: 0.32,
            bell: BellState {
                p: Vec3::new(bell_x, 0.0, 0.0),
                v: zero(),
                held_by: held_by.map(|s| s.to_string()),
                thrown_by: thrown_by.map(|s| s.to_string()),
                touched_since_throw: false,
                pass_chain: pass_chain.iter().map(|s| s.to_string()).collect(),
            },
            players,
        }
    }

    #[test]
    fn fall_touched_faith_ring_2pts_attacker() {
        let sim = make_sim(None, &["hp"], None, 0.0, vec![make_player("hp", TeamSide::Home)]);
        let ev = SimEvent::BellThroughRing {
            end: RingEnd::PlusX,
            touched: true,
            loop_tier: LoopTier::None,
        };
        let r = score_for(&ev, RingEnd::PlusX, &sim).unwrap();
        assert_eq!(r.kind, ScoreKind::Fall);
        assert_eq!(r.points, 2);
        assert_eq!(r.team, TeamSide::Home);
        assert!(!r.ends_inning);
    }

    #[test]
    fn rise_free_ring_5pts_attacker() {
        let sim = make_sim(None, &["ap"], None, 0.0, vec![make_player("ap", TeamSide::Away)]);
        let ev = SimEvent::BellThroughRing {
            end: RingEnd::MinusX,
            touched: true,
            loop_tier: LoopTier::None,
        };
        let r = score_for(&ev, RingEnd::PlusX, &sim).unwrap();
        assert_eq!(r.kind, ScoreKind::Rise);
        assert_eq!(r.points, 5);
        assert_eq!(r.team, TeamSide::Away);
        assert!(!r.ends_inning);
    }

    #[test]
    fn loop_untouched_7pts_ends_inning() {
        let sim = make_sim(None, &["hp"], None, 0.0, vec![make_player("hp", TeamSide::Home)]);
        let ev = SimEvent::BellThroughRing {
            end: RingEnd::PlusX,
            touched: false,
            loop_tier: LoopTier::Loop,
        };
        let r = score_for(&ev, RingEnd::PlusX, &sim).unwrap();
        assert_eq!(r.kind, ScoreKind::Loop);
        assert_eq!(r.points, 7);
        assert!(r.ends_inning);
    }

    #[test]
    fn loop_greater_than_curl_always() {
        let sim = make_sim(None, &["hp"], None, 0.0, vec![make_player("hp", TeamSide::Home)]);
        let loop_pts = score_for(
            &SimEvent::BellThroughRing {
                end: RingEnd::PlusX,
                touched: false,
                loop_tier: LoopTier::Loop,
            },
            RingEnd::PlusX,
            &sim,
        )
        .unwrap()
        .points;
        let faith_curl = score_for(
            &SimEvent::BellThroughRing {
                end: RingEnd::PlusX,
                touched: false,
                loop_tier: LoopTier::Curl,
            },
            RingEnd::PlusX,
            &sim,
        )
        .unwrap()
        .points;
        let free_curl = score_for(
            &SimEvent::BellThroughRing {
                end: RingEnd::MinusX,
                touched: false,
                loop_tier: LoopTier::Curl,
            },
            RingEnd::PlusX,
            &sim,
        )
        .unwrap()
        .points;
        assert!(faith_curl < loop_pts);
        assert!(free_curl < loop_pts);
    }

    #[test]
    fn faith_curl_4pts() {
        let sim = make_sim(None, &["hp"], None, 0.0, vec![make_player("hp", TeamSide::Home)]);
        let ev = SimEvent::BellThroughRing {
            end: RingEnd::PlusX,
            touched: false,
            loop_tier: LoopTier::Curl,
        };
        let r = score_for(&ev, RingEnd::PlusX, &sim).unwrap();
        assert_eq!(r.kind, ScoreKind::Curl);
        assert_eq!(r.points, 4);
    }

    #[test]
    fn free_curl_6pts_capped_below_loop() {
        let sim = make_sim(None, &["ap"], None, 0.0, vec![make_player("ap", TeamSide::Away)]);
        let ev = SimEvent::BellThroughRing {
            end: RingEnd::MinusX,
            touched: false,
            loop_tier: LoopTier::Curl,
        };
        let r = score_for(&ev, RingEnd::PlusX, &sim).unwrap();
        assert_eq!(r.kind, ScoreKind::Curl);
        assert_eq!(r.points, 6);
        assert!(r.points < 7);
    }

    #[test]
    fn ground_attacker_skinned_1pt_defense() {
        let sim = make_sim(
            None,
            &["hp"],
            None,
            0.0,
            vec![
                make_player("hp", TeamSide::Home),
                make_player("ap", TeamSide::Away),
            ],
        );
        let ev = SimEvent::PlayerSkinned { id: "hp".into() };
        let r = ground_for(&ev, &sim).unwrap();
        assert_eq!(r.kind, ScoreKind::Ground);
        assert_eq!(r.points, 1);
        assert_eq!(r.team, TeamSide::Away);
        assert!(!r.ends_inning);
    }

    #[test]
    fn ground_defender_skinned_null() {
        let sim = make_sim(
            None,
            &["hp"],
            None,
            0.0,
            vec![
                make_player("hp", TeamSide::Home),
                make_player("ap", TeamSide::Away),
            ],
        );
        let ev = SimEvent::PlayerSkinned { id: "ap".into() };
        assert!(ground_for(&ev, &sim).is_none());
    }

    #[test]
    fn pass_chain_last_entry_determines_team() {
        let sim = make_sim(
            None,
            &["hA", "aB", "hC"],
            None,
            0.0,
            vec![
                make_player("hA", TeamSide::Home),
                make_player("aB", TeamSide::Away),
                make_player("hC", TeamSide::Home),
            ],
        );
        let ev = SimEvent::BellThroughRing {
            end: RingEnd::PlusX,
            touched: true,
            loop_tier: LoopTier::None,
        };
        let r = score_for(&ev, RingEnd::PlusX, &sim).unwrap();
        assert_eq!(r.team, TeamSide::Home);
    }

    #[test]
    fn score_for_null_when_no_attribution() {
        let sim = make_sim(None, &[], None, 0.0, vec![]);
        let ev = SimEvent::BellThroughRing {
            end: RingEnd::PlusX,
            touched: true,
            loop_tier: LoopTier::None,
        };
        assert!(score_for(&ev, RingEnd::PlusX, &sim).is_none());
    }
}

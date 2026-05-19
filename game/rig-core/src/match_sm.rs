//! match_sm.rs — RIG match-rules layer.
//!
//! 1:1 port of `src/match/MatchStateMachine.ts` (FROZEN contract). No DOM, no
//! time, no rng — pure deterministic reducer / state machine.
//!
//! Field geometry (sport.md §3.1): field is 640 m long; goal rings at ±320 m.
//! Three gates divide the half-field:
//!   first : 0.25 × 320 =  80 m
//!   deep  : 0.55 × 320 = 176 m
//!   mouth : 0.85 × 320 = 272 m
//! home attacks toward +x; away attacks toward -x. Signed gate x = GATE_ABS·dir.
//!
//! Cast/downs, contest, scoring, innings and the `MatchUpdate` return shape are
//! ported verbatim from the TS (see method-level comments for source spans).

use crate::contest::{ContestDirection, ContestState, ContestWinner};
use crate::scoring::{
    ground_for, score_for, RingEnd, ScoreResult, SimEvent, SimState, TeamSide,
};
use crate::tuning;

// ─── Gate geometry (MatchStateMachine.ts lines 54–80) ────────────────────────

/// Half-field length (goal ring absolute x). TS `GATE_X = 320`; here
/// `tuning::GATE_X == L/2 == 320.0`.
const GATE_X: f64 = tuning::GATE_X;

/// Mirrors TS `'first' | 'deep' | 'mouth'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    First,
    Deep,
    Mouth,
}

/// TS `GATE_ABS` record.
fn gate_abs(g: Gate) -> f64 {
    match g {
        Gate::First => GATE_X * 0.25, // 80 m
        Gate::Deep => GATE_X * 0.55,  // 176 m
        Gate::Mouth => GATE_X * 0.85, // 272 m
    }
}

/// TS `GATE_ORDER = ['first','deep','mouth']`.
const GATE_ORDER: [Gate; 3] = [Gate::First, Gate::Deep, Gate::Mouth];

fn gate_order_index(g: Gate) -> usize {
    match g {
        Gate::First => 0,
        Gate::Deep => 1,
        Gate::Mouth => 2,
    }
}

/// Signed gate x: `dir` +1 toward +x, -1 toward -x. (TS `gateX`)
fn gate_x(g: Gate, dir: f64) -> f64 {
    gate_abs(g) * dir
}

/// Direction of attack for the current possession. (TS `attackDir`)
fn attack_dir(possession: TeamSide) -> f64 {
    match possession {
        TeamSide::Home => 1.0,
        TeamSide::Away => -1.0,
    }
}

// ─── State shapes (mirror of TS MatchState) ──────────────────────────────────

/// Mirrors TS `MatchPhase`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchPhase {
    Set,
    Live,
    Contest,
    Dead,
    InningBreak,
    Spine,
    Final,
}

/// Mirrors TS `MatchState['cast']`. `throws_left` domain is `0|1|2|3`.
#[derive(Clone, Debug, PartialEq)]
pub struct Cast {
    pub throws_left: u8,
    pub gate: Gate,
    pub spot_x: f64,
}

/// Mirrors TS `MatchState['contest']`'s inline object (the live contest).
#[derive(Clone, Debug, PartialEq)]
pub struct MatchContest {
    pub thrower: String,
    pub contester: String,
    pub count: u8,
    pub radius: f64,
    pub direction: ContestDirection,
}

/// Mirrors TS `MatchState`.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchState {
    pub inning: i64,
    pub spine: bool,
    pub possession: TeamSide,
    pub faith_end: RingEnd,
    pub cast: Cast,
    pub contest: Option<MatchContest>,
    pub score_home: i64,
    pub score_away: i64,
    pub phase: MatchPhase,
    pub message: String,
    pub winner: Option<TeamSide>,
}

/// Mirrors TS `MatchUpdate.resets` element (`'pushoff' | 'set'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reset {
    Pushoff,
    Set,
}

/// Mirrors TS `MatchUpdate`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatchUpdate {
    pub resets: Vec<Reset>,
    pub scored: Option<ScoreResult>,
    pub turnover: Option<Turnover>,
    pub inning_end: Option<InningEnd>,
    pub spine_start: bool,
    pub winner: Option<TeamSide>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turnover {
    pub spot_x: f64,
    pub team: TeamSide,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InningEnd {
    pub inning: i64,
}

/// Mirrors TS `{ winner: 'thrower' | 'contester' }` arg to `resolveContest`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContestOutcome {
    pub winner: ContestWinner,
}

// ─── Initial states (MatchStateMachine.ts lines 82–107) ──────────────────────

fn initial_cast(possession: TeamSide) -> Cast {
    let dir = attack_dir(possession);
    Cast {
        throws_left: 3,
        gate: Gate::First,
        spot_x: gate_x(Gate::First, dir),
    }
}

fn make_initial_state(faith_end: RingEnd, first_possession: TeamSide) -> MatchState {
    MatchState {
        inning: 1,
        spine: false,
        possession: first_possession,
        faith_end,
        cast: initial_cast(first_possession),
        contest: None,
        score_home: 0,
        score_away: 0,
        phase: MatchPhase::Set,
        message: String::new(),
        winner: None,
    }
}

// ─── Main state machine (mirror of TS class MatchStateMachine) ───────────────

#[derive(Clone)]
pub struct MatchStateMachine {
    state: MatchState,
}

impl MatchStateMachine {
    /// `faith_end` — which ring is the spinward/Faith end.
    /// `first_possession` — which team has the opening cast.
    pub fn new(faith_end: RingEnd, first_possession: TeamSide) -> Self {
        Self {
            state: make_initial_state(faith_end, first_possession),
        }
    }

    /// TS `get state`.
    pub fn state(&self) -> &MatchState {
        &self.state
    }

    /// TS `get pendingContest`: the frozen contest iff phase == Contest.
    pub fn pending_contest(&self) -> Option<&MatchContest> {
        if self.state.phase == MatchPhase::Contest {
            self.state.contest.as_ref()
        } else {
            None
        }
    }

    /// Consume a batch of SimEvents from one sim step. (TS `consume`, 155–164)
    pub fn consume(&mut self, events: &[SimEvent], sim: &SimState) -> MatchUpdate {
        let mut update = MatchUpdate::default();
        for ev in events {
            if self.state.winner.is_some() {
                break;
            }
            self.handle_event(ev, sim, &mut update);
        }
        update
    }

    /// Re-cast a dead / inning-break ball back to LIVE, preserving the cast.
    /// (TS `resumeLive`, 172–183)
    pub fn resume_live(&mut self) {
        let phase = self.state.phase;
        if self.state.winner.is_some()
            || phase == MatchPhase::Final
            || phase == MatchPhase::Live
            || phase == MatchPhase::Contest
        {
            return;
        }
        self.state.phase = MatchPhase::Live;
    }

    /// Resolve a pending contest. (TS `resolveContest`, 188–206)
    pub fn resolve_contest(&mut self, result: ContestOutcome, _sim: &SimState) -> MatchUpdate {
        let mut update = MatchUpdate::default();
        self.resolve_contest_into(result, &mut update);
        update
    }

    /// Resolve a pending contest, merging effects into an existing update
    /// (so a physics-driven contest resolved mid-`consume` still propagates
    /// its turnover / inning-end to the caller).
    fn resolve_contest_into(&mut self, result: ContestOutcome, update: &mut MatchUpdate) {
        if self.state.phase != MatchPhase::Contest || self.state.contest.is_none() {
            return;
        }
        match result.winner {
            ContestWinner::Thrower => {
                self.state.contest = None;
                self.state.phase = MatchPhase::Live;
                self.state.message = "Contest complete — cast continues".to_string();
            }
            ContestWinner::Contester => {
                self.apply_turnover(update);
            }
        }
    }

    // ─── Private: event dispatch (TS _handleEvent, 210–250) ──────────────────

    fn handle_event(&mut self, ev: &SimEvent, sim: &SimState, update: &mut MatchUpdate) {
        if self.state.phase == MatchPhase::Final || self.state.winner.is_some() {
            return;
        }

        if let SimEvent::FoulGarrote { .. } = ev {
            self.on_foul_garrote(update);
            return;
        }

        let phase = self.state.phase;
        if phase == MatchPhase::Dead || phase == MatchPhase::InningBreak {
            return;
        }

        // PART B — physics-resolved contest. A `ContestStarted` emitted from
        // real loose-bell play (sim_world.rs) puts the match in Contest
        // phase. The duel is then resolved by the ACTUAL physics, with NO
        // rng (determinism is sacred — `resolve_contest`'s dice path stays
        // for the headless statistical layer only): whoever the sim awards
        // the loose bell to decides it. A catch by the CONTESTER's team is
        // a won contest → turnover; a catch by the THROWER's team, or the
        // bell going dead/through-ring, means the contester failed → the
        // cast continues (thrower wins). Deterministic, id-driven.
        if phase == MatchPhase::Contest {
            if let Some(c) = self.state.contest.clone() {
                match ev {
                    SimEvent::BellCaught { by } => {
                        let by_team = self.team_of_sim(sim, by);
                        let contester_team = self.team_of_sim(sim, &c.contester);
                        let winner = if by_team.is_some() && by_team == contester_team {
                            ContestWinner::Contester
                        } else {
                            ContestWinner::Thrower
                        };
                        self.resolve_contest_into(ContestOutcome { winner }, update);
                        return;
                    }
                    SimEvent::BellMissed { .. } => {
                        // Loose bell died with no one securing it — the
                        // contester did not win possession: cast continues.
                        self.resolve_contest_into(
                            ContestOutcome { winner: ContestWinner::Thrower },
                            update,
                        );
                        return;
                    }
                    SimEvent::BellThroughRing { .. } => {
                        // Score during a contest: clear the contest, then
                        // score normally (the through-ring path ends the
                        // inning anyway).
                        self.state.phase = MatchPhase::Live;
                        self.state.contest = None;
                        self.on_bell_through_ring(ev, sim, update);
                        return;
                    }
                    SimEvent::FoulGarrote { .. } => {
                        self.on_foul_garrote(update);
                        return;
                    }
                    _ => {
                        // Bobble/clatter/skin/etc.: contest still in flight.
                        return;
                    }
                }
            }
        }

        match ev {
            SimEvent::BellThroughRing { .. } => self.on_bell_through_ring(ev, sim, update),
            SimEvent::ContestStarted { .. } => {
                if phase == MatchPhase::Live {
                    self.on_contest_started(ev);
                }
            }
            SimEvent::PlayerSkinned { .. } => self.on_player_skinned(ev, sim),
            SimEvent::BellCaught { .. }
            | SimEvent::BellBobble { .. }
            | SimEvent::BellClatter { .. } => {
                if phase == MatchPhase::Live {
                    // A clean catch by a TEAMMATE (same team as the
                    // possessing/attacking side) is a COMPLETED PASS, not a
                    // failed throw — it must not burn a down on its own.
                    // Bobble/clatter and any opponent catch are still
                    // failed/contested throws and spend the down.
                    let completed_pass = match ev {
                        SimEvent::BellCaught { by } => {
                            self.team_of_sim(sim, by) == Some(self.state.possession)
                        }
                        _ => false,
                    };
                    self.on_throw_spent(sim, update, completed_pass);
                }
            }
            SimEvent::BellMissed { .. } => {
                if phase == MatchPhase::Live {
                    self.on_bell_missed(sim, update);
                }
            }
            SimEvent::BellSkin => {
                // Bell bouncing off skin: no throw consumed, no score.
            }
            // foul_garrote handled above.
            SimEvent::FoulGarrote { .. } => {}
        }
    }

    // ─── Private: specific events ────────────────────────────────────────────

    /// TS `_onBellThroughRing` (254–279).
    fn on_bell_through_ring(&mut self, ev: &SimEvent, sim: &SimState, update: &mut MatchUpdate) {
        let result = match score_for(ev, self.state.faith_end, sim) {
            None => {
                self.state.message = "Bell through ring — no attribution".to_string();
                return;
            }
            Some(r) => r,
        };

        if result.team == TeamSide::Home {
            self.state.score_home += result.points;
        }
        if result.team == TeamSide::Away {
            self.state.score_away += result.points;
        }

        update.scored = Some(result);
        let team = match result.team {
            TeamSide::Home => "home",
            TeamSide::Away => "away",
        };
        self.state.message = format!("{} +{} ({})", result.kind.upper(), result.points, team);

        self.end_inning(update, false);
    }

    /// TS `_onContestStarted` (281–300).
    fn on_contest_started(&mut self, ev: &SimEvent) {
        let (thrower, contester) = match ev {
            SimEvent::ContestStarted { thrower, contester } => (thrower.clone(), contester.clone()),
            _ => return,
        };
        let gate = self.state.cast.gate;
        let direction = if gate == Gate::Mouth {
            ContestDirection::Cross
        } else {
            ContestDirection::Fair
        };
        let radius = gate_abs(gate);

        self.state.phase = MatchPhase::Contest;
        let gate_name = match gate {
            Gate::First => "first",
            Gate::Deep => "deep",
            Gate::Mouth => "mouth",
        };
        self.state.message = format!("Contest: {} vs {} at {}", thrower, contester, gate_name);
        self.state.contest = Some(MatchContest {
            thrower,
            contester,
            count: 0,
            radius,
            direction,
        });
    }

    /// TS `_onPlayerSkinned` (302–319).
    fn on_player_skinned(&mut self, ev: &SimEvent, sim: &SimState) {
        let result = match ground_for(ev, sim) {
            None => return,
            Some(r) => r,
        };
        if result.team == TeamSide::Home {
            self.state.score_home += result.points;
        }
        if result.team == TeamSide::Away {
            self.state.score_away += result.points;
        }
        let team = match result.team {
            TeamSide::Home => "home",
            TeamSide::Away => "away",
        };
        self.state.message = format!("GROUND +1 ({})", team);
        // Ground does not end inning or change possession.
    }

    /// Team of a player id in the sim, or `None` if unknown.
    fn team_of_sim(&self, sim: &SimState, id: &str) -> Option<TeamSide> {
        sim.players.iter().find(|p| p.id == id).map(|p| p.team)
    }

    /// TS `_onThrowSpent` (321–356).
    fn on_throw_spent(
        &mut self,
        sim: &SimState,
        update: &mut MatchUpdate,
        completed_pass: bool,
    ) {
        let dir = attack_dir(self.state.possession);
        let bell_x = sim.bell.p.x;

        let mut cleared_idx: i64 = -1;
        for (i, g) in GATE_ORDER.iter().enumerate() {
            let gx = gate_x(*g, dir);
            let passed = if dir == 1.0 { bell_x >= gx } else { bell_x <= gx };
            if passed {
                cleared_idx = i as i64;
            } else {
                break;
            }
        }
        let cur_idx = gate_order_index(self.state.cast.gate) as i64;

        if cleared_idx >= cur_idx {
            let new_idx =
                ((cleared_idx + 1).min(GATE_ORDER.len() as i64 - 1)) as usize;
            let new_gate = GATE_ORDER[new_idx];
            self.state.cast = Cast {
                throws_left: 3,
                gate: new_gate,
                spot_x: gate_x(new_gate, dir),
            };
            let gate_name = match new_gate {
                Gate::First => "first",
                Gate::Deep => "deep",
                Gate::Mouth => "mouth",
            };
            self.state.message = format!("Gate cleared → {}", gate_name);
        } else if completed_pass {
            // Completed pass that did not clear a gate: possession retained
            // by a teammate, the cast LIVES — do NOT burn a down. Track the
            // ball's new spot so later gate progress measures from here.
            self.state.cast.spot_x = bell_x;
            self.state.message = "Pass completed — cast continues".to_string();
        } else {
            // Did not clear current gate and the throw was NOT retained by a
            // teammate (failed/contested/opponent) — burn a throw.
            let new_left = self.state.cast.throws_left - 1;
            self.state.cast.throws_left = new_left;
            self.state.cast.spot_x = bell_x;
            if new_left == 0 {
                self.apply_turnover(update);
            }
        }
    }

    /// TS `_onBellMissed` (365–387).
    fn on_bell_missed(&mut self, sim: &SimState, update: &mut MatchUpdate) {
        let bell_x = sim.bell.p.x;
        let spot_x = bell_x.clamp(-GATE_X, GATE_X);

        // A miss is a FAILED throw — always burns one. TS subtracts then
        // checks `<= 0` (throws_left is 0|1|2|3 so this is 0). We mirror the
        // numeric check using i64 to allow the `<= 0` semantics exactly.
        let new_left: i64 = self.state.cast.throws_left as i64 - 1;
        if new_left <= 0 {
            self.state.cast.spot_x = spot_x;
            self.apply_turnover(update);
            return;
        }
        self.state.cast.throws_left = new_left as u8;
        self.state.cast.spot_x = spot_x;
        self.state.phase = MatchPhase::Dead;
        self.state.message = format!("Missed throw — {} left", new_left);
        update.resets.push(Reset::Set);
    }

    /// TS `_onFoulGarrote` (389–399).
    fn on_foul_garrote(&mut self, update: &mut MatchUpdate) {
        self.state.cast = initial_cast(self.state.possession);
        self.state.phase = MatchPhase::Live;
        let poss = match self.state.possession {
            TeamSide::Home => "home",
            TeamSide::Away => "away",
        };
        self.state.message = format!("Foul garrote — fresh cast for {}", poss);
        update.resets.push(Reset::Set);
    }

    // ─── Private: turnover & inning management ───────────────────────────────

    /// TS `_applyTurnover` (403–425).
    fn apply_turnover(&mut self, update: &mut MatchUpdate) {
        let spot_x = self.state.cast.spot_x;
        let new_possession = self.state.possession.other();

        update.turnover = Some(Turnover {
            spot_x,
            team: new_possession,
        });

        let mut cast = initial_cast(new_possession);
        cast.spot_x = spot_x;
        self.state.possession = new_possession;
        self.state.contest = None;
        self.state.cast = cast;
        self.state.phase = MatchPhase::Dead;
        let poss = match new_possession {
            TeamSide::Home => "home",
            TeamSide::Away => "away",
        };
        self.state.message = format!("Turnover → {} at x={:.1}", poss, spot_x);

        // Turnover-and-clear → inning ends; do NOT alternate possession again.
        self.end_inning(update, true);
    }

    /// TS `_endInning` (427–497).
    fn end_inning(&mut self, update: &mut MatchUpdate, preserve_possession: bool) {
        update.inning_end = Some(InningEnd {
            inning: self.state.inning,
        });

        if self.state.spine {
            // SUDDEN-DEATH CORRECTNESS: a spine inning ENDS the match only
            // when a team has actually pulled ahead. A turnover-and-clear in
            // a still-tied spine inning does NOT default-win for Away (the
            // old `score_home > score_away` → false bug): the spine
            // CONTINUES with the other team casting. Only a real score
            // (score_home != score_away) resolves it; the scorer wins.
            if self.state.score_home == self.state.score_away {
                let new_inning = self.state.inning + 1;
                let next = self.state.possession.other();
                self.state.inning = new_inning;
                self.state.possession = next;
                self.state.contest = None;
                self.state.cast = initial_cast(next);
                self.state.phase = MatchPhase::Spine;
                let poss = match next {
                    TeamSide::Home => "home",
                    TeamSide::Away => "away",
                };
                self.state.message =
                    format!("Spine still tied — {} to cast (sudden death)", poss);
                update.resets.push(Reset::Pushoff);
                update.resets.push(Reset::Set);
                return;
            }
            let winner = if self.state.score_home > self.state.score_away {
                TeamSide::Home
            } else {
                TeamSide::Away
            };
            self.state.winner = Some(winner);
            self.state.phase = MatchPhase::Final;
            let w = match winner {
                TeamSide::Home => "home",
                TeamSide::Away => "away",
            };
            self.state.message = format!("Spine won by {}", w);
            update.winner = Some(winner);
            return;
        }

        if self.state.inning >= 9 {
            if self.state.score_home == self.state.score_away {
                // Tie → spine.
                let free_end = if self.state.faith_end == RingEnd::PlusX {
                    RingEnd::MinusX
                } else {
                    RingEnd::PlusX
                };
                let spine_dir = if free_end == RingEnd::PlusX { 1.0 } else { -1.0 };
                let spine_first = self.state.possession.other();
                self.state.spine = true;
                self.state.phase = MatchPhase::Spine;
                self.state.inning += 1;
                self.state.possession = spine_first;
                self.state.contest = None;
                self.state.cast = Cast {
                    throws_left: 3,
                    gate: Gate::First,
                    spot_x: gate_x(Gate::First, spine_dir),
                };
                self.state.message =
                    "Tie after 9 innings — spine (sudden death at Free end)".to_string();
                update.spine_start = true;
                update.resets.push(Reset::Pushoff);
                update.resets.push(Reset::Set);
            } else {
                let winner = if self.state.score_home > self.state.score_away {
                    TeamSide::Home
                } else {
                    TeamSide::Away
                };
                self.state.winner = Some(winner);
                self.state.phase = MatchPhase::Final;
                self.state.message = format!(
                    "Final: home {} – away {}",
                    self.state.score_home, self.state.score_away
                );
                update.winner = Some(winner);
            }
        } else {
            let new_inning = self.state.inning + 1;
            let new_possession = if preserve_possession {
                self.state.possession
            } else {
                self.state.possession.other()
            };
            self.state.inning = new_inning;
            self.state.possession = new_possession;
            self.state.contest = None;
            self.state.cast = initial_cast(new_possession);
            self.state.phase = MatchPhase::InningBreak;
            let poss = match new_possession {
                TeamSide::Home => "home",
                TeamSide::Away => "away",
            };
            self.state.message = format!("Inning {} — {} to cast", new_inning, poss);
            update.resets.push(Reset::Pushoff);
            update.resets.push(Reset::Set);
        }
    }
}

// ─── Tests (mirror test/match.test.ts MSM describe blocks) ───────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::scoring::{BellState, LoopTier, PlayerSim, RiggerRole};

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

    fn make_sim(bell_x: f64, pass_chain: &[&str], players: Vec<PlayerSim>) -> SimState {
        SimState {
            tick: 0,
            omega: 0.32,
            bell: BellState {
                p: Vec3::new(bell_x, 0.0, 0.0),
                v: zero(),
                held_by: None,
                thrown_by: None,
                touched_since_throw: false,
                pass_chain: pass_chain.iter().map(|s| s.to_string()).collect(),
            },
            players,
        }
    }

    fn to_live(msm: &mut MatchStateMachine) {
        msm.consume(
            &[SimEvent::FoulGarrote {
                by: "__setup__".into(),
            }],
            &make_sim(0.0, &[], vec![]),
        );
    }

    // ── cast & turnover ──

    /// RE-BASELINE (Part D): "three unproductive throws → turnover" now
    /// means three FAILED throws. A `bell_caught` by a TEAMMATE is a
    /// COMPLETED PASS that retains possession and must NOT burn a down (the
    /// down-burn bug fix); only a failed/none-retained throw spends one. A
    /// missed throw (`bell_missed`) is the canonical failed throw and always
    /// burns; three of them exhaust the cast and turn the ball over.
    #[test]
    fn three_unproductive_throws_turnover() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        assert_eq!(msm.state().phase, MatchPhase::Live);

        let sim = make_sim(5.0, &["hp"], vec![make_player("hp", TeamSide::Home)]);
        let miss = SimEvent::BellMissed { end: RingEnd::PlusX };

        msm.consume(std::slice::from_ref(&miss), &sim);
        assert_eq!(msm.state().cast.throws_left, 2);
        msm.resume_live();
        msm.consume(std::slice::from_ref(&miss), &sim);
        assert_eq!(msm.state().cast.throws_left, 1);
        msm.resume_live();
        let upd = msm.consume(std::slice::from_ref(&miss), &sim);
        assert!(upd.turnover.is_some());
        assert_eq!(upd.turnover.unwrap().team, TeamSide::Away);
        assert!(upd.inning_end.is_some());
    }

    /// PART D — the down-burn bug fix, isolated: a completed pass (catch by
    /// a TEAMMATE that does not clear a gate) does NOT burn a down, while a
    /// failed throw (a missed throw, or a catch by an OPPONENT) does.
    #[test]
    fn completed_pass_keeps_down_failed_throw_burns_it() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        // Possession is Home. A teammate (Home) catch short of a gate is a
        // completed pass → cast continues, throws_left untouched.
        let sim_team = make_sim(
            5.0,
            &["hp2"],
            vec![
                make_player("hp", TeamSide::Home),
                make_player("hp2", TeamSide::Home),
                make_player("ap", TeamSide::Away),
            ],
        );
        msm.consume(&[SimEvent::BellCaught { by: "hp2".into() }], &sim_team);
        assert_eq!(
            msm.state().cast.throws_left,
            3,
            "completed pass to a teammate must NOT burn a down"
        );
        // A catch by an OPPONENT is a failed/contested throw → burns one.
        msm.consume(&[SimEvent::BellCaught { by: "ap".into() }], &sim_team);
        assert_eq!(
            msm.state().cast.throws_left,
            2,
            "an opponent catch is a failed throw and spends a down"
        );
        // A missed throw also burns.
        msm.resume_live();
        msm.consume(&[SimEvent::BellMissed { end: RingEnd::PlusX }], &sim_team);
        assert_eq!(msm.state().cast.throws_left, 1);
    }

    #[test]
    fn clearing_gate_resets_throws_advances_gate() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        let sim = make_sim(90.0, &["hp"], vec![make_player("hp", TeamSide::Home)]);
        msm.consume(&[SimEvent::BellCaught { by: "hp".into() }], &sim);
        assert_eq!(msm.state().cast.gate, Gate::Deep);
        assert_eq!(msm.state().cast.throws_left, 3);
    }

    #[test]
    fn gate_cleared_at_mouth_no_readvance() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        let sim = make_sim(290.0, &["hp"], vec![make_player("hp", TeamSide::Home)]);
        msm.consume(&[SimEvent::BellCaught { by: "hp".into() }], &sim);
        assert_eq!(msm.state().cast.gate, Gate::Mouth);
        assert_eq!(msm.state().cast.throws_left, 3);
    }

    #[test]
    fn bobble_and_clatter_consume_throw() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        let sim = make_sim(5.0, &["hp"], vec![make_player("hp", TeamSide::Home)]);
        msm.consume(&[SimEvent::BellBobble { by: "hp".into() }], &sim);
        assert_eq!(msm.state().cast.throws_left, 2);
        msm.consume(
            &[SimEvent::BellClatter {
                by: Some("hp".into()),
            }],
            &sim,
        );
        assert_eq!(msm.state().cast.throws_left, 1);
    }

    // ── scoring events ──

    #[test]
    fn fall_scores_2_home_ends_inning() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        let sim = make_sim(0.0, &["hp"], vec![make_player("hp", TeamSide::Home)]);
        let upd = msm.consume(
            &[SimEvent::BellThroughRing {
                end: RingEnd::PlusX,
                touched: true,
                loop_tier: LoopTier::None,
            }],
            &sim,
        );
        let s = upd.scored.unwrap();
        assert_eq!(s.kind, crate::scoring::ScoreKind::Fall);
        assert_eq!(s.points, 2);
        assert_eq!(msm.state().score_home, 2);
        assert!(upd.inning_end.is_some());
    }

    #[test]
    fn loop_scores_7_ends_inning_immediately() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        let sim = make_sim(0.0, &["hp"], vec![make_player("hp", TeamSide::Home)]);
        let upd = msm.consume(
            &[SimEvent::BellThroughRing {
                end: RingEnd::PlusX,
                touched: false,
                loop_tier: LoopTier::Loop,
            }],
            &sim,
        );
        let s = upd.scored.unwrap();
        assert_eq!(s.kind, crate::scoring::ScoreKind::Loop);
        assert_eq!(s.points, 7);
        assert!(s.ends_inning);
        assert!(upd.inning_end.is_some());
    }

    #[test]
    fn ground_plus1_defense_no_inning_end() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        let sim = make_sim(
            0.0,
            &["hp"],
            vec![
                make_player("hp", TeamSide::Home),
                make_player("ap", TeamSide::Away),
            ],
        );
        let upd = msm.consume(&[SimEvent::PlayerSkinned { id: "hp".into() }], &sim);
        assert_eq!(msm.state().score_away, 1);
        assert!(upd.inning_end.is_none());
    }

    // ── contest integration ──

    #[test]
    fn contest_started_pending_thrower_wins_cast_continues() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        msm.consume(
            &[SimEvent::ContestStarted {
                thrower: "p1".into(),
                contester: "p2".into(),
            }],
            &make_sim(0.0, &[], vec![]),
        );
        assert_eq!(msm.state().phase, MatchPhase::Contest);
        assert!(msm.pending_contest().is_some());

        let sim = make_sim(5.0, &["p1"], vec![make_player("p1", TeamSide::Home)]);
        msm.resolve_contest(
            ContestOutcome {
                winner: ContestWinner::Thrower,
            },
            &sim,
        );
        assert_eq!(msm.state().phase, MatchPhase::Live);
        assert!(msm.state().contest.is_none());
    }

    #[test]
    fn contest_contester_wins_turnover_inning_ends() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        msm.consume(
            &[SimEvent::ContestStarted {
                thrower: "p1".into(),
                contester: "p2".into(),
            }],
            &make_sim(0.0, &[], vec![]),
        );
        let sim = make_sim(5.0, &["p1"], vec![make_player("p1", TeamSide::Home)]);
        let upd = msm.resolve_contest(
            ContestOutcome {
                winner: ContestWinner::Contester,
            },
            &sim,
        );
        assert!(upd.turnover.is_some());
        assert_eq!(msm.state().possession, TeamSide::Away);
        assert!(upd.inning_end.is_some());
    }

    // ── innings, spine, winner ──

    fn score_inning(msm: &mut MatchStateMachine, team: TeamSide) {
        if msm.state().phase != MatchPhase::Live {
            to_live(msm);
        }
        let pid = match team {
            TeamSide::Home => "home-p",
            TeamSide::Away => "away-p",
        };
        let sim = make_sim(0.0, &[pid], vec![make_player(pid, team)]);
        msm.consume(
            &[SimEvent::BellThroughRing {
                end: RingEnd::PlusX,
                touched: true,
                loop_tier: LoopTier::None,
            }],
            &sim,
        );
    }

    fn turnover_inning(msm: &mut MatchStateMachine) {
        if msm.state().phase != MatchPhase::Live {
            to_live(msm);
        }
        let team = msm.state().possession;
        let pid = match team {
            TeamSide::Home => "home-p",
            TeamSide::Away => "away-p",
        };
        // RE-BASELINE (Part D): a turnover-by-cast-exhaustion must come from
        // FAILED throws, not teammate catches. A `bell_caught` by a teammate
        // is now a COMPLETED PASS that retains possession and does NOT burn
        // a down (the down-burn bug fix). Three failed throws (`bell_missed`,
        // which always burns) is the real way a cast turns the ball over;
        // resumeLive re-arms the dead ball between misses like the runtime.
        let sim = make_sim(5.0, &[pid], vec![make_player(pid, team)]);
        let miss = SimEvent::BellMissed { end: RingEnd::PlusX };
        msm.consume(std::slice::from_ref(&miss), &sim);
        msm.resume_live();
        msm.consume(std::slice::from_ref(&miss), &sim);
        msm.resume_live();
        msm.consume(std::slice::from_ref(&miss), &sim);
    }

    #[test]
    fn home_scores_9_innings_home_wins_final() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        for _ in 0..9 {
            score_inning(&mut msm, TeamSide::Home);
        }
        assert_eq!(msm.state().winner, Some(TeamSide::Home));
        assert_eq!(msm.state().phase, MatchPhase::Final);
        assert!(msm.state().score_home > msm.state().score_away);
    }

    #[test]
    fn tied_after_9_spine_phase() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        for i in 0..8 {
            score_inning(
                &mut msm,
                if i % 2 == 0 {
                    TeamSide::Home
                } else {
                    TeamSide::Away
                },
            );
        }
        assert_eq!(msm.state().inning, 9);
        turnover_inning(&mut msm);
        assert!(msm.state().spine);
        assert_eq!(msm.state().phase, MatchPhase::Spine);
        assert_eq!(msm.state().score_home, msm.state().score_away);
    }

    #[test]
    fn spine_first_score_decides_winner() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        for i in 0..8 {
            score_inning(
                &mut msm,
                if i % 2 == 0 {
                    TeamSide::Home
                } else {
                    TeamSide::Away
                },
            );
        }
        turnover_inning(&mut msm);
        assert!(msm.state().spine);
        to_live(&mut msm);

        let spine_poss = msm.state().possession;
        let pid = match spine_poss {
            TeamSide::Home => "home-sp",
            TeamSide::Away => "away-sp",
        };
        let spine_end = match spine_poss {
            TeamSide::Home => RingEnd::PlusX,
            TeamSide::Away => RingEnd::MinusX,
        };
        let sim = make_sim(0.0, &[pid], vec![make_player(pid, spine_poss)]);
        msm.consume(
            &[SimEvent::BellThroughRing {
                end: spine_end,
                touched: true,
                loop_tier: LoopTier::None,
            }],
            &sim,
        );
        assert_eq!(msm.state().winner, Some(spine_poss));
        assert_eq!(msm.state().phase, MatchPhase::Final);
    }

    /// PART D — spine sudden-death default-win bug fix: a tied spine inning
    /// that ends on a turnover (turnover-and-clear) must NOT default-win for
    /// Away. The spine CONTINUES until a team actually scores; only then
    /// does that team win.
    #[test]
    fn spine_turnover_continues_spine_then_score_wins() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        for i in 0..8 {
            score_inning(
                &mut msm,
                if i % 2 == 0 { TeamSide::Home } else { TeamSide::Away },
            );
        }
        turnover_inning(&mut msm); // tie after 9 → spine
        assert!(msm.state().spine);
        assert_eq!(msm.state().phase, MatchPhase::Spine);
        assert!(msm.state().winner.is_none());

        // A turnover in the (still-tied) spine inning: must NOT end the
        // match — the old bug resolved `score_home > score_away` → false
        // and handed Away a default win. Spine must CONTINUE.
        turnover_inning(&mut msm);
        assert!(
            msm.state().winner.is_none(),
            "a tied-spine turnover must not default-win the match"
        );
        assert!(msm.state().spine, "still spine");
        assert_eq!(msm.state().phase, MatchPhase::Spine);
        assert_eq!(msm.state().score_home, msm.state().score_away);

        // Now a real score in spine decides it for the scorer.
        to_live(&mut msm);
        let poss = msm.state().possession;
        let pid = match poss {
            TeamSide::Home => "home-spx",
            TeamSide::Away => "away-spx",
        };
        let end = match poss {
            TeamSide::Home => RingEnd::PlusX,
            TeamSide::Away => RingEnd::MinusX,
        };
        let sim = make_sim(0.0, &[pid], vec![make_player(pid, poss)]);
        msm.consume(
            &[SimEvent::BellThroughRing {
                end,
                touched: true,
                loop_tier: LoopTier::None,
            }],
            &sim,
        );
        assert_eq!(msm.state().winner, Some(poss));
        assert_eq!(msm.state().phase, MatchPhase::Final);
    }

    #[test]
    fn events_after_final_ignored() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        for _ in 0..9 {
            score_inning(&mut msm, TeamSide::Home);
        }
        let score_after = msm.state().score_home;
        let sim = make_sim(0.0, &["hp"], vec![make_player("hp", TeamSide::Home)]);
        msm.consume(
            &[SimEvent::BellThroughRing {
                end: RingEnd::PlusX,
                touched: true,
                loop_tier: LoopTier::None,
            }],
            &sim,
        );
        assert_eq!(msm.state().score_home, score_after);
    }

    #[test]
    fn away_wins_if_away_outscores_home() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        for _ in 0..9 {
            score_inning(&mut msm, TeamSide::Away);
        }
        assert_eq!(msm.state().winner, Some(TeamSide::Away));
    }

    // ── pure / deterministic ──

    #[test]
    fn same_event_sequence_identical_state() {
        fn run() -> MatchState {
            let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
            to_live(&mut msm);
            let sim = make_sim(5.0, &["p1"], vec![make_player("p1", TeamSide::Home)]);
            msm.consume(&[SimEvent::BellCaught { by: "p1".into() }], &sim);
            msm.consume(
                &[SimEvent::ContestStarted {
                    thrower: "p1".into(),
                    contester: "p2".into(),
                }],
                &sim,
            );
            msm.resolve_contest(
                ContestOutcome {
                    winner: ContestWinner::Thrower,
                },
                &sim,
            );
            let score_sim = make_sim(0.0, &["p1"], vec![make_player("p1", TeamSide::Home)]);
            msm.consume(
                &[SimEvent::BellThroughRing {
                    end: RingEnd::PlusX,
                    touched: true,
                    loop_tier: LoopTier::None,
                }],
                &score_sim,
            );
            msm.state().clone()
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn does_not_mutate_sim_state() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        let sim = make_sim(5.0, &["hp"], vec![make_player("hp", TeamSide::Home)]);
        let orig_x = sim.bell.p.x;
        let orig_len = sim.bell.pass_chain.len();
        for _ in 0..3 {
            msm.consume(&[SimEvent::BellCaught { by: "hp".into() }], &sim);
        }
        assert_eq!(sim.bell.p.x, orig_x);
        assert_eq!(sim.bell.pass_chain.len(), orig_len);
    }

    #[test]
    fn loop_immediately_ends_inning_same_consume() {
        let mut msm = MatchStateMachine::new(RingEnd::PlusX, TeamSide::Home);
        to_live(&mut msm);
        let sim = make_sim(0.0, &["hp"], vec![make_player("hp", TeamSide::Home)]);
        let upd = msm.consume(
            &[SimEvent::BellThroughRing {
                end: RingEnd::PlusX,
                touched: false,
                loop_tier: LoopTier::Loop,
            }],
            &sim,
        );
        let s = upd.scored.unwrap();
        assert_eq!(s.kind, crate::scoring::ScoreKind::Loop);
        assert!(s.ends_inning);
        assert!(upd.inning_end.is_some());
    }

    #[test]
    fn faith_end_minus_x_flips_faith_free() {
        let mut msm = MatchStateMachine::new(RingEnd::MinusX, TeamSide::Home);
        to_live(&mut msm);
        let sim = make_sim(0.0, &["hp"], vec![make_player("hp", TeamSide::Home)]);
        let upd = msm.consume(
            &[SimEvent::BellThroughRing {
                end: RingEnd::MinusX,
                touched: true,
                loop_tier: LoopTier::None,
            }],
            &sim,
        );
        let s = upd.scored.unwrap();
        assert_eq!(s.kind, crate::scoring::ScoreKind::Fall);
        assert_eq!(s.points, 2);
    }
}

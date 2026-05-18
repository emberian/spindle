//! 1:1 faithful Rust port of src/ai/decide/ThrowScore.ts (TS lines 1-111).
//!
//! Per-target throw scoring for the AI. Given a potential receiver and the
//! current game state, compute a scalar "throw score" used by role
//! policies to pick the best recipient. Integrates ScoreEV (end-zone
//! option value) with TeamProfile biases. No RNG here — the call-site
//! injects noise for difficulty variance.
//!
//! TS `number` → `f64`. `REG.L` → `crate::tuning::L`, `GATE_X` →
//! `crate::tuning::GATE_X`, `REG.R` → `crate::tuning::R`.

use crate::ai::profile::TeamProfile;
use crate::ai::score_ev::{compute_score_ev, Preferred, ScoreEvContext};
use crate::ai::types::{FaithEnd, MatchState, PlayerSim};
use crate::ai::orientation::{attack_ring_x, attack_sign};
use crate::math::Vec3;
use crate::tuning::{GATE_X, L, R};

/// Port of TS `ThrowCandidate` (TS:15-25).
#[derive(Clone, Debug)]
pub struct ThrowCandidate {
    pub receiver: PlayerSim,
    /// World position the throw is aimed at (lead intercept).
    pub aim_pos: Vec3,
    /// Estimated flight time in seconds.
    pub flight_time: f64,
    /// How open the receiver is [0, 1] — 1 = totally uncontested.
    pub openness: f64,
    /// Axis-radius of receiver position.
    pub receiver_radius: f64,
}

/// Port of TS `ThrowScoreResult['scoringIntent']` union (TS:32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoringIntent {
    Fall,
    Rise,
    Loop,
    Curl,
    Advance,
}

impl From<Preferred> for ScoringIntent {
    fn from(p: Preferred) -> Self {
        match p {
            Preferred::Fall => ScoringIntent::Fall,
            Preferred::Rise => ScoringIntent::Rise,
            Preferred::Loop => ScoringIntent::Loop,
            Preferred::Curl => ScoringIntent::Curl,
        }
    }
}

/// Port of TS `ThrowScoreResult` (TS:27-33).
#[derive(Clone, Debug)]
pub struct ThrowScoreResult {
    pub candidate: ThrowCandidate,
    /// Final scalar score — higher = better throw.
    pub score: f64,
    /// Preferred scoring intent if the receiver is at/near the goal mouth.
    pub scoring_intent: ScoringIntent,
}

/// Score a single throw candidate.
///
/// Scoring factors: openness (primary), axial yardage gain toward the
/// target ring, ScoreEV for receivers near the goal mouth, profile-aligned
/// radius, and a flight-time penalty.
///
/// Port of TS `scoreThrow` (TS:45-110). `addNoise` defaults to 0 in TS;
/// Rust has no default args so callers pass `0.0` explicitly.
pub fn score_throw(
    thrower: &PlayerSim,
    candidate: ThrowCandidate,
    match_: &MatchState,
    profile: TeamProfile,
    add_noise: f64,
) -> ThrowScoreResult {
    // TS:52 destructure.
    let aim_pos = candidate.aim_pos;
    let openness = candidate.openness;
    let flight_time = candidate.flight_time;
    let receiver_radius = candidate.receiver_radius;

    // ORIENTATION-CORRECT: the attacking ring is fixed by the THROWER's
    // team (home → +X, away → -X), independent of faithEnd. faithEnd only
    // labels which physical ring scores as Faith (2) vs Free (5).
    let sgn = attack_sign(thrower.team);
    let our_ring_x = attack_ring_x(thrower.team);
    let faith_x = match match_.faith_end {
        FaithEnd::PlusX => GATE_X,
        FaithEnd::MinusX => -GATE_X,
    };
    // Is OUR attacking ring the Faith (2-pt) ring?
    //
    // TS: `Math.sign(ourRingX) === Math.sign(faithX)`. our_ring_x and
    // faith_x are both ±GATE_X (GATE_X = L/2 = 320, never 0), so each is
    // strictly nonzero and JS `Math.sign` agrees with Rust `signum` on
    // these inputs — equivalently, equal-sign iff the product is positive.
    let attack_faith = (our_ring_x * faith_x) > 0.0;

    // Distances to the physical Faith / Free rings (for EV classification).
    let dist_to_faith_ring = (aim_pos.x - faith_x).abs();
    let dist_to_free_ring = (aim_pos.x - -faith_x).abs();

    // Axial progress toward OUR attacking ring (orientation-correct).
    let thrower_x = thrower.p.x;
    let axial_gain = sgn * (aim_pos.x - thrower_x);

    let normalized_gain = (axial_gain / (L * 0.25)).clamp(-0.5, 1.5);

    // Openness is the dominant term.
    let openness_score = openness * 0.45;

    // Axial gain toward goal.
    let gain_score = normalized_gain * (0.2 + profile.aggression * 0.15);

    // Flight-time penalty (short = safe, long = risky).
    let time_penalty = (flight_time * 0.03).min(0.2);

    // Radius alignment: Fall teams want deep (high radius), Rise teams
    // want high (low radius).
    let receiver_radius_frac = receiver_radius / R;
    let radius_score = if attack_faith {
        receiver_radius_frac * 0.1 * (1.0 - profile.free_end_bias) // fall: deep bonus
    } else {
        (1.0 - receiver_radius_frac) * 0.1 * profile.free_end_bias // rise: axis bonus
    };

    // Near-goal EV bonus.
    let mut scoring_intent = ScoringIntent::Advance;
    let mut ev_bonus = 0.0;
    let is_near_goal = dist_to_faith_ring.min(dist_to_free_ring) < L * 0.12;
    if is_near_goal {
        let ev_ctx = ScoreEvContext {
            dist_to_faith_ring,
            dist_to_free_ring,
            radius_from_axis: receiver_radius,
            match_,
            profile,
        };
        let ev = compute_score_ev(&ev_ctx);
        // Math.max(a,b,c,d) chained.
        let best_ev = ev
            .ev_fall
            .max(ev.ev_rise)
            .max(ev.ev_loop)
            .max(ev.ev_curl);
        ev_bonus = (best_ev * 0.06).min(0.3);
        scoring_intent = ev.preferred.into();
    }

    let raw_score =
        openness_score + gain_score + radius_score + ev_bonus - time_penalty + add_noise;

    ThrowScoreResult {
        candidate,
        score: raw_score,
        scoring_intent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{
        Cast, FaithEnd, Gate, MatchPhase, RiggerRole, TeamSide,
    };
    use crate::math::Quat;

    fn match_stub(faith_end: FaithEnd) -> MatchState {
        MatchState {
            inning: 1.0,
            spine: false,
            possession: TeamSide::Home,
            faith_end,
            cast: Cast {
                throws_left: 3,
                gate: Gate::First,
                spot_x: 0.0,
            },
            contest: None,
            score_home: 0.0,
            score_away: 0.0,
            phase: MatchPhase::Live,
            message: String::new(),
            winner: None,
        }
    }

    fn player(team: TeamSide, x: f64) -> PlayerSim {
        PlayerSim {
            id: "p".to_string(),
            team,
            role: RiggerRole::Reach,
            p: Vec3::new(x, 0.0, 0.0),
            v: Vec3::new(0.0, 0.0, 0.0),
            q: Quat::ident(),
            line: None,
            dv_budget: 0.0,
            contact_ref: None,
            grounded: false,
        }
    }

    fn cand(aim_x: f64, openness: f64, ft: f64, rr: f64) -> ThrowCandidate {
        ThrowCandidate {
            receiver: player(TeamSide::Home, aim_x),
            aim_pos: Vec3::new(aim_x, 0.0, 0.0),
            flight_time: ft,
            openness,
            receiver_radius: rr,
        }
    }

    #[test]
    fn score_is_finite() {
        let m = match_stub(FaithEnd::PlusX);
        let thrower = player(TeamSide::Home, 0.0);
        for &fe in &[FaithEnd::PlusX, FaithEnd::MinusX] {
            let m = match_stub(fe);
            for ox in [-300.0, -50.0, 0.0, 50.0, 300.0] {
                for op in [0.0, 0.5, 1.0] {
                    let r = score_throw(
                        &thrower,
                        cand(ox, op, 5.0, 20.0),
                        &m,
                        TeamProfile::baseline(),
                        0.0,
                    );
                    assert!(r.score.is_finite(), "score {}", r.score);
                }
            }
        }
        let _ = m;
    }

    #[test]
    fn determinism() {
        let m = match_stub(FaithEnd::PlusX);
        let thrower = player(TeamSide::Home, 0.0);
        let a = score_throw(
            &thrower,
            cand(300.0, 0.7, 4.0, 25.0),
            &m,
            TeamProfile::baseline(),
            0.0,
        );
        let b = score_throw(
            &thrower,
            cand(300.0, 0.7, 4.0, 25.0),
            &m,
            TeamProfile::baseline(),
            0.0,
        );
        assert_eq!(a.score, b.score);
        assert_eq!(a.scoring_intent, b.scoring_intent);
    }

    #[test]
    fn add_noise_is_pure_offset() {
        // raw_score is `... + add_noise`; changing only add_noise must
        // shift the score by exactly that delta (true by construction).
        let m = match_stub(FaithEnd::PlusX);
        let thrower = player(TeamSide::Home, 0.0);
        let base = score_throw(
            &thrower,
            cand(100.0, 0.5, 3.0, 10.0),
            &m,
            TeamProfile::baseline(),
            0.0,
        );
        let noised = score_throw(
            &thrower,
            cand(100.0, 0.5, 3.0, 10.0),
            &m,
            TeamProfile::baseline(),
            0.25,
        );
        assert!((noised.score - base.score - 0.25).abs() < 1e-12);
    }

    #[test]
    fn near_goal_sets_scoring_intent() {
        // Aim right at the +X ring with home attacking +X: near-goal path
        // runs and scoring_intent leaves the default `Advance`.
        let m = match_stub(FaithEnd::PlusX);
        let thrower = player(TeamSide::Home, 0.0);
        let r = score_throw(
            &thrower,
            cand(GATE_X, 0.8, 2.0, 30.0),
            &m,
            TeamProfile::baseline(),
            0.0,
        );
        assert_ne!(r.scoring_intent, ScoringIntent::Advance);
        assert!(r.score.is_finite());
    }
}

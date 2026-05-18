//! 1:1 faithful Rust port of src/ai/decide/ScoreEV.ts (TS lines 1-145).
//!
//! Expected-value calculations for Fall / Rise / Loop / Curl scoring
//! attempts. Parameterised by a `TeamProfile` so different franchise
//! philosophies produce different shot selections. No RNG — the call-site
//! injects variance if desired.
//!
//! TS `number` → `f64`. `REG.L` → `crate::tuning::L`, `REG.R` →
//! `crate::tuning::R` (RegConstants.ts: L=640, R=45 — identical literals,
//! so f64 arithmetic is bit-identical to the TS oracle).

use crate::ai::profile::TeamProfile;
use crate::ai::types::MatchState;
use crate::tuning::{L, R};

// ── Scoring values (canon §6 rules.md) — TS:10-13 ────────────────────────────
pub const SCORE_FALL: f64 = 2.0;
pub const SCORE_RISE: f64 = 5.0;
pub const SCORE_LOOP: f64 = 7.0;
/// awarded to defense
pub const SCORE_GROUND: f64 = 1.0;

// ── Completion-probability models ─────────────────────────────────────────────
// Rough sigmoid-shaped models keyed on distance to goal ring and current
// axis-radius. Outer values calibrated to match the "Rise rarely scores in
// a season" and "Fall is the bread" canon.

/// Estimated probability of a clean Fall (spinward Faith-end score).
/// Easier than Rise; higher from depth (deep = heavy, committed but true).
/// `p_fall = base * distFactor * radiusFactor`
///
/// Port of TS:25-34.
pub fn p_fall(dist_to_faith_ring: f64, radius_from_axis: f64) -> f64 {
    // Base: 80% if right at the gate, falling off with range.
    let dist_factor = (1.0 - dist_to_faith_ring / (L * 0.55)).max(0.0);
    // Deep play (high radius) strengthens a Fall — "deep is faithful".
    let radius_factor = 0.65 + 0.35 * (radius_from_axis / R).min(1.0);
    (0.82 * dist_factor * radius_factor).min(1.0)
}

/// Estimated probability of a clean Rise (antispinward Free-end score).
/// Harder — rises against the world's curve. Most players never score one.
///
/// Port of TS:40-49.
pub fn p_rise(dist_to_free_ring: f64, radius_from_axis: f64) -> f64 {
    // Base: only ~35% even close to the ring.
    let dist_factor = (1.0 - dist_to_free_ring / (L * 0.4)).max(0.0);
    // Being closer to the axis helps on the Free end — high is free.
    let radius_factor = 0.75 - 0.25 * (radius_from_axis / R).min(1.0);
    (0.35 * dist_factor * radius_factor).min(1.0)
}

/// Estimated probability of a Loop completing (7-pt score).
/// Extremely difficult; requires very specific positions and spin window.
///
/// Port of TS:55-68.
pub fn p_loop(radius_from_axis: f64, dist_to_ring: f64, loop_propensity: f64) -> f64 {
    // Canon-rare on purpose: ANY score ends the inning, so if loops are easy
    // every inning collapses to one spinner-huck. Keeping the Loop a hard,
    // special event makes innings long multi-cast possession battles.
    let radius_factor = (1.0 - radius_from_axis / (R * 0.7)).max(0.0);
    let dist_factor = (1.0 - dist_to_ring / (L * 0.35)).max(0.0);
    (0.08 * radius_factor * dist_factor * (0.4 + loop_propensity)).min(0.6)
}

/// Curl: a partial-loop that curves back from the Free end.
/// Worth 5 (a Rise) if it makes it through; modeled as a lower-prob Rise.
///
/// Port of TS:74-81.
pub fn p_curl(radius_from_axis: f64, dist_to_free_ring: f64, loop_propensity: f64) -> f64 {
    let base = p_rise(dist_to_free_ring, radius_from_axis);
    base * (0.5 + 0.5 * loop_propensity)
}

// ── EV calculators ─────────────────────────────────────────────────────────────

/// Port of TS `ScoreEVContext` (TS:85-91). `match` (a Rust keyword) → `match_`.
#[derive(Clone, Debug)]
pub struct ScoreEvContext<'a> {
    pub dist_to_faith_ring: f64,
    pub dist_to_free_ring: f64,
    pub radius_from_axis: f64,
    pub match_: &'a MatchState,
    pub profile: TeamProfile,
}

/// Port of TS `ScoreEVResult['preferred']` union (TS:98).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preferred {
    Fall,
    Rise,
    Loop,
    Curl,
}

/// Port of TS `ScoreEVResult` (TS:93-99).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreEvResult {
    pub ev_fall: f64,
    pub ev_rise: f64,
    pub ev_loop: f64,
    pub ev_curl: f64,
    pub preferred: Preferred,
}

/// Compute expected-value for each scoring option at the current field
/// position.
///
/// E[Fall] = p_fall·2, E[Rise] = p_rise·5, E[Loop] = p_loop·7,
/// E[Curl] = p_curl·5 (same ring value as Rise).
///
/// Team profile biases are applied multiplicatively after the
/// physics-based EV so that a rise-chaos team still shoots loops even when
/// the raw math says no.
///
/// Port of TS `computeScoreEV` (TS:112-144).
pub fn compute_score_ev(ctx: &ScoreEvContext) -> ScoreEvResult {
    // TS:113 destructure — `match` unused in the body (passed through ctx).
    let dist_to_faith_ring = ctx.dist_to_faith_ring;
    let dist_to_free_ring = ctx.dist_to_free_ring;
    let radius_from_axis = ctx.radius_from_axis;
    let profile = ctx.profile;

    let pf = p_fall(dist_to_faith_ring, radius_from_axis);
    let pr = p_rise(dist_to_free_ring, radius_from_axis);
    let pl = p_loop(
        radius_from_axis,
        dist_to_faith_ring.min(dist_to_free_ring),
        profile.loop_propensity,
    );
    let pc = p_curl(radius_from_axis, dist_to_free_ring, profile.loop_propensity);

    // Raw EVs.
    let raw_fall = pf * SCORE_FALL;
    let raw_rise = pr * SCORE_RISE;
    let raw_loop = pl * SCORE_LOOP;
    let raw_curl = pc * SCORE_RISE;

    // Bias adjustments from team profile (scale, not add, to preserve
    // relative ordering).
    let fall_bias = 1.0 + (1.0 - profile.free_end_bias) * 0.4; // Fall teams inflate Faith EV
    let free_bias = 1.0 + profile.free_end_bias * 0.6; // Rise teams inflate Free EV
    let loop_bias = 1.0 + profile.loop_propensity * 0.8; // Loop specialists inflate Loop EV

    let ev_fall = raw_fall * fall_bias;
    let ev_rise = raw_rise * free_bias;
    let ev_loop = raw_loop * loop_bias;
    let ev_curl = raw_curl * free_bias * (0.6 + profile.loop_propensity * 0.4);

    // Preferred option: highest EV. Branch order mirrors TS:137-141 exactly
    // (strict `>`, so ties keep the earlier option).
    let mut preferred = Preferred::Fall;
    let mut best = ev_fall;
    if ev_rise > best {
        best = ev_rise;
        preferred = Preferred::Rise;
    }
    if ev_loop > best {
        best = ev_loop;
        preferred = Preferred::Loop;
    }
    if ev_curl > best {
        // TS does not reassign `best` here (it is unused afterward); kept
        // identical — only `preferred` flips.
        let _ = best;
        preferred = Preferred::Curl;
    }

    ScoreEvResult {
        ev_fall,
        ev_rise,
        ev_loop,
        ev_curl,
        preferred,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{
        Cast, FaithEnd, Gate, MatchPhase, MatchState, TeamSide,
    };

    fn match_stub() -> MatchState {
        MatchState {
            inning: 1.0,
            spine: false,
            possession: TeamSide::Home,
            faith_end: FaithEnd::PlusX,
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

    #[test]
    fn probabilities_are_unit_clamped() {
        // Sweep distance and radius; every model output stays in [0, 1]
        // (true by construction: max(0,..) floors, .min(1) / .min(0.6)
        // caps, and the curl scale factor 0.5+0.5*lp ∈ [0.5,1] applied to
        // a [0,1] base stays in [0,1] for lp ∈ [0,1]).
        for di in 0..=20 {
            let d = (di as f64) * 40.0;
            for ri in 0..=10 {
                let r = (ri as f64) * 6.0;
                for lpi in 0..=4 {
                    let lp = (lpi as f64) * 0.25;
                    let pf = p_fall(d, r);
                    let pr = p_rise(d, r);
                    let pl = p_loop(r, d, lp);
                    let pc = p_curl(r, d, lp);
                    assert!((0.0..=1.0).contains(&pf), "pf {pf}");
                    assert!((0.0..=1.0).contains(&pr), "pr {pr}");
                    assert!((0.0..=0.6).contains(&pl), "pl {pl}");
                    assert!((0.0..=1.0).contains(&pc), "pc {pc}");
                }
            }
        }
    }

    #[test]
    fn determinism() {
        let m = match_stub();
        let ctx = ScoreEvContext {
            dist_to_faith_ring: 30.0,
            dist_to_free_ring: 600.0,
            radius_from_axis: 20.0,
            match_: &m,
            profile: TeamProfile::baseline(),
        };
        let a = compute_score_ev(&ctx);
        let b = compute_score_ev(&ctx);
        assert_eq!(a, b);
    }

    #[test]
    fn p_loop_monotonic_in_propensity() {
        // TS multiplies by (0.4 + loopPropensity) with all other factors
        // non-negative, so p_loop is non-decreasing in loop_propensity
        // (until the 0.6 clamp, where it stays equal).
        let mut prev = -1.0;
        for lpi in 0..=10 {
            let lp = (lpi as f64) * 0.1;
            let pl = p_loop(5.0, 10.0, lp);
            assert!(pl >= prev - 1e-12, "pl {pl} prev {prev}");
            prev = pl;
        }
    }

    #[test]
    fn ev_results_finite() {
        let m = match_stub();
        for fe in [0.0f64, 0.5, 1.0] {
            for lp in [0.0f64, 0.5, 1.0] {
                let mut prof = TeamProfile::baseline();
                prof.free_end_bias = fe;
                prof.loop_propensity = lp;
                let ctx = ScoreEvContext {
                    dist_to_faith_ring: 25.0,
                    dist_to_free_ring: 590.0,
                    radius_from_axis: 15.0,
                    match_: &m,
                    profile: prof,
                };
                let r = compute_score_ev(&ctx);
                assert!(r.ev_fall.is_finite());
                assert!(r.ev_rise.is_finite());
                assert!(r.ev_loop.is_finite());
                assert!(r.ev_curl.is_finite());
            }
        }
    }
}

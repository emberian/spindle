//! contest.rs — pure one-on-one duel resolution for RIG.
//!
//! 1:1 port of `src/match/Contest.ts` (FROZEN contract).
//!
//! A contest ("a one") resolves over up to THREE exchanges (best-of-3, first
//! to 2 exchange-wins). Canon (rules.md §5 / sport.md §3.4):
//!   - Thrower wins the count → complete; cast continues.
//!   - Contester wins the count → turnover at that radius/direction.
//!   - Depth (radius from axis) governs decisiveness:
//!       deep (large radius) = low variance, radius-heavy outcome
//!       high (small radius) = higher variance
//!
//! Determinism is SACRED: `rng_next` is injected by the caller; the caller
//! owns the seed. This function is pure — same inputs ⇒ same output.

use crate::scoring::TeamSide;

// ─── ContestState (mirror of TS ContestState) ────────────────────────────────

/// Mirrors the TS `direction: 'fair' | 'cross'` union.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContestDirection {
    Fair,
    Cross,
}

/// Mirrors TS `ContestState`. `count` is `0 | 1 | 2 | 3` in TS; kept as `u8`
/// with the same domain (the value is carried but not read by resolve).
#[derive(Clone, Debug, PartialEq)]
pub struct ContestState {
    pub thrower: String,
    pub contester: String,
    pub count: u8,
    pub radius: f64,
    pub direction: ContestDirection,
}

/// Mirrors TS `ContestResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContestResult {
    pub winner: ContestWinner,
    /// How many exchanges were needed (2–3 for best-of-3).
    pub exchanges: u32,
}

/// Mirrors TS `winner: 'thrower' | 'contester'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContestWinner {
    Thrower,
    Contester,
}

/// Mirrors the TS `MatchState` snapshot param. It is `_matchSnap` (unused) in
/// the TS body; carried here for signature parity / future gradient hooks.
#[derive(Clone, Copy, Debug)]
pub struct MatchSnap {
    pub possession: TeamSide,
}

/// Resolve a full RIG contest (up to 3 exchanges).
///
/// Depth model (verbatim from Contest.ts lines 58–80):
///   `p = 0.5 + 0.25 * clamp(radius / R, 0, 1)`
///   `p -= 0.05` if `direction == Cross`
///   `p = clamp(p, 0.25, 0.85)`
/// Best-of-3: first to 2 exchange-wins takes the contest.
///
/// `radius` is `contest.radius` (metres). `rng_next` returns a value in
/// `[0, 1)`.
pub fn resolve_contest(
    contest: &ContestState,
    _match_snap: &MatchSnap,
    mut rng_next: impl FnMut() -> f64,
) -> ContestResult {
    // Regulation skin radius (from RegConstants, inlined for purity — matches
    // the TS `const R = 45;`). crate::tuning::R is also 45.0.
    let r: f64 = 45.0;

    let mut p = 0.5 + 0.25 * (contest.radius / r).clamp(0.0, 1.0);
    if contest.direction == ContestDirection::Cross {
        p -= 0.05;
    }
    p = p.clamp(0.25, 0.85);

    let mut thrower_wins: u32 = 0;
    let mut contester_wins: u32 = 0;
    let mut exchanges: u32 = 0;

    while thrower_wins < 2 && contester_wins < 2 {
        exchanges += 1;
        if rng_next() < p {
            thrower_wins += 1;
        } else {
            contester_wins += 1;
        }
    }

    ContestResult {
        winner: if thrower_wins >= 2 {
            ContestWinner::Thrower
        } else {
            ContestWinner::Contester
        },
        exchanges,
    }
}

// ─── Tests (mirror test/match.test.ts "Contest.resolveContest" describe) ─────

#[cfg(test)]
mod tests {
    use super::*;

    fn snap() -> MatchSnap {
        MatchSnap {
            possession: TeamSide::Home,
        }
    }

    fn contest(radius_m: f64, direction: ContestDirection) -> ContestState {
        ContestState {
            thrower: "T".into(),
            contester: "C".into(),
            count: 0,
            radius: radius_m,
            direction,
        }
    }

    /// Sequence-based RNG mirroring the TS `makeSeqRng`.
    fn seq_rng(values: Vec<f64>) -> impl FnMut() -> f64 {
        let mut i = 0usize;
        move || {
            let v = values[i % values.len()];
            i += 1;
            v
        }
    }

    #[test]
    fn same_inputs_same_result() {
        let c = contest(30.0, ContestDirection::Fair);
        let r1 = resolve_contest(&c, &snap(), seq_rng(vec![0.3, 0.8, 0.3, 0.8]));
        let r2 = resolve_contest(&c, &snap(), seq_rng(vec![0.3, 0.8, 0.3, 0.8]));
        assert_eq!(r1, r2);
    }

    #[test]
    fn exchanges_in_2_3() {
        let c = contest(20.0, ContestDirection::Fair);
        for seed in 0..50i64 {
            let mut n = seed;
            let rng = move || {
                let v = ((n * 1009 + 7).rem_euclid(97)) as f64 / 97.0;
                n += 1;
                v
            };
            let r = resolve_contest(&c, &snap(), rng);
            assert!(r.exchanges >= 2);
            assert!(r.exchanges <= 3);
        }
    }

    #[test]
    fn winner_always_thrower_or_contester() {
        let c = contest(10.0, ContestDirection::Cross);
        for v in 0..=10 {
            let r = resolve_contest(&c, &snap(), seq_rng(vec![v as f64 / 10.0]));
            assert!(matches!(
                r.winner,
                ContestWinner::Thrower | ContestWinner::Contester
            ));
        }
    }

    #[test]
    fn deep_favours_thrower_over_high() {
        let deep = contest(45.0, ContestDirection::Fair);
        let high = contest(0.0, ContestDirection::Fair);
        let deep_r = resolve_contest(&deep, &snap(), || 0.5);
        let high_r = resolve_contest(&high, &snap(), || 0.5);
        assert_eq!(deep_r.winner, ContestWinner::Thrower);
        assert_eq!(high_r.winner, ContestWinner::Contester);
    }
}

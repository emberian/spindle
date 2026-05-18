//! TeamProfile + Difficulty — the league/teams.ts AI surface.
//!
//! The struct + Difficulty are the frozen substrate. `style_to_profile`
//! is the big mechanical branch table (teams.ts:602+) — a self-contained
//! swarm port task; the baseline here is the TS function's initial state
//! (all 0.5) so the substrate compiles and decide-leaf tests can build
//! profiles before the full table lands.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeamProfile {
    pub free_end_bias: f64,
    pub loop_propensity: f64,
    pub aggression: f64,
    pub grapple_risk: f64,
    pub contest_aggression: f64,
    pub snatch_vs_clatter: f64,
    pub away_point_bias: f64,
    pub variance: f64,
}

impl TeamProfile {
    /// The pre-branch initial state of styleToProfile (teams.ts:604-611).
    pub fn baseline() -> Self {
        TeamProfile {
            free_end_bias: 0.5,
            loop_propensity: 0.5,
            aggression: 0.5,
            grapple_risk: 0.5,
            contest_aggression: 0.5,
            snatch_vs_clatter: 0.5,
            away_point_bias: 0.5,
            variance: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Difficulty {
    Rookie,
    Pro,
    Legend,
}

/// 1:1 port of styleToProfile(styleTag, cylinderClass) — teams.ts:602.
/// TODO(port-inc3-swarm): replace the body with the full branch table
/// (style family overrides + cylinder-class modifiers). The signature
/// and the all-0.5 baseline are frozen; only the branches get filled.
pub fn style_to_profile(style_tag: &str, cylinder_class: &str) -> TeamProfile {
    // Start from the all-0.5 baseline (teams.ts:604-611).
    let mut p = TeamProfile::baseline();

    // ── cylinder-class baselines (teams.ts:613-658) ─────────────────────────
    match cylinder_class {
        "big-slow" => {
            // Fall: patient, deep, Faithful-end power
            p.free_end_bias = 0.2;
            p.loop_propensity = 0.15;
            p.aggression = 0.6;
            p.grapple_risk = 0.3;
            p.contest_aggression = 0.65;
            p.snatch_vs_clatter = 0.35;
            p.away_point_bias = 0.3;
            p.variance = 0.2;
        }
        "small-fast" => {
            // Rise: high-free, all Freewing, chasing 5s and 7s
            p.free_end_bias = 0.8;
            p.loop_propensity = 0.75;
            p.aggression = 0.75;
            p.grapple_risk = 0.8;
            p.contest_aggression = 0.7;
            p.snatch_vs_clatter = 0.65;
            p.away_point_bias = 0.4;
            p.variance = 0.8;
        }
        "mid" => {
            // Ground/line: broker rig, tempo, away draws
            p.free_end_bias = 0.4;
            p.loop_propensity = 0.25;
            p.aggression = 0.4;
            p.grapple_risk = 0.35;
            p.contest_aggression = 0.45;
            p.snatch_vs_clatter = 0.7;
            p.away_point_bias = 0.65;
            p.variance = 0.3;
        }
        "neutral" => {
            // Adaptive / reference: all-balanced, no home-field shape
            p.free_end_bias = 0.5;
            p.loop_propensity = 0.35;
            p.aggression = 0.4;
            p.grapple_risk = 0.35;
            p.contest_aggression = 0.45;
            p.snatch_vs_clatter = 0.6;
            p.away_point_bias = 0.6;
            p.variance = 0.25;
        }
        // TS switch with no matching case: locals keep their 0.5 init.
        _ => {}
    }

    // ── style-tag adjustments (teams.ts:660-811) ────────────────────────────
    match style_tag {
        "fall-dynasty" => {
            // Granite Anchor, long casts
            p.aggression += 0.1;
            p.contest_aggression += 0.1;
            p.variance -= 0.05;
        }
        "fall-grind" => {
            // Clatter-heavy defense
            p.snatch_vs_clatter -= 0.2;
            p.aggression -= 0.05;
            p.variance -= 0.05;
        }
        "fall-tempo" => {
            // Beautiful clean passing
            p.snatch_vs_clatter += 0.1;
            p.away_point_bias += 0.1;
            p.variance += 0.05;
        }
        "fall-loop" => {
            // Engineers doing Coriolis math for fun
            p.loop_propensity += 0.2;
            p.grapple_risk += 0.1;
            p.variance += 0.1;
        }
        "fall-isolated" => {
            // Stubborn, far-edge
            p.aggression -= 0.05;
            p.variance -= 0.05;
        }
        "line" => {
            // Tether craft, tempo, away-points
            p.away_point_bias += 0.1;
            p.grapple_risk += 0.1;
            p.snatch_vs_clatter += 0.05;
        }
        "ground-away" => {
            // Infuriating broker rig, wins on road
            p.away_point_bias += 0.15;
            p.aggression -= 0.1;
            p.variance -= 0.05;
        }
        "ground-broker" => {
            // Perfected draw game
            p.away_point_bias += 0.2;
            p.snatch_vs_clatter += 0.1;
            p.aggression -= 0.1;
        }
        "ground-grief" | "ground-haunted" => {
            // Defiant, low-scoring
            p.aggression -= 0.05;
            p.variance -= 0.05;
            p.free_end_bias -= 0.05;
        }
        "ground-defiant" => {
            // Low-scoring, beloved
            p.aggression -= 0.05;
            p.variance -= 0.05;
        }
        "ground-travel" => {
            // Always away — lean into the away-point game
            p.away_point_bias += 0.15;
            p.variance += 0.05;
        }
        "ground-ceremonial" | "ground-reference" => {
            // Reference spin, no home-advantage skew
            p.variance -= 0.1;
            p.away_point_bias += 0.05;
        }
        "adaptive-ground" => {
            // No home, no home disadvantage
            p.away_point_bias += 0.1;
            p.variance += 0.05;
        }
        "audio-ground" => {
            // Slow, hypnotic; built for audio
            p.aggression -= 0.1;
            p.variance -= 0.05;
            p.away_point_bias += 0.1;
        }
        "ground-mournful" => {
            // Technically gorgeous, underfunded
            p.snatch_vs_clatter += 0.1;
            p.variance -= 0.05;
        }
        "ground-junior" => {
            // Junior-broker / dull-effective
            p.aggression -= 0.1;
            p.variance -= 0.1;
        }
        "rise-chaos" => {
            // All Freewing, chasing 5s and 7s
            p.free_end_bias += 0.1;
            p.loop_propensity += 0.1;
            p.variance += 0.1;
        }
        "rise-power" => {
            // Wins 7s by force
            p.aggression += 0.1;
            p.grapple_risk += 0.05;
            p.free_end_bias += 0.05;
        }
        "rise-exhausted" => {
            // Magnificent but tired
            p.variance += 0.05;
            p.aggression -= 0.05;
        }
        "rise-flock" => {
            // Coordinated high game
            p.contest_aggression += 0.1;
            p.snatch_vs_clatter += 0.1;
            p.variance -= 0.05;
        }
        "rise-political" => {
            // Furious, electric
            p.aggression += 0.1;
            p.grapple_risk += 0.1;
            p.variance += 0.05;
        }
        "rise-lowlight" => {
            // Audio legends
            p.variance += 0.05;
            p.away_point_bias += 0.05;
        }
        "rise-lunatic" => {
            // Purest chaos rig
            p.variance += 0.1;
            p.free_end_bias += 0.05;
            p.grapple_risk += 0.1;
        }
        "rise-contrarian" => {
            // Attacks Free end by preference
            p.free_end_bias += 0.1;
            p.grapple_risk += 0.05;
            p.aggression += 0.05;
        }
        "rise-ceremonial" => {
            // Tradition as a weapon
            p.loop_propensity += 0.1;
            p.variance -= 0.05;
        }
        "rise-dirty" => {
            // Foul-prone, magnetic
            p.grapple_risk += 0.1;
            p.aggression += 0.1;
            p.contest_aggression += 0.05;
            p.variance += 0.05;
        }
        // TS switch with no matching case: no style adjustment applied.
        _ => {}
    }

    // clamp all to [0, 1] (teams.ts:813-824)
    let clamp = |v: f64| v.max(0.0).min(1.0);
    p.free_end_bias = clamp(p.free_end_bias);
    p.loop_propensity = clamp(p.loop_propensity);
    p.aggression = clamp(p.aggression);
    p.grapple_risk = clamp(p.grapple_risk);
    p.contest_aggression = clamp(p.contest_aggression);
    p.snatch_vs_clatter = clamp(p.snatch_vs_clatter);
    p.away_point_bias = clamp(p.away_point_bias);
    p.variance = clamp(p.variance);
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_in_unit(p: &TeamProfile) -> bool {
        let f = [
            p.free_end_bias,
            p.loop_propensity,
            p.aggression,
            p.grapple_risk,
            p.contest_aggression,
            p.snatch_vs_clatter,
            p.away_point_bias,
            p.variance,
        ];
        f.iter().all(|&v| v >= 0.0 && v <= 1.0)
    }

    #[test]
    fn fields_stay_in_unit_interval() {
        let styles = [
            "fall-dynasty",
            "fall-grind",
            "fall-loop",
            "line",
            "ground-broker",
            "ground-haunted",
            "ground-reference",
            "rise-chaos",
            "rise-lunatic",
            "rise-dirty",
            "unknown-style",
        ];
        let classes = ["big-slow", "small-fast", "mid", "neutral", "bogus"];
        for s in styles {
            for c in classes {
                let p = style_to_profile(s, c);
                assert!(all_in_unit(&p), "out of [0,1] for {s}/{c}: {p:?}");
            }
        }
    }

    #[test]
    fn distinctive_style_differs_from_baseline() {
        // small-fast cylinder class alone reshapes the profile.
        let p = style_to_profile("rise-chaos", "small-fast");
        assert_ne!(p, TeamProfile::baseline());
        // fall-dynasty/big-slow has high contest_aggression (0.65 + 0.1).
        let q = style_to_profile("fall-dynasty", "big-slow");
        assert!((q.contest_aggression - 0.75).abs() < 1e-9);
        assert!((q.aggression - 0.70).abs() < 1e-9);
    }

    #[test]
    fn unknown_style_and_class_returns_baseline() {
        // No cylinder-class match and no style-tag match: locals keep 0.5.
        let p = style_to_profile("not-a-style", "not-a-class");
        assert_eq!(p, TeamProfile::baseline());
    }
}

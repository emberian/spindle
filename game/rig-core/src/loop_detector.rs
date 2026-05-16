//! Loop / Curl detector — port of TS sim/LoopDetector.ts.
//! Tracks cumulative heading turn of the (y,z) velocity cross-section since
//! release; classifies as Loop (closed arc) or Curl (notable curve) at ring
//! crossings.

use crate::math::Vec3;

use std::f64::consts::PI;

/// Radians of cumulative turn required to score a Loop.
pub const LOOP_TURN: f64 = 2.2;
/// Radians of cumulative turn required to score a Curl.
pub const CURL_TURN: f64 = 1.1;

/// Distance (m) within which the bell must return to its release (y,z) to
/// count as a closed-arc Loop rather than a mere Curl.
const CLOSE_RADIUS: f64 = 22.0;

/// Classification at a ring crossing.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LoopTier {
    Loop,
    Curl,
    None,
}

/// Per-bell state machine for Loop / Curl detection.
pub struct LoopTracker {
    active: bool,
    touched: bool,
    release_y: f64,
    release_z: f64,
    prev_heading: f64,
    cum_turn: f64,
    has_prev: bool,
}

impl LoopTracker {
    /// Create a new tracker (idle until `on_release` is called).
    pub fn new() -> Self {
        Self {
            active: false,
            touched: true,
            release_y: 0.0,
            release_z: 0.0,
            prev_heading: 0.0,
            cum_turn: 0.0,
            has_prev: false,
        }
    }

    /// Call when the bell is released (thrown, leaves a hand).
    pub fn on_release(&mut self, p: Vec3) {
        self.active = true;
        self.touched = false;
        self.release_y = p.y;
        self.release_z = p.z;
        self.cum_turn = 0.0;
        self.has_prev = false;
    }

    /// Call when anyone / anything contacts the bell in flight.
    pub fn on_touch(&mut self) {
        self.touched = true;
        self.active = false;
    }

    /// Call every sim step while the bell is free.
    /// Accumulates the absolute wrapped heading change of the (y,z) velocity.
    pub fn update(&mut self, v: Vec3) {
        if !self.active {
            return;
        }
        let speed = v.y.hypot(v.z);
        if speed < 1e-4 {
            return;
        }
        let heading = v.z.atan2(v.y);
        if self.has_prev {
            let mut d = heading - self.prev_heading;
            while d > PI {
                d -= 2.0 * PI;
            }
            while d < -PI {
                d += 2.0 * PI;
            }
            self.cum_turn += d.abs();
        }
        self.prev_heading = heading;
        self.has_prev = true;
    }

    /// Classify at a ring crossing: Loop (approx-closed), Curl (big curve),
    /// or None.
    pub fn tier(&self, p: Vec3) -> LoopTier {
        if self.touched {
            return LoopTier::None;
        }
        if self.cum_turn >= LOOP_TURN {
            let dy = p.y - self.release_y;
            let dz = p.z - self.release_z;
            if dy.hypot(dz) <= CLOSE_RADIUS {
                return LoopTier::Loop;
            }
        }
        if self.cum_turn >= CURL_TURN {
            return LoopTier::Curl;
        }
        LoopTier::None
    }

    /// Back-compat: strict-ish loop check (= `tier == Loop`).
    pub fn is_loop(&self, p: Vec3) -> bool {
        self.tier(p) == LoopTier::Loop
    }

    /// True when no contact has been registered since the last release.
    pub fn untouched(&self) -> bool {
        !self.touched
    }

    /// Cumulative heading turn (radians) accumulated since release.
    pub fn turn(&self) -> f64 {
        self.cum_turn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Feed `n` velocity samples that each rotate the (y,z) heading by `d_rad`
    /// around the origin at unit speed.
    fn feed_circular(tr: &mut LoopTracker, steps: usize, d_rad: f64) {
        let mut heading = 0.0_f64;
        for _ in 0..steps {
            let v = Vec3::new(0.0, heading.cos(), heading.sin());
            tr.update(v);
            heading += d_rad;
        }
    }

    // --- mirrors loop.test.ts "a hard-curving fed velocity stream" ---

    #[test]
    fn hard_curve_winds_loop_turn_and_stays_untouched() {
        // 200 steps each rotating ~2π/100 rad ≈ 0.063 rad → total ≈ 12.6 rad >> LOOP_TURN
        let mut tr = LoopTracker::new();
        tr.on_release(Vec3::new(0.0, 0.0, 0.0));
        feed_circular(&mut tr, 200, 2.0 * PI / 100.0);
        assert!(tr.untouched(), "should still be untouched");
        assert!(
            tr.turn() >= LOOP_TURN,
            "cum_turn {} should be >= LOOP_TURN {}",
            tr.turn(),
            LOOP_TURN
        );
    }

    // --- mirrors loop.test.ts "a near-straight axial throw" ---

    #[test]
    fn near_straight_stays_below_curl_turn() {
        // Velocity nearly constant in (y,z): tiny heading drift — total turn << CURL_TURN.
        let mut tr = LoopTracker::new();
        let release = Vec3::new(0.0, 1.0, 0.0);
        tr.on_release(release);

        // 120 steps of v=(32, 0, 0.2) — (y,z) heading barely changes.
        for _ in 0..120 {
            tr.update(Vec3::new(32.0, 0.0, 0.2));
        }
        let end = Vec3::new(0.0, 1.0, 0.0);
        assert!(
            tr.turn() < CURL_TURN,
            "cum_turn {} should be < CURL_TURN {}",
            tr.turn(),
            CURL_TURN
        );
        assert_eq!(tr.tier(end), LoopTier::None);
    }

    // --- mirrors loop.test.ts "contact mid-flight voids both Loop and Curl" ---

    #[test]
    fn on_touch_makes_tier_none() {
        let mut tr = LoopTracker::new();
        tr.on_release(Vec3::new(0.0, 2.0, 0.0));
        // Feed a very curvy stream so cum_turn would be well above LOOP_TURN.
        feed_circular(&mut tr, 200, 2.0 * PI / 100.0);
        tr.on_touch();
        assert!(!tr.untouched(), "touched flag should be set");
        assert_eq!(
            tr.tier(Vec3::new(0.0, 2.0, 0.0)),
            LoopTier::None,
            "touched bell must not score Loop or Curl"
        );
    }

    // --- extra: inactive before on_release accumulates nothing ---

    #[test]
    fn inactive_before_release_accumulates_nothing() {
        let mut tr = LoopTracker::new();
        feed_circular(&mut tr, 300, 0.1);
        // touched starts true, so tier must be None regardless of cum_turn
        assert_eq!(tr.turn(), 0.0);
        assert_eq!(tr.tier(Vec3::new(0.0, 0.0, 0.0)), LoopTier::None);
    }
}

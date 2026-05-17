//! Rig-line spring-damper constraint — port of TS sim/Grapple.ts.
//!
//! Continuous (Fluidity Spike) model. The rope is a one-sided radial
//! spring-damper — TENSION ONLY, it never pushes inward when slack:
//!   • deep slack  (stretch ≤ −SLACK_BAND): no force — true free swing.
//!   • slack band  (−SLACK_BAND < stretch ≤ 0): a gentle pre-tension spring
//!              eases the line into tautness instead of snapping.
//!   • taut        (stretch > 0): F = −LINE_K·stretch − LINE_C·v_radial,
//!              clamped so it can only pull toward the anchor.
//! The reel ONLY eases `rest_len` toward the target — it never sets position
//! and never rescales tangential velocity. The angular-momentum slingshot is
//! therefore fully EMERGENT from the spring + free-flight integration.
//!
//! Semi-implicit (symplectic) Euler: velocities are updated from the spring
//! force this step (ω_n·h ≈ 0.018 ≪ 2 ⇒ unconditionally stable here).

use crate::math::Vec3;
use crate::tuning::{LINE_C, LINE_K, LINE_SLACK_BAND, LINE_SLACK_K};

// ── public constants ──────────────────────────────────────────────────────────

pub const TETHER_MIN: f64 = 3.0;
pub const TETHER_MAX: f64 = 60.0;
pub const REEL_RATE: f64 = 14.0; // m/s of rope-length change

// ── public types ─────────────────────────────────────────────────────────────

/// A rigid body reduced to position, velocity and inverse mass.
/// `inv_mass = 0` means immovable (spar / skin / ring).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub p: Vec3,
    pub v: Vec3,
    pub inv_mass: f64,
}

/// State of a single rig line.
///
/// When `anchor` passed to [`resolve_line`] is `Some(b)` the anchor is the
/// moving player body `b`; `anchor_pos` is then unused.  When `anchor` is
/// `None` the anchor is the static world point `anchor_pos`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    pub anchor_pos: Vec3, // world-space anchor used for a static attachment
    pub rest_len: f64,
    pub taut: bool,
}

// ── constraint solver ─────────────────────────────────────────────────────────

/// Resolve one rig-line constraint for one simulation step.
///
/// * `player` — the body attached to this end of the line.
/// * `line`   — mutable line state (rest_len, taut).
/// * `anchor` — `Some(b)` for a player↔player line (force applied to `b`
///              equal-and-opposite); `None` for a static world anchor at
///              `line.anchor_pos` (infinite mass, zero velocity).
/// * `reel`   — `−1` reel in, `0` hold, `+1` reel out.
/// * `h`      — timestep (seconds).
pub fn resolve_line(
    player: &mut Body,
    line: &mut Line,
    anchor: Option<&mut Body>,
    reel: i32,
    h: f64,
) {
    // Anchor position and velocity — borrow once here before splitting.
    let (a_p, a_v, a_inv_mass): (Vec3, Vec3, f64) = match &anchor {
        Some(b) => (b.p, b.v, b.inv_mass),
        None => (line.anchor_pos, Vec3::new(0.0, 0.0, 0.0), 0.0),
    };

    let d = player.p.sub(a_p); // anchor → player
    let len = d.len();
    if len < 1e-6 {
        line.taut = false;
        return;
    }
    let n = d.scale(1.0 / len); // unit vector, anchor → player

    // ── Reel: ONLY ease rest_len toward the target. No position set, no
    //    velocity rescale — the slingshot must emerge from the spring. ──
    if reel != 0 {
        let target = (line.rest_len + reel as f64 * REEL_RATE * h)
            .max(TETHER_MIN)
            .min(TETHER_MAX);
        line.rest_len = target;
    }

    let stretch = len - line.rest_len;

    // ── Deep slack: line does nothing (free ballistic swing). ──
    if stretch <= -LINE_SLACK_BAND {
        line.taut = false;
        return;
    }

    let inv_sum = player.inv_mass + a_inv_mass;
    if inv_sum <= 0.0 {
        line.taut = stretch > 0.0;
        return;
    }

    // Radial relative velocity (positive ⇒ separating from the anchor).
    let rv = player.v.sub(a_v);
    let v_rad = rv.dot(n);

    // Scalar radial force along n. Tension pulls toward the anchor ⇒ this is
    // ≤ 0 (along −n). We clamp to ≤ 0 so the rope can NEVER push outward.
    let f = if stretch > 0.0 {
        // Taut: full spring-damper.
        line.taut = true;
        let raw = -LINE_K * stretch - LINE_C * v_rad;
        raw.min(0.0)
    } else {
        // Slack band: gentle pre-tension only (stretch ≤ 0 ⇒ already ≤ 0).
        line.taut = false;
        LINE_SLACK_K * stretch
    };

    // Semi-implicit Euler velocity update; equal-and-opposite on the anchor.
    let impulse = f * h; // N·s along n
    player.v = player.v.add(n.scale(impulse * player.inv_mass));
    if let Some(b) = anchor {
        b.v = b.v.add(n.scale(-impulse * b.inv_mass));
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const H: f64 = 1.0 / 240.0;

    /// Deep-slack line (well inside radius) must not change the velocity.
    #[test]
    fn slack_no_force() {
        let mut p = Body {
            p: Vec3::new(5.0, 0.0, 0.0),
            v: Vec3::new(2.0, 1.0, 0.0),
            inv_mass: 1.0,
        };
        let mut line = Line {
            anchor_pos: Vec3::new(0.0, 0.0, 0.0),
            rest_len: 20.0,
            taut: false,
        };
        let v0 = p.v;
        resolve_line(&mut p, &mut line, None, 0, H);
        assert_eq!(p.v, v0);
        assert!(!line.taut);
    }

    /// Taut static anchor: a separating-radial spring force decelerates the
    /// outward (radial) motion while tangential is untouched in one step.
    #[test]
    fn taut_static_decelerates_radial_keeps_tangential() {
        // Player 10.1 m out on +x (stretch +0.1), moving outward + tangential.
        let mut p = Body {
            p: Vec3::new(10.1, 0.0, 0.0),
            v: Vec3::new(6.0, 4.0, 0.0),
            inv_mass: 1.0,
        };
        let mut line = Line {
            anchor_pos: Vec3::new(0.0, 0.0, 0.0),
            rest_len: 10.0,
            taut: false,
        };
        resolve_line(&mut p, &mut line, None, 0, H);
        // Radial spring force is inward (−x) ⇒ outward speed must drop.
        assert!(p.v.x < 6.0, "radial speed should drop, got {}", p.v.x);
        // Tangential (y) is untouched in a single radial-force step.
        assert!(
            (p.v.y - 4.0).abs() < 1e-12,
            "tangential should be preserved, got {}",
            p.v.y
        );
        assert!(line.taut);
    }

    /// The rope is tension-only: a slack body moving inward gets NO outward push.
    #[test]
    fn slack_band_never_pushes_outward() {
        // 9.7 m out, rest_len 10 ⇒ stretch −0.3 (inside the 0.6 m band).
        let mut p = Body {
            p: Vec3::new(9.7, 0.0, 0.0),
            v: Vec3::new(-2.0, 0.0, 0.0), // moving inward
            inv_mass: 1.0,
        };
        let mut line = Line {
            anchor_pos: Vec3::new(0.0, 0.0, 0.0),
            rest_len: 10.0,
            taut: false,
        };
        resolve_line(&mut p, &mut line, None, 0, H);
        // Pre-tension pulls gently inward (−x); never pushes outward (+x).
        assert!(p.v.x <= -2.0, "must not be pushed outward, got {}", p.v.x);
        assert!(!line.taut);
    }

    /// Player↔player taut: total linear momentum must be conserved (the spring
    /// force is equal-and-opposite ⇒ the slingshot stays "free").
    #[test]
    fn player_player_conserves_momentum() {
        let mut anchor_body = Body {
            p: Vec3::new(0.0, 0.0, 0.0),
            v: Vec3::new(0.0, 0.0, 0.0),
            inv_mass: 1.0 / 80.0,
        };
        let mut p = Body {
            p: Vec3::new(12.5, 0.0, 0.0),
            v: Vec3::new(9.0, 0.0, 0.0),
            inv_mass: 1.0 / 75.0,
        };
        let mut line = Line {
            anchor_pos: Vec3::new(0.0, 0.0, 0.0),
            rest_len: 12.0,
            taut: false,
        };

        let m_a = 80.0_f64;
        let m_p = 75.0_f64;
        let p0 = m_a * anchor_body.v.x + m_p * p.v.x;

        resolve_line(&mut p, &mut line, Some(&mut anchor_body), 0, H);

        let p1 = m_a * anchor_body.v.x + m_p * p.v.x;
        assert!(
            (p1 - p0).abs() < 1e-6,
            "linear momentum not conserved: Δp = {}",
            p1 - p0
        );
        assert!(anchor_body.v.x > 0.0, "anchor should recoil");
        assert!(p.v.x < 9.0, "player should decelerate outward");
    }

    /// Reel now ONLY eases rest_len toward the target — it never teleports
    /// position and never rescales velocity. (The slingshot is verified to
    /// be EMERGENT via real free-flight integration in player.rs:
    /// `reel_in_slingshot_is_emergent`.)
    #[test]
    fn reel_only_eases_rest_len_no_teleport() {
        let mut p = Body {
            p: Vec3::new(20.0, 0.0, 0.0),
            v: Vec3::new(0.0, 5.0, 0.0),
            inv_mass: 1.0,
        };
        let mut line = Line {
            anchor_pos: Vec3::new(0.0, 0.0, 0.0),
            rest_len: 20.0,
            taut: false,
        };

        // One reel-in step: rest_len shrinks by exactly REEL_RATE·h.
        let p_before = p.p;
        resolve_line(&mut p, &mut line, None, -1, H);
        let expected_rest = 20.0 - REEL_RATE * H;
        assert!(
            (line.rest_len - expected_rest).abs() < 1e-12,
            "rest_len must ease by REEL_RATE·h, got {}",
            line.rest_len
        );
        // Position is NOT teleported onto the new radius (no p = a + n·target).
        assert_eq!(p.p, p_before, "reel must never set position");

        // Reel-in for ~1 s: rest_len monotonically eases toward TETHER_MIN
        // and the line becomes taut (len now exceeds the shrunken rest_len),
        // so the spring engages — the slingshot work is done by the spring,
        // not by a scripted v_tan rescale.
        for _ in 0..240 {
            resolve_line(&mut p, &mut line, None, -1, H);
        }
        assert!(
            line.rest_len < 20.0 && line.rest_len >= TETHER_MIN,
            "rest_len eased inward, got {}",
            line.rest_len
        );
        assert!(line.taut, "shrunk rest_len ⇒ line is taut and pulling");
    }
}

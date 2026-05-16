//! Rig-line impulse constraint — port of TS sim/Grapple.ts.
//!
//! Three states per frame:
//!   • slack  (len < rest_len): no force — free ballistic swing within radius
//!   • taut   (len ≥ rest_len): cancel only the separating radial relative
//!              velocity via impulse; equal-and-opposite on a player anchor
//!              ⇒ momentum-conserving slingshot, no special case.
//!   • reel   : shrink/grow rest_len; conserve angular momentum about the
//!              anchor (L = v_t · r) ⇒ tangential speed rises on reel-in —
//!              the emergent slingshot, not scripted.

use crate::math::Vec3;

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
/// * `anchor` — `Some(b)` for a player↔player line (impulse applied to `b`
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

    // ── Reel: adjust rest_len, conserving angular momentum about the anchor ──
    if reel != 0 {
        let target = (line.rest_len + reel as f64 * REEL_RATE * h)
            .max(TETHER_MIN)
            .min(TETHER_MAX);

        if (target - line.rest_len).abs() > f64::EPSILON {
            let rel_v = player.v.sub(a_v);
            let v_rad = rel_v.dot(n);
            let v_tan_vec = rel_v.sub(n.scale(v_rad)); // tangential component

            // Scale tangential speed by r_old/r_new (angular-momentum conservation).
            let scale = if line.rest_len > 1e-6 {
                line.rest_len / target
            } else {
                1.0
            };
            let new_tan = v_tan_vec.norm().scale(v_tan_vec.len() * scale);

            // Radial velocity follows the reel direction (closing when reeling in).
            let new_rad = -(reel as f64) * REEL_RATE;
            let new_rel_v = new_tan.add(n.scale(new_rad));

            player.v = a_v.add(new_rel_v);
            player.p = a_p.add(n.scale(target));
            line.rest_len = target;
            line.taut = true;
            return;
        }
    }

    // ── Slack: inside the radius, the line does nothing (free swing) ──────────
    if len < line.rest_len - 1e-4 {
        line.taut = false;
        return;
    }

    // ── Taut: cancel only the separating radial relative velocity ────────────
    line.taut = true;

    let rv = player.v.sub(a_v);
    let v_rel = rv.dot(n); // > 0 means separating
    if v_rel <= 0.0 {
        // Swinging inward or along the line — nothing to do.
        return;
    }

    let inv_sum = player.inv_mass + a_inv_mass;
    if inv_sum <= 0.0 {
        return;
    }

    let j = -v_rel / inv_sum; // impulse magnitude along n

    player.v = player.v.add(n.scale(j * player.inv_mass));

    // Equal-and-opposite impulse on a player anchor.
    if let Some(b) = anchor {
        b.v = b.v.add(n.scale(-j * b.inv_mass));
    }

    // Soft Baumgarte position correction — prevents slow drift past rest_len.
    let err = len - line.rest_len;
    if err > 0.0 {
        let corr = (0.2 * err) / h.max(1e-4);
        player.v = player.v.add(n.scale(-corr * (player.inv_mass / inv_sum) * h));
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const H: f64 = 1.0 / 240.0;

    /// Slack line (inside radius) must not change the player's velocity.
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

    /// Taut static anchor: separating radial velocity removed, tangential kept.
    #[test]
    fn taut_static_removes_radial_keeps_tangential() {
        // Player 10 m out on +x, moving outward (+x) and along +y (tangential).
        let mut p = Body {
            p: Vec3::new(10.0, 0.0, 0.0),
            v: Vec3::new(6.0, 4.0, 0.0),
            inv_mass: 1.0,
        };
        let mut line = Line {
            anchor_pos: Vec3::new(0.0, 0.0, 0.0),
            rest_len: 10.0,
            taut: false,
        };
        resolve_line(&mut p, &mut line, None, 0, H);
        assert!(
            p.v.x.abs() < 1e-4,
            "radial (outward) should be cancelled, got {}",
            p.v.x
        );
        assert!(
            (p.v.y - 4.0).abs() < 1e-4,
            "tangential should be preserved, got {}",
            p.v.y
        );
        assert!(line.taut);
    }

    /// Player↔player taut: total linear momentum must be conserved.
    #[test]
    fn player_player_conserves_momentum() {
        let mut anchor_body = Body {
            p: Vec3::new(0.0, 0.0, 0.0),
            v: Vec3::new(0.0, 0.0, 0.0),
            inv_mass: 1.0 / 80.0,
        };
        let mut p = Body {
            p: Vec3::new(12.0, 0.0, 0.0),
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

    /// Reel-in increases tangential speed: v_t · r ≈ conserved (L conservation).
    #[test]
    fn reel_in_raises_tangential_speed() {
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

        // Reel in for ~0.5 s (120 steps).
        for _ in 0..120 {
            resolve_line(&mut p, &mut line, None, -1, H);
        }

        let r = p.p.len();
        // Motion stays in the x-y plane; tangential = y and z components.
        let v_tan = (p.v.y * p.v.y + p.v.z * p.v.z).sqrt();

        assert!(r < 20.0, "should have been pulled inward, r = {}", r);
        assert!(v_tan > 5.0, "tangential speed should have risen, v_tan = {}", v_tan);
        // L = v_t · r should be close to the initial value of 5 · 20 = 100.
        let l = v_tan * r;
        assert!(
            (l - 100.0).abs() < 1.0,
            "angular momentum not conserved: v_t·r = {}",
            l
        );
    }
}

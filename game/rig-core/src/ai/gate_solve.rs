//! 1:1 faithful Rust port of src/ai/decide/GateSolve.ts.
//!
//! Does a real scoring throw exist from here toward MY ring?
//!
//! Physics (canon): axial (x) motion is INERTIAL — constant vx. A throw
//! from x0 reaches the attacking ring at x = ring_x after t = |ring_x - x0|
//! / |vx|. The cross-section ζ = y + i·z evolves under the roulette ODE,
//! whose exact solution gives a CLOSED-FORM world launch that zeroes the
//! ring-plane radius:
//!
//! ```text
//! vy = -y0/t + w*z0
//! vz = -z0/t - w*y0
//! ```
//!
//! We sweep only the axial speed vx (which sets the flight time); for each
//! vx the cross-plane launch is the closed form above. We keep the
//! skin-safe solution whose arrival radius is smallest. Deterministic — no
//! RNG.
//!
//! Flight verification uses the SACRED canon integrator
//! `crate::trajectory::rk4_step` (the exact fn the sim uses) so a solved
//! throw is verified against the true simulation physics.

use crate::ai::orientation::{attack_ring_x, attack_sign};
use crate::ai::types::TeamSide;
use crate::math::Vec3;
use crate::trajectory::{rk4_step, PointState};
use crate::tuning::{GATE_RADIUS, GATE_X, L, R};

// The sim launches a thrown bell at  v_bell = thrower.v + dir·speed  with
// THROW_MIN ≤ speed ≤ THROW_MAX. To realise our desired free-bell velocity
// v0 EXACTLY, the caller must aim along (v0 − thrower.v) and charge so the
// release speed equals |v0 − thrower.v|. We surface that here so the solver
// only returns physically achievable shots.
pub const THROW_MIN_SPEED: f64 = 9.0;
pub const THROW_MAX_SPEED: f64 = 34.0;

/// A ring-threading launch solution.
#[derive(Clone, Copy, Debug)]
pub struct GateSolution {
    /// Launch velocity (world) that threads the attacking ring.
    pub v0: Vec3,
    /// Throw vector the player must impart: v0 − thrower.v (aim = its unit).
    pub throw_vec: Vec3,
    /// Release speed required = |throw_vec| (already ≤ THROW_MAX_SPEED).
    pub release_speed: f64,
    /// Predicted arrival radius at the ring plane (m).
    pub arrive_rho: f64,
    /// Flight time to the ring (s).
    pub flight_time: f64,
    /// Axial speed component used.
    pub vx: f64,
}

const SOLVE_H: f64 = 1.0 / 240.0; // match the canonical RIG sim timestep

struct RingHit {
    rho: f64,
    max_r: f64,
    t: f64,
}

/// Verify a candidate launch in the SAME rk4 integrator the sim uses:
/// return the cross-section radius at the moment x crosses `ring_x`, and
/// the max radius over the flight (skin-safety). Returns None if it never
/// reaches the plane.
fn verify_at_ring(
    p0: Vec3,
    v0: Vec3,
    omega: f64,
    ring_x: f64,
    sign: f64,
    t_flight: f64,
) -> Option<RingHit> {
    if sign * v0.x <= 0.01 {
        return None;
    }
    let mut s = PointState { p: p0, v: v0 };
    let mut prev_p = p0;
    let mut max_r = (p0.y * p0.y + p0.z * p0.z).sqrt();
    let max_steps = (t_flight * 1.4 * 240.0).ceil() as i64 + 8;
    for i in 1..=max_steps {
        let prev = s;
        s = rk4_step(s, omega, SOLVE_H);
        let r = (s.p.y * s.p.y + s.p.z * s.p.z).sqrt();
        if r > max_r {
            max_r = r;
        }
        let crossed = (sign > 0.0 && prev_p.x < ring_x && s.p.x >= ring_x)
            || (sign < 0.0 && prev_p.x > ring_x && s.p.x <= ring_x);
        if crossed {
            // Linear-interpolate to the exact crossing for an accurate rho.
            let denom = s.p.x - prev.p.x;
            let frac = if denom.abs() > 1e-12 {
                (ring_x - prev.p.x) / denom
            } else {
                0.5
            };
            let cy = prev.p.y + frac * (s.p.y - prev.p.y);
            let cz = prev.p.z + frac * (s.p.z - prev.p.z);
            return Some(RingHit {
                rho: (cy * cy + cz * cz).sqrt(),
                max_r,
                t: i as f64 * SOLVE_H,
            });
        }
        prev_p = s.p;
    }
    None
}

/// Search for a launch velocity from `from` that threads the team's
/// attacking ring. For each candidate axial speed the cross-plane launch is
/// the exact closed-form that nulls the ring-plane radius (v_rot = −ζ₀/t).
/// Returns the best skin-safe ring solution, or None if none reaches the
/// ring in a believable flight.
///
/// Deterministic — no RNG. Symmetric in team (uses Orientation only).
pub fn solve_gate_throw(
    from: Vec3,
    team: TeamSide,
    omega: f64,
    base_speed: f64,
    _loop_propensity: f64,
    // Thrower's current velocity (the sim ADDS it to the launch — we must
    // cancel it). Pass `Vec3::new(0,0,0)` for callers that don't supply it
    // (the TS default `{x:0,y:0,z:0}`).
    from_vel: Vec3,
) -> Option<GateSolution> {
    let ring_x = attack_ring_x(team);
    let sign = attack_sign(team);
    let to_ring_x = ring_x - from.x;
    let dx = to_ring_x.abs();

    // Need real axial separation to fly through the plane; if we're
    // basically on the ring plane there is no meaningful gate throw.
    if dx < 6.0 {
        return None;
    }
    // Too far for any believable flight (whole field is 640 m).
    if dx > L * 1.05 {
        return None;
    }

    let y0 = from.y;
    let z0 = from.z;

    // HARD CONSTRAINT: the sim kills a free bell after 30 s. Cap flight time
    // well under it and SCORE candidates to prefer shorter flights.
    const MAX_FLIGHT: f64 = 24.0; // s — safe margin below the 30 s limit

    // Axial-speed sweep. Slower → longer flight → more Coriolis winding;
    // faster → flatter. All thread the ring by construction; keep the
    // smallest-rho skin-safe one. (Order preserved verbatim from the TS.)
    let speeds: Vec<f64> = [
        10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 32.0, base_speed,
    ]
    .into_iter()
    .filter(|&s| s > 6.0 && s < THROW_MAX_SPEED + 8.0)
    .collect();

    let mut best: Option<GateSolution> = None;
    let mut best_cost: f64 = f64::INFINITY;

    for vx in speeds {
        let t = dx / vx;
        // Keep flight time physically sane.
        if t < 1.0 || t > MAX_FLIGHT {
            continue;
        }

        // EXACT closed-form cross-plane launch that zeroes |ζ| at the ring.
        //   v_rot = −ζ₀/t  ⇒  vy = −y₀/t + ω·z₀ ,  vz = −z₀/t − ω·y₀
        let v0 = Vec3::new(
            to_ring_x.signum() * vx,
            -y0 / t + omega * z0,
            -z0 / t - omega * y0,
        );

        // The player must impart (v0 − thrower.v); the sim adds thrower.v.
        let throw_vec = Vec3::new(
            v0.x - from_vel.x,
            v0.y - from_vel.y,
            v0.z - from_vel.z,
        );
        let release_speed = (throw_vec.x * throw_vec.x
            + throw_vec.y * throw_vec.y
            + throw_vec.z * throw_vec.z)
            .sqrt();
        // Unthrowable by the sim's release-speed band → skip.
        if release_speed < THROW_MIN_SPEED || release_speed > THROW_MAX_SPEED {
            continue;
        }

        let hit = match verify_at_ring(from, v0, omega, ring_x, sign, t) {
            Some(h) => h,
            None => continue,
        };
        // Reject if the flight ever touches the skin.
        if hit.max_r >= R {
            continue;
        }

        // Cost: thread accuracy first, then strongly prefer staying in the
        // calm (small max cross-section radius); a light flight-time term
        // keeps it safely under the dead-ball timeout among similar options.
        let cost = hit.rho + hit.max_r * 0.6 + hit.t * 0.05;
        if cost < best_cost {
            best_cost = cost;
            best = Some(GateSolution {
                v0,
                throw_vec,
                release_speed,
                arrive_rho: hit.rho,
                flight_time: hit.t,
                vx,
            });
        }
    }

    // Only return if it actually threads the ring (rho ≤ gate radius).
    if let Some(b) = best {
        if b.arrive_rho <= GATE_RADIUS {
            return Some(b);
        }
    }
    None
}

/// Mirror of `export const GATE_RADIUS = REG.gateRadius`.
pub const GATE_RADIUS_EXPORT: f64 = GATE_RADIUS;
/// Mirror of `export const GATE_HALF = GATE_X`.
pub const GATE_HALF: f64 = GATE_X;

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = crate::tuning::OMEGA;
    const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);

    /// Deterministic: identical inputs → identical output. No RNG.
    #[test]
    fn deterministic() {
        let from = Vec3::new(-200.0, 3.0, -2.0);
        let a = solve_gate_throw(from, TeamSide::Home, W, 24.0, 0.5, ZERO);
        let b = solve_gate_throw(from, TeamSide::Home, W, 24.0, 0.5, ZERO);
        match (a, b) {
            (Some(x), Some(y)) => {
                assert_eq!(x.v0, y.v0);
                assert_eq!(x.vx, y.vx);
                assert_eq!(x.arrive_rho, y.arrive_rho);
                assert_eq!(x.flight_time, y.flight_time);
                assert_eq!(x.release_speed, y.release_speed);
            }
            (None, None) => {}
            _ => panic!("nondeterministic Some/None"),
        }
    }

    /// On the ring plane → None (dx < 6).
    #[test]
    fn no_axial_separation_returns_none() {
        // Home attacks +GATE_X (= +320). Place very close to that plane.
        let from = Vec3::new(GATE_X - 1.0, 2.0, 1.0);
        assert!(solve_gate_throw(from, TeamSide::Home, W, 24.0, 0.5, ZERO).is_none());
    }

    /// Too far → None (dx > L*1.05).
    #[test]
    fn too_far_returns_none() {
        // Away attacks -320; put it absurdly far on the +x side.
        let from = Vec3::new(GATE_X + L, 0.0, 0.0);
        assert!(solve_gate_throw(from, TeamSide::Away, W, 24.0, 0.5, ZERO).is_none());
    }

    /// True-by-construction: when a solution is returned, re-simulating the
    /// launch with the SACRED canon integrator from `from` at `v0` arrives
    /// at the ring plane within the gate radius. Tested for BOTH teams to
    /// confirm Orientation symmetry.
    #[test]
    fn solution_threads_ring_resimulated() {
        for &team in &[TeamSide::Home, TeamSide::Away] {
            let sign = attack_sign(team);
            let ring_x = attack_ring_x(team);
            // Start 250 m short of our ring, slightly off-axis.
            let from = Vec3::new(ring_x - sign * 250.0, 4.0, -3.0);
            if let Some(sol) = solve_gate_throw(team_from(from), team, W, 24.0, 0.5, ZERO)
            {
                // Re-roll with the canon sim integrator until x crosses ring.
                let mut s = PointState {
                    p: from,
                    v: sol.v0,
                };
                let mut prev = s;
                let max_steps = (sol.flight_time * 1.4 * 240.0).ceil() as i64 + 8;
                let mut arrived = false;
                for _ in 1..=max_steps {
                    prev = s;
                    s = rk4_step(s, W, SOLVE_H);
                    let crossed = (sign > 0.0
                        && prev.p.x < ring_x
                        && s.p.x >= ring_x)
                        || (sign < 0.0 && prev.p.x > ring_x && s.p.x <= ring_x);
                    if crossed {
                        let denom = s.p.x - prev.p.x;
                        let frac = if denom.abs() > 1e-12 {
                            (ring_x - prev.p.x) / denom
                        } else {
                            0.5
                        };
                        let cy = prev.p.y + frac * (s.p.y - prev.p.y);
                        let cz = prev.p.z + frac * (s.p.z - prev.p.z);
                        let rho = (cy * cy + cz * cz).sqrt();
                        assert!(
                            rho <= GATE_RADIUS + 1e-6,
                            "team {team:?} re-sim rho {rho} exceeds gate radius"
                        );
                        assert!(sol.release_speed <= THROW_MAX_SPEED);
                        assert!(sol.release_speed >= THROW_MIN_SPEED);
                        arrived = true;
                        break;
                    }
                }
                assert!(arrived, "team {team:?} solution never reached ring plane");
            }
        }
    }

    // Helper: keep `from` as-is (the closure above wants the same point the
    // solver was given). Separate fn so the test reads cleanly.
    fn team_from(p: Vec3) -> Vec3 {
        p
    }
}

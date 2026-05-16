//! 6-DOF bell with internal mass-ring (symmetric top) — port of TS sim/Bell.ts.
//! Body frame: ring axis = body +x̂. Inertia I = diag(I_AXIAL, I_TRANS, I_TRANS).
//! A true spin sits on the principal axis (ω×(Iω) ≈ 0) and stays true;
//! an off-axis impulse induces gyroscopic precession and tumble.

use crate::math::{Quat, Vec3};
use crate::trajectory::accel;

// Thin-ring inertia, normalised.
const I_AXIAL: f64 = 1.0; // about the ring axis (body x)
const I_TRANS: f64 = 0.5; // about a diameter (body y,z)
const SPIN_GOOD: f64 = 8.0; // rad/s where the chime is fully voiced
const SPIN_MIN: f64 = 1.5; // below this it cannot ring at all

/// Full rigid-body state: centre-of-mass position/velocity + orientation/spin.
#[derive(Clone, Copy, Debug)]
pub struct BellBody {
    pub p: Vec3,
    pub v: Vec3,
    pub q: Quat, // orientation body → world (unit quaternion)
    pub w: Vec3, // angular velocity in BODY frame (rad/s)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Euler's equation, body frame, free flight (τ_ext = 0):
///   I·ω̇ = −ω × (I·ω)
fn ang_accel_body(w: Vec3) -> Vec3 {
    let iw = Vec3::new(I_AXIAL * w.x, I_TRANS * w.y, I_TRANS * w.z);
    let gyro = w.cross(iw);
    Vec3::new(-gyro.x / I_AXIAL, -gyro.y / I_TRANS, -gyro.z / I_TRANS)
}

/// All four time-derivatives for (p, v, q, w).
fn deriv_bell(s: BellBody, omega: f64) -> (Vec3, Vec3, Quat, Vec3) {
    let w_world = s.q.rotate(s.w);
    let dq = s.q.deriv(w_world);
    (s.v, accel(s.p, s.v, omega), dq, ang_accel_body(s.w))
}

/// Linear blend:  s + k * f  (componentwise over all four fields).
fn add_scaled(s: BellBody, dp: Vec3, dv: Vec3, dq: Quat, dw: Vec3, f: f64) -> BellBody {
    BellBody {
        p: s.p.add(dp.scale(f)),
        v: s.v.add(dv.scale(f)),
        q: Quat::new(
            s.q.x + dq.x * f,
            s.q.y + dq.y * f,
            s.q.z + dq.z * f,
            s.q.w + dq.w * f,
        ),
        w: s.w.add(dw.scale(f)),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Advance the bell by one RK4 step of size `h` seconds.
/// The quaternion is renormalised at the end of every step.
pub fn step_bell(s: BellBody, omega: f64, h: f64) -> BellBody {
    let (k1p, k1v, k1q, k1w) = deriv_bell(s, omega);
    let s2 = add_scaled(s, k1p, k1v, k1q, k1w, h / 2.0);

    let (k2p, k2v, k2q, k2w) = deriv_bell(s2, omega);
    let s3 = add_scaled(s, k2p, k2v, k2q, k2w, h / 2.0);

    let (k3p, k3v, k3q, k3w) = deriv_bell(s3, omega);
    let s4 = add_scaled(s, k3p, k3v, k3q, k3w, h);

    let (k4p, k4v, k4q, k4w) = deriv_bell(s4, omega);

    let c = h / 6.0;
    BellBody {
        p: Vec3::new(
            s.p.x + c * (k1p.x + 2.0 * k2p.x + 2.0 * k3p.x + k4p.x),
            s.p.y + c * (k1p.y + 2.0 * k2p.y + 2.0 * k3p.y + k4p.y),
            s.p.z + c * (k1p.z + 2.0 * k2p.z + 2.0 * k3p.z + k4p.z),
        ),
        v: Vec3::new(
            s.v.x + c * (k1v.x + 2.0 * k2v.x + 2.0 * k3v.x + k4v.x),
            s.v.y + c * (k1v.y + 2.0 * k2v.y + 2.0 * k3v.y + k4v.y),
            s.v.z + c * (k1v.z + 2.0 * k2v.z + 2.0 * k3v.z + k4v.z),
        ),
        q: Quat::new(
            s.q.x + c * (k1q.x + 2.0 * k2q.x + 2.0 * k3q.x + k4q.x),
            s.q.y + c * (k1q.y + 2.0 * k2q.y + 2.0 * k3q.y + k4q.y),
            s.q.z + c * (k1q.z + 2.0 * k2q.z + 2.0 * k3q.z + k4q.z),
            s.q.w + c * (k1q.w + 2.0 * k2q.w + 2.0 * k3q.w + k4q.w),
        )
        .norm(),
        w: Vec3::new(
            s.w.x + c * (k1w.x + 2.0 * k2w.x + 2.0 * k3w.x + k4w.x),
            s.w.y + c * (k1w.y + 2.0 * k2w.y + 2.0 * k3w.y + k4w.y),
            s.w.z + c * (k1w.z + 2.0 * k2w.z + 2.0 * k3w.z + k4w.z),
        ),
    }
}

/// Spin-trueness ∈ [0,1]: alignment of ω with the body ring axis (body x̂).
/// Returns 0 if |ω| < 1e-6.
pub fn trueness(w: Vec3) -> f64 {
    let sp = w.len();
    if sp < 1e-6 {
        return 0.0;
    }
    (w.x.abs() / sp).clamp(0.0, 1.0)
}

/// Audible chime ∈ [0,1]: trueness gated by enough spin to voice.
pub fn chime(w: Vec3) -> f64 {
    let sp = w.len();
    trueness(w) * smoothstep(SPIN_MIN, SPIN_GOOD, sp)
}

/// Apply an off-axis body-frame angular impulse (e.g. a defender's knock).
/// Each component is divided by the corresponding principal inertia.
pub fn clatter(s: &mut BellBody, body_impulse: Vec3) {
    s.w.x += body_impulse.x / I_AXIAL;
    s.w.y += body_impulse.y / I_TRANS;
    s.w.z += body_impulse.z / I_TRANS;
}

// ---------------------------------------------------------------------------
// Private utility
// ---------------------------------------------------------------------------

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Tests — mirror of test/bell.test.ts
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::analytic;

    const W: f64 = 0.32;
    const H: f64 = 1.0 / 240.0;

    fn mk(w: Vec3) -> BellBody {
        BellBody {
            p: Vec3::new(0.0, 4.0, -2.0),
            v: Vec3::new(18.0, -3.0, 5.0),
            q: Quat::ident(),
            w,
        }
    }

    fn default_bell() -> BellBody {
        mk(Vec3::new(30.0, 0.0, 0.0))
    }

    #[test]
    fn com_follows_analytic_coriolis_path() {
        let mut s = default_bell();
        let p0 = s.p;
        let v0 = s.v;
        for i in 1..=(240 * 5) {
            s = step_bell(s, W, H);
            let t = i as f64 * H;
            let a = analytic(p0, v0, W, t);
            assert!((s.p.y - a.y).abs() < 1e-2, "y mismatch at t={t}: {} vs {}", s.p.y, a.y);
            assert!((s.p.z - a.z).abs() < 1e-2, "z mismatch at t={t}: {} vs {}", s.p.z, a.z);
            assert!(
                (s.p.x - v0.x * t).abs() < 1e-4,
                "x axial mismatch at t={t}: {} vs {}",
                s.p.x,
                v0.x * t
            );
        }
    }

    #[test]
    fn true_spin_stays_true() {
        let mut s = mk(Vec3::new(28.0, 0.0, 0.0));
        for _ in 0..(240 * 4) {
            s = step_bell(s, W, H);
        }
        assert!(trueness(s.w) > 0.999, "trueness = {}", trueness(s.w));
        assert!(chime(s.w) > 0.9, "chime = {}", chime(s.w));
    }

    #[test]
    fn off_axis_spin_tumbles_and_precesses() {
        let mut s = mk(Vec3::new(14.0, 9.0, 6.0));
        let t0 = trueness(s.w);
        let mut moved = false;
        let mut prev_wy = s.w.y;
        for _ in 0..(240 * 3) {
            s = step_bell(s, W, H);
            if (s.w.y - prev_wy).abs() > 1e-3 {
                moved = true;
            }
            prev_wy = s.w.y;
        }
        assert!(t0 < 0.85, "initial trueness should be < 0.85, got {t0}");
        assert!(chime(s.w) < 0.9, "chime should drop, got {}", chime(s.w));
        assert!(moved, "gyroscopic precession should move ω_y");
    }

    #[test]
    fn angular_momentum_magnitude_conserved() {
        let lm = |b: &BellBody| -> f64 {
            // |I·ω| as a body-frame proxy
            (I_AXIAL * b.w.x).hypot((I_TRANS * b.w.y).hypot(I_TRANS * b.w.z))
        };
        let mut s = mk(Vec3::new(14.0, 9.0, 6.0));
        let l0 = lm(&s);
        for _ in 0..(240 * 3) {
            s = step_bell(s, W, H);
        }
        let l1 = lm(&s);
        assert!((l1 - l0).abs() < 0.1, "|I·ω| drift: {l0} → {l1}");
    }

    #[test]
    fn clatter_drops_trueness() {
        let mut s = mk(Vec3::new(28.0, 0.0, 0.0));
        assert!(trueness(s.w) > 0.999);
        clatter(&mut s, Vec3::new(0.0, 12.0, 8.0));
        assert!(trueness(s.w) < 0.92, "trueness after clatter = {}", trueness(s.w));
    }
}

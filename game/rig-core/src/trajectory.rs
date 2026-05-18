//! Pure Coriolis/centrifugal predictor — port of TS sim/trajectory.ts.
//! Frame: +X = spin axis (inertial along X); (y,z) is the rotating plane.
//!   a = ( 0,  ω²·y + 2ω·vz,  ω²·z − 2ω·vy )

use crate::math::Vec3;

pub fn accel(p: Vec3, v: Vec3, omega: f64) -> Vec3 {
    Vec3::new(
        0.0,
        omega * omega * p.y + 2.0 * omega * v.z,
        omega * omega * p.z - 2.0 * omega * v.y,
    )
}

#[derive(Clone, Copy, Debug)]
pub struct PointState {
    pub p: Vec3,
    pub v: Vec3,
}

pub fn rk4_step(s: PointState, omega: f64, h: f64) -> PointState {
    let deriv = |st: PointState| (st.v, accel(st.p, st.v, omega));
    let add = |st: PointState, dp: Vec3, dv: Vec3, f: f64| PointState {
        p: st.p.add(dp.scale(f)),
        v: st.v.add(dv.scale(f)),
    };
    let (k1p, k1v) = deriv(s);
    let (k2p, k2v) = deriv(add(s, k1p, k1v, h / 2.0));
    let (k3p, k3v) = deriv(add(s, k2p, k2v, h / 2.0));
    let (k4p, k4v) = deriv(add(s, k3p, k3v, h));
    PointState {
        p: Vec3::new(
            s.p.x + (h / 6.0) * (k1p.x + 2.0 * k2p.x + 2.0 * k3p.x + k4p.x),
            s.p.y + (h / 6.0) * (k1p.y + 2.0 * k2p.y + 2.0 * k3p.y + k4p.y),
            s.p.z + (h / 6.0) * (k1p.z + 2.0 * k2p.z + 2.0 * k3p.z + k4p.z),
        ),
        v: Vec3::new(
            s.v.x + (h / 6.0) * (k1v.x + 2.0 * k2v.x + 2.0 * k3v.x + k4v.x),
            s.v.y + (h / 6.0) * (k1v.y + 2.0 * k2v.y + 2.0 * k3v.y + k4v.y),
            s.v.z + (h / 6.0) * (k1v.z + 2.0 * k2v.z + 2.0 * k3v.z + k4v.z),
        ),
    }
}

/// Analytic closed form: x inertial; ζ(t) = (ζ0 + (ζ̇0 + iω·ζ0)·t)·e^(−iω t).
pub fn analytic(p0: Vec3, v0: Vec3, omega: f64, t: f64) -> Vec3 {
    let ar = p0.y;
    let ai = p0.z;
    let br = v0.y - omega * p0.z;
    let bi = v0.z + omega * p0.y;
    let cr = ar + br * t;
    let ci = ai + bi * t;
    let c = (omega * t).cos();
    let s = (omega * t).sin();
    Vec3::new(p0.x + v0.x * t, cr * c + ci * s, ci * c - cr * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = 0.32;
    const H: f64 = 1.0 / 240.0;

    #[test]
    fn axial_is_inertial() {
        let mut s = PointState {
            p: Vec3::new(0.0, 3.0, 0.0),
            v: Vec3::new(25.0, 0.0, 4.0),
        };
        for i in 1..=240 * 8 {
            s = rk4_step(s, W, H);
            let t = i as f64 * H;
            assert!((s.p.x - 25.0 * t).abs() < 1e-6);
        }
    }

    #[test]
    fn rk4_tracks_analytic() {
        let p0 = Vec3::new(0.0, 4.0, -2.0);
        let v0 = Vec3::new(18.0, -3.0, 5.0);
        let mut s = PointState { p: p0, v: v0 };
        for i in 1..=240 * 6 {
            s = rk4_step(s, W, H);
            let a = analytic(p0, v0, W, i as f64 * H);
            assert!((s.p.y - a.y).abs() < 1e-3, "y {} vs {}", s.p.y, a.y);
            assert!((s.p.z - a.z).abs() < 1e-3, "z {} vs {}", s.p.z, a.z);
        }
    }
}

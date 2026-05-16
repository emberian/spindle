//! The Number, in Rust — same gameplay inequalities as TS sim/tuning.ts,
//! enforced by `cargo test` so the gate holds in both toolchains.

pub const OMEGA: f64 = 0.32;
pub const R: f64 = 45.0;
pub const L: f64 = 640.0;
pub const GATE_RADIUS: f64 = 8.0;
pub const EARTH_G: f64 = 9.80665;
pub const GATE_X: f64 = L / 2.0;

const STRONG_THROW: f64 = 30.0;
const LAUNCH_RHO: f64 = 2.0;
const LAUNCH_VPERP: f64 = 2.0;
const LOOP_RHO: f64 = 1.5;
const LOOP_VPERP: f64 = 4.0;
pub const LOOP_TURN_TARGET: f64 = 2.2;

pub fn time_to_skin(omega: f64, r: f64, rho0: f64, v_perp: f64) -> f64 {
    let k = v_perp + omega * rho0;
    if k <= 1e-9 {
        return f64::INFINITY;
    }
    if r <= rho0 {
        return 0.0;
    }
    (r * r - rho0 * rho0).sqrt() / k
}

pub struct TuningReport {
    pub axial_reach: f64,
    pub traversable: bool,
    pub loop_wind: f64,
    pub loop_flight: f64,
    pub loop_achievable: bool,
    pub g_skin_fraction: f64,
    pub gradient_ok: bool,
    pub pass: bool,
}

pub fn evaluate(omega: f64, r: f64, l: f64, earth_g: f64) -> TuningReport {
    let t_skin = time_to_skin(omega, r, LAUNCH_RHO, LAUNCH_VPERP);
    let axial_reach = STRONG_THROW * t_skin;
    let traversable = axial_reach >= 0.5 * l;

    let loop_flight = time_to_skin(omega, r, LOOP_RHO, LOOP_VPERP);
    let loop_wind = omega * loop_flight;
    let loop_achievable = loop_wind >= LOOP_TURN_TARGET && loop_wind <= 6.5;

    let g_skin = omega * omega * r;
    let g_skin_fraction = g_skin / earth_g;
    let gradient_ok = (0.1..=0.6).contains(&g_skin_fraction);

    TuningReport {
        axial_reach,
        traversable,
        loop_wind,
        loop_flight,
        loop_achievable,
        g_skin_fraction,
        gradient_ok,
        pass: traversable && loop_achievable && gradient_ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reg_constants_pass_the_gate() {
        let r = evaluate(OMEGA, R, L, EARTH_G);
        assert!(r.traversable, "axial_reach {}", r.axial_reach);
        assert!(r.loop_wind >= 2.2 && r.loop_wind <= 6.5, "loop_wind {}", r.loop_wind);
        assert!(r.loop_flight > 4.0 && r.loop_flight < 16.0, "loop_flight {}", r.loop_flight);
        assert!(
            r.g_skin_fraction >= 0.1 && r.g_skin_fraction <= 0.6,
            "g {}",
            r.g_skin_fraction
        );
        assert!(r.pass);
    }
}

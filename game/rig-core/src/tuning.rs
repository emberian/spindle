//! The Number, in Rust — same gameplay inequalities as TS sim/tuning.ts,
//! enforced by `cargo test` so the gate holds in both toolchains.

pub const OMEGA: f64 = 0.32;
pub const R: f64 = 45.0;
pub const L: f64 = 640.0;
pub const GATE_RADIUS: f64 = 8.0;
pub const EARTH_G: f64 = 9.80665;
pub const GATE_X: f64 = L / 2.0;

// ── FEEL constants (Fluidity Spike) ───────────────────────────────────────────
// Continuous player physics. NOT part of tuning::evaluate (gameplay gate).
// All springs integrated with semi-implicit (symplectic) Euler at h = 1/240 s;
// every ω_n·h ≪ 2 so they are unconditionally stable here. Same literals are
// mirrored verbatim in RegConstants.ts so Rust and TS compute identical f64.

/// Taut rig-line radial spring stiffness (N/m equivalent over PLAYER_MASS).
/// ω_n = sqrt(LINE_K / 78) ≈ 4.24 rad/s ⇒ ω_n·h ≈ 0.018 ≪ 2.
pub const LINE_K: f64 = 1400.0;
/// Taut rig-line radial damping. ζ ≈ LINE_C / (2·sqrt(LINE_K·78)) ≈ 0.55
/// (near-critical: snappy, no ring-out, no overshoot). Reeling-in still
/// slingshots (~16–30 m/s cruise, controllable); the "carrier can't move"
/// bug was the onboarding eating clicks + grapple gated on !holdingBell,
/// NOT this damping.
pub const LINE_C: f64 = 520.0;
/// Slack pre-tension band width (m) below rest_len where a gentle spring
/// pre-loads the rope so it eases into tautness instead of snapping.
pub const LINE_SLACK_BAND: f64 = 0.6;
/// Pre-tension stiffness inside the slack band (soft — ~1/12 of LINE_K).
pub const LINE_SLACK_K: f64 = 120.0;

/// POWERED HOOK (locomotion fix). The rig line is a Xonotic-style winch,
/// not a passive pendulum rope: reeling-IN actively drags the rigger
/// toward the anchor. Without this the riggers cannot self-propel —
/// "the calm" is the near-axis WEIGHTLESS volume (centrifugal ≈ ω²r ≈
/// 0.8 m/s² at play radius), so there is no gravity to pendulum against
/// and a passive radial spring on a near-rest body produces ~no motion.
/// The winch is the rigger's only real propulsion; it must be strong.
/// `REEL_PULL_SPEED` = the cruise speed it drags you toward the anchor;
/// `REEL_PULL_ACCEL` = how hard the motor accelerates you to it
/// (accel-capped ⇒ deterministic + unconditionally stable, no spring
/// stiffness blow-up). Reach cruise in ≈ SPEED/ACCEL ≈ 0.22 s.
pub const REEL_PULL_SPEED: f64 = 26.0; // m/s toward the anchor while reeling in
pub const REEL_PULL_ACCEL: f64 = 120.0; // m/s² winch acceleration toward anchor

/// GRAPPLE LATENCY (the catch fix). The rig claw is NOT instantaneous: when
/// a line is fired the claw travels from the player toward the fire target at
/// `CLAW_SPEED` m/s and only ATTACHES (line becomes active/anchored) after
/// `ceil(distance / CLAW_SPEED / h)` ticks. While the claw is in flight the
/// line is committed to that target and re-aiming is ignored until it lands
/// or an explicit release cancels it. After a release (or a missed/expired
/// claw) the player cannot fire again for `REFIRE_COOLDOWN_TICKS`.
///
/// `CLAW_SPEED` = 80 m/s ⇒ at h = 1/240 s the flight is `ceil(distance·3)`
/// ticks: a 20 m shot lands in 60 ticks ≈ 0.25 s, 30 m ≈ 0.375 s, 40 m ≈
/// 0.5 s — slow enough to read on screen, fast enough to still be a viable
/// committed dive. Tick-counted (integer ceil against the sim tick), NO
/// wall clock, NO rng ⇒ fully deterministic.
pub const CLAW_SPEED: f64 = 80.0; // m/s — claw travel speed from player to target
/// Re-fire lockout after a release / missed claw. 48 ticks at 240 Hz = 0.2 s
/// — long enough that the AI must RIDE its committed line instead of
/// spamming a fresh prediction every tick, short enough not to feel sticky.
pub const REFIRE_COOLDOWN_TICKS: u64 = 48;

/// Soft-grounding penetration push-out spring (rho ≥ R).
/// ω_n = sqrt(GROUND_K / 78) ≈ 8.0 rad/s ⇒ ω_n·h ≈ 0.033 ≪ 2.
pub const GROUND_K: f64 = 5000.0;
/// Grounding push-out damping (near-critical on the radial DOF).
pub const GROUND_C: f64 = 1250.0;
/// Per-step tangential friction multiplier applied to the grounded
/// tangential velocity. PRECOMPUTED so neither side calls exp() per step:
/// = exp(-FRICTION_RATE · h) with FRICTION_RATE = 6 /s, h = 1/240
/// → exp(-0.025) computed once, frozen as a literal for bit-identical f64.
pub const FRICTION_DECAY: f64 = 0.9753099120283326;

/// Eased-actuator ramp lengths (ticks). TOTAL Δv is unchanged; the budget is
/// consumed immediately on trigger so cap/determinism semantics are preserved.
pub const PUSHOFF_RAMP_TICKS: u32 = 18;
pub const THRUMBLER_RAMP_TICKS: u32 = 12;

/// Stiff stick-spring used for `contact` (replaces the inv_mass=0 freeze).
/// ω_n = sqrt(CONTACT_K / 78) ≈ 8.77 rad/s ⇒ ω_n·h ≈ 0.037 ≪ 2.
pub const CONTACT_K: f64 = 6000.0;
/// Stick-spring damping (near-critical → no buzz).
pub const CONTACT_C: f64 = 1400.0;

/// Held-bell follow spring (bell tracks holder's hand softly instead of a raw
/// p/v copy). Bell mass ≈ 1 (treated as unit in sim) so ω_n = sqrt(HOLD_K)
/// ≈ 28.3 rad/s ⇒ ω_n·h ≈ 0.118 ≪ 2 (still stable, tight tracking).
pub const HOLD_K: f64 = 800.0;
/// Held-bell follow damping (near-critical for unit mass).
pub const HOLD_C: f64 = 56.0;

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

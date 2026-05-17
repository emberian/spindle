// "The Reg" — the mandated reference constants of the sport. Canon explicitly
// says the Reg fixes ONE negotiated reference spin so the game plays identically
// on every cylinder, which means ω is ours to *choose by design*. These values
// are the output of the tuning solver (see tuning.ts) and are asserted to
// satisfy the three gameplay inequalities by test/tuning.test.ts (CI gate).
//
// Frame convention: world +X = the spin axis = the 640 m length of the calm.
// The (y,z) plane is the rotating cross-section (centrifugal + Coriolis live
// here). Motion parallel to +X is inertial. Faith ring at +x end, Free at -x.

export const REG = {
  // Retuned (P2.5) so an *approximate* closed Loop is achievable in a
  // snappy ~8 s gentle near-axis orbit instead of an ~18 s knife-edge.
  omega: 0.32, // rad/s (spin period ≈ 19.6 s)
  R: 45, // m — skin radius (tighter calm → loop closes sooner, gradient sane)
  L: 640, // m — axial field length (gate rings at x = ±L/2)
  gateRadius: 8, // m — scoring ring radius
  EARTH_G: 9.80665,
} as const;

export const GATE_X = REG.L / 2; // ±320 m

// ── FEEL constants (Fluidity Spike) ───────────────────────────────────────────
// Continuous player physics. NOT part of the tuning gate. Every value is the
// verbatim mirror of tuning.rs so Rust and TS compute bit-identical f64. All
// springs use semi-implicit (symplectic) Euler at h = 1/240 s with ω_n·h ≪ 2.
export const FEEL = {
  LINE_K: 1400, // taut rig-line radial spring  (ω_n·h ≈ 0.018)
  LINE_C: 520, // taut rig-line radial damping (ζ ≈ 0.55, near-critical; reel-in still slingshots ~16–30 m/s cruise)
  LINE_SLACK_BAND: 0.6, // m — slack pre-tension band below rest_len
  LINE_SLACK_K: 120, // soft pre-tension stiffness in the band

  GROUND_K: 5000, // skin penetration push-out spring (ω_n·h ≈ 0.033)
  GROUND_C: 1250, // grounding push-out damping (near-critical radial)
  // exp(-6 * (1/240)) precomputed once; no per-step exp() either side.
  FRICTION_DECAY: 0.9753099120283326,

  PUSHOFF_RAMP_TICKS: 18, // eased pushoff ramp length (ticks)
  THRUMBLER_RAMP_TICKS: 12, // eased thrumbler ramp length (ticks)

  CONTACT_K: 6000, // stiff stick-spring for `contact` (ω_n·h ≈ 0.037)
  CONTACT_C: 1400, // stick-spring damping (near-critical)

  HOLD_K: 800, // held-bell follow spring (ω_n·h ≈ 0.118)
  HOLD_C: 56, // held-bell follow damping (near-critical, unit mass)
} as const;

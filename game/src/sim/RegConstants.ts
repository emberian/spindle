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

  // PLAYER–PLAYER SOFT-BODY COLLISION — verbatim mirror of tuning.rs.
  // A rigger has a radius; overlapping pairs are pushed apart by a
  // momentum-conserving soft spring + damper applied equal-and-opposite
  // along the contact normal (one O(n²) pass, fixed order, deterministic).
  // PLAYER_RADIUS 0.6 m (rigger ~1.2 m across — small vs 45 m skin / 8 m
  // ring so ring/loop geometry is untouched, big enough to screen/pick).
  // COLLIDE_K 4000 ⇒ ω_n = sqrt(4000/39) ≈ 10.1 rad/s, ω_n·h ≈ 0.042 ≪ 2.
  // COLLIDE_C 550 ⇒ ζ ≈ 0.70 on the reduced mass (soft stand-off, no bounce).
  PLAYER_RADIUS: 0.6, // m — soft-body collision radius
  COLLIDE_K: 4000, // penetration push-apart spring
  COLLIDE_C: 550, // collision normal damping (near-critical, reduced mass)

  // PART B — CONTEST + GARROTE detection — verbatim mirror of tuning.rs.
  // CONTEST_RADIUS 6 m: a loose bell with two opposing non-grounded
  // riggers both inside it is a 1:1 contest (emitted once per loose-bell
  // episode). GARROTE_RADIUS 1.0 m = PLAYER_RADIUS + 0.4 line margin: a
  // fired line's taut segment sweeping this close to an opposing body
  // centre is a garrote foul. Segment-vs-point, fixed id order, no rng.
  CONTEST_RADIUS: 6.0, // m — loose-bell 1:1 contest trigger
  GARROTE_RADIUS: 1.0, // m — line-vs-opponent-body foul distance

  PUSHOFF_RAMP_TICKS: 18, // eased pushoff ramp length (ticks)
  THRUMBLER_RAMP_TICKS: 12, // eased thrumbler ramp length (ticks)

  CONTACT_K: 6000, // stiff stick-spring for `contact` (ω_n·h ≈ 0.037)
  CONTACT_C: 1400, // stick-spring damping (near-critical)

  HOLD_K: 800, // held-bell follow spring (ω_n·h ≈ 0.118)
  HOLD_C: 56, // held-bell follow damping (near-critical, unit mass)

  // GRAPPLE LATENCY (the catch fix) — verbatim mirror of tuning.rs.
  // The claw is NOT instantaneous: a fired line travels from the player to
  // the target at CLAW_SPEED m/s and only ATTACHES after
  // ceil(distance / CLAW_SPEED / h) ticks. CLAW_SPEED = 80 ⇒ at h = 1/240
  // a 20 m shot lands in 60 ticks ≈ 0.25 s, 40 m ≈ 0.5 s. After a release
  // / missed claw the player cannot re-fire for REFIRE_COOLDOWN_TICKS
  // (48 ticks = 0.2 s). Integer tick counts, no wall clock.
  CLAW_SPEED: 80, // m/s — claw travel speed from player to target
  REFIRE_COOLDOWN_TICKS: 48, // ticks — re-fire lockout after release/miss

  // COMMITTED-DIVE TERMINAL EASE (the catch fix) — verbatim mirror of
  // tuning.rs DIVE_TERMINAL_RADIUS. The decisive dive fires its anchor
  // downrange ALONG the predicted bell velocity at the catch point so the
  // winch pulls the rigger PARALLEL to the bell's path (velocity-matched
  // rendezvous, not a crossing flyby). Within DIVE_TERMINAL_RADIUS m of
  // that predicted catch point the rigger stops hard-winching (reel = 0,
  // it does NOT release — latency makes re-acquire costly) so its speed
  // bleeds toward the bell's, dropping |bell_v − player_v| into the
  // committed absorb window (COMMIT_CATCH_SPEED 38) ⇒ try_catch_ex Caught.
  // = COMMIT_ARM_REACH (7.0) + 2.0 margin. Tick/geometry only.
  DIVE_TERMINAL_RADIUS: 9, // m — winch-ease radius around the predicted catch point
} as const;

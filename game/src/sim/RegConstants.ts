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

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
  omega: 0.15, // rad/s (spin period ≈ 41.9 s)
  R: 100, // m — skin radius (cross-axis bound)
  L: 640, // m — axial field length (gate rings at x = ±L/2)
  gateRadius: 9, // m — scoring ring radius
  EARTH_G: 9.80665,
} as const;

export const GATE_X = REG.L / 2; // ±320 m

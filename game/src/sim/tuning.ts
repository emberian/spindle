// The ω/R/L tuning solver. v0.1's central failure was an *empirical* ω
// (0.22 unplayable -> 0.06 patch). Here tuning is a principled function of
// three gameplay targets; CI fails if the shipped Reg constants violate them.
//
// Cross-section dynamics (y,z) in the rotating frame, complex ζ = y + i·z:
//   ζ̈ = ω²·ζ − 2iω·ζ̇      (centrifugal + Coriolis)
// Solution: ζ(t) = (ζ0 + (ζ̇0 + iω·ζ0)·t)·e^(−iω t)  — a roulette/involute.
// |ζ(t)| (the cross-axis radius ρ) grows ~ linearly·exp envelope; we solve
// numerically for the time it reaches the skin R.

export interface TuningInput {
  omega: number;
  R: number;
  L: number;
  earthG: number;
}

export interface TuningReport {
  // (1) Traversability: a strong axial throw must cross half the field
  //     before cross-axis drift skins it.
  tSkin: number; // s, for a representative near-axis launch
  axialReach: number; // m at strong-throw speed
  traversable: boolean;
  // (2) Loop must be dramatic but controllable: ω·τ in a band.
  omegaTauLoop: number;
  loopDramatic: boolean;
  // (3) Gradient felt but not lethal.
  gSkinFraction: number; // g at the literal skin, in Earth-g
  gradientOk: boolean;
  pass: boolean;
}

// Representative test throw: launched near the axis, modest cross-axis speed.
const STRONG_THROW = 30; // m/s axial component of a hard throw
const LAUNCH_RHO = 2; // m off-axis at release
const LAUNCH_VPERP = 2; // m/s cross-axis at release
const LOOP_FLIGHT = 5; // s — the flight time of a dramatic loop

// Time for cross-axis radius to reach R, by closed-form |ζ(t)|.
export function timeToSkin(omega: number, R: number, rho0: number, vPerp: number): number {
  // ζ0 = rho0 (real axis WLOG), ζ̇0 = i·vPerp (purely tangential push).
  // ζ(t) = (ζ0 + (ζ̇0 + iω ζ0) t) e^{-iωt};  |e^{-iωt}| = 1, so
  // ρ(t) = |ζ0 + (ζ̇0 + iω ζ0) t| = |(rho0 + i(vPerp + ω·rho0)·t)|
  //      = sqrt(rho0² + (vPerp + ω·rho0)²·t²)   (monotonic in t)
  const k = vPerp + omega * rho0;
  if (k <= 1e-9) return Infinity;
  // sqrt(rho0² + k²t²) = R  ->  t = sqrt(R² - rho0²)/k
  if (R <= rho0) return 0;
  return Math.sqrt(R * R - rho0 * rho0) / k;
}

export function evaluateTuning(inp: TuningInput): TuningReport {
  const { omega, R, L, earthG } = inp;

  const tSkin = timeToSkin(omega, R, LAUNCH_RHO, LAUNCH_VPERP);
  const axialReach = STRONG_THROW * tSkin;
  const traversable = axialReach >= 0.5 * L; // reach at least half the field

  const omegaTauLoop = omega * LOOP_FLIGHT;
  const loopDramatic = omegaTauLoop >= 0.5 && omegaTauLoop <= 1.2;

  const gSkin = omega * omega * R; // m/s² at the skin
  const gSkinFraction = gSkin / earthG;
  const gradientOk = gSkinFraction >= 0.1 && gSkinFraction <= 0.6;

  return {
    tSkin,
    axialReach,
    traversable,
    omegaTauLoop,
    loopDramatic,
    gSkinFraction,
    gradientOk,
    pass: traversable && loopDramatic && gradientOk,
  };
}

// Inverse helper: given desired ω·τ_loop and g-fraction, derive ω and R.
export function solveConstants(
  targetOmegaTauLoop: number,
  targetGFraction: number,
  earthG: number,
  loopFlight = LOOP_FLIGHT,
): { omega: number; R: number } {
  const omega = targetOmegaTauLoop / loopFlight;
  const R = (targetGFraction * earthG) / (omega * omega);
  return { omega, R };
}

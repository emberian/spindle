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
  // (2) An *approximate* closed Loop must be achievable: a gentle near-axis
  //     throw winds enough heading (≈ LOOP_TURN_TARGET rad) before skinning.
  loopWind: number; // rad of heading wound by the gentle loop throw
  loopAchievable: boolean;
  loopFlight: number; // s the gentle loop stays airborne
  // (3) Gradient felt but not lethal.
  gSkinFraction: number; // g at the literal skin, in Earth-g
  gradientOk: boolean;
  pass: boolean;
}

// Representative throws.
const STRONG_THROW = 30; // m/s axial component of a hard throw
const LAUNCH_RHO = 2; // m off-axis at release
const LAUNCH_VPERP = 2; // m/s cross-axis at release
// The gentle "loop" throw: near the axis, low cross-axis speed so it can
// linger and wind a coming-around arc without skinning.
const LOOP_RHO = 1.5;
const LOOP_VPERP = 4;
export const LOOP_TURN_TARGET = 2.2; // rad — "comes back around" enough to read

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

  // The gentle loop throw lingers ~ this long before the skin; heading winds
  // at ≈ ω, so it accumulates ≈ ω·t of turn — "comes back around".
  const loopFlight = timeToSkin(omega, R, LOOP_RHO, LOOP_VPERP);
  const loopWind = omega * loopFlight;
  const loopAchievable = loopWind >= LOOP_TURN_TARGET && loopWind <= 6.5;

  const gSkin = omega * omega * R; // m/s² at the skin
  const gSkinFraction = gSkin / earthG;
  const gradientOk = gSkinFraction >= 0.1 && gSkinFraction <= 0.6;

  return {
    tSkin,
    axialReach,
    traversable,
    loopWind,
    loopAchievable,
    loopFlight,
    gSkinFraction,
    gradientOk,
    pass: traversable && loopAchievable && gradientOk,
  };
}

// Inverse helper: given desired ω·τ_loop and g-fraction, derive ω and R.
export function solveConstants(
  targetLoopWind: number,
  targetGFraction: number,
  earthG: number,
  loopFlight = 8,
): { omega: number; R: number } {
  const omega = targetLoopWind / loopFlight;
  const R = (targetGFraction * earthG) / (omega * omega);
  return { omega, R };
}

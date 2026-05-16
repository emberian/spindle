// PURE Coriolis/centrifugal point predictor — shared by throw-lead, loop
// detection, and grapple slingshot planning. No state, no RNG, no DOM.
// Frame: +X = spin axis (inertial along X); (y,z) is the rotating plane.
//
//   a = ( 0,  ω²·y + 2ω·vz,  ω²·z − 2ω·vy )
//
// Ported/derived from the verified v0.1 web/engine.js Coriolis block, lifted
// to 3D with the canonical "axial motion is inertial" invariant.

import type { Vec3 } from './vec';

export interface PointState {
  p: Vec3;
  v: Vec3;
}

export function accel(p: Vec3, v: Vec3, omega: number): Vec3 {
  return {
    x: 0, // axial: inertial (canon invariant)
    y: omega * omega * p.y + 2 * omega * v.z,
    z: omega * omega * p.z - 2 * omega * v.y,
  };
}

// One RK4 step of the free-flight point ODE.
export function rk4Step(s: PointState, omega: number, h: number): PointState {
  const deriv = (st: PointState) => ({
    dp: st.v,
    dv: accel(st.p, st.v, omega),
  });
  const add = (st: PointState, k: { dp: Vec3; dv: Vec3 }, f: number): PointState => ({
    p: { x: st.p.x + k.dp.x * f, y: st.p.y + k.dp.y * f, z: st.p.z + k.dp.z * f },
    v: { x: st.v.x + k.dv.x * f, y: st.v.y + k.dv.y * f, z: st.v.z + k.dv.z * f },
  });
  const k1 = deriv(s);
  const k2 = deriv(add(s, k1, h / 2));
  const k3 = deriv(add(s, k2, h / 2));
  const k4 = deriv(add(s, k3, h));
  return {
    p: {
      x: s.p.x + (h / 6) * (k1.dp.x + 2 * k2.dp.x + 2 * k3.dp.x + k4.dp.x),
      y: s.p.y + (h / 6) * (k1.dp.y + 2 * k2.dp.y + 2 * k3.dp.y + k4.dp.y),
      z: s.p.z + (h / 6) * (k1.dp.z + 2 * k2.dp.z + 2 * k3.dp.z + k4.dp.z),
    },
    v: {
      x: s.v.x + (h / 6) * (k1.dv.x + 2 * k2.dv.x + 2 * k3.dv.x + k4.dv.x),
      y: s.v.y + (h / 6) * (k1.dv.y + 2 * k2.dv.y + 2 * k3.dv.y + k4.dv.y),
      z: s.v.z + (h / 6) * (k1.dv.z + 2 * k2.dv.z + 2 * k3.dv.z + k4.dv.z),
    },
  };
}

// Predict a path: array of positions sampled every `h` for `steps`.
export function predictPath(
  p0: Vec3,
  v0: Vec3,
  omega: number,
  h: number,
  steps: number,
): Vec3[] {
  let s: PointState = { p: { ...p0 }, v: { ...v0 } };
  const out: Vec3[] = [{ ...p0 }];
  for (let i = 0; i < steps; i++) {
    s = rk4Step(s, omega, h);
    out.push({ ...s.p });
  }
  return out;
}

// Analytic closed form (the reference the RK4 sim is validated against):
//   x(t) = x0 + vx0·t                              (inertial)
//   ζ(t) = (ζ0 + (ζ̇0 + iω·ζ0)·t)·e^(−iω t)        ζ = y + i·z
export function analytic(p0: Vec3, v0: Vec3, omega: number, t: number): Vec3 {
  // ζ0 = y0 + i z0 ; ζ̇0 = vy0 + i vz0
  const Ar = p0.y,
    Ai = p0.z;
  // B = ζ̇0 + iω·ζ0  -> real = vy0 - ω·z0 ; imag = vz0 + ω·y0
  const Br = v0.y - omega * p0.z;
  const Bi = v0.z + omega * p0.y;
  // (A + B t)
  const Cr = Ar + Br * t;
  const Ci = Ai + Bi * t;
  // multiply by e^{-iωt} = cos(ωt) - i sin(ωt)
  const c = Math.cos(omega * t);
  const sgn = Math.sin(omega * t);
  const y = Cr * c + Ci * sgn; // (Cr + iCi)(c - i s) real
  const z = Ci * c - Cr * sgn; // imag
  return { x: p0.x + v0.x * t, y, z };
}

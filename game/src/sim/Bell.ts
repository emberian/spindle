// The bell: a 6-DOF rigid sphere with an internal mass-ring (a symmetric
// top). Deep-sim fidelity: full orientation + body-frame angular velocity,
// Euler's equation with the gyroscopic term, RK4. "Rings true" when spin is
// aligned with the ring axis; a clatter is an off-axis angular impulse.
//
// Body frame: ring axis = body +x̂. Inertia I = diag(Iaxial, It, It).
// A near-true spin sits on the principal axis so ω×(Iω) ≈ 0 and it *stays*
// true — physically self-consistent with the canon "rings when it spins
// true". Knock it off axis and gyroscopic precession makes it tumble.

import type { Vec3, Quat } from './vec';
import { qmul, qnorm, vnorm } from './vec';
import { accel } from './trajectory';

// Thin-ring inertia, normalised (absolute scale only sets torque response).
const I_AXIAL = 1.0; // about the ring axis (body x)
const I_TRANS = 0.5; // about a diameter (body y,z)
const SPIN_GOOD = 8; // rad/s where the chime is fully voiced
const SPIN_MIN = 1.5; // below this it can't ring at all

export interface BellBody {
  p: Vec3;
  v: Vec3;
  q: Quat; // orientation (body->world)
  w: Vec3; // angular velocity, BODY frame
}

// Rotate a body-frame vector into world using quaternion q.
function rotate(q: Quat, b: Vec3): Vec3 {
  // v' = q * (0,b) * q⁻¹  ; q is unit so q⁻¹ = conj
  const t = qmul(q, { x: b.x, y: b.y, z: b.z, w: 0 });
  const r = qmul(t, { x: -q.x, y: -q.y, z: -q.z, w: q.w });
  return { x: r.x, y: r.y, z: r.z };
}

// Euler's equation, body frame, free flight (τ_ext = 0):
//   I·ω̇ = −ω × (I·ω)
function angAccelBody(w: Vec3): Vec3 {
  const Iw = { x: I_AXIAL * w.x, y: I_TRANS * w.y, z: I_TRANS * w.z };
  const gyro = {
    x: w.y * Iw.z - w.z * Iw.y,
    y: w.z * Iw.x - w.x * Iw.z,
    z: w.x * Iw.y - w.y * Iw.x,
  };
  return { x: -gyro.x / I_AXIAL, y: -gyro.y / I_TRANS, z: -gyro.z / I_TRANS };
}

interface Deriv {
  dp: Vec3;
  dv: Vec3;
  dq: Quat;
  dw: Vec3;
}

function deriv(s: BellBody, omega: number): Deriv {
  const wWorld = rotate(s.q, s.w);
  const dqh = qmul({ x: wWorld.x, y: wWorld.y, z: wWorld.z, w: 0 }, s.q);
  return {
    dp: s.v,
    dv: accel(s.p, s.v, omega), // rotating-frame translational (sphere, no aero)
    dq: { x: 0.5 * dqh.x, y: 0.5 * dqh.y, z: 0.5 * dqh.z, w: 0.5 * dqh.w },
    dw: angAccelBody(s.w),
  };
}

function addB(s: BellBody, k: Deriv, f: number): BellBody {
  return {
    p: { x: s.p.x + k.dp.x * f, y: s.p.y + k.dp.y * f, z: s.p.z + k.dp.z * f },
    v: { x: s.v.x + k.dv.x * f, y: s.v.y + k.dv.y * f, z: s.v.z + k.dv.z * f },
    q: { x: s.q.x + k.dq.x * f, y: s.q.y + k.dq.y * f, z: s.q.z + k.dq.z * f, w: s.q.w + k.dq.w * f },
    w: { x: s.w.x + k.dw.x * f, y: s.w.y + k.dw.y * f, z: s.w.z + k.dw.z * f },
  };
}

export function stepBell(s: BellBody, omega: number, h: number): BellBody {
  const k1 = deriv(s, omega);
  const k2 = deriv(addB(s, k1, h / 2), omega);
  const k3 = deriv(addB(s, k2, h / 2), omega);
  const k4 = deriv(addB(s, k3, h), omega);
  const out: BellBody = {
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
    q: qnorm({
      x: s.q.x + (h / 6) * (k1.dq.x + 2 * k2.dq.x + 2 * k3.dq.x + k4.dq.x),
      y: s.q.y + (h / 6) * (k1.dq.y + 2 * k2.dq.y + 2 * k3.dq.y + k4.dq.y),
      z: s.q.z + (h / 6) * (k1.dq.z + 2 * k2.dq.z + 2 * k3.dq.z + k4.dq.z),
      w: s.q.w + (h / 6) * (k1.dq.w + 2 * k2.dq.w + 2 * k3.dq.w + k4.dq.w),
    }),
    w: {
      x: s.w.x + (h / 6) * (k1.dw.x + 2 * k2.dw.x + 2 * k3.dw.x + k4.dw.x),
      y: s.w.y + (h / 6) * (k1.dw.y + 2 * k2.dw.y + 2 * k3.dw.y + k4.dw.y),
      z: s.w.z + (h / 6) * (k1.dw.z + 2 * k2.dw.z + 2 * k3.dw.z + k4.dw.z),
    },
  };
  return out;
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Spin-trueness ∈ [0,1]: alignment of ω with the body ring axis. */
export function trueness(w: Vec3): number {
  const sp = Math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
  if (sp < 1e-6) return 0;
  const align = Math.abs(w.x) / sp; // ring axis = body x
  return Math.max(0, Math.min(1, align));
}

/** Audible chime ∈ [0,1]: trueness gated by enough spin to voice. */
export function chime(w: Vec3): number {
  const sp = Math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
  return trueness(w) * smoothstep(SPIN_MIN, SPIN_GOOD, sp);
}

/** A defender knocks it off true: add an off-axis body-frame angular impulse. */
export function clatter(s: BellBody, bodyImpulse: Vec3): void {
  s.w = {
    x: s.w.x + bodyImpulse.x / I_AXIAL,
    y: s.w.y + bodyImpulse.y / I_TRANS,
    z: s.w.z + bodyImpulse.z / I_TRANS,
  };
}

export { rotate as rotateBodyToWorld, vnorm };

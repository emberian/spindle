// Minimal deterministic Vec3 / Quat math. No three.js, no DOM — sim-pure.
// Operation order is fixed so results are bit-stable across machines.

export type Vec3 = { x: number; y: number; z: number };
export type Quat = { x: number; y: number; z: number; w: number };

export const v3 = (x = 0, y = 0, z = 0): Vec3 => ({ x, y, z });
export const vclone = (a: Vec3): Vec3 => ({ x: a.x, y: a.y, z: a.z });
export const vadd = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const vsub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const vscale = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });
export const vdot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const vcross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
export const vlen = (a: Vec3): number => Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
export const vnorm = (a: Vec3): Vec3 => {
  const l = vlen(a);
  return l < 1e-12 ? { x: 0, y: 0, z: 0 } : { x: a.x / l, y: a.y / l, z: a.z / l };
};

// Quaternion (x,y,z,w). qmul applies a then b (b ⊗ a).
export const qident = (): Quat => ({ x: 0, y: 0, z: 0, w: 1 });
export const qmul = (b: Quat, a: Quat): Quat => ({
  w: b.w * a.w - b.x * a.x - b.y * a.y - b.z * a.z,
  x: b.w * a.x + b.x * a.w + b.y * a.z - b.z * a.y,
  y: b.w * a.y - b.x * a.z + b.y * a.w + b.z * a.x,
  z: b.w * a.z + b.x * a.y - b.y * a.x + b.z * a.w,
});
export const qnorm = (q: Quat): Quat => {
  const l = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) || 1;
  return { x: q.x / l, y: q.y / l, z: q.z / l, w: q.w / l };
};
// dq/dt = 0.5 * (0, ωworld) ⊗ q  — used by the rigid-body integrator (P1).
export const qderiv = (q: Quat, wWorld: Vec3): Quat => {
  const h = qmul({ x: wWorld.x, y: wWorld.y, z: wWorld.z, w: 0 }, q);
  return { x: 0.5 * h.x, y: 0.5 * h.y, z: 0.5 * h.z, w: 0.5 * h.w };
};

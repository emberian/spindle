//! Deterministic f64 vector math (mirror of the TS sim/vec.ts).

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn add(self, b: Vec3) -> Vec3 {
        Vec3::new(self.x + b.x, self.y + b.y, self.z + b.z)
    }
    pub fn sub(self, b: Vec3) -> Vec3 {
        Vec3::new(self.x - b.x, self.y - b.y, self.z - b.z)
    }
    pub fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn dot(self, b: Vec3) -> f64 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }
    pub fn len(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn norm(self) -> Vec3 {
        let l = self.len();
        if l < 1e-12 {
            Vec3::new(0.0, 0.0, 0.0)
        } else {
            Vec3::new(self.x / l, self.y / l, self.z / l)
        }
    }
    pub fn cross(self, b: Vec3) -> Vec3 {
        Vec3::new(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x,
        )
    }
}

/// Quaternion (x,y,z,w). Mirrors TS sim/vec.ts conventions exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Quat {
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }
    pub const fn ident() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }
    /// qmul(b, a) = b ⊗ a  (apply a then b) — identical to TS `qmul`.
    pub fn mul(b: Quat, a: Quat) -> Quat {
        Quat {
            w: b.w * a.w - b.x * a.x - b.y * a.y - b.z * a.z,
            x: b.w * a.x + b.x * a.w + b.y * a.z - b.z * a.y,
            y: b.w * a.y - b.x * a.z + b.y * a.w + b.z * a.x,
            z: b.w * a.z + b.x * a.y - b.y * a.x + b.z * a.w,
        }
    }
    pub fn norm(self) -> Quat {
        let l = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        let l = if l == 0.0 { 1.0 } else { l };
        Quat::new(self.x / l, self.y / l, self.z / l, self.w / l)
    }
    /// dq/dt = 0.5 · (0, ωworld) ⊗ q
    pub fn deriv(self, w_world: Vec3) -> Quat {
        let h = Quat::mul(Quat::new(w_world.x, w_world.y, w_world.z, 0.0), self);
        Quat::new(0.5 * h.x, 0.5 * h.y, 0.5 * h.z, 0.5 * h.w)
    }
    /// Rotate a body-frame vector into world: q ⊗ (0,b) ⊗ q⁻¹ (q unit).
    pub fn rotate(self, b: Vec3) -> Vec3 {
        let t = Quat::mul(self, Quat::new(b.x, b.y, b.z, 0.0));
        let r = Quat::mul(t, Quat::new(-self.x, -self.y, -self.z, self.w));
        Vec3::new(r.x, r.y, r.z)
    }
}

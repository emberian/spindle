//! Bit-exact port of src/ai/index.ts `makeRng` / `sfc32` / `splitmix32`.
//!
//! IMPORTANT: this is the AI's single-draw sfc32 (`(t>>>0) / 4294967296`),
//! deliberately NOT crate::rng::Substream (which is a 53-bit two-u32-draw
//! uniform for the sim/replay path). The AI's determinism contract is
//! "same (seed, tick, team_index) → identical f64 stream", so this must
//! reproduce the TS arithmetic exactly: u32-wrapping ops, `Math.imul` =
//! u32 wrapping_mul, `>>>` = logical shift, `>>>0` = `as u32`.

/// splitmix32 — verbatim port (index.ts:35-42).
fn splitmix32(seed: u32) -> u32 {
    let mut s = if seed == 0 { 1 } else { seed };
    s = s.wrapping_add(0x9e37_79b9);
    let mut t = s;
    t = (t ^ (t >> 16)).wrapping_mul(0x21f0_aaad);
    t = (t ^ (t >> 15)).wrapping_mul(0x735a_2d97);
    t ^ (t >> 15)
}

/// sfc32 generator state — `next()` is index.ts:24-32 verbatim.
#[derive(Clone, Debug)]
pub struct AiRng {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

impl AiRng {
    /// `makeRng(seed, tick, team_index)` (index.ts:44-55): mix → 4×
    /// splitmix32 words → sfc32 → discard 16 (the rng.ts Substream warm-up).
    ///
    /// `tick` and `team_index` are JS `number`s coerced by `>>> 0`
    /// (ToUint32). For the integer domain used here (tick < 2^32,
    /// team_index small) JS float `*` then `>>>0` equals u32 wrapping_mul.
    pub fn make(seed: u32, tick: u32, team_index: u32) -> Self {
        let mixed = seed
            ^ tick.wrapping_mul(0x9e37_79b9)
            ^ team_index.wrapping_mul(0x6c62_272e);
        let a = splitmix32(mixed);
        let b = splitmix32(a);
        let c = splitmix32(b);
        let d = splitmix32(c);
        let mut g = AiRng { a, b, c, d };
        for _ in 0..16 {
            g.next();
        }
        g
    }

    /// Uniform [0,1) — single u32 draw, exactly `(t>>>0) / 4294967296`.
    pub fn next(&mut self) -> f64 {
        let t = self.a.wrapping_add(self.b).wrapping_add(self.d);
        self.d = self.d.wrapping_add(1);
        self.a = self.b ^ (self.b >> 9);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = (self.c << 21) | (self.c >> 11);
        self.c = self.c.wrapping_add(t);
        (t as f64) / 4294967296.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let mut x = AiRng::make(12345, 240, 7);
        let mut y = AiRng::make(12345, 240, 7);
        for _ in 0..64 {
            assert_eq!(x.next().to_bits(), y.next().to_bits());
        }
    }

    #[test]
    fn unit_range_and_decorrelated() {
        let mut a = AiRng::make(999, 1, 0);
        let mut b = AiRng::make(999, 2, 0);
        let mut diff = false;
        for _ in 0..32 {
            let va = a.next();
            let vb = b.next();
            assert!((0.0..1.0).contains(&va), "out of range: {va}");
            assert!((0.0..1.0).contains(&vb));
            if va != vb {
                diff = true;
            }
        }
        assert!(diff, "different ticks must decorrelate");
    }
}

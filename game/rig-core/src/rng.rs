//! sfc32 + splitmix32 seeding — a BIT-EXACT port of the TS sim/rng.ts so the
//! Rust core and the TS oracle produce identical streams (replay parity).
//! All ops are u32 wrapping, mirroring JS `Math.imul` / `>>> 0`.

use std::collections::HashMap;

fn splitmix32_next(a: &mut u32) -> u32 {
    *a = a.wrapping_add(0x9e37_79b9);
    let mut t = *a;
    t = (t ^ (t >> 16)).wrapping_mul(0x21f0_aaad);
    t = (t ^ (t >> 15)).wrapping_mul(0x735a_2d97);
    t ^ (t >> 15)
}

fn name_hash(name: &str) -> u32 {
    let mut h: u32 = 2166136261;
    for b in name.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

fn hash2(a: u32, b: u32) -> u32 {
    let mut h = a ^ 0x9e37_79b9;
    h = (h ^ (h >> 16)).wrapping_mul(0x85eb_ca6b);
    h = h.wrapping_add(b);
    h = (h ^ (h >> 13)).wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

#[derive(Clone)]
pub struct Substream {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    pub draws: u64,
}

impl Substream {
    pub fn new(seed: u32) -> Self {
        let mut sm = if seed == 0 { 1 } else { seed };
        let a = splitmix32_next(&mut sm);
        let b = splitmix32_next(&mut sm);
        let c = splitmix32_next(&mut sm);
        let d = splitmix32_next(&mut sm);
        let mut s = Self { a, b, c, d, draws: 0 };
        for _ in 0..16 {
            s.u32();
        }
        s
    }

    fn u32(&mut self) -> u32 {
        let t = self.a.wrapping_add(self.b).wrapping_add(self.d);
        self.d = self.d.wrapping_add(1);
        self.a = self.b ^ (self.b >> 9);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = (self.c << 21) | (self.c >> 11);
        self.c = self.c.wrapping_add(t);
        t
    }

    /// Uniform [0,1) — identical f64 to the TS impl (53-bit, two u32 draws).
    pub fn next(&mut self) -> f64 {
        self.draws += 1;
        let hi = (self.u32() >> 5) as f64; // 27 bits
        let lo = (self.u32() >> 6) as f64; // 26 bits
        (hi * 67108864.0 + lo) / 9007199254740992.0
    }
    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next() * (max - min)
    }
    pub fn int(&mut self, min: i64, max: i64) -> i64 {
        (self.range(min as f64, max as f64)).floor() as i64
    }
    pub fn chance(&mut self, p: f64) -> bool {
        self.next() < p
    }
}

pub struct Rng {
    seed: u32,
    streams: HashMap<String, Substream>,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self {
            seed,
            streams: HashMap::new(),
        }
    }
    pub fn sub(&mut self, name: &str) -> &mut Substream {
        let seed = self.seed;
        self.streams
            .entry(name.to_string())
            .or_insert_with(|| Substream::new(hash2(seed, name_hash(name))))
    }
    pub fn cursor(&self) -> Vec<(String, u64)> {
        let mut v: Vec<_> = self
            .streams
            .iter()
            .map(|(k, s)| (k.clone(), s.draws))
            .collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_ts_oracle_known_answers() {
        // Captured from the TS impl: new Rng(12345).sub('contest').next() x5
        let expected = [
            0.96806102826403,
            0.14561917944373737,
            0.9415351546655758,
            0.03234286412563092,
            0.26646868810857627,
        ];
        let mut r = Rng::new(12345);
        let s = r.sub("contest");
        for e in expected {
            let v = s.next();
            assert!((v - e).abs() < 1e-15, "got {v}, want {e}");
        }
    }

    #[test]
    fn uniform_mean() {
        let mut r = Rng::new(7);
        let s = r.sub("aiJitter");
        let mut sum = 0.0;
        for _ in 0..100_000 {
            sum += s.next();
        }
        let mean = sum / 100_000.0;
        assert!((mean - 0.498694).abs() < 1e-6, "mean {mean}");
    }

    #[test]
    fn substreams_decorrelate() {
        let mut r = Rng::new(99);
        let u: Vec<f64> = (0..1000).map(|_| r.sub("a").next()).collect();
        let v: Vec<f64> = (0..1000).map(|_| r.sub("b").next()).collect();
        let same = u.iter().zip(&v).filter(|(x, y)| (*x - *y).abs() < 1e-12).count();
        assert_eq!(same, 0);
    }
}

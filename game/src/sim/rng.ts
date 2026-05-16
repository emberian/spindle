// Seeded RNG with NAMED SUBSTREAMS. No global instance (a determinism
// hazard). The cursor is serialisable so a replay re-seeds exactly.
//
// Generator: sfc32 (Small Fast Counting, 32-bit) — passes PractRand to
// multi-TB, ~2^128 state, far better equidistribution/period than an LCG.
// Seeded via splitmix32 so even nearby seeds decorrelate fully. All math is
// 32-bit (Math.imul / >>>0) so results are bit-identical across machines.

function splitmix32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x9e3779b9) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 16), 0x21f0aaad) >>> 0;
    t = Math.imul(t ^ (t >>> 15), 0x735a2d97) >>> 0;
    return (t ^ (t >>> 15)) >>> 0;
  };
}

function nameHash(name: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

export class Substream {
  private a: number;
  private b: number;
  private c: number;
  private d: number;
  public draws = 0;

  constructor(seed: number) {
    // Mix the seed into four 32-bit words of sfc32 state.
    const sm = splitmix32((seed >>> 0) || 1);
    this.a = sm();
    this.b = sm();
    this.c = sm();
    this.d = sm();
    // Warm up to wash out seeding structure.
    for (let i = 0; i < 16; i++) this.u32();
  }

  private u32(): number {
    // sfc32 core.
    const t = (((this.a + this.b) >>> 0) + this.d) >>> 0;
    this.d = (this.d + 1) >>> 0;
    this.a = (this.b ^ (this.b >>> 9)) >>> 0;
    this.b = (this.c + (this.c << 3)) >>> 0;
    this.c = ((this.c << 21) | (this.c >>> 11)) >>> 0;
    this.c = (this.c + t) >>> 0;
    return t >>> 0;
  }

  /** Uniform float in [0,1). 53-bit mantissa from two 32-bit draws. */
  next(): number {
    this.draws++;
    const hi = this.u32() >>> 5; // 27 bits
    const lo = this.u32() >>> 6; // 26 bits
    return (hi * 67108864 + lo) / 9007199254740992; // / 2^53
  }
  range(min: number, max: number): number {
    return min + this.next() * (max - min);
  }
  int(min: number, max: number): number {
    return Math.floor(this.range(min, max));
  }
  chance(p: number): boolean {
    return this.next() < p;
  }
}

function hash2(a: number, b: number): number {
  let h = (a ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
  h = (h + (b >>> 0)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

export class Rng {
  private streams = new Map<string, Substream>();
  constructor(public readonly seed: number) {}
  /** Get (or lazily create) a named substream, deterministically seeded. */
  sub(name: string): Substream {
    let s = this.streams.get(name);
    if (!s) {
      s = new Substream(hash2(this.seed, nameHash(name)));
      this.streams.set(name, s);
    }
    return s;
  }
  /** Serialisable cursor (substream -> draw count) for replay re-seeding. */
  cursor(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [k, v] of this.streams) out[k] = v.draws;
    return out;
  }
}

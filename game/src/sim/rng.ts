// Seeded RNG with NAMED SUBSTREAMS. There is deliberately no global instance
// (frisqueendom's global `Random` is a determinism hazard we do not port).
// Each consumer takes an explicit substream; the draw cursor is serialisable
// so a replay re-seeds to the exact state.

function hash2(a: number, b: number): number {
  // splitmix-ish 32-bit mix of (seed, name-hash)
  let h = (a ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
  h = (h + (b >>> 0)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
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
  private state: number;
  public draws = 0;
  constructor(seed: number) {
    this.state = (Math.abs(seed | 0) || 1) >>> 0;
  }
  next(): number {
    // LCG (Numerical Recipes constants); deterministic across machines.
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    this.draws++;
    return this.state / 4294967296;
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

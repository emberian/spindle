// BellPulse — time-varying modulation for the bell mesh so you can SEE it ring.
//
// When chime is high (ringing true) the bell breathes: gentle sine oscillation
// on scale and emissive, like a steady heartbeat. When chime drops (clattering)
// the modulation becomes noisy and irregular — the bell looks sick/unstable.
//
// Outputs three values each frame:
//   scaleMod     — multiply into bellMesh.scale (centered at 1.0, ±3-5%)
//   emissiveMod  — multiply into emissiveIntensity (centered at 1.0, ±10-20%)
//   trailWidthMod— multiply into trail HALF_W (centered at 1.0, ±5-8%)

export class BellPulse {
  // Internal phase accumulators (continuous, never reset)
  private phase = 0;
  private noisePhase = 0;
  private lastChime = 1;

  // Cheap deterministic noise state (xorshift32)
  private noiseSeed = 0x1337cafe;

  /** Call once per frame. dt in seconds, chime in [0,1], spinRate in rad/s. */
  update(dt: number, chime: number, _spinRate: number): void {
    this.lastChime = chime;

    // Primary oscillation rate: when ringing true, pulse at ~2.5 Hz (calm breath).
    // When clattering, the base rate jitters faster (~6 Hz).
    const trueRate = 2.5;
    const clatterRate = 6.0;
    const rate = trueRate + (clatterRate - trueRate) * (1 - chime);

    this.phase += dt * rate * Math.PI * 2;
    // Keep phase bounded to avoid floating-point drift over long sessions
    if (this.phase > 1e5) this.phase -= 1e5;

    // Noise phase advances at a different rate for variety
    this.noisePhase += dt * 13.7;
    if (this.noisePhase > 1e5) this.noisePhase -= 1e5;
  }

  /** Scale multiplier for the bell mesh (centered at 1.0). */
  get scaleMod(): number {
    const chime = this.lastChime;
    const clean = Math.sin(this.phase);
    const noisy = clean * 0.5 + this.noise1() * 0.5;

    // 3% amplitude when true, 5% when clattering
    const amplitude = 0.03 + 0.02 * (1 - chime);
    const signal = chime * clean + (1 - chime) * noisy;
    return 1.0 + signal * amplitude;
  }

  /** Emissive intensity multiplier (centered at 1.0). */
  get emissiveMod(): number {
    const chime = this.lastChime;
    const clean = Math.sin(this.phase);
    const flicker = this.noise2() * 0.7 + Math.sin(this.phase * 2.3) * 0.3;

    // 12% amplitude when true, 20% when clattering
    const amplitude = 0.12 + 0.08 * (1 - chime);
    const signal = chime * clean + (1 - chime) * flicker;
    return 1.0 + signal * amplitude;
  }

  /** Trail width multiplier (centered at 1.0). */
  get trailWidthMod(): number {
    const chime = this.lastChime;
    // Phase-shifted so it doesn't mirror scale exactly
    const clean = Math.sin(this.phase + 1.2);
    const jittery = clean * 0.4 + this.noise3() * 0.6;

    // 5% amplitude when true, 8% when clattering
    const amplitude = 0.05 + 0.03 * (1 - chime);
    const signal = chime * clean + (1 - chime) * jittery;
    return 1.0 + signal * amplitude;
  }

  // ── noise sources ──────────────────────────────────────────────────────────

  private xorshift(): number {
    let x = this.noiseSeed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.noiseSeed = x;
    return (x & 0x7fffffff) / 0x3fffffff - 1; // [-1, 1]
  }

  private noise1(): number {
    return Math.sin(this.noisePhase * 3.17) * 0.6 + this.xorshift() * 0.4;
  }

  private noise2(): number {
    return Math.sin(this.noisePhase * 5.43 + 2.1) * 0.5 + this.xorshift() * 0.5;
  }

  private noise3(): number {
    return Math.sin(this.noisePhase * 4.01 + 0.7) * 0.7 + this.xorshift() * 0.3;
  }
}

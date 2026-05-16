// Synth-only audio (no asset files). The chime IS a game system: its timbre
// purity tracks the bell's spin-trueness. Ambience hum + crowd that HUSHES
// on a loop/contest and ERUPTS on a score (canon: 40k go silent for a Loop).
// Must be created on a user gesture (autoplay policy); headless = silent
// (constructor guarded), which is fine for CI screenshots.

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private crowd!: GainNode;
  private hum!: GainNode;
  // chime voices
  private fund!: OscillatorNode;
  private partial!: OscillatorNode;
  private noise!: AudioBufferSourceNode;
  private chimeGain!: GainNode;
  private noiseGain!: GainNode;
  private partialGain!: GainNode;
  private pan!: StereoPannerNode;
  ok = false;

  start(): void {
    if (this.ok) return;
    try {
      this.ctx = new (window.AudioContext || (window as never as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    } catch {
      return;
    }
    const c = this.ctx;
    this.master = c.createGain();
    this.master.gain.value = 0.5;
    this.master.connect(c.destination);

    // Cylinder hum: two low detuned saws through a lowpass.
    this.hum = c.createGain();
    this.hum.gain.value = 0.12;
    const lp = c.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 160;
    for (const f of [42, 43.7]) {
      const o = c.createOscillator();
      o.type = 'sawtooth';
      o.frequency.value = f;
      o.connect(lp);
      o.start();
    }
    lp.connect(this.hum);
    this.hum.connect(this.master);

    // Crowd bed: filtered noise.
    this.crowd = c.createGain();
    this.crowd.gain.value = 0.06;
    const nb = c.createBuffer(1, c.sampleRate * 2, c.sampleRate);
    const nd = nb.getChannelData(0);
    for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
    const cn = c.createBufferSource();
    cn.buffer = nb;
    cn.loop = true;
    const bp = c.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 700;
    bp.Q.value = 0.7;
    cn.connect(bp);
    bp.connect(this.crowd);
    this.crowd.connect(this.master);
    cn.start();

    // Chime: fundamental + one partial + noise (purity = trueness).
    this.pan = c.createStereoPanner();
    this.pan.connect(this.master);
    this.chimeGain = c.createGain();
    this.chimeGain.gain.value = 0;
    this.chimeGain.connect(this.pan);
    this.partialGain = c.createGain();
    this.partialGain.gain.value = 0;
    this.partialGain.connect(this.pan);
    this.noiseGain = c.createGain();
    this.noiseGain.gain.value = 0;
    this.noiseGain.connect(this.pan);

    this.fund = c.createOscillator();
    this.fund.type = 'sine';
    this.fund.frequency.value = 520;
    this.fund.connect(this.chimeGain);
    this.fund.start();
    this.partial = c.createOscillator();
    this.partial.type = 'triangle';
    this.partial.frequency.value = 520 * 2.76; // inharmonic → "metallic" when loud
    this.partial.connect(this.partialGain);
    this.partial.start();
    this.noise = c.createBufferSource();
    this.noise.buffer = nb;
    this.noise.loop = true;
    this.noise.connect(this.noiseGain);
    this.noise.start();

    this.ok = true;
  }

  /** Per-frame: voice the chime from trueness, position by screen pan. */
  setBell(chime: number, spinRate: number, pan: number, inFlight: boolean): void {
    if (!this.ok || !this.ctx) return;
    const t = this.ctx.currentTime;
    const audible = inFlight ? 1 : 0.25;
    this.pan.pan.setTargetAtTime(Math.max(-1, Math.min(1, pan)), t, 0.05);
    this.fund.frequency.setTargetAtTime(360 + spinRate * 9, t, 0.05);
    // pure when true; partial+noise rise as it clatters
    this.chimeGain.gain.setTargetAtTime(0.18 * chime * audible, t, 0.04);
    this.partialGain.gain.setTargetAtTime(0.10 * (1 - chime) * audible, t, 0.04);
    this.noiseGain.gain.setTargetAtTime(0.05 * (1 - chime) * (1 - chime) * audible, t, 0.04);
  }

  /** hush ∈ [0,1]: duck the crowd (loop/contest). */
  setHush(hush: number): void {
    if (!this.ok || !this.ctx) return;
    this.crowd.gain.setTargetAtTime(0.06 * (1 - 0.92 * hush), this.ctx.currentTime, 0.12);
  }

  erupt(big: number): void {
    if (!this.ok || !this.ctx) return;
    const t = this.ctx.currentTime;
    this.crowd.gain.cancelScheduledValues(t);
    this.crowd.gain.setValueAtTime(0.06, t);
    this.crowd.gain.linearRampToValueAtTime(0.06 + 0.5 * big, t + 0.15);
    this.crowd.gain.exponentialRampToValueAtTime(0.06, t + 2.2);
  }
}

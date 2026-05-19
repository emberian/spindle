// AudioEngine — soulful synth soundscape for zero-g cylinder sport.
// The bell IS a game system: riggers READ the sport by ear.
// "On a Loop, stadiums of forty thousand go completely silent to hear
//  if the bell is still ringing." Pure WebAudio — NO asset files.
// Init only on user gesture (autoplay); headless/no-device = silent, no throw.
//
// ─── PUBLIC API ──────────────────────────────────────────────────────────────
//   start(): void
//     Call once on first user gesture. Safe to call again (idempotent).
//
//   setBell(chime, spinRate, pan, inFlight): void
//     Call every frame.
//     chime     0..1  – 1 = spinning true (pure bell); 0 = tumbling (clatter)
//     spinRate  ≥0    – angular magnitude, modulates pitch (subtle)
//     pan      -1..1  – screen position
//     inFlight  bool  – bell louder in flight, quieter when held
//
//   setHush(hush): void
//     hush 0..1 – ducks crowd toward silence as a loop develops.
//     At hush=1 only the chime survives; crowd is near-silent.
//
//   event(name): void
//     One-shot sound events. Supported names:
//       'catch'          – soft catching thunk (ball landing in grip)
//       'grapple'        – heavier body grapple thud
//       'release'        – sharp click, ball leaving hand
//       'throw'          – rising whoosh, launch burst
//       'clatter'        – sharp detuned metallic hit (bell struck badly)
//       'score_fall'     – score for falling-ring: solid satisfying clunk
//       'score_rise'     – score for rising-ring: big bright burst + crowd
//       'score_loop'     – LOOP score: beat of silence → massive crowd eruption
//       'score_ground'   – score on ground-gate: wry low plop
//       'turnover'       – possession change: downward sting
//       'loop_building'  – held-breath tension riser (loop about to close)
// ─────────────────────────────────────────────────────────────────────────────

export class AudioEngine {
  private ctx: AudioContext | null = null;

  // Bus structure
  private master!: GainNode;
  private crowdBus!: GainNode;      // crowd ambience
  private humBus!: GainNode;        // cylinder hum
  private chimeBus!: GainNode;      // spatialised bell chain
  private oneShotBus!: GainNode;    // event sounds

  // Bell voice nodes
  private bellPan!: StereoPannerNode;
  private fundOsc!: OscillatorNode;             // clean fundamental
  private harm2Osc!: OscillatorNode;            // 2nd harmonic (true)
  private harm3Osc!: OscillatorNode;            // gentle 3rd
  private inharmonicOsc!: OscillatorNode;       // detuned metallic partial (clatter)
  private inharmonic2Osc!: OscillatorNode;      // second detuned partial
  private noiseSource!: AudioBufferSourceNode;  // clatter noise
  private fundGain!: GainNode;
  private harm2Gain!: GainNode;
  private harm3Gain!: GainNode;
  private inharmonicGain!: GainNode;
  private inharmonic2Gain!: GainNode;
  private noiseGain!: GainNode;
  private flutterLfo!: OscillatorNode;          // amplitude flutter when clattering
  private flutterGain!: GainNode;               // depth of flutter modulation
  private flutterDepth!: GainNode;              // controlled by chime

  // Crowd — spatial ring (6 sections around the cylinder rim)
  private crowdGain!: GainNode;                 // master inner gain (setHush targets this)
  private crowdSections!: {
    source: AudioBufferSourceNode;
    panner: PannerNode;
    gain: GainNode;
    angle: number;               // current azimuth radians
    baseAngle: number;           // initial azimuth radians
    lfoRate: number;             // per-section LFO frequency
  }[];

  // Shared noise buffer (reused for crowd + bell noise)
  private noiseBuffer!: AudioBuffer;

  // Crowd rotation
  private crowdRotationId = 0;   // rAF id for rotation tick

  // Hush baseline
  private readonly CROWD_BASE = 0.07;
  private readonly CROWD_SECTIONS = 6;
  private readonly CROWD_RING_RADIUS = 10;   // units from listener
  private readonly CROWD_RING_Y = 4;         // elevation (above = rim)
  private readonly CROWD_ROTATION_PERIOD = 30; // seconds per revolution

  ok = false;

  // ── start ────────────────────────────────────────────────────────────────

  start(): void {
    if (this.ok) return;
    try {
      const AC = window.AudioContext
        || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new AC();
    } catch {
      return; // headless / permission denied — silent
    }
    const c = this.ctx;

    // ── Master bus (soft limiter via waveshaper) ──────────────────────────
    this.master = c.createGain();
    this.master.gain.value = 0.72;
    const limiter = this.makeLimiter(c);
    this.master.connect(limiter);
    limiter.connect(c.destination);

    // ── Build shared noise buffer (2 s white) ────────────────────────────
    this.noiseBuffer = this.makeNoiseBuffer(c, 2.0);

    // ── Cylinder hum bed ─────────────────────────────────────────────────
    //    Two detuned saws deep in the low end, heavy lowpass — the resonant
    //    body of the spinning cylinder felt more than heard.
    this.humBus = c.createGain();
    this.humBus.gain.value = 0.14;
    const humLp = c.createBiquadFilter();
    humLp.type = 'lowpass';
    humLp.frequency.value = 130;
    humLp.Q.value = 0.5;
    const humLp2 = c.createBiquadFilter();
    humLp2.type = 'lowpass';
    humLp2.frequency.value = 220;
    for (const [f, det] of [[41, 0], [41.8, 0], [82, 0]] as [number, number][]) {
      const o = c.createOscillator();
      o.type = 'sawtooth';
      o.frequency.value = f + det;
      o.connect(humLp);
      o.start();
    }
    // very subtle high mid shimmer — air inside the cylinder
    const shimmerOsc = c.createOscillator();
    shimmerOsc.type = 'sine';
    shimmerOsc.frequency.value = 312;
    const shimmerGain = c.createGain();
    shimmerGain.gain.value = 0.012;
    shimmerOsc.connect(shimmerGain);
    shimmerGain.connect(humLp2);
    shimmerOsc.start();
    humLp.connect(this.humBus);
    humLp2.connect(this.humBus);
    this.humBus.connect(this.master);

    // ── Crowd bed — spatial ring of 6 HRTF sections ────────────────────
    //    Each section is an independent noise source → two bandpass filters
    //    → per-section gain → PannerNode (HRTF) → crowdGain → crowdBus.
    //    Positioned in a ring above the listener (cylinder rim audience).
    this.crowdBus = c.createGain();
    this.crowdBus.gain.value = 1.0;

    this.crowdGain = c.createGain();
    this.crowdGain.gain.value = this.CROWD_BASE;

    // Set up AudioListener at origin, facing forward (Z-negative)
    const listener = c.listener;
    if (listener.positionX) {
      // Modern API (AudioParam)
      listener.positionX.value = 0;
      listener.positionY.value = 0;
      listener.positionZ.value = 0;
      listener.forwardX.value = 0;
      listener.forwardY.value = 0;
      listener.forwardZ.value = -1;
      listener.upX.value = 0;
      listener.upY.value = 1;
      listener.upZ.value = 0;
    } else {
      // Legacy API
      listener.setPosition(0, 0, 0);
      listener.setOrientation(0, 0, -1, 0, 1, 0);
    }

    // Per-section bandpass frequency offsets to create organic variation
    const bp1Freqs = [530, 560, 590, 610, 640, 570];
    const bp2Freqs = [1150, 1180, 1220, 1260, 1300, 1170];
    const lfoRates = [0.06, 0.073, 0.055, 0.082, 0.065, 0.078];

    // Per-section gain: divide total level among 6 sections
    const perSectionGain = 1.0 / this.CROWD_SECTIONS;

    this.crowdSections = [];

    for (let i = 0; i < this.CROWD_SECTIONS; i++) {
      const angle = (i / this.CROWD_SECTIONS) * Math.PI * 2;

      // Noise source with phase offset (start at different point in buffer)
      const src = c.createBufferSource();
      src.buffer = this.noiseBuffer;
      src.loop = true;

      // Two bandpass filters with per-section variation
      const bp1 = c.createBiquadFilter();
      bp1.type = 'bandpass';
      bp1.frequency.value = bp1Freqs[i];
      bp1.Q.value = 0.6;

      const bp2 = c.createBiquadFilter();
      bp2.type = 'bandpass';
      bp2.frequency.value = bp2Freqs[i];
      bp2.Q.value = 0.5;

      // Per-section gain node
      const sGain = c.createGain();
      sGain.gain.value = perSectionGain;

      // Per-section LFO for organic life
      const lfo = c.createOscillator();
      lfo.frequency.value = lfoRates[i];
      const lfoGain = c.createGain();
      lfoGain.gain.value = 0.002; // subtle per-section modulation
      lfo.connect(lfoGain);
      lfoGain.connect(sGain.gain);
      lfo.start();

      // HRTF PannerNode
      const panner = c.createPanner();
      panner.panningModel = 'HRTF';
      panner.distanceModel = 'inverse';
      panner.refDistance = 5;
      panner.maxDistance = 50;
      panner.rolloffFactor = 1;
      panner.coneInnerAngle = 360;
      panner.coneOuterAngle = 360;

      // Position in ring
      const x = Math.cos(angle) * this.CROWD_RING_RADIUS;
      const z = Math.sin(angle) * this.CROWD_RING_RADIUS;
      panner.positionX.value = x;
      panner.positionY.value = this.CROWD_RING_Y;
      panner.positionZ.value = z;

      // Wire: source → bp1/bp2 → sGain → panner → crowdGain
      src.connect(bp1);
      src.connect(bp2);
      bp1.connect(sGain);
      bp2.connect(sGain);
      sGain.connect(panner);
      panner.connect(this.crowdGain);

      // Start with phase offset so each section is de-correlated
      const offset = (i / this.CROWD_SECTIONS) * this.noiseBuffer.duration;
      src.start(0, offset);

      this.crowdSections.push({
        source: src,
        panner,
        gain: sGain,
        angle,
        baseAngle: angle,
        lfoRate: lfoRates[i],
      });
    }

    this.crowdGain.connect(this.crowdBus);
    this.crowdBus.connect(this.master);

    // Start subtle crowd rotation (cylinder spin hint)
    this.startCrowdRotation();

    // ── Chime voice ───────────────────────────────────────────────────────
    //    A tuned bell voice. Purity encodes spin-trueness.
    //    Pure: clean fundamental + gentle 2nd + 3rd harmonics.
    //    Clattering: inharmonic partials (metallic), noise, amplitude flutter.
    //
    //    Bell pitch: ~A4 area but lower/darker = 440 Hz base, nudged by spinRate.
    //    The partials are tuned to real bell ratios (not integer).
    //
    //    Chain: oscs → individual gains → bellPan → chimeBus → master
    //           flutter LFO modulates the inharmonic gains (not fundamental)
    //
    this.bellPan = c.createStereoPanner();
    this.chimeBus = c.createGain();
    this.chimeBus.gain.value = 1.0;
    this.bellPan.connect(this.chimeBus);
    this.chimeBus.connect(this.master);

    // Fundamental — pure sine
    this.fundGain   = c.createGain(); this.fundGain.gain.value = 0;
    this.fundOsc    = c.createOscillator();
    this.fundOsc.type = 'sine';
    this.fundOsc.frequency.value = 440;
    this.fundOsc.connect(this.fundGain);
    this.fundGain.connect(this.bellPan);
    this.fundOsc.start();

    // 2nd harmonic (bell ratio ≈ 2.0 × fund, slightly sharp for shimmer)
    this.harm2Gain  = c.createGain(); this.harm2Gain.gain.value = 0;
    this.harm2Osc   = c.createOscillator();
    this.harm2Osc.type = 'sine';
    this.harm2Osc.frequency.value = 440 * 2.02;
    this.harm2Osc.connect(this.harm2Gain);
    this.harm2Gain.connect(this.bellPan);
    this.harm2Osc.start();

    // 3rd harmonic (bell ratio ≈ 3.0, soft)
    this.harm3Gain  = c.createGain(); this.harm3Gain.gain.value = 0;
    this.harm3Osc   = c.createOscillator();
    this.harm3Osc.type = 'sine';
    this.harm3Osc.frequency.value = 440 * 3.01;
    this.harm3Osc.connect(this.harm3Gain);
    this.harm3Gain.connect(this.bellPan);
    this.harm3Osc.start();

    // Inharmonic partial 1 — classic metallic bell clatter ratio ~2.756
    this.inharmonicGain  = c.createGain(); this.inharmonicGain.gain.value = 0;
    this.inharmonicOsc   = c.createOscillator();
    this.inharmonicOsc.type = 'triangle';
    this.inharmonicOsc.frequency.value = 440 * 2.756;
    this.inharmonicOsc.connect(this.inharmonicGain);
    this.inharmonicGain.connect(this.bellPan);
    this.inharmonicOsc.start();

    // Inharmonic partial 2 — beating against partial 1 creates roughness
    this.inharmonic2Gain  = c.createGain(); this.inharmonic2Gain.gain.value = 0;
    this.inharmonic2Osc   = c.createOscillator();
    this.inharmonic2Osc.type = 'triangle';
    this.inharmonic2Osc.frequency.value = 440 * 2.891; // slightly detuned from 2.756 → beating
    this.inharmonic2Osc.connect(this.inharmonic2Gain);
    this.inharmonic2Gain.connect(this.bellPan);
    this.inharmonic2Osc.start();

    // Noise component through bandpass — grain of the clatter
    this.noiseGain = c.createGain(); this.noiseGain.gain.value = 0;
    this.noiseSource = c.createBufferSource();
    this.noiseSource.buffer = this.noiseBuffer;
    this.noiseSource.loop = true;
    const noiseBp = c.createBiquadFilter();
    noiseBp.type = 'bandpass';
    noiseBp.frequency.value = 1800;
    noiseBp.Q.value = 1.2;
    this.noiseSource.connect(noiseBp);
    noiseBp.connect(this.noiseGain);
    this.noiseGain.connect(this.bellPan);
    this.noiseSource.start();

    // Amplitude flutter LFO for clattering — modulates inharmonic+noise depth
    // The flutter itself is a slow irregular modulation made from two LFOs.
    this.flutterLfo    = c.createOscillator();
    this.flutterLfo.frequency.value = 7.3;  // Hz — audible flutter speed
    this.flutterDepth  = c.createGain();    // controlled by (1-chime)²
    this.flutterDepth.gain.value = 0;
    this.flutterGain   = c.createGain();
    this.flutterGain.gain.value = 0.04;     // absolute flutter amplitude
    this.flutterLfo.connect(this.flutterGain);
    this.flutterGain.connect(this.flutterDepth);
    // flutter modulates the inharmonic gain
    this.flutterDepth.connect(this.inharmonicGain.gain);
    this.flutterDepth.connect(this.noiseGain.gain);
    this.flutterLfo.start();

    // ── One-shot bus ──────────────────────────────────────────────────────
    this.oneShotBus = c.createGain();
    this.oneShotBus.gain.value = 1.0;
    this.oneShotBus.connect(this.master);

    this.ok = true;
  }

  // ── stop (cleanup) ────────────────────────────────────────────────────────

  stop(): void {
    if (this.crowdRotationId) {
      cancelAnimationFrame(this.crowdRotationId);
      this.crowdRotationId = 0;
    }
  }

  // ── setBell ──────────────────────────────────────────────────────────────

  setBell(chime: number, spinRate: number, pan: number, inFlight: boolean): void {
    if (!this.ok || !this.ctx) return;
    const c = this.ctx;
    const t = c.currentTime;
    const smooth = 0.04;   // seconds for smooth param ramping

    // Clamp inputs
    chime    = Math.max(0, Math.min(1, chime));
    spinRate = Math.max(0, spinRate);
    pan      = Math.max(-1, Math.min(1, pan));

    // Volume: louder in flight, quieter when held — always present enough
    const audible = inFlight ? 1.0 : 0.22;

    // Pan — smooth spatial movement
    this.bellPan.pan.setTargetAtTime(pan, t, smooth * 1.5);

    // Pitch: base 440 Hz, spinRate nudges it slightly (spinning faster = higher)
    // Range: spinRate 0..~20 → 440..510 Hz (subtle, evocative)
    const freq = 440 + spinRate * 3.5;
    this.fundOsc.frequency.setTargetAtTime(freq, t, smooth);
    this.harm2Osc.frequency.setTargetAtTime(freq * 2.02, t, smooth);
    this.harm3Osc.frequency.setTargetAtTime(freq * 3.01, t, smooth);
    this.inharmonicOsc.frequency.setTargetAtTime(freq * 2.756, t, smooth);
    this.inharmonic2Osc.frequency.setTargetAtTime(freq * 2.891, t, smooth);

    // Timbral morphing:
    // chime=1: clean fundamental dominant, harmonics gentle, no noise/flutter
    // chime=0: fundamental fades, inharmonics & noise rise, flutter appears
    const clatter = 1 - chime;
    const clatter2 = clatter * clatter;

    this.fundGain.gain.setTargetAtTime(0.20 * chime * audible, t, smooth);
    this.harm2Gain.gain.setTargetAtTime(0.07 * chime * audible, t, smooth);
    this.harm3Gain.gain.setTargetAtTime(0.035 * chime * audible, t, smooth);

    // Inharmonics: base level from (1-chime), flutter adds on top via LFO
    const inharmonicBase = 0.11 * clatter * audible;
    this.inharmonicGain.gain.setTargetAtTime(inharmonicBase, t, smooth);
    this.inharmonic2Gain.gain.setTargetAtTime(0.07 * clatter * audible, t, smooth);
    this.noiseGain.gain.setTargetAtTime(0.045 * clatter2 * audible, t, smooth);

    // Flutter depth: proportional to clatter²
    this.flutterDepth.gain.setTargetAtTime(0.035 * clatter2 * audible, t, smooth);
  }

  // ── setHush ──────────────────────────────────────────────────────────────

  setHush(hush: number): void {
    if (!this.ok || !this.ctx) return;
    hush = Math.max(0, Math.min(1, hush));
    // At hush=1 crowd drops to ~4% of base (near silent but not zero — keeps
    // the breath-held tension; full silence comes from score_loop beat).
    const targetGain = this.CROWD_BASE * (1 - 0.94 * hush);
    this.crowdGain.gain.setTargetAtTime(targetGain, this.ctx.currentTime, 0.15);
  }

  // ── event ────────────────────────────────────────────────────────────────

  event(name: string): void {
    if (!this.ok || !this.ctx) return;
    const c = this.ctx;
    const t = c.currentTime;

    switch (name) {

      // ── catch: soft catching thunk — the bell lands in the grip
      case 'catch': {
        // Low thud: short noise burst through lowpass
        this.thud(c, t, 120, 0.9, 0.004, 0.12, 0.18);
        break;
      }

      // ── grapple: heavier body-contact thud — bigger, slower
      case 'grapple': {
        this.thud(c, t, 80, 1.3, 0.005, 0.18, 0.25);
        break;
      }

      // ── release: sharp click — ball leaving hand
      case 'release': {
        // Very short click: impulse through highpass
        this.click(c, t);
        break;
      }

      // ── throw: rising whoosh — launch burst
      case 'throw': {
        this.whoosh(c, t, 0.3);
        break;
      }

      // ── clatter: sharp detuned metallic strike (bad spin event)
      case 'clatter': {
        this.metalStrike(c, t, 0.22);
        break;
      }

      // ── score_fall: score for falling-ring — solid satisfying clunk + crowd bump
      case 'score_fall': {
        this.thud(c, t, 95, 1.1, 0.003, 0.14, 0.55);
        // pitched tone burst
        this.toneStab(c, t + 0.05, 220, 0.15, 0.0, 0.4);
        this.crowdErupt(c, t + 0.1, 0.55, 2.5);
        break;
      }

      // ── score_rise: score for rising-ring — big bright burst + crowd eruption
      case 'score_rise': {
        this.toneStab(c, t, 880, 0.28, 0.0, 0.45);
        this.toneStab(c, t + 0.025, 1100, 0.18, 0.0, 0.38);
        this.toneStab(c, t + 0.05, 660, 0.22, 0.0, 0.60);
        this.crowdErupt(c, t + 0.08, 0.80, 3.0);
        break;
      }

      // ── score_loop: LOOP score — beat of silence → massive crowd eruption
      case 'score_loop': {
        // Kill crowd briefly (hard cut to near-zero)
        this.crowdGain.gain.cancelScheduledValues(t);
        this.crowdGain.gain.setValueAtTime(0.0, t);
        // Beat of silence: 0.55 s
        this.crowdGain.gain.setValueAtTime(0.0, t + 0.55);
        // Eruption: ramp to massive then decay
        this.crowdGain.gain.linearRampToValueAtTime(
          this.CROWD_BASE + 0.65, t + 0.55 + 0.18
        );
        this.crowdGain.gain.exponentialRampToValueAtTime(
          this.CROWD_BASE, t + 0.55 + 4.5
        );
        // Bell tone on the eruption moment for drama
        this.toneStab(c, t + 0.55, 440, 0.35, 0.0, 0.9);
        this.toneStab(c, t + 0.60, 880, 0.20, 0.0, 0.6);
        break;
      }

      // ── score_ground: score on ground-gate — wry low plop
      case 'score_ground': {
        // Descending pitch blip — wry/comic
        this.pitchDrop(c, t, 180, 60, 0.18, 0.45);
        this.crowdErupt(c, t + 0.15, 0.22, 1.5);
        break;
      }

      // ── turnover: possession change — downward sting
      case 'turnover': {
        // Descending minor sting — three notes falling
        this.toneStab(c, t,       330, 0.14, 0.0, 0.22);
        this.toneStab(c, t + 0.1, 277, 0.13, 0.0, 0.22);
        this.toneStab(c, t + 0.2, 220, 0.16, 0.0, 0.35);
        // small crowd murmur
        this.crowdErupt(c, t + 0.05, 0.12, 1.0);
        break;
      }

      // ── loop_building: held breath — tension riser
      case 'loop_building': {
        // Slow rising filtered noise swell — held breath of stadium
        this.tensionRiser(c, t, 1.8);
        break;
      }

      default:
        // Unknown event — ignore silently
        break;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private synthesis helpers
  // ─────────────────────────────────────────────────────────────────────────

  /** Short noise burst through lowpass — a thud/thunk. */
  private thud(
    c: AudioContext, t: number,
    lpFreq: number, gainPeak: number,
    attack: number, decay: number, duration: number
  ): void {
    const nb = this.makeNoiseBuffer(c, Math.max(duration + 0.05, 0.3));
    const src = c.createBufferSource();
    src.buffer = nb;
    const lp = c.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = lpFreq;
    lp.Q.value = 0.8;
    const g = c.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gainPeak, t + attack);
    g.gain.exponentialRampToValueAtTime(0.001, t + attack + decay);
    src.connect(lp);
    lp.connect(g);
    g.connect(this.oneShotBus);
    src.start(t);
    src.stop(t + duration + 0.05);
  }

  /** Very short click transient. */
  private click(c: AudioContext, t: number): void {
    const nb = this.makeNoiseBuffer(c, 0.04);
    const src = c.createBufferSource();
    src.buffer = nb;
    const hp = c.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 3000;
    const g = c.createGain();
    g.gain.setValueAtTime(0.35, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.025);
    src.connect(hp);
    hp.connect(g);
    g.connect(this.oneShotBus);
    src.start(t);
    src.stop(t + 0.04);
  }

  /** Rising whoosh — launch/throw. */
  private whoosh(c: AudioContext, t: number, duration: number): void {
    const nb = this.makeNoiseBuffer(c, duration + 0.1);
    const src = c.createBufferSource();
    src.buffer = nb;
    const bp = c.createBiquadFilter();
    bp.type = 'bandpass';
    bp.Q.value = 1.5;
    bp.frequency.setValueAtTime(400, t);
    bp.frequency.exponentialRampToValueAtTime(2400, t + duration);
    const g = c.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.28, t + 0.05);
    g.gain.exponentialRampToValueAtTime(0.001, t + duration);
    src.connect(bp);
    bp.connect(g);
    g.connect(this.oneShotBus);
    src.start(t);
    src.stop(t + duration + 0.05);
  }

  /** Sharp metallic strike — inharmonic clatter. */
  private metalStrike(c: AudioContext, t: number, gainPeak: number): void {
    for (const [freq, rel] of [
      [440 * 2.756, 0.0],
      [440 * 2.891, 0.008],
      [440 * 4.13,  0.016],
    ] as [number, number][]) {
      const o = c.createOscillator();
      o.type = 'triangle';
      o.frequency.value = freq;
      const g = c.createGain();
      g.gain.setValueAtTime(0, t + rel);
      g.gain.linearRampToValueAtTime(gainPeak * 0.7, t + rel + 0.002);
      g.gain.exponentialRampToValueAtTime(0.001, t + rel + 0.22);
      o.connect(g);
      g.connect(this.oneShotBus);
      o.start(t + rel);
      o.stop(t + rel + 0.25);
    }
    // noise burst
    this.thud(c, t, 2200, gainPeak * 0.5, 0.001, 0.04, 0.08);
  }

  /** Tone stab: sine burst with sharp attack and decay. */
  private toneStab(
    c: AudioContext, t: number,
    freq: number, gainPeak: number,
    attack: number, decay: number
  ): void {
    const o = c.createOscillator();
    o.type = 'sine';
    o.frequency.value = freq;
    const g = c.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gainPeak, t + Math.max(attack, 0.001));
    g.gain.exponentialRampToValueAtTime(0.001, t + Math.max(attack, 0.001) + decay);
    o.connect(g);
    g.connect(this.oneShotBus);
    o.start(t);
    o.stop(t + Math.max(attack, 0.001) + decay + 0.05);
  }

  /** Descending pitch drop — comic/wry sound. */
  private pitchDrop(
    c: AudioContext, t: number,
    startFreq: number, endFreq: number,
    gainPeak: number, duration: number
  ): void {
    const o = c.createOscillator();
    o.type = 'sine';
    o.frequency.setValueAtTime(startFreq, t);
    o.frequency.exponentialRampToValueAtTime(endFreq, t + duration);
    const g = c.createGain();
    g.gain.setValueAtTime(gainPeak, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + duration);
    o.connect(g);
    g.connect(this.oneShotBus);
    o.start(t);
    o.stop(t + duration + 0.05);
  }

  /** Crowd eruption: spike crowd gain then decay back.
   *  With spatial ring: one random section leads by 50-100ms (nearest crowd reacts first),
   *  others follow with slight stagger. The master gain still gets the main spike envelope. */
  private crowdErupt(
    _c: AudioContext, t: number,
    spike: number, decay: number
  ): void {
    const peak = this.CROWD_BASE + spike;

    // Master gain envelope (overall eruption)
    this.crowdGain.gain.cancelScheduledValues(t);
    this.crowdGain.gain.setValueAtTime(this.crowdGain.gain.value, t);
    this.crowdGain.gain.linearRampToValueAtTime(peak, t + 0.12);
    this.crowdGain.gain.exponentialRampToValueAtTime(this.CROWD_BASE, t + 0.12 + decay);

    // Directional stagger: pick a random lead section, others offset 50-100ms behind
    if (this.crowdSections) {
      const leadIdx = Math.floor(Math.random() * this.CROWD_SECTIONS);
      const perSectionBase = 1.0 / this.CROWD_SECTIONS;
      // The lead section gets a brief solo gain boost before others catch up
      const boostPeak = perSectionBase * 2.5;  // momentarily louder

      for (let i = 0; i < this.CROWD_SECTIONS; i++) {
        const section = this.crowdSections[i];
        // Distance from lead section (0..3 wrapping) determines delay
        const dist = Math.min(
          Math.abs(i - leadIdx),
          this.CROWD_SECTIONS - Math.abs(i - leadIdx)
        );
        const delay = dist * (0.025 + Math.random() * 0.025); // 25-50ms per step

        section.gain.gain.cancelScheduledValues(t);
        section.gain.gain.setValueAtTime(perSectionBase, t);
        section.gain.gain.linearRampToValueAtTime(boostPeak, t + delay + 0.06);
        section.gain.gain.exponentialRampToValueAtTime(
          perSectionBase, t + delay + 0.06 + Math.min(decay * 0.4, 0.8)
        );
      }
    }
  }

  /** Tension riser: slow filtered noise swell — the held breath. */
  private tensionRiser(c: AudioContext, t: number, duration: number): void {
    const nb = this.makeNoiseBuffer(c, duration + 0.2);
    const src = c.createBufferSource();
    src.buffer = nb;
    const bp = c.createBiquadFilter();
    bp.type = 'bandpass';
    bp.Q.value = 3.0;
    bp.frequency.setValueAtTime(200, t);
    bp.frequency.linearRampToValueAtTime(900, t + duration);
    const g = c.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.09, t + duration * 0.6);
    g.gain.linearRampToValueAtTime(0.0, t + duration);
    src.connect(bp);
    bp.connect(g);
    g.connect(this.oneShotBus);
    src.start(t);
    src.stop(t + duration + 0.1);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Crowd rotation — subtle cylinder spin hint
  // ─────────────────────────────────────────────────────────────────────────

  /** Slowly rotate panner positions to hint at cylinder spin.
   *  ~1 revolution per CROWD_ROTATION_PERIOD seconds. Nearly imperceptible. */
  private startCrowdRotation(): void {
    if (!this.ctx) return;
    const angularVelocity = (Math.PI * 2) / this.CROWD_ROTATION_PERIOD;
    let lastTime = this.ctx.currentTime;

    const tick = () => {
      if (!this.ctx || !this.crowdSections) return;
      const now = this.ctx.currentTime;
      const dt = now - lastTime;
      lastTime = now;

      const dAngle = angularVelocity * dt;

      for (const section of this.crowdSections) {
        section.angle += dAngle;
        const x = Math.cos(section.angle) * this.CROWD_RING_RADIUS;
        const z = Math.sin(section.angle) * this.CROWD_RING_RADIUS;
        section.panner.positionX.value = x;
        section.panner.positionZ.value = z;
        // Y stays constant (rim elevation)
      }

      this.crowdRotationId = requestAnimationFrame(tick);
    };

    this.crowdRotationId = requestAnimationFrame(tick);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Utility
  // ─────────────────────────────────────────────────────────────────────────

  /** White noise buffer of given length in seconds. */
  private makeNoiseBuffer(c: AudioContext, seconds: number): AudioBuffer {
    const len = Math.ceil(c.sampleRate * seconds);
    const buf = c.createBuffer(1, len, c.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    return buf;
  }

  /** Soft waveshaper limiter — prevents harsh clipping without altering timbre. */
  private makeLimiter(c: AudioContext): WaveShaperNode {
    const ws = c.createWaveShaper();
    const n = 256;
    const curve = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = (i * 2) / n - 1;
      // soft-knee tanh limiter
      curve[i] = (Math.tanh(x * 2.5) / Math.tanh(2.5));
    }
    ws.curve = curve;
    ws.oversample = '4x';
    return ws;
  }
}

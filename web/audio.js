// audio.js — AX.Audio
// WebAudio synthesized sounds only — no external files.
// Initialize on first user gesture (call AX.Audio.init() from START click).
window.AX = window.AX || {};

AX.Audio = (function () {
  'use strict';

  let ctx = null;
  let masterGain = null;
  let ambientNode = null;
  let ambientGain = null;
  let ambientDucked = false;

  // ── init ──────────────────────────────────────────────────────────────────
  function init() {
    if (ctx) return;
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) {
      console.warn('AX.Audio: WebAudio not supported', e);
      return;
    }

    masterGain = ctx.createGain();
    masterGain.gain.value = 0.7;
    masterGain.connect(ctx.destination);

    startAmbient();
  }

  // ── resume on interaction (browsers block auto-play) ──────────────────────
  function ensureRunning() {
    if (ctx && ctx.state === 'suspended') ctx.resume();
  }

  // ── ambient cylinder hum ──────────────────────────────────────────────────
  function startAmbient() {
    if (!ctx) return;
    ambientGain = ctx.createGain();
    ambientGain.gain.value = 0.08;
    ambientGain.connect(masterGain);

    // Two detuned oscillators for beating texture
    const freqs = [55, 56.2, 82.5];
    for (const f of freqs) {
      const osc = ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.value = f;
      const g = ctx.createGain();
      g.gain.value = 1 / freqs.length;

      // Subtle slow LFO for movement
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 0.07 + Math.random() * 0.05;
      const lfoGain = ctx.createGain();
      lfoGain.gain.value = 0.015;
      lfo.connect(lfoGain);
      lfoGain.connect(osc.frequency);
      lfo.start();

      osc.connect(g);
      g.connect(ambientGain);
      osc.start();
    }

    // Distant noise layer
    const bufLen = ctx.sampleRate * 2;
    const buf = ctx.createBuffer(1, bufLen, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < bufLen; i++) data[i] = (Math.random() * 2 - 1) * 0.4;
    const noise = ctx.createBufferSource();
    noise.buffer = buf;
    noise.loop = true;
    const noiseFilter = ctx.createBiquadFilter();
    noiseFilter.type = 'bandpass';
    noiseFilter.frequency.value = 120;
    noiseFilter.Q.value = 0.3;
    const noiseGain = ctx.createGain();
    noiseGain.gain.value = 0.03;
    noise.connect(noiseFilter);
    noiseFilter.connect(noiseGain);
    noiseGain.connect(ambientGain);
    noise.start();
  }

  function duckAmbient(ducked) {
    if (!ambientGain || ambientDucked === ducked) return;
    ambientDucked = ducked;
    const target = ducked ? 0.02 : 0.08;
    ambientGain.gain.cancelScheduledValues(ctx.currentTime);
    ambientGain.gain.setTargetAtTime(target, ctx.currentTime, 0.4);
  }

  // ── synthesis helpers ─────────────────────────────────────────────────────
  function envelope(gainNode, attack, decay, sustainLevel, release, duration, t0) {
    const g = gainNode.gain;
    g.cancelScheduledValues(t0);
    g.setValueAtTime(0, t0);
    g.linearRampToValueAtTime(1, t0 + attack);
    g.linearRampToValueAtTime(sustainLevel, t0 + attack + decay);
    g.setValueAtTime(sustainLevel, t0 + duration - release);
    g.linearRampToValueAtTime(0, t0 + duration);
  }

  function makeOsc(freq, type, t0, duration, opts = {}) {
    if (!ctx) return;
    const g = ctx.createGain();
    g.connect(masterGain);
    const osc = ctx.createOscillator();
    osc.type = type || 'sine';
    osc.frequency.setValueAtTime(freq, t0);
    if (opts.sweep) osc.frequency.linearRampToValueAtTime(opts.sweep, t0 + duration);
    osc.connect(g);
    osc.start(t0);
    osc.stop(t0 + duration + 0.01);
    return { osc, gain: g };
  }

  function noise(t0, duration, filterFreq, filterQ, gainLevel) {
    if (!ctx) return;
    const bufLen = Math.ceil(ctx.sampleRate * Math.min(duration + 0.05, 2));
    const buf = ctx.createBuffer(1, bufLen, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < bufLen; i++) data[i] = Math.random() * 2 - 1;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = filterFreq || 400;
    filter.Q.value = filterQ || 1;
    const g = ctx.createGain();
    g.gain.value = gainLevel || 0.3;
    src.connect(filter);
    filter.connect(g);
    g.connect(masterGain);
    src.start(t0);
    src.stop(t0 + duration + 0.01);
    return { src, gain: g };
  }

  // ── event sounds ──────────────────────────────────────────────────────────
  const sounds = {

    pushoff() {
      // Soft thunk — low impact transient
      ensureRunning();
      const t = ctx.currentTime;
      const { gain: g1 } = makeOsc(80, 'sine', t, 0.18, { sweep: 40 });
      envelope(g1, 0.002, 0.05, 0.3, 0.1, 0.18, t);
      g1.gain.value *= 0.4;

      const { gain: g2 } = makeOsc(140, 'triangle', t, 0.1);
      envelope(g2, 0.001, 0.04, 0.1, 0.06, 0.1, t);
      g2.gain.value *= 0.2;
    },

    catch() {
      ensureRunning();
      const t = ctx.currentTime;
      const { gain: g } = makeOsc(220, 'sine', t, 0.14, { sweep: 180 });
      envelope(g, 0.001, 0.04, 0.4, 0.09, 0.14, t);
      g.gain.value *= 0.35;
    },

    throw(charge = 0.5) {
      ensureRunning();
      const t = ctx.currentTime;
      const baseFreq = 200 + charge * 600;
      const dur = 0.08 + charge * 0.25;
      // Whoosh
      const n = noise(t, dur * 1.5, 800 + charge * 2000, 0.8, 0.15 + charge * 0.25);
      if (n) {
        envelope(n.gain, 0.005, 0.05, 1, dur * 0.5, dur * 1.5, t);
      }
      // Tone tail
      const { gain: g } = makeOsc(baseFreq, 'sawtooth', t, dur * 0.8, { sweep: baseFreq * 0.5 });
      envelope(g, 0.002, 0.03, 0.5, dur * 0.4, dur * 0.8, t);
      g.gain.value *= 0.18;
    },

    score1() {
      ensureRunning();
      const t = ctx.currentTime;
      // Bright chime — single note
      for (const [freq, delay, dur] of [[880,0,0.5],[1100,0.06,0.4],[1320,0.12,0.35]]) {
        const { gain: g } = makeOsc(freq, 'sine', t + delay, dur);
        envelope(g, 0.001, 0.03, 0.6, 0.25, dur, t + delay);
        g.gain.value *= 0.35;
      }
      // Crowd swell
      const n = noise(t, 1.2, 600, 0.4, 0.18);
      if (n) envelope(n.gain, 0.15, 0.3, 1, 0.5, 1.2, t);
    },

    score2() {
      ensureRunning();
      const t = ctx.currentTime;
      // Bigger chime — major chord
      const notes = [
        [880,0,0.7],[1100,0.04,0.65],[1320,0.08,0.6],[1760,0.12,0.55],
        [880*1.5,0.18,0.5],[2200,0.22,0.45],
      ];
      for (const [freq, delay, dur] of notes) {
        const { gain: g } = makeOsc(freq, 'sine', t + delay, dur);
        envelope(g, 0.001, 0.04, 0.7, 0.3, dur, t + delay);
        g.gain.value *= 0.3;
      }
      // Crowd swell bigger
      const n = noise(t, 1.8, 500, 0.35, 0.25);
      if (n) envelope(n.gain, 0.1, 0.4, 1, 0.6, 1.8, t);
    },

    long() {
      // Long launch score — same as score2 + extra sparkle
      sounds.score2();
      const t = ctx.currentTime + 0.3;
      for (const [freq, delay] of [[3200,0],[2800,0.08],[3600,0.16]]) {
        const { gain: g } = makeOsc(freq, 'sine', t + delay, 0.4);
        envelope(g, 0.001, 0.02, 0.5, 0.3, 0.4, t + delay);
        g.gain.value *= 0.15;
      }
    },

    turnover() {
      ensureRunning();
      const t = ctx.currentTime;
      // Descending sting
      const { gain: g1 } = makeOsc(440, 'sawtooth', t, 0.35, { sweep: 220 });
      envelope(g1, 0.002, 0.05, 0.6, 0.2, 0.35, t);
      g1.gain.value *= 0.25;
      const { gain: g2 } = makeOsc(300, 'sine', t + 0.1, 0.3, { sweep: 180 });
      envelope(g2, 0.002, 0.05, 0.5, 0.15, 0.3, t + 0.1);
      g2.gain.value *= 0.2;
    },

    skin() {
      ensureRunning();
      const t = ctx.currentTime;
      // Percussive thud
      const { gain: g } = makeOsc(60, 'sine', t, 0.12, { sweep: 30 });
      envelope(g, 0.001, 0.03, 0.4, 0.07, 0.12, t);
      g.gain.value *= 0.5;
      const n = noise(t, 0.1, 300, 1, 0.2);
      if (n) envelope(n.gain, 0.001, 0.02, 0.5, 0.06, 0.1, t);
    },

    highframe() {
      ensureRunning();
      const t = ctx.currentTime;
      // Rising arpeggio sting
      for (const [freq, delay] of [[440,0],[550,0.08],[660,0.16],[880,0.24],[1100,0.34]]) {
        const { gain: g } = makeOsc(freq, 'triangle', t + delay, 0.3);
        envelope(g, 0.001, 0.04, 0.6, 0.2, 0.3, t + delay);
        g.gain.value *= 0.3;
      }
    },

    freespin() {
      ensureRunning();
      const t = ctx.currentTime;
      // Swirling effect — chorus-like
      for (let i = 0; i < 5; i++) {
        const freq = 330 * Math.pow(2, i / 7);
        const delay = i * 0.07;
        const { gain: g } = makeOsc(freq, 'sine', t + delay, 0.8);
        envelope(g, 0.01, 0.1, 0.5, 0.4, 0.8, t + delay);
        g.gain.value *= 0.2;
      }
      // Pitch shimmer
      const n = noise(t, 1.0, 1200, 1.5, 0.12);
      if (n) envelope(n.gain, 0.05, 0.2, 1, 0.5, 1.0, t);
    },

    win() {
      ensureRunning();
      const t = ctx.currentTime;
      // Fanfare
      const fanfare = [
        [523,0,0.25],[659,0,0.25],[784,0,0.25],
        [1047,0.2,0.6],[1047,0.82,0.15],[1047,0.98,0.5],
      ];
      for (const [freq, delay, dur] of fanfare) {
        const { gain: g } = makeOsc(freq, 'square', t + delay, dur);
        envelope(g, 0.005, 0.04, 0.5, 0.15, dur, t + delay);
        g.gain.value *= 0.2;
        // Harmonics
        const { gain: g2 } = makeOsc(freq * 2, 'sine', t + delay, dur);
        envelope(g2, 0.005, 0.04, 0.3, 0.15, dur, t + delay);
        g2.gain.value *= 0.1;
      }
      const n = noise(t, 2.0, 600, 0.4, 0.2);
      if (n) envelope(n.gain, 0.2, 0.4, 1, 0.8, 2.0, t);
    },

    lose() {
      ensureRunning();
      const t = ctx.currentTime;
      // Descending minor sting
      const loseNotes = [
        [466,0,0.3],[392,0.18,0.3],[330,0.36,0.4],[277,0.54,0.6],
      ];
      for (const [freq, delay, dur] of loseNotes) {
        const { gain: g } = makeOsc(freq, 'triangle', t + delay, dur);
        envelope(g, 0.002, 0.05, 0.6, 0.3, dur, t + delay);
        g.gain.value *= 0.28;
      }
    },

    tick() {
      ensureRunning();
      const t = ctx.currentTime;
      const { gain: g } = makeOsc(1200, 'sine', t, 0.04);
      envelope(g, 0.001, 0.01, 0.5, 0.02, 0.04, t);
      g.gain.value *= 0.12;
    },
  };

  // ── public event dispatcher ───────────────────────────────────────────────
  function event(name, extra) {
    if (!ctx) return;
    ensureRunning();

    if (name === 'freespin' || name === 'long') {
      // Duck ambient during exciting moments
      duckAmbient(true);
      setTimeout(() => duckAmbient(false), 3000);
    }

    const fn = sounds[name];
    if (fn) fn(extra);
    else console.warn('AX.Audio: unknown event', name);
  }

  return { init, event };
}());

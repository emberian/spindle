// Bloom (the glow that makes the trail/rings sing) + a controllable scene
// dim for the Loop "the stadium goes silent" moment. EffectComposer from
// three/examples/jsm — bundled statically by Vite, no extra deps.
//
// Loop moment (setLoopGlow at g=1):
//   • Bloom cranks up: the glowing closed arc becomes the only bright thing.
//   • Scene desaturates to greyscale: colour drains out of everything else.
//   • A deep vignette closes in: the ring floats in near-darkness.
//   Combined effect: breathtaking, unambiguous — a visual held breath.
//
// Broadcast aesthetic:
//   Subtle scanlines, chromatic aberration, and rare signal glitch communicate
//   "you are experiencing an interstellar audio feed visualized." Felt, not
//   seen — the texture of the medium, not a distraction.

import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass }     from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass }     from 'three/examples/jsm/postprocessing/ShaderPass.js';

// ── Finish shader: vignette + dim + desat + radial focus + broadcast aesthetic
// Applied AFTER bloom so vignette corners are dark even on bright bloom.
// The broadcast layer (scanlines, chromatic aberration, signal glitch) is all
// in this one pass — no additional render targets, no perf cost.
const FinishShader = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    // dim:         0..1 — multiplicative darkening of the whole frame
    dim:       { value: 0.0 },
    // desat:       0..1 — 0 = full colour, 1 = greyscale
    desat:     { value: 0.0 },
    // vig:         0..1 — vignette strength (0 = none, 1 = black corners)
    vig:       { value: 0.0 },
    // vigRadius:   normalised radius of vignette falloff
    vigRadius: { value: 0.75 },
    // bellScreen:  normalised screen-space position of the bell (0-1, 0-1)
    bellScreen: { value: new THREE.Vector2(0.5, 0.5) },
    // loopG:       0..1 — raw loop intensity (drives radial focus)
    loopG:     { value: 0.0 },
    // ── Broadcast / transmission uniforms ────────────────────────────────────
    time:          { value: 0.0 },           // seconds (performance.now/1000)
    resolution:    { value: new THREE.Vector2(1, 1) }, // px
    chromaStrength:{ value: 0.001 },         // UV-space offset at edges
    signalQuality: { value: 1.0 },           // 0..1 from bell chime
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float dim;
    uniform float desat;
    uniform float vig;
    uniform float vigRadius;
    uniform vec2 bellScreen;
    uniform float loopG;
    uniform float time;
    uniform vec2 resolution;
    uniform float chromaStrength;
    uniform float signalQuality;
    varying vec2 vUv;

    // Cheap pseudo-random from a seed float
    float hash(float n) { return fract(sin(n) * 43758.5453123); }

    void main() {
      // ── Chromatic aberration (transmission dispersion) ──────────────────
      // Offset R outward, B inward from centre. Strength ramps with distance
      // from centre and inversely with signalQuality.
      vec2 centre = vUv - 0.5;
      float dist = length(centre);
      float caStr = chromaStrength * (1.0 + (1.0 - signalQuality) * 2.0);
      vec2 caOffset = centre * dist * caStr;

      // ── Signal glitch (rare horizontal displacement) ────────────────────
      // Only fires when signalQuality < 0.4, and even then very rarely
      // (~1 frame every 5-10 seconds). A single-scanline UV offset.
      float glitchOffset = 0.0;
      if (signalQuality < 0.4) {
        // Quantise time to frames (~60fps) so the glitch lasts 1-2 frames
        float frameId = floor(time * 60.0);
        float trigger = hash(frameId * 0.017);
        // Fire roughly once per 360-600 frames (6-10s at 60fps)
        if (trigger < 0.003) {
          float scanY = floor(vUv.y * resolution.y);
          float band = hash(frameId * 0.031);
          float bandCentre = band * resolution.y;
          if (abs(scanY - bandCentre) < 3.0) {
            glitchOffset = (hash(frameId * 0.053) - 0.5) * 0.01;
          }
        }
      }

      vec2 uvR = vUv + caOffset + vec2(glitchOffset, 0.0);
      vec2 uvG = vUv + vec2(glitchOffset, 0.0);
      vec2 uvB = vUv - caOffset + vec2(glitchOffset, 0.0);

      vec4 c;
      c.r = texture2D(tDiffuse, uvR).r;
      c.g = texture2D(tDiffuse, uvG).g;
      c.b = texture2D(tDiffuse, uvB).b;
      c.a = 1.0;

      // 1. Desaturate (drain colour from everything except the bloomed loop)
      float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
      c.rgb = mix(c.rgb, vec3(luma), desat);

      // 2. Scene dim (multiply-down the whole frame)
      c.rgb *= (1.0 - dim);

      // 3. Radial focus: during Loop (loopG > 0), everything except a circle
      //    around the bell darkens and desaturates. The bell trail stays
      //    incandescent; the surrounding scene fades/softens. At g=0 the
      //    focusRadius is wide (everything sharp); at g=1 it's tight.
      if (loopG > 0.0) {
        float rfDist = length(vUv - bellScreen);
        float focusRadius = mix(0.8, 0.15, loopG);
        float focus = 1.0 - smoothstep(focusRadius, focusRadius + 0.3, rfDist);
        // Dim the unfocused area (multiplicative darken)
        c.rgb *= mix(1.0, focus, loopG * 0.7);
        // Desaturate the unfocused area
        float focusLuma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
        float focusDesat = (1.0 - focus) * loopG * 0.5;
        c.rgb = mix(c.rgb, vec3(focusLuma), focusDesat);
      }

      // 4. Vignette: smooth radial fall-off, deepened by vig parameter.
      //    At vig=0 it's off; at vig=1 it turns the corners nearly black.
      vec2 uv2 = vUv - 0.5;
      float r2 = dot(uv2, uv2) * 4.0; // 0 centre → ~1 corners
      // Use a power curve so it feels like a real lens shadow, not a hard disk
      float shadow = 1.0 - smoothstep(vigRadius * vigRadius,
                                       vigRadius * vigRadius + 0.4,
                                       r2);
      // Constant base vignette even at vig=0: subtle, tasteful. Trimmed
      // 0.18 → 0.10 so riggers toward the frame edges stay clearly legible
      // (legibility wins over darkness) while still a tasteful lens shadow.
      float baseVig = 0.10;
      float loopVig = vig * 0.72; // extra darkness during loop moment
      c.rgb *= mix(1.0, shadow, baseVig + loopVig);

      // ── Scanlines (texture of the medium) ──────────────────────────────
      // Faint horizontal lines drifting slowly upward — the CRT of an audio
      // feed reconstructed into image. Barely perceptible: felt, not seen.
      float scanSpeed = 0.5; // px/sec drift upward
      float pixelY = vUv.y * resolution.y + time * scanSpeed;
      float scanline = sin(pixelY * 3.14159265) * 0.5 + 0.5; // ~2px period
      // Base opacity: 0.03 when signal is clean, rises to 0.07 when degraded
      float scanOpacity = mix(0.03, 0.07, 1.0 - signalQuality);
      c.rgb *= 1.0 - scanline * scanOpacity;

      gl_FragColor = c;
    }
  `,
};

// ── Bloom tiers (pick based on renderer capability / pixel budget) ────────────
// Full:  render bloom at native res — gorgeous, costs ~30% of frame on bloom
// Half:  bloom at half res — 60fps safe on mid GPUs, indistinguishable in motion
const BLOOM_RESOLUTION_DIVISOR = 2; // 1=full, 2=half

export class PostFX {
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private finishPass: ShaderPass;

  // Baseline bloom — strong enough to make the trail/rings sing, not nuclear
  private readonly BASE_STRENGTH  = 0.26; // visible glow, still well below white-out
  private readonly BASE_RADIUS    = 0.4;
  // Threshold nudged 0.70 → 0.78: the scene is now brighter (legibility fix),
  // so only the truly hot accents (goal rings, bell) bloom — figures stay
  // crisp, high-contrast bodies rather than getting washed into glow.
  private readonly BASE_THRESHOLD = 0.78;

  // Loop peak bloom — a mild lift, NOT a screen-whiting blaze. The old 3.8
  // (designed for the deleted loop-cam) was THE recurring blowout: every
  // Loop ramped bloom to nuclear with no special framing to contain it.
  private readonly LOOP_STRENGTH  = 0.4;
  private readonly LOOP_RADIUS    = 0.5;
  private readonly LOOP_THRESHOLD = 0.55;

  constructor(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera) {
    this.composer = new EffectComposer(renderer);
    this.composer.addPass(new RenderPass(scene, camera));

    const w = renderer.domElement.width  / BLOOM_RESOLUTION_DIVISOR;
    const h = renderer.domElement.height / BLOOM_RESOLUTION_DIVISOR;

    this.bloom = new UnrealBloomPass(
      new THREE.Vector2(w, h),
      this.BASE_STRENGTH,
      this.BASE_RADIUS,
      this.BASE_THRESHOLD,
    );
    this.composer.addPass(this.bloom);

    this.finishPass = new ShaderPass(FinishShader as never);
    this.composer.addPass(this.finishPass);
  }

  setSize(w: number, h: number): void {
    this.composer.setSize(w, h);
    this.bloom.resolution.set(
      w / BLOOM_RESOLUTION_DIVISOR,
      h / BLOOM_RESOLUTION_DIVISOR,
    );
    // Keep resolution uniform in sync for scanline density calculation
    const u = this.finishPass.uniforms as {
      resolution: { value: THREE.Vector2 };
    };
    u.resolution.value.set(w, h);
  }

  // loopGlow ∈ [0,1]:
  //   0 = normal play (good bloom, tasteful vignette, full colour)
  //   1 = loop is airborne (bloom roars, radial focus tightens on the bell,
  //       the glowing orbit is the only bright thing — the visual held breath)
  setLoopGlow(g: number): void {
    const u = this.finishPass.uniforms as {
      dim:   { value: number };
      desat: { value: number };
      vig:   { value: number };
      loopG: { value: number };
    };

    // Bloom: interpolate from base to loop peak
    // Use an ease-in curve so the first half of the ramp is subtle and the
    // top half is dramatic — the gasp builds and then hits.
    const ge = g * g * (3.0 - 2.0 * g); // smoothstep ease
    this.bloom.strength  = this.BASE_STRENGTH  + (this.LOOP_STRENGTH  - this.BASE_STRENGTH)  * ge;
    this.bloom.radius    = this.BASE_RADIUS    + (this.LOOP_RADIUS    - this.BASE_RADIUS)    * ge;
    this.bloom.threshold = this.BASE_THRESHOLD + (this.LOOP_THRESHOLD - this.BASE_THRESHOLD) * ge;

    // Radial focus: the raw g drives the shader's radial dim/desat centred
    // on the bell's screen position. At g=0 it's invisible; at g=1 the
    // focus is tight and dramatic — but never black, the bell trail punches
    // through because it's the brightest thing in the scene.
    u.loopG.value = ge;

    // Legacy dim/desat stay zeroed — the radial focus replaces the old
    // global darken. A whisper of vignette still deepens the edges.
    u.dim.value   = 0.0;
    u.desat.value = 0.0;
    u.vig.value   = ge * 0.12;
  }

  // Set the bell's normalised screen-space position (0-1, 0-1) for the
  // radial focus effect. Called each frame from the render loop after
  // projecting bellMesh.position through the camera.
  setBellScreen(x: number, y: number): void {
    const u = this.finishPass.uniforms as {
      bellScreen: { value: THREE.Vector2 };
    };
    u.bellScreen.value.set(x, y);
  }

  // ── Broadcast uniforms: fed each frame from the render loop ────────────────
  // time:          seconds (monotonic clock)
  // signalQuality: 0..1 mapped from bell chime (1 = clean, 0 = degraded)
  setBroadcast(time: number, signalQuality: number): void {
    const u = this.finishPass.uniforms as {
      time:          { value: number };
      signalQuality: { value: number };
    };
    u.time.value = time;
    u.signalQuality.value = signalQuality;
  }

  render(): void {
    this.composer.render();
  }
}

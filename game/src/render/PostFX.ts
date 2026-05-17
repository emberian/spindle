// Bloom (the glow that makes the trail/rings sing) + a controllable scene
// dim for the Loop "the stadium goes silent" moment. EffectComposer from
// three/examples/jsm — bundled statically by Vite, no extra deps.
//
// Loop moment (setLoopGlow at g=1):
//   • Bloom cranks up: the glowing closed arc becomes the only bright thing.
//   • Scene desaturates to greyscale: colour drains out of everything else.
//   • A deep vignette closes in: the ring floats in near-darkness.
//   Combined effect: breathtaking, unambiguous — a visual held breath.

import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass }     from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass }     from 'three/examples/jsm/postprocessing/ShaderPass.js';

// ── Finish shader: vignette + scene-dim + desaturate ─────────────────────────
// Applied AFTER bloom so vignette corners are dark even on bright bloom.
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
    varying vec2 vUv;

    void main() {
      vec4 c = texture2D(tDiffuse, vUv);

      // 1. Desaturate (drain colour from everything except the bloomed loop)
      float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
      c.rgb = mix(c.rgb, vec3(luma), desat);

      // 2. Scene dim (multiply-down the whole frame)
      c.rgb *= (1.0 - dim);

      // 3. Vignette: smooth radial fall-off, deepened by vig parameter.
      //    At vig=0 it's off; at vig=1 it turns the corners nearly black.
      vec2 uv2 = vUv - 0.5;
      float r2 = dot(uv2, uv2) * 4.0; // 0 centre → ~1 corners
      // Use a power curve so it feels like a real lens shadow, not a hard disk
      float shadow = 1.0 - smoothstep(vigRadius * vigRadius,
                                       vigRadius * vigRadius + 0.4,
                                       r2);
      // Constant base vignette even at vig=0: subtle, tasteful
      float baseVig = 0.18;
      float loopVig = vig * 0.72; // extra darkness during loop moment
      c.rgb *= mix(1.0, shadow, baseVig + loopVig);

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
  private readonly BASE_STRENGTH  = 0.85;
  private readonly BASE_RADIUS    = 0.60;
  private readonly BASE_THRESHOLD = 0.26; // a bit higher → only bright things bloom (less wash)

  // Loop peak bloom
  private readonly LOOP_STRENGTH  = 3.8;  // blazing
  private readonly LOOP_RADIUS    = 0.85; // wide, corona-like
  private readonly LOOP_THRESHOLD = 0.05; // grab almost everything

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
  }

  // loopGlow ∈ [0,1]:
  //   0 = normal play (good bloom, tasteful vignette, full colour)
  //   1 = loop is airborne (bloom roars, colour drains, vignette deepens,
  //       the glowing orbit is the only bright thing — the visual held breath)
  setLoopGlow(g: number): void {
    const u = this.finishPass.uniforms as {
      dim:   { value: number };
      desat: { value: number };
      vig:   { value: number };
    };

    // Bloom: interpolate from base to loop peak
    // Use an ease-in curve so the first half of the ramp is subtle and the
    // top half is dramatic — the gasp builds and then hits.
    const ge = g * g * (3.0 - 2.0 * g); // smoothstep ease
    this.bloom.strength  = this.BASE_STRENGTH  + (this.LOOP_STRENGTH  - this.BASE_STRENGTH)  * ge;
    this.bloom.radius    = this.BASE_RADIUS    + (this.LOOP_RADIUS    - this.BASE_RADIUS)    * ge;
    this.bloom.threshold = this.BASE_THRESHOLD + (this.LOOP_THRESHOLD - this.BASE_THRESHOLD) * ge;

    // Scene dim: gentle until g>0.5, then drops the room
    u.dim.value   = ge * 0.55;

    // Desaturate: colour drains as bloom fills — the loop is all that matters
    u.desat.value = ge * 0.80;

    // Vignette: the frame closes in as the loop rises
    u.vig.value   = ge;
  }

  render(): void {
    this.composer.render();
  }
}

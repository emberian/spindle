// Bloom (the glow that makes the trail/rings sing) + a controllable scene
// dim for the Loop "the stadium goes silent" moment. EffectComposer from
// three/examples/jsm — bundled statically by Vite, no extra deps.

import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';

const DimShader = {
  uniforms: { tDiffuse: { value: null as THREE.Texture | null }, dim: { value: 0 }, vig: { value: 0.25 } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float dim; uniform float vig; varying vec2 vUv;
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      vec2 d = vUv - 0.5;
      float v = smoothstep(0.85, vig, dot(d,d)*2.0);
      c.rgb *= (1.0 - dim);
      c.rgb *= mix(1.0, 0.55, v);
      gl_FragColor = c;
    }`,
};

export class PostFX {
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private dimPass: ShaderPass;
  private baseBloom = 0.9;

  constructor(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera) {
    this.composer = new EffectComposer(renderer);
    this.composer.addPass(new RenderPass(scene, camera));
    this.bloom = new UnrealBloomPass(
      new THREE.Vector2(innerWidth, innerHeight),
      this.baseBloom,
      0.55,
      0.2,
    );
    this.composer.addPass(this.bloom);
    this.dimPass = new ShaderPass(DimShader as never);
    this.composer.addPass(this.dimPass);
  }

  setSize(w: number, h: number): void {
    this.composer.setSize(w, h);
    this.bloom.resolution.set(w, h);
  }

  // loopGlow ∈ [0,1]: ramps bloom up and the rest of the scene down so the
  // glowing closed arc is the only bright thing — the visual "silence".
  setLoopGlow(g: number): void {
    this.bloom.strength = this.baseBloom + g * 2.1;
    (this.dimPass.uniforms as { dim: { value: number } }).dim.value = g * 0.34;
  }

  render(): void {
    this.composer.render();
  }
}

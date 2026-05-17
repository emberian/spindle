// The landing / explainer. First thing you see: what RIG is, why it's
// interesting (the physics + the fiction), the controls, and a PLAY button.
// Pure DOM overlay, on the dossier palette. API:
//   new LandingScreen(root); show(onPlay); hide();

import { PAL } from './palette';

const CSS = `
#rig-landing{position:absolute;inset:0;z-index:60;display:none;overflow-y:auto;
  background:radial-gradient(120% 90% at 50% 0%, #161a24 0%, ${PAL.bgCss} 60%);
  color:${PAL.paperCss};font-family:ui-monospace,"Space Mono",monospace;}
#rig-landing .wrap{max-width:880px;margin:0 auto;padding:7vh 28px 12vh;}
#rig-landing .mark{font-size:clamp(56px,11vw,128px);font-weight:800;letter-spacing:.18em;
  text-align:center;background:linear-gradient(90deg,${PAL.cyanCss},${PAL.orangeCss});
  -webkit-background-clip:text;background-clip:text;color:transparent;margin:0;}
#rig-landing .tag{text-align:center;color:${PAL.dimCss};letter-spacing:.34em;
  text-transform:uppercase;font-size:13px;margin:.4em 0 0;}
#rig-landing .lede{text-align:center;color:${PAL.paperCss};font-size:clamp(15px,2vw,19px);
  line-height:1.7;margin:2.4em auto 0;max-width:680px;}
#rig-landing .lede b{color:${PAL.cyanCss};}
#rig-landing .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:18px;margin:2.8em 0;}
#rig-landing .card{border:1px solid ${PAL.dimCss}55;border-radius:6px;padding:16px 18px;
  background:#0d0f16;}
#rig-landing .card h3{margin:0 0 6px;color:${PAL.cyanCss};font-size:13px;
  letter-spacing:.16em;text-transform:uppercase;}
#rig-landing .card p{margin:0;font-size:13.5px;line-height:1.6;color:#c9ccd2;}
#rig-landing .curio{border-left:2px solid ${PAL.orangeCss};padding:10px 0 10px 18px;
  margin:2.4em 0;color:#aeb2bb;font-size:13px;line-height:1.7;}
#rig-landing .curio code{color:${PAL.paperCss};background:#1c2030;padding:1px 6px;border-radius:3px;}
#rig-landing .ctl{display:flex;flex-wrap:wrap;gap:10px 26px;justify-content:center;
  color:${PAL.dimCss};font-size:12.5px;letter-spacing:.06em;margin:1.6em 0 2.6em;}
#rig-landing .ctl b{color:${PAL.paperCss};}
#rig-landing .acts{display:flex;flex-wrap:wrap;gap:14px 18px;justify-content:center;align-items:center;}
#rig-landing .play{display:block;padding:18px 64px;font:inherit;
  font-size:20px;font-weight:800;letter-spacing:.28em;color:${PAL.bgCss};cursor:pointer;
  border:none;border-radius:6px;background:linear-gradient(90deg,${PAL.cyanCss},${PAL.orangeCss});}
#rig-landing .play:hover{filter:brightness(1.12);}
#rig-landing .watch{display:block;padding:17px 40px;font:inherit;font-size:14px;
  font-weight:700;letter-spacing:.24em;text-transform:uppercase;color:${PAL.cyanCss};
  cursor:pointer;border:1px solid ${PAL.cyanCss}88;border-radius:6px;background:transparent;}
#rig-landing .watch:hover{background:${PAL.cyanCss}1a;border-color:${PAL.cyanCss};}
#rig-landing .links{text-align:center;margin-top:2.4em;font-size:12px;color:${PAL.dimCss};}
#rig-landing .links a{color:${PAL.cyanCss};text-decoration:none;margin:0 12px;}
#rig-landing .foot{text-align:center;margin-top:3em;color:${PAL.dimCss};
  font-size:11px;letter-spacing:.3em;text-transform:uppercase;}
`;

const CARDS: [string, string][] = [
  ['The Calm', "Play happens in the weightless tube on the spin axis of an O'Neill cylinder. The field is a volume; \"down\" is a direction you spend."],
  ['The Loop', 'Throw against the spin and the Coriolis force bends the bell into a closed arc back through the goal. Seven points. Most riggers never score one.'],
  ['The Line', 'You cannot fly. You move by pushing off structure or by throwing a powered grapple and hauling — arena-shooter movement in zero-g.'],
  ['No Clock', "Nine innings, never a clock. Pick one of 32 canon franchises and play The Jump — a 16-team single-elimination bracket — to a champion."],
];

export class LandingScreen {
  private el: HTMLElement;
  private onPlay: (() => void) | null = null;
  private onSpectate: (() => void) | null = null;
  private onReplay: (() => void) | null = null;

  constructor(root: HTMLElement) {
    if (!document.getElementById('rig-landing-css')) {
      const s = document.createElement('style');
      s.id = 'rig-landing-css';
      s.textContent = CSS;
      document.head.appendChild(s);
    }
    this.el = document.createElement('div');
    this.el.id = 'rig-landing';
    this.el.innerHTML = `
      <div class="wrap">
        <h1 class="mark">RIG</h1>
        <p class="tag">The game played in the calm</p>
        <p class="lede">A zero-gravity sport <b>derived</b>, not decorated, from the
          physics of a rotating space habitat — and canon in the <b>Vivere Astra</b>
          universe. The rulebook falls out of one equation. You play it here, in your
          browser, against an AI that plays the way its home cylinder spun it.</p>
        <div class="grid">${CARDS.map(
          ([h, p]) => `<div class="card"><h3>${h}</h3><p>${p}</p></div>`,
        ).join('')}</div>
        <p class="curio">For the curious: the cross-section obeys
          <code>ζ̈ = ω²ζ − 2iω ζ̇</code> — a double root at <code>−iω</code>, so every
          trajectory is a roulette of a circle. The Loop is the closed member of that
          family. The whole sim is a deterministic Rust→WASM core; a match replays
          bit-for-bit from its seed (the in-fiction "re-call").</p>
        <div class="ctl">
          <span><b>MOVE</b> grapple + push-off</span>
          <span><b>MOUSE</b> aim</span>
          <span><b>HOLD</b> charge a throw</span>
          <span><b>SPIN</b> bend the bell</span>
        </div>
        <div class="acts">
          <button class="play" id="rig-play-btn">PLAY</button>
          <button class="watch" id="rig-watch-btn">Watch a match</button>
          <button class="watch" id="rig-recall-btn">Re-calls</button>
        </div>
        <p class="links">
          <a href="classic/">Classic (the 2D original)</a> ·
          <a href="https://github.com/emberian/spindle" target="_blank" rel="noopener">Source &amp; the dossier</a>
        </p>
        <p class="foot">No clock · The chime is still ringing</p>
      </div>`;
    root.style.position = 'relative';
    root.appendChild(this.el);
    this.el.querySelector<HTMLButtonElement>('#rig-play-btn')!.addEventListener('click', () => {
      const cb = this.onPlay;
      this.hide();
      cb?.();
    });
    this.el.querySelector<HTMLButtonElement>('#rig-watch-btn')!.addEventListener('click', () => {
      const cb = this.onSpectate;
      this.hide();
      cb?.();
    });
    this.el.querySelector<HTMLButtonElement>('#rig-recall-btn')!.addEventListener('click', () => {
      const cb = this.onReplay;
      this.hide();
      cb?.();
    });
  }

  show(onPlay: () => void, onSpectate?: () => void, onReplay?: () => void): void {
    this.onPlay = onPlay;
    this.onSpectate = onSpectate ?? null;
    this.onReplay = onReplay ?? null;
    this.el.style.display = 'block';
    this.el.scrollTop = 0;
  }

  hide(): void {
    this.el.style.display = 'none';
    this.onPlay = null;
    this.onSpectate = null;
    this.onReplay = null;
  }
}

// HUD — DOM/canvas overlay reading MatchState + SimState.
// Renders: inning, score with team tints, cast throws-left pips, contest
// count, possession indicator, bell-trueness meter (shivers on clatter),
// reticle crosshair, charge ring, and predicted-arc ghost hooks.
//
// No gameplay logic — purely a read-only view of sim+match state.

import type { SimState, MatchState } from '../sim/types';
import type { InputView } from '../input/InputManager';
import { PAL } from './palette';

// ── Layout constants ───────────────────────────────────────────────────────

const RETICLE_R   = 18;   // px
const CHARGE_R    = 26;   // px  (charge ring, slightly outside reticle)
const GHOST_DOT_R =  3;   // px

// ── Helpers ────────────────────────────────────────────────────────────────

function hex(n: number): string {
  return '#' + n.toString(16).padStart(6, '0');
}

function arc(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  r: number,
  start: number, end: number,
  color: string,
  lineW: number,
): void {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth   = lineW;
  ctx.beginPath();
  ctx.arc(cx, cy, r, start, end);
  ctx.stroke();
  ctx.restore();
}

// ── HUD ────────────────────────────────────────────────────────────────────

export class HUD {
  private root:   HTMLElement;
  private canvas: HTMLCanvasElement;
  private ctx:    CanvasRenderingContext2D;

  // DOM panels
  private scoreEl:     HTMLElement;
  private inningEl:    HTMLElement;
  private pipsEl:      HTMLElement;
  private contestEl:   HTMLElement;
  private possEl:      HTMLElement;
  private truenessEl:  HTMLElement;
  private messageEl:   HTMLElement;

  // Shiver state for trueness meter
  private lastChime    = 1;
  private shiverOffset = 0;

  constructor(root: HTMLElement) {
    this.root = root;

    // Canvas for reticle, charge ring, arc ghost (covers full viewport)
    this.canvas = document.createElement('canvas');
    Object.assign(this.canvas.style, {
      position: 'absolute',
      inset:    '0',
      pointerEvents: 'none',
      zIndex:   '10',
    });
    this.ctx = this.canvas.getContext('2d')!;

    // Resize canvas to root
    const ro = new ResizeObserver(() => this._resize());
    ro.observe(root);
    this._resize();

    // ── HUD DOM panels ──────────────────────────────────────────────────
    const panel = document.createElement('div');
    Object.assign(panel.style, {
      position:     'absolute',
      inset:        '0',
      pointerEvents:'none',
      zIndex:       '9',
      fontFamily:   'monospace, "Space Mono", sans-serif',
      fontSize:     '13px',
      color:        PAL.paperCss,
      userSelect:   'none',
    });

    // Score: top-center
    this.scoreEl = this._makeEl({
      position:  'absolute',
      top:       '16px',
      left:      '50%',
      transform: 'translateX(-50%)',
      fontSize:  '22px',
      letterSpacing: '2px',
      textShadow: `0 0 8px ${PAL.dimCss}`,
    });

    // Inning: just below score
    this.inningEl = this._makeEl({
      position:  'absolute',
      top:       '44px',
      left:      '50%',
      transform: 'translateX(-50%)',
      fontSize:  '11px',
      color:     PAL.dimCss,
    });

    // Throws-left pips: top-left
    this.pipsEl = this._makeEl({
      position: 'absolute',
      top:      '16px',
      left:     '20px',
      fontSize: '16px',
    });

    // Contest: below pips
    this.contestEl = this._makeEl({
      position:  'absolute',
      top:       '40px',
      left:      '20px',
      fontSize:  '12px',
      color:     PAL.orangeCss,
    });

    // Possession: top-right
    this.possEl = this._makeEl({
      position:  'absolute',
      top:       '16px',
      right:     '20px',
      fontSize:  '13px',
    });

    // Trueness meter: bottom-left
    this.truenessEl = this._makeEl({
      position:     'absolute',
      bottom:       '24px',
      left:         '20px',
      width:        '100px',
      height:       '10px',
      background:   PAL.bgCss,
      border:       `1px solid ${PAL.dimCss}`,
      overflow:     'hidden',
    });
    const fill = document.createElement('div');
    fill.id = '_chime_fill';
    Object.assign(fill.style, {
      height: '100%',
      width:  '100%',
      background: PAL.cyanCss,
      transformOrigin: 'left',
    });
    this.truenessEl.appendChild(fill);

    // Match message: bottom-center
    this.messageEl = this._makeEl({
      position:  'absolute',
      bottom:    '40px',
      left:      '50%',
      transform: 'translateX(-50%)',
      fontSize:  '12px',
      color:     PAL.dimCss,
      whiteSpace:'nowrap',
    });

    panel.append(
      this.scoreEl, this.inningEl, this.pipsEl, this.contestEl,
      this.possEl, this.truenessEl, this.messageEl,
    );
    root.style.position = 'relative';
    root.appendChild(panel);
    root.appendChild(this.canvas);
  }

  // ── Main render ────────────────────────────────────────────────────────

  render(sim: SimState, match: MatchState, input: InputView): void {
    this._updateDOM(sim, match);
    this._paintCanvas(sim, match, input);
  }

  // ── DOM update ─────────────────────────────────────────────────────────

  private _updateDOM(sim: SimState, match: MatchState): void {
    // Score — home tinted cyan, away tinted orange
    const homeTint  = hex(PAL.cyan);
    const awayTint  = hex(PAL.orange);
    this.scoreEl.innerHTML =
      `<span style="color:${homeTint}">${match.scoreHome}</span>` +
      ` <span style="color:${PAL.dimCss}">–</span> ` +
      `<span style="color:${awayTint}">${match.scoreAway}</span>`;

    // Inning
    const inningLabel = match.spine ? 'SPINE' : `INN ${match.inning}`;
    const phaseLabel  = match.phase.toUpperCase().replace('_', ' ');
    this.inningEl.textContent = `${inningLabel}  ${phaseLabel}`;

    // Throws-left pips (filled circles)
    const PIP_ON  = '●';
    const PIP_OFF = '○';
    const t = match.cast.throwsLeft;
    const pips = [PIP_ON, PIP_ON, PIP_ON]
      .map((_p, i) => `<span style="color:${i < t ? PAL.cyanCss : PAL.dimCss}">${i < t ? PIP_ON : PIP_OFF}</span>`)
      .join(' ');
    this.pipsEl.innerHTML = `CAST ${pips}`;

    // Contest count
    if (match.contest) {
      const c = match.contest;
      const countPips = Array.from({ length: 3 }, (_, i) =>
        `<span style="color:${i < c.count ? PAL.orangeCss : PAL.dimCss}">◆</span>`,
      ).join(' ');
      this.contestEl.innerHTML = `CONTEST ${countPips} ${c.direction.toUpperCase()}`;
      this.contestEl.style.display = 'block';
    } else {
      this.contestEl.style.display = 'none';
    }

    // Possession
    const possColor = match.possession === 'home' ? homeTint : awayTint;
    const possLabel = match.possession.toUpperCase();
    this.possEl.innerHTML = `<span style="color:${possColor}">▶ ${possLabel}</span>`;

    // Trueness meter — shiver on clatter
    const chime = sim.bell.chime;
    const fell  = chime < this.lastChime - 0.15;
    if (fell) {
      this.shiverOffset = (Math.random() - 0.5) * 6;
    } else {
      this.shiverOffset *= 0.7;
    }
    this.lastChime = chime;

    const fill = this.truenessEl.querySelector('#_chime_fill') as HTMLElement;
    if (fill) {
      fill.style.transform      = `scaleX(${chime}) translateX(${this.shiverOffset}px)`;
      fill.style.background     = chime > 0.6 ? PAL.cyanCss : PAL.orangeCss;
      fill.style.transition     = fell ? 'none' : 'transform 0.1s';
    }
    this.truenessEl.title = `bell trueness: ${(chime * 100).toFixed(0)}%`;

    // Match message
    this.messageEl.textContent = match.message;
  }

  // ── Canvas overlay (reticle, charge ring, arc ghost) ──────────────────

  private _paintCanvas(sim: SimState, match: MatchState, input: InputView): void {
    void sim; void match; // may use later
    const w  = this.canvas.width;
    const h  = this.canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    // ── Arc ghost ───────────────────────────────────────────────────────
    // The ghost is in world space — we project it via a simple perspective
    // hint for now. The orchestrator can wire in a Three.js project helper
    // for accurate screen-space dots; here we draw them as a stylistic
    // scatter near the reticle center as a charge-level indicator.
    if (input.ghostArc.length > 1 && input.chargeLevel > 0.01) {
      // Draw a dot-trail converging outward from center
      const n = input.ghostArc.length;
      for (let i = 1; i < n; i++) {
        const t    = i / n;
        const fade = 0.15 + 0.5 * (1 - t) * input.chargeLevel;
        // Fan out from reticle with a crude projection hint
        // (orchestrator will replace with proper THREE.Vector3.project)
        const angle = Math.atan2(
          input.ghostArc[i].z - input.ghostArc[0].z,
          input.ghostArc[i].y - input.ghostArc[0].y,
        );
        const dist = Math.min(
          Math.hypot(
            input.ghostArc[i].y - input.ghostArc[0].y,
            input.ghostArc[i].z - input.ghostArc[0].z,
          ) * 0.5,
          80,
        );
        const gx = cx + Math.cos(angle) * dist * t;
        const gy = cy + Math.sin(angle) * dist * t;
        ctx.beginPath();
        ctx.arc(gx, gy, GHOST_DOT_R, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(26,166,183,${fade})`;
        ctx.fill();
      }
    }

    // ── Reticle ─────────────────────────────────────────────────────────
    const reticle = input.reticleState;
    if (reticle !== 'none') {
      let col: string;
      if      (reticle === 'valid')       col = PAL.cyanCss;
      else if (reticle === 'out_of_range') col = PAL.dimCss;
      else                                col = PAL.orangeCss;

      const r = RETICLE_R;
      const GAP = 4;
      const LEN = 8;
      // Cross-hair gaps
      ctx.strokeStyle = col;
      ctx.lineWidth   = 1.5;
      // Top
      ctx.beginPath(); ctx.moveTo(cx, cy - r - GAP); ctx.lineTo(cx, cy - r - GAP - LEN); ctx.stroke();
      // Bottom
      ctx.beginPath(); ctx.moveTo(cx, cy + r + GAP); ctx.lineTo(cx, cy + r + GAP + LEN); ctx.stroke();
      // Left
      ctx.beginPath(); ctx.moveTo(cx - r - GAP, cy); ctx.lineTo(cx - r - GAP - LEN, cy); ctx.stroke();
      // Right
      ctx.beginPath(); ctx.moveTo(cx + r + GAP, cy); ctx.lineTo(cx + r + GAP + LEN, cy); ctx.stroke();

      // Outer circle (faint)
      arc(ctx, cx, cy, r + 2, 0, Math.PI * 2, col + '44', 1);
    } else {
      // Dot only when no anchor in range
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = PAL.dimCss + 'aa';
      ctx.fill();
    }

    // ── Charge ring ──────────────────────────────────────────────────────
    if (input.chargeLevel > 0.01) {
      const startAngle = -Math.PI / 2;
      const endAngle   = startAngle + Math.PI * 2 * input.chargeLevel;
      // Background ring
      arc(ctx, cx, cy, CHARGE_R, 0, Math.PI * 2, PAL.dimCss + '44', 3);
      // Filled arc
      const chargeColor = input.chargeLevel > 0.85 ? PAL.orangeCss : PAL.cyanCss;
      arc(ctx, cx, cy, CHARGE_R, startAngle, endAngle, chargeColor, 3);

      // Pulse on high charge
      if (input.chargeLevel > 0.85) {
        const pulse = 0.4 + 0.3 * Math.sin(Date.now() * 0.012);
        arc(ctx, cx, cy, CHARGE_R + 4, startAngle, endAngle,
          `rgba(212,96,42,${pulse})`, 1.5);
      }
    }
  }

  // ── Internals ──────────────────────────────────────────────────────────

  private _resize(): void {
    const rect = this.root.getBoundingClientRect();
    this.canvas.width  = rect.width  || window.innerWidth;
    this.canvas.height = rect.height || window.innerHeight;
  }

  private _makeEl(style: Partial<CSSStyleDeclaration>): HTMLElement {
    const el = document.createElement('div');
    Object.assign(el.style, style);
    return el;
  }
}

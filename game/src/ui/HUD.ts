// HUD — DOM/canvas overlay reading MatchState + SimState.
//
// Mission: make match PROGRESSION unmistakable at a glance, readable even
// at 4× spectate speed. Top-level readouts:
//   1. Possession + attack direction  (who has the bell, which ring driven to)
//   2. Cast clock                     (throw pips, gate, axial field bar)
//   3. Bell state                     (HELD / IN FLIGHT / DEAD / CONTEST)
//   4. Play-by-play ribbon            (last ~5 notable events, derived here)
//   5. Score-by-kind tally            (Fall/Rise/Loop/Curl/Ground per side)
// Plus the original spectator/player chrome: score, inning, contest count,
// bell-trueness meter, and the human-play reticle / ghost-arc / charge ring.
//
// No gameplay logic — purely a read-only view of sim+match state. All
// progression-derived signals (play-by-play, score-kind) are reconstructed
// from cross-frame transitions in match.message / scores, debounced here.
//
// Geometry: spin axis = +X. Rings at x = ±GATE_X. RULE (verified in
// src/match/MatchStateMachine.ts `attackDir`): HOME always attacks +X,
// AWAY always attacks −X — independent of faithEnd. faithEnd only labels
// which ring scores Faith vs Free.

import type { SimState, MatchState } from '../sim/types';
import type { InputView } from '../input/InputManager';
import { GATE_X } from '../sim/RegConstants';
import { PAL } from './palette';
import { PlayByPlay } from './PlayByPlay';
import type { PbpEvent } from './PlayByPlay';

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

/** HOME attacks +X, AWAY attacks −X. Returns the target ring's X sign. */
function attackSign(possession: 'home' | 'away'): 1 | -1 {
  return possession === 'home' ? 1 : -1;
}

type ScoreKind = 'fall' | 'rise' | 'loop' | 'curl' | 'ground';
const KIND_ORDER: ScoreKind[] = ['fall', 'rise', 'loop', 'curl', 'ground'];

// ── HUD ────────────────────────────────────────────────────────────────────

export class HUD {
  private root:   HTMLElement;
  private canvas: HTMLCanvasElement;
  private ctx:    CanvasRenderingContext2D;

  // DOM panels (original chrome)
  private scoreEl:     HTMLElement;
  private inningEl:    HTMLElement;
  private pipsEl:      HTMLElement;
  private contestEl:   HTMLElement;
  private truenessEl:  HTMLElement;
  private messageEl:   HTMLElement;

  // New progression readouts
  private driveEl:     HTMLElement;   // possession + attack direction banner
  private bellEl:      HTMLElement;   // bell state (HELD / IN FLIGHT / …)
  private fieldEl:     HTMLElement;   // axial field bar container
  private fieldBellEl: HTMLElement;   // bell marker within field bar
  private fieldRingHi: HTMLElement;   // highlighted target ring cap
  private gateEl:      HTMLElement;   // current cast gate (first/deep/mouth)
  private tallyEl:     HTMLElement;   // score-by-kind tally
  private pbp:         PlayByPlay;

  // Shiver state for trueness meter
  private lastChime    = 1;
  private shiverOffset = 0;

  // ── Cross-frame transition tracking (for play-by-play + tally) ──────────
  private lastMessage = '';
  private lastScoreHome = 0;
  private lastScoreAway = 0;
  private lastInning = 1;
  private lastPhase = '';
  private lastContestOn = false;
  private seq = 0;
  private kindTally: Record<'home' | 'away', Record<ScoreKind, number>> = {
    home: { fall: 0, rise: 0, loop: 0, curl: 0, ground: 0 },
    away: { fall: 0, rise: 0, loop: 0, curl: 0, ground: 0 },
  };

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

    // Possession + attack-direction drive banner: below inning, top-center.
    // The single biggest "who is doing what" signal — big and colour-coded.
    this.driveEl = this._makeEl({
      position:  'absolute',
      top:       '64px',
      left:      '50%',
      transform: 'translateX(-50%)',
      fontSize:  '15px',
      fontWeight:'bold',
      letterSpacing: '1px',
      whiteSpace:'nowrap',
      textShadow: `0 0 6px ${PAL.bgCss}`,
    });

    // Throws-left pips: top-left
    this.pipsEl = this._makeEl({
      position: 'absolute',
      top:      '16px',
      left:     '20px',
      fontSize: '16px',
    });

    // Cast gate: below pips
    this.gateEl = this._makeEl({
      position:  'absolute',
      top:       '40px',
      left:      '20px',
      fontSize:  '12px',
      color:     PAL.dimCss,
    });

    // Contest: below gate
    this.contestEl = this._makeEl({
      position:  'absolute',
      top:       '60px',
      left:      '20px',
      fontSize:  '12px',
      color:     PAL.orangeCss,
    });

    // Bell state: top-right — the single biggest "is anything happening".
    this.bellEl = this._makeEl({
      position:  'absolute',
      top:       '16px',
      right:     '20px',
      fontSize:  '15px',
      fontWeight:'bold',
      letterSpacing: '1px',
      textAlign: 'right',
      whiteSpace:'nowrap',
    });

    // Score-by-kind tally: right side, under bell state.
    this.tallyEl = this._makeEl({
      position:  'absolute',
      top:       '44px',
      right:     '20px',
      fontSize:  '10px',
      lineHeight:'14px',
      color:     PAL.dimCss,
      textAlign: 'right',
      whiteSpace:'nowrap',
    });

    // Axial field bar: bottom-center. Shows bell.x in [−GATE_X,+GATE_X];
    // target ring cap highlighted in the possessing team's colour.
    this.fieldEl = this._makeEl({
      position:  'absolute',
      bottom:    '18px',
      left:      '50%',
      transform: 'translateX(-50%)',
      width:     '420px',
      height:    '14px',
      background: PAL.bgCss,
      border:    `1px solid ${PAL.dimCss}`,
      boxSizing: 'border-box',
    });
    // midline tick
    const mid = document.createElement('div');
    Object.assign(mid.style, {
      position:  'absolute',
      left:      '50%',
      top:       '0',
      width:     '1px',
      height:    '100%',
      background: PAL.dimCss,
    } as Partial<CSSStyleDeclaration>);
    this.fieldEl.appendChild(mid);
    // highlighted target ring cap (positioned each frame)
    this.fieldRingHi = document.createElement('div');
    Object.assign(this.fieldRingHi.style, {
      position:  'absolute',
      top:       '-2px',
      width:     '6px',
      height:    '18px',
      background: PAL.cyanCss,
      boxShadow: `0 0 6px ${PAL.cyanCss}`,
    } as Partial<CSSStyleDeclaration>);
    this.fieldEl.appendChild(this.fieldRingHi);
    // bell marker
    this.fieldBellEl = document.createElement('div');
    Object.assign(this.fieldBellEl.style, {
      position:  'absolute',
      top:       '50%',
      width:     '8px',
      height:    '8px',
      borderRadius: '50%',
      background: PAL.paperCss,
      transform: 'translate(-50%,-50%)',
      transition: 'left 0.08s linear',
    } as Partial<CSSStyleDeclaration>);
    this.fieldEl.appendChild(this.fieldBellEl);

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

    // Match message: bottom-center (just above the field bar)
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
      this.scoreEl, this.inningEl, this.driveEl,
      this.pipsEl, this.gateEl, this.contestEl,
      this.bellEl, this.tallyEl,
      this.fieldEl, this.truenessEl, this.messageEl,
    );

    // Play-by-play ribbon lives in the (pointer-events:none) panel.
    this.pbp = new PlayByPlay(panel);

    root.style.position = 'relative';
    root.appendChild(panel);
    root.appendChild(this.canvas);
  }

  // ── Main render (FROZEN signature — must not change) ────────────────────

  render(sim: SimState, match: MatchState, input: InputView): void {
    this._detectTransitions(sim, match);
    this._updateDOM(sim, match);
    this._paintCanvas(sim, match, input);
  }

  // ── Transition detection → play-by-play + tally ────────────────────────

  private _detectTransitions(sim: SimState, match: MatchState): void {
    // Score change → a point was scored. Parse the score message which the
    // state machine sets as `${KIND} +${pts} (${team})` or
    // `GROUND +1 (${team})`.
    const dHome = match.scoreHome - this.lastScoreHome;
    const dAway = match.scoreAway - this.lastScoreAway;
    if (dHome !== 0 || dAway !== 0) {
      const side: 'home' | 'away' = dHome > 0 ? 'home' : 'away';
      const pts = dHome > 0 ? dHome : dAway;
      const kind = this._parseKind(match.message);
      if (kind) this.kindTally[side][kind] += 1;
      const tag = kind ? kind.toUpperCase() : 'SCORE';
      this._emit({
        tag,
        text: `${side.toUpperCase()} +${pts}  (${match.scoreHome}-${match.scoreAway})`,
        side,
        seq: ++this.seq,
      });
    }

    // Message transitions for non-scoring notable events. Dedup on the
    // message string so a held message across frames only fires once.
    const msg = match.message;
    if (msg && msg !== this.lastMessage) {
      const score = (dHome !== 0 || dAway !== 0);
      if (!score) {
        const ev = this._classifyMessage(msg);
        if (ev) this._emit(ev);
      }
    }

    // Inning rollover (independent of message; robust signal).
    if (match.inning !== this.lastInning && !match.spine) {
      this._emit({
        tag: 'INNING',
        text: `Inning ${match.inning} begins`,
        side: 'neutral',
        seq: ++this.seq,
      });
    }

    // Phase → spine / final milestones.
    if (match.phase !== this.lastPhase) {
      if (match.phase === 'spine') {
        this._emit({ tag: 'SPINE', text: 'Sudden-death spine', side: 'neutral', seq: ++this.seq });
      } else if (match.phase === 'final' && match.winner) {
        this._emit({
          tag: 'FINAL',
          text: `${match.winner.toUpperCase()} wins ${match.scoreHome}-${match.scoreAway}`,
          side: match.winner,
          seq: ++this.seq,
        });
      }
    }

    // Contest start (edge), independent of message wording.
    const contestOn = !!match.contest;
    if (contestOn && !this.lastContestOn) {
      this._emit({ tag: 'CONTEST', text: 'Bell contested', side: 'neutral', seq: ++this.seq });
    }

    this.lastMessage   = msg;
    this.lastScoreHome = match.scoreHome;
    this.lastScoreAway = match.scoreAway;
    this.lastInning    = match.inning;
    this.lastPhase     = match.phase;
    this.lastContestOn = contestOn;
    void sim;
  }

  private _parseKind(msg: string): ScoreKind | null {
    const m = msg.match(/^([A-Z]+)\s*\+/);
    if (!m) return null;
    const k = m[1].toLowerCase();
    return (KIND_ORDER as string[]).includes(k) ? (k as ScoreKind) : null;
  }

  private _classifyMessage(msg: string): PbpEvent | null {
    const lower = msg.toLowerCase();
    let tag = 'PLAY';
    let side: 'home' | 'away' | 'neutral' = 'neutral';
    if (lower.startsWith('turnover')) {
      tag = 'TURNOVER';
      if (lower.includes('home')) side = 'home';
      else if (lower.includes('away')) side = 'away';
    } else if (lower.startsWith('foul')) {
      tag = 'FOUL';
    } else if (lower.startsWith('missed')) {
      tag = 'MISS';
    } else if (lower.startsWith('gate cleared')) {
      tag = 'GATE';
    } else if (lower.startsWith('contest')) {
      tag = 'CONTEST';
    } else if (lower.startsWith('bell through')) {
      tag = 'RING';
    } else {
      // Skip noise we already surface via dedicated readouts/edges.
      return null;
    }
    return { tag, text: msg, side, seq: ++this.seq };
  }

  private _emit(ev: PbpEvent): void {
    this.pbp.push(ev);
  }

  // ── DOM update ─────────────────────────────────────────────────────────

  private _updateDOM(sim: SimState, match: MatchState): void {
    const homeTint = hex(PAL.cyan);
    const awayTint = hex(PAL.orange);

    // Score — home tinted cyan, away tinted orange
    this.scoreEl.innerHTML =
      `<span style="color:${homeTint}">HOME ${match.scoreHome}</span>` +
      ` <span style="color:${PAL.dimCss}">–</span> ` +
      `<span style="color:${awayTint}">${match.scoreAway} AWAY</span>`;

    // Inning + phase
    const inningLabel = match.spine ? 'SPINE' : `INN ${match.inning}/9`;
    const phaseLabel  = match.phase.toUpperCase().replace('_', ' ');
    this.inningEl.textContent = `${inningLabel}  ·  ${phaseLabel}`;

    // ── Drive banner: possession + attack direction ────────────────────
    const poss = match.possession;
    const possColor = poss === 'home' ? homeTint : awayTint;
    const sign = attackSign(poss);
    const targetX: 1 | -1 = sign;
    // Which scoring kind does the attacked ring give? faithEnd labels the
    // Faith (=2) ring; the other ring is Free (=5).
    const targetSignTok: '+x' | '-x' = targetX === 1 ? '+x' : '-x';
    const ringLabel = match.faithEnd === targetSignTok ? 'FAITH' : 'FREE';
    const dirArrow  = sign === 1 ? '▶' : '◀';
    const dirAxis   = sign === 1 ? '+X' : '−X';
    // Render as "HOME ▶ +X FAITH" (away → "FREE −X ◀ AWAY") so the arrow
    // always points the way the bell should travel on screen.
    if (sign === 1) {
      this.driveEl.innerHTML =
        `<span style="color:${possColor}">${poss.toUpperCase()}</span>` +
        `<span style="color:${possColor}"> ${dirArrow} </span>` +
        `<span style="color:${PAL.paperCss}">${dirAxis} ${ringLabel}</span>`;
    } else {
      this.driveEl.innerHTML =
        `<span style="color:${PAL.paperCss}">${ringLabel} ${dirAxis}</span>` +
        `<span style="color:${possColor}"> ${dirArrow} </span>` +
        `<span style="color:${possColor}">${poss.toUpperCase()}</span>`;
    }

    // ── Throws-left pips ────────────────────────────────────────────────
    const PIP_ON  = '●';
    const PIP_OFF = '○';
    const t = match.cast.throwsLeft;
    const pips = [0, 1, 2]
      .map((i) => `<span style="color:${i < t ? possColor : PAL.dimCss}">${i < t ? PIP_ON : PIP_OFF}</span>`)
      .join(' ');
    this.pipsEl.innerHTML = `CAST ${pips}`;

    // ── Cast gate ───────────────────────────────────────────────────────
    this.gateEl.textContent = `GATE: ${match.cast.gate.toUpperCase()}`;

    // ── Contest count ───────────────────────────────────────────────────
    if (match.contest) {
      const c = match.contest;
      const countPips = [0, 1, 2].map((i) =>
        `<span style="color:${i < c.count ? PAL.orangeCss : PAL.dimCss}">◆</span>`,
      ).join(' ');
      this.contestEl.innerHTML = `CONTEST ${countPips} ${c.direction.toUpperCase()}`;
      this.contestEl.style.display = 'block';
    } else {
      this.contestEl.style.display = 'none';
    }

    // ── Bell state — the biggest "is anything happening" signal ─────────
    const heldBy = sim.bell.heldBy;
    let bellTxt: string;
    let bellCol: string;
    if (match.phase === 'contest' || match.contest) {
      bellTxt = '⚔ CONTEST';
      bellCol = PAL.orangeCss;
    } else if (match.phase === 'dead' || match.phase === 'set') {
      bellTxt = '○ DEAD · RE-CAST';
      bellCol = PAL.dimCss;
    } else if (match.phase === 'inning_break') {
      bellTxt = '— INNING BREAK';
      bellCol = PAL.dimCss;
    } else if (match.phase === 'final') {
      bellTxt = '■ FINAL';
      bellCol = PAL.paperCss;
    } else if (heldBy) {
      // Held — colour by the holding side if we can infer it.
      const holder = sim.players.find((p) => p.id === heldBy);
      const hside = holder?.team;
      bellCol = hside === 'home' ? homeTint : hside === 'away' ? awayTint : PAL.paperCss;
      const who = hside ? hside.toUpperCase() : heldBy;
      bellTxt = `● HELD · ${who}`;
    } else {
      bellTxt = '✦ IN FLIGHT';
      bellCol = PAL.paperCss;
    }
    this.bellEl.innerHTML = `<span style="color:${bellCol}">${bellTxt}</span>`;

    // ── Score-by-kind tally ─────────────────────────────────────────────
    const th = this.kindTally.home;
    const ta = this.kindTally.away;
    const anyKind = KIND_ORDER.some((k) => th[k] || ta[k]);
    if (anyKind) {
      const rowFor = (k: ScoreKind) => {
        const h = th[k], a = ta[k];
        if (!h && !a) return '';
        return `<div>` +
          `<span style="color:${PAL.dimCss}">${k.toUpperCase().padEnd(6)}</span> ` +
          `<span style="color:${homeTint}">${h}</span>` +
          `<span style="color:${PAL.dimCss}">·</span>` +
          `<span style="color:${awayTint}">${a}</span></div>`;
      };
      this.tallyEl.innerHTML = KIND_ORDER.map(rowFor).join('');
      this.tallyEl.style.display = 'block';
    } else {
      this.tallyEl.style.display = 'none';
    }

    // ── Axial field bar: bell.x ∈ [−GATE_X, +GATE_X] ───────────────────
    const W = this.fieldEl.clientWidth || 420;
    const bx = Math.max(-GATE_X, Math.min(GATE_X, sim.bell.p.x));
    const frac = (bx + GATE_X) / (2 * GATE_X); // 0 at −X ring, 1 at +X ring
    this.fieldBellEl.style.left = `${(frac * W).toFixed(1)}px`;
    this.fieldBellEl.style.background =
      heldBy ? possColor : PAL.paperCss;
    // Highlight the attacked ring cap (left edge for −X, right for +X).
    this.fieldRingHi.style.background = possColor;
    this.fieldRingHi.style.boxShadow = `0 0 6px ${possColor}`;
    this.fieldRingHi.style.left = targetX === 1 ? `${W - 6}px` : `0px`;

    // ── Trueness meter — shiver on clatter ─────────────────────────────
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
      fill.style.transform  = `scaleX(${chime}) translateX(${this.shiverOffset}px)`;
      fill.style.background  = chime > 0.6 ? PAL.cyanCss : PAL.orangeCss;
      fill.style.transition  = fell ? 'none' : 'transform 0.1s';
    }
    this.truenessEl.title = `bell trueness: ${(chime * 100).toFixed(0)}%`;

    // ── Match message ──────────────────────────────────────────────────
    this.messageEl.textContent = match.message;
  }

  // ── Canvas overlay (reticle, charge ring, arc ghost) ──────────────────

  private _paintCanvas(sim: SimState, match: MatchState, input: InputView): void {
    void sim; void match;
    const w  = this.canvas.width;
    const h  = this.canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    // ── Arc ghost ───────────────────────────────────────────────────────
    // World-space ghost projected via a stylistic scatter near reticle
    // centre as a charge-level indicator (orchestrator may wire a real
    // THREE.Vector3.project later — not required here).
    if (input.ghostArc.length > 1) {
      const n = input.ghostArc.length;
      for (let i = 1; i < n; i++) {
        const t    = i / n;
        const fade = 0.15 + 0.5 * (1 - t) * input.chargeLevel;
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
    if (reticle !== 'free') {
      let col: string;
      if      (reticle === 'anchor-lock')  col = PAL.cyanCss;
      else if (reticle === 'out_of_range') col = PAL.dimCss;
      else if (reticle === 'charging')     col = PAL.orangeCss;
      else                                 col = PAL.paperCss;

      const r = RETICLE_R;
      const GAP = 4;
      const LEN = 8;
      ctx.strokeStyle = col;
      ctx.lineWidth   = 1.5;
      ctx.beginPath(); ctx.moveTo(cx, cy - r - GAP); ctx.lineTo(cx, cy - r - GAP - LEN); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, cy + r + GAP); ctx.lineTo(cx, cy + r + GAP + LEN); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - r - GAP, cy); ctx.lineTo(cx - r - GAP - LEN, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx + r + GAP, cy); ctx.lineTo(cx + r + GAP + LEN, cy); ctx.stroke();

      arc(ctx, cx, cy, r + 2, 0, Math.PI * 2, col + '44', 1);
    } else {
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = PAL.dimCss + 'aa';
      ctx.fill();
    }

    // ── Charge ring ──────────────────────────────────────────────────────
    if (input.chargeLevel > 0.01) {
      const startAngle = -Math.PI / 2;
      const endAngle   = startAngle + Math.PI * 2 * input.chargeLevel;
      arc(ctx, cx, cy, CHARGE_R, 0, Math.PI * 2, PAL.dimCss + '44', 3);
      const chargeColor = input.chargeLevel > 0.85 ? PAL.orangeCss : PAL.cyanCss;
      arc(ctx, cx, cy, CHARGE_R, startAngle, endAngle, chargeColor, 3);

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

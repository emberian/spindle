// PlayByPlay — a stadium-ribbon style scrolling event log for RIG.
//
// Pure DOM, read-only. The HUD feeds it parsed events (it does NOT parse
// match state itself — HUD owns transition detection and debouncing). This
// component only owns presentation: a short fixed-height ribbon showing the
// last ~5 notable events, newest on top, gently fading older lines.
//
// On palette, monospace, pointer-events:none, never blocks the canvas.

import { PAL } from './palette';

/** A notable match event, already de-duped/debounced by the HUD. */
export interface PbpEvent {
  /** Short ALL-CAPS tag, e.g. 'FALL', 'TURNOVER', 'INNING', 'CONTEST'. */
  tag: string;
  /** Human-readable line body. */
  text: string;
  /** Side colour accent: 'home' (cyan), 'away' (orange) or 'neutral'. */
  side: 'home' | 'away' | 'neutral';
  /** Monotonic id for keying / dedupe by the caller. */
  seq: number;
}

const MAX_LINES = 5;

export class PlayByPlay {
  private wrap: HTMLElement;
  private rows: HTMLElement[] = [];
  private events: PbpEvent[] = [];

  constructor(parent: HTMLElement) {
    this.wrap = document.createElement('div');
    Object.assign(this.wrap.style, {
      position: 'absolute',
      left: '20px',
      bottom: '60px',
      width: '320px',
      display: 'flex',
      flexDirection: 'column-reverse', // newest pushed to the top visually
      gap: '2px',
      pointerEvents: 'none',
      fontFamily: 'monospace, "Space Mono", sans-serif',
      fontSize: '11px',
      lineHeight: '15px',
      letterSpacing: '0.5px',
    } as Partial<CSSStyleDeclaration>);

    for (let i = 0; i < MAX_LINES; i++) {
      const row = document.createElement('div');
      Object.assign(row.style, {
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        background: 'rgba(17,19,26,0.55)',
        borderLeft: `3px solid ${PAL.dimCss}`,
        padding: '1px 8px',
        opacity: '0',
        transition: 'opacity 0.25s',
      } as Partial<CSSStyleDeclaration>);
      this.rows.push(row);
      this.wrap.appendChild(row);
    }

    parent.appendChild(this.wrap);
  }

  /** Push a new event onto the ribbon. Caller guarantees it's notable+unique. */
  push(ev: PbpEvent): void {
    this.events.push(ev);
    if (this.events.length > MAX_LINES) this.events.shift();
    this._paint();
  }

  /** Re-render rows from the current event buffer. */
  private _paint(): void {
    for (let i = 0; i < MAX_LINES; i++) {
      const row = this.rows[i];
      // events[] is oldest..newest; row 0 (visually bottom via column-reverse
      // = top) should hold the newest. Map newest → first row.
      const ev = this.events[this.events.length - 1 - i];
      if (!ev) {
        row.style.opacity = '0';
        row.textContent = '';
        continue;
      }
      const accent =
        ev.side === 'home' ? PAL.cyanCss
        : ev.side === 'away' ? PAL.orangeCss
        : PAL.dimCss;
      // Older lines fade out so the eye tracks the freshest action.
      const fade = 1 - i * 0.18;
      row.style.opacity = String(Math.max(0.25, fade));
      row.style.borderLeftColor = accent;
      row.innerHTML =
        `<span style="color:${accent};font-weight:bold">${ev.tag}</span>` +
        `<span style="color:${PAL.paperCss}"> ${ev.text}</span>`;
    }
  }
}

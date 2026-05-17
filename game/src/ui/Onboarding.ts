// Onboarding — first-match overlay teaching the Coriolis loop.
//
// Shown once when a match begins. Dismissed by:
//   • Any keyboard key press
//   • Mouse click on the "Begin" button
//   • Calling .hide() from the orchestrator (e.g. on first fireLineAt)
//
// Voice: dossier / field-manual. Palette: canonical RIG colours.
// No gameplay logic — purely DOM. Zero dependencies on sim/input.

import { PAL } from './palette';

// ── Layout constants ──────────────────────────────────────────────────────────

const FONT_STACK = '"Space Mono", "Courier New", monospace';

// ── Onboarding ────────────────────────────────────────────────────────────────

export class Onboarding {
  private root:    HTMLElement;
  private overlay: HTMLDivElement | null = null;
  private onDismiss?: () => void;

  constructor(root: HTMLElement) {
    this.root = root;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  show(onDismiss?: () => void): void {
    if (this.overlay) return; // already visible
    this.onDismiss = onDismiss;
    this.overlay   = this._build();
    this.root.appendChild(this.overlay);

    // Auto-dismiss on any key press
    const onKey = (e: KeyboardEvent) => {
      // Ignore bare modifier presses — wait for a real key
      if (['Control','Alt','Meta','Shift','CapsLock'].includes(e.key)) return;
      this._dismiss();
      window.removeEventListener('keydown', onKey);
    };
    window.addEventListener('keydown', onKey);

    // Also auto-dismiss the moment the player interacts with the game
    // (a canvas pointerdown) — clicking is the natural first instinct.
    const onPtr = () => {
      this._dismiss();
      window.removeEventListener('pointerdown', onPtr, true);
    };
    window.addEventListener('pointerdown', onPtr, true);

    // Store cleanup ref so hide() can also remove the listeners
    (this.overlay as any)._cleanupKey = () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onPtr, true);
    };

    // Fade in
    requestAnimationFrame(() => {
      if (this.overlay) this.overlay.style.opacity = '1';
    });
  }

  hide(): void {
    this._dismiss();
  }

  // ── Private ───────────────────────────────────────────────────────────────

  private _dismiss(): void {
    if (!this.overlay) return;
    const el = this.overlay;

    // Clean up keydown listener if still registered
    if (typeof (el as any)._cleanupKey === 'function') {
      (el as any)._cleanupKey();
    }

    // Fade out then remove
    el.style.opacity    = '0';
    el.style.transition = 'opacity 0.4s ease';
    setTimeout(() => {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, 420);

    this.overlay = null;
    this.onDismiss?.();
  }

  private _build(): HTMLDivElement {
    // ── Backdrop ─────────────────────────────────────────────────────────
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
      position:       'absolute',
      inset:          '0',
      zIndex:         '100',
      display:        'flex',
      alignItems:     'center',
      justifyContent: 'center',
      background:     'rgba(17,19,26,0.88)',
      backdropFilter: 'blur(3px)',
      opacity:        '0',
      transition:     'opacity 0.3s ease',
      // The backdrop must NOT eat input — it was a full-screen z=100 layer
      // that swallowed every canvas click, so the game was uncontrollable
      // until dismissed. Clicks now pass straight through to the canvas;
      // only the card itself is interactive.
      pointerEvents:  'none',
      fontFamily:     FONT_STACK,
    });

    // ── Card ─────────────────────────────────────────────────────────────
    const card = document.createElement('div');
    Object.assign(card.style, {
      position:     'relative',
      // NOTE: deliberately NOT pointer-events:auto. The whole overlay is
      // click-through so it can never eat a game click; the global keydown/
      // pointerdown auto-dismiss (in show()) closes it on the player's first
      // interaction, and that same click still reaches the canvas.
      maxWidth:     '480px',
      width:        '90%',
      padding:      '36px 40px 32px',
      background:   PAL.bgCss,
      border:       `1px solid ${PAL.dimCss}44`,
      boxShadow:    `0 0 40px ${PAL.cyanCss}18, 0 2px 24px rgba(0,0,0,0.6)`,
      color:        PAL.paperCss,
    });

    // ── Accent line (top) ─────────────────────────────────────────────────
    const topLine = document.createElement('div');
    Object.assign(topLine.style, {
      position:   'absolute',
      top:        '0',
      left:       '0',
      right:      '0',
      height:     '2px',
      background: `linear-gradient(90deg, ${PAL.cyanCss} 0%, ${PAL.dimCss}00 100%)`,
    });
    card.appendChild(topLine);

    // ── Header ────────────────────────────────────────────────────────────
    const header = document.createElement('div');
    Object.assign(header.style, {
      fontSize:      '10px',
      letterSpacing: '3px',
      color:         PAL.dimCss,
      marginBottom:  '8px',
      textTransform: 'uppercase',
    });
    header.textContent = 'FIELD MANUAL — RIGGER INDUCTION';
    card.appendChild(header);

    const title = document.createElement('h1');
    Object.assign(title.style, {
      margin:        '0 0 24px',
      fontSize:      '18px',
      fontWeight:    'normal',
      color:         PAL.cyanCss,
      letterSpacing: '1px',
      lineHeight:    '1.4',
    });
    title.textContent = 'THE CALM HAS NO FLOOR AND NO CLOCK.';
    card.appendChild(title);

    // ── Body lines ────────────────────────────────────────────────────────
    const body = document.createElement('div');
    Object.assign(body.style, {
      fontSize:     '13px',
      lineHeight:   '1.85',
      color:        PAL.paperCss,
      marginBottom: '28px',
    });

    type BodyLine = { text: string; accent?: string; after?: string };
    const lines: BodyLine[] = [
      { text: 'You are the cyan rigger under the ', accent: '▼ marker' },
      { text: 'You cannot fly. You move by line.' },
      { text: '' },
      { text: 'Hold ', accent: 'G', after: ' — auto-grapple toward the ring.' },
      { text: 'No aiming: it hauls you goal-ward.' },
      { text: '(RIGHT mouse = manual grapple, heavy snap.)' },
      { text: '' },
      { text: 'When you hold the bell the game ', accent: 'solves the shot' },
      { text: 'for you: a bold arc shows it thread the ring.' },
      { text: 'When it says ', accent: 'SHOT READY', after: ', press & release' },
      { text: 'LEFT to launch. No shot? Hold G to get closer.' },
    ];

    for (const ln of lines) {
      const p = document.createElement('p');
      Object.assign(p.style, { margin: '0', minHeight: '1em' });

      if (ln.text === '') {
        p.style.marginTop = '4px';
      } else if (ln.accent) {
        // Build mixed-colour line: prefix + accent span + optional suffix
        p.appendChild(document.createTextNode(ln.text));
        const span = document.createElement('span');
        Object.assign(span.style, { color: PAL.cyanCss });
        span.textContent = ln.accent;
        p.appendChild(span);
        if (ln.after) {
          p.appendChild(document.createTextNode(ln.after));
        }
      } else {
        p.textContent = ln.text;
      }
      body.appendChild(p);
    }
    card.appendChild(body);

    // ── Controls strip ────────────────────────────────────────────────────
    const controls = document.createElement('div');
    Object.assign(controls.style, {
      display:        'grid',
      gridTemplateColumns: '1fr 1fr',
      gap:            '4px 16px',
      fontSize:       '11px',
      color:          PAL.dimCss,
      marginBottom:   '28px',
      borderTop:      `1px solid ${PAL.dimCss}33`,
      paddingTop:     '16px',
    });

    const bindings: [string, string][] = [
      ['G (hold)',  'Auto-grapple toward the ring — no aiming'],
      ['LEFT MB',   'Launch the SOLVED shot (when SHOT READY)'],
      ['RIGHT MB',  'Manual grapple (heavy snap) · release = let go'],
      ['W / S',     'Reel line in / out (fine control)'],
      ['A / D',     'Spin (curl the bell)'],
      ['SHIFT',     'Push off a surface'],
      ['SPACE',     'Thrumbler nudge'],
    ];
    for (const [key, desc] of bindings) {
      const keyEl = document.createElement('span');
      Object.assign(keyEl.style, {
        color:         PAL.paperCss,
        textAlign:     'right',
        paddingRight:  '8px',
      });
      keyEl.textContent = key;

      const descEl = document.createElement('span');
      descEl.textContent = desc;

      controls.appendChild(keyEl);
      controls.appendChild(descEl);
    }
    card.appendChild(controls);

    // ── Begin button ──────────────────────────────────────────────────────
    const btn = document.createElement('button');
    Object.assign(btn.style, {
      display:       'block',
      width:         '100%',
      padding:       '10px 0',
      background:    'transparent',
      border:        `1px solid ${PAL.cyanCss}`,
      color:         PAL.cyanCss,
      fontFamily:    FONT_STACK,
      fontSize:      '12px',
      letterSpacing: '2px',
      cursor:        'pointer',
      transition:    'background 0.15s, color 0.15s',
      outline:       'none',
      textTransform: 'uppercase',
    });
    btn.textContent = 'Begin — or press any key';

    btn.addEventListener('mouseover', () => {
      btn.style.background = PAL.cyanCss;
      btn.style.color      = PAL.bgCss;
    });
    btn.addEventListener('mouseout', () => {
      btn.style.background = 'transparent';
      btn.style.color      = PAL.cyanCss;
    });
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      this._dismiss();
    });
    card.appendChild(btn);

    // ── Footnote ──────────────────────────────────────────────────────────
    const footnote = document.createElement('div');
    Object.assign(footnote.style, {
      marginTop:     '14px',
      fontSize:      '10px',
      color:         PAL.dimCss,
      textAlign:     'center',
      letterSpacing: '1px',
    });
    footnote.textContent = 'MOVE YOUR MOUSE TO AIM  ·  CLICK TO LOCK CURSOR';
    card.appendChild(footnote);

    overlay.appendChild(card);
    return overlay;
  }
}

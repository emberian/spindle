// SpectateControls — unobtrusive in-match overlay for AI-vs-AI spectating.
//
// Renders a fixed bottom-right control bar with:
//   • Speed buttons: 0.5× / 1× / 2× / 4× (calls onSpeed; active one highlighted)
//   • CAM button (onCycleCam)
//   • EXIT button (onExit)
//
// Does not block the canvas; pointer-events are restricted to the bar element.
// Optional keyboard shortcuts: [ (slow), ] (fast), C (cam), Escape (exit).
// Does NOT preventDefault on WASD / Shift / Space / arrows.
//
// Multiple show/hide cycles are safe; all listeners are removed on hide().

import { PAL } from './palette';

// ── Style constants ────────────────────────────────────────────────────────────

const FF = `'Courier New', 'Menlo', 'Consolas', monospace, sans-serif`;

const SPEED_STEPS: number[] = [0.5, 1, 2, 4];

// Keys the game owns — never preventDefault on these
const GAME_KEYS = new Set([
  'w', 'a', 's', 'd',
  'W', 'A', 'S', 'D',
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
  'Shift', ' ',
]);

// ── CSS injection (one-time, idempotent) ───────────────────────────────────────

const STYLE_ID = '__rig_spectate_controls_css';

function injectCSS(): void {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement('style');
  s.id = STYLE_ID;
  s.textContent = `
/* ── RIG SpectateControls ──────────────────────────────────────────── */

#rig-spec-controls {
  position: absolute;
  bottom: 20px;
  right: 20px;
  z-index: 20;
  display: none;
  align-items: center;
  gap: 6px;
  background: ${PAL.bgCss}cc;
  border: 1px solid ${PAL.dimCss}44;
  border-radius: 3px;
  padding: 6px 10px;
  font-family: ${FF};
  font-size: 11px;
  letter-spacing: 0.18em;
  user-select: none;
  /* Only intercept pointer events on the bar itself, not the canvas behind */
  pointer-events: auto;
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
  opacity: 0.72;
  transition: opacity 0.2s;
}

#rig-spec-controls:hover {
  opacity: 1;
}

/* Thin separator between groups */
.rig-sc-sep {
  width: 1px;
  height: 18px;
  background: ${PAL.dimCss}44;
  flex-shrink: 0;
}

/* Base button style */
.rig-sc-btn {
  background: transparent;
  border: 1px solid ${PAL.dimCss}55;
  color: ${PAL.dimCss};
  font-family: ${FF};
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 4px 9px;
  cursor: pointer;
  border-radius: 2px;
  transition: background 0.1s, color 0.1s, border-color 0.1s;
  line-height: 1;
}

.rig-sc-btn:hover {
  border-color: ${PAL.dimCss}99;
  color: ${PAL.paperCss};
  background: ${PAL.dimCss}18;
}

/* Active speed button */
.rig-sc-btn.speed-active {
  border-color: ${PAL.cyanCss};
  color: ${PAL.cyanCss};
  background: ${PAL.cyanCss}18;
}

/* CAM button */
.rig-sc-btn.cam {
  color: ${PAL.dimCss};
}

.rig-sc-btn.cam:hover {
  color: ${PAL.paperCss};
}

/* EXIT button */
.rig-sc-btn.exit {
  border-color: ${PAL.orangeCss}66;
  color: ${PAL.orangeCss}cc;
}

.rig-sc-btn.exit:hover {
  background: ${PAL.orangeCss}18;
  border-color: ${PAL.orangeCss};
  color: ${PAL.orangeCss};
}

/* Speed label prefix */
.rig-sc-label {
  font-size: 9px;
  letter-spacing: 0.26em;
  color: ${PAL.dimCss}99;
  text-transform: uppercase;
  padding-right: 2px;
}
`;
  document.head.appendChild(s);
}

// ── SpectateControls ───────────────────────────────────────────────────────────

export class SpectateControls {
  private bar: HTMLElement;
  private speedBtns: Map<number, HTMLButtonElement> = new Map();
  private activeSpeed: number = 1;

  // Keyboard listener — stored so we can remove it on hide()
  private _keyListener: ((e: KeyboardEvent) => void) | null = null;

  // Per-show handlers
  private handlers: {
    onSpeed: (mult: number) => void;
    onCycleCam: () => void;
    onExit: () => void;
  } | null = null;

  constructor(root: HTMLElement) {
    injectCSS();

    // ── Bar element ────────────────────────────────────────────────────────
    this.bar = document.createElement('div');
    this.bar.id = 'rig-spec-controls';

    // Speed label prefix
    const speedLabel = document.createElement('span');
    speedLabel.className = 'rig-sc-label';
    speedLabel.textContent = 'SPD';
    this.bar.appendChild(speedLabel);

    // Speed buttons
    for (const mult of SPEED_STEPS) {
      const btn = document.createElement('button');
      btn.className = 'rig-sc-btn';
      btn.textContent = mult === 0.5 ? '½×' : `${mult}×`;
      btn.title = `${mult}× speed`;
      btn.addEventListener('click', () => {
        this._applySpeed(mult);
        this.handlers?.onSpeed(mult);
      });
      this.speedBtns.set(mult, btn);
      this.bar.appendChild(btn);
    }

    // Separator
    const sep1 = document.createElement('div');
    sep1.className = 'rig-sc-sep';
    this.bar.appendChild(sep1);

    // CAM button
    const camBtn = document.createElement('button');
    camBtn.className = 'rig-sc-btn cam';
    camBtn.textContent = 'CAM';
    camBtn.title = 'Cycle cinematic camera angle (C)';
    camBtn.addEventListener('click', () => {
      this.handlers?.onCycleCam();
    });
    this.bar.appendChild(camBtn);

    // Separator
    const sep2 = document.createElement('div');
    sep2.className = 'rig-sc-sep';
    this.bar.appendChild(sep2);

    // EXIT button
    const exitBtn = document.createElement('button');
    exitBtn.className = 'rig-sc-btn exit';
    exitBtn.textContent = 'EXIT';
    exitBtn.title = 'Stop watching and return (Esc)';
    exitBtn.addEventListener('click', () => {
      if (!this.handlers) return;
      const cb = this.handlers.onExit; // capture before hide()
      this.hide();
      cb();
    });
    this.bar.appendChild(exitBtn);

    root.style.position = 'relative';
    root.appendChild(this.bar);
  }

  // ── Public API ────────────────────────────────────────────────────────────

  show(handlers: {
    onSpeed: (mult: number) => void;
    onCycleCam: () => void;
    onExit: () => void;
  }): void {
    this.handlers = handlers;
    this.activeSpeed = 1;
    this._applySpeed(1); // reset to 1× visually

    this.bar.style.display = 'flex';

    // Install keyboard shortcuts
    const listener = (e: KeyboardEvent) => this._onKey(e);
    this._keyListener = listener;
    window.addEventListener('keydown', listener, { capture: false });
  }

  hide(): void {
    this.bar.style.display = 'none';
    this.handlers = null;

    if (this._keyListener) {
      window.removeEventListener('keydown', this._keyListener, { capture: false });
      this._keyListener = null;
    }
  }

  /** Reflect the active speed in the UI (orchestrator may call this externally). */
  setSpeed(mult: number): void {
    this._applySpeed(mult);
  }

  // ── Internal helpers ──────────────────────────────────────────────────────

  private _applySpeed(mult: number): void {
    this.activeSpeed = mult;
    for (const [step, btn] of this.speedBtns) {
      btn.classList.toggle('speed-active', step === mult);
    }
  }

  private _onKey(e: KeyboardEvent): void {
    // Never steal game-owned keys
    if (GAME_KEYS.has(e.key)) return;

    switch (e.key) {
      case '[': {
        // Step speed down
        const idx = SPEED_STEPS.indexOf(this.activeSpeed);
        if (idx > 0) {
          const next = SPEED_STEPS[idx - 1]!;
          this._applySpeed(next);
          this.handlers?.onSpeed(next);
        }
        break;
      }
      case ']': {
        // Step speed up
        const idx = SPEED_STEPS.indexOf(this.activeSpeed);
        if (idx < SPEED_STEPS.length - 1) {
          const next = SPEED_STEPS[idx + 1]!;
          this._applySpeed(next);
          this.handlers?.onSpeed(next);
        }
        break;
      }
      case 'c':
      case 'C': {
        this.handlers?.onCycleCam();
        break;
      }
      case 'Escape': {
        if (!this.handlers) break;
        const cb = this.handlers.onExit; // capture before hide()
        this.hide();
        cb();
        break;
      }
    }
  }
}

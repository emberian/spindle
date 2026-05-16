// TitleScreen — RIG title + team-select DOM overlay.
// Pure DOM/CSS; no Three.js, no gameplay logic.
// Append to a root element, call show(onPick) to display, hide() to dismiss.

import { TEAMS } from '../league/teams';
import type { Franchise } from '../league/teams';
import { PAL } from './palette';

// ── Layout / style constants ───────────────────────────────────────────────

const FF = `'Courier New', 'Menlo', 'Consolas', monospace, sans-serif`;

const CONF_LABELS: Record<string, string> = {
  sol:  'SOL — FIRST SPIN',
  near: 'NEAR REACH — FIRST LIGHT',
  far:  'FAR REACH & THE BELT — DEEP JUMP',
};

// ── CSS injection (one-time, idempotent) ───────────────────────────────────

const STYLE_ID = '__rig_title_css';

function injectCSS(): void {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement('style');
  s.id = STYLE_ID;
  s.textContent = `
/* ── RIG TitleScreen ──────────────────────────────────────────── */

#rig-title-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: ${PAL.bgCss};
  color: ${PAL.paperCss};
  font-family: ${FF};
  display: flex;
  flex-direction: column;
  align-items: stretch;
  overflow: hidden;
}

/* ── Header / wordmark ───────────────────────────────────────── */

#rig-title-header {
  flex: 0 0 auto;
  text-align: center;
  padding: 44px 24px 28px;
  border-bottom: 1px solid ${PAL.dimCss}44;
  user-select: none;
}

#rig-wordmark {
  font-size: clamp(52px, 10vw, 96px);
  font-weight: 900;
  letter-spacing: 0.25em;
  line-height: 1;
  /* Cyan/orange split — left half cyan, right half orange */
  background: linear-gradient(
    90deg,
    ${PAL.cyanCss} 0%,
    ${PAL.cyanCss} 50%,
    ${PAL.orangeCss} 50%,
    ${PAL.orangeCss} 100%
  );
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

#rig-tagline {
  margin-top: 10px;
  font-size: clamp(11px, 1.8vw, 14px);
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: ${PAL.dimCss};
}

#rig-select-label {
  margin-top: 20px;
  font-size: clamp(10px, 1.4vw, 12px);
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: ${PAL.cyanCss}bb;
}

/* ── Scrollable grid body ─────────────────────────────────────── */

#rig-title-body {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 24px 20px 100px;
  /* Thin scrollbar on webkit */
  scrollbar-width: thin;
  scrollbar-color: ${PAL.dimCss}55 transparent;
}

#rig-title-body::-webkit-scrollbar { width: 6px; }
#rig-title-body::-webkit-scrollbar-track { background: transparent; }
#rig-title-body::-webkit-scrollbar-thumb { background: ${PAL.dimCss}55; border-radius: 3px; }

/* ── Conference section ───────────────────────────────────────── */

.rig-conf-section {
  margin-bottom: 32px;
}

.rig-conf-heading {
  font-size: 10px;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: ${PAL.dimCss};
  padding: 6px 0 12px;
  border-bottom: 1px solid ${PAL.dimCss}33;
  margin-bottom: 14px;
}

/* ── Team card grid ───────────────────────────────────────────── */

.rig-team-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 10px;
}

/* ── Individual team card ─────────────────────────────────────── */

.rig-card {
  background: ${PAL.bgCss};
  border: 1px solid ${PAL.dimCss}44;
  border-radius: 2px;
  padding: 12px 12px 10px;
  cursor: pointer;
  transition: border-color 0.1s, background 0.1s;
  position: relative;
  outline: none;
  user-select: none;
}

.rig-card:hover {
  border-color: ${PAL.dimCss}99;
  background: #191c25;
}

.rig-card:focus-visible {
  border-color: ${PAL.cyanCss};
  box-shadow: 0 0 0 1px ${PAL.cyanCss}44;
}

.rig-card.selected {
  border-color: ${PAL.cyanCss};
  background: #0f1e22;
  box-shadow: 0 0 0 1px ${PAL.cyanCss}66, inset 0 0 24px ${PAL.cyanCss}0a;
}

/* Jump seed badge — top-right corner */
.rig-seed-badge {
  position: absolute;
  top: 7px;
  right: 8px;
  font-size: 9px;
  letter-spacing: 0.12em;
  color: ${PAL.orangeCss};
  background: ${PAL.orangeCss}18;
  border: 1px solid ${PAL.orangeCss}55;
  border-radius: 2px;
  padding: 1px 5px;
  font-weight: 700;
}

/* Color swatches row */
.rig-swatches {
  display: flex;
  gap: 5px;
  margin-bottom: 8px;
}

.rig-swatch {
  width: 18px;
  height: 10px;
  border-radius: 1px;
  border: 1px solid rgba(255,255,255,0.08);
  flex: 0 0 auto;
}

/* Team name */
.rig-card-name {
  font-size: clamp(11px, 1.5vw, 13px);
  line-height: 1.3;
  color: ${PAL.paperCss};
  font-weight: 700;
  letter-spacing: 0.04em;
  margin-bottom: 5px;
  padding-right: 28px; /* clear seed badge */
}

/* Meta row: faction · record */
.rig-card-meta {
  font-size: 10px;
  letter-spacing: 0.08em;
  color: ${PAL.dimCss};
  display: flex;
  gap: 8px;
  align-items: baseline;
}

.rig-card-faction {
  color: ${PAL.dimCss}cc;
}

.rig-card-record {
  color: ${PAL.dimCss}99;
}

/* ── Sticky footer / confirm bar ──────────────────────────────── */

#rig-title-footer {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(to top, ${PAL.bgCss} 60%, transparent);
  pointer-events: none;
  z-index: 2;
}

#rig-confirm-btn {
  pointer-events: auto;
  background: transparent;
  border: 1px solid ${PAL.cyanCss};
  color: ${PAL.cyanCss};
  font-family: ${FF};
  font-size: 13px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  padding: 12px 36px;
  cursor: pointer;
  border-radius: 2px;
  transition: background 0.12s, color 0.12s, opacity 0.12s;
  opacity: 0.3;
}

#rig-confirm-btn:not(:disabled):hover {
  background: ${PAL.cyanCss}22;
}

#rig-confirm-btn.ready {
  opacity: 1;
}

#rig-confirm-btn:disabled {
  cursor: default;
}
`;
  document.head.appendChild(s);
}

// ── TitleScreen ───────────────────────────────────────────────────────────

export class TitleScreen {
  private overlay: HTMLElement;
  private confirmBtn: HTMLButtonElement;
  private selected: Franchise | null = null;
  private cards: Map<string, HTMLElement> = new Map();
  private onPick: ((f: Franchise) => void) | null = null;

  constructor(root: HTMLElement) {
    injectCSS();

    // ── Outer overlay ───────────────────────────────────────────────────
    this.overlay = document.createElement('div');
    this.overlay.id = 'rig-title-overlay';
    this.overlay.style.display = 'none';

    // ── Header ──────────────────────────────────────────────────────────
    const header = document.createElement('div');
    header.id = 'rig-title-header';

    const wordmark = document.createElement('div');
    wordmark.id = 'rig-wordmark';
    wordmark.textContent = 'RIG';

    const tagline = document.createElement('div');
    tagline.id = 'rig-tagline';
    tagline.textContent = 'the game played in the calm';

    const selectLabel = document.createElement('div');
    selectLabel.id = 'rig-select-label';
    selectLabel.textContent = 'select a franchise to enter the jump';

    header.append(wordmark, tagline, selectLabel);

    // ── Scrollable body with conference groups ──────────────────────────
    const body = document.createElement('div');
    body.id = 'rig-title-body';

    const conferences: Array<'sol' | 'near' | 'far'> = ['sol', 'near', 'far'];
    for (const conf of conferences) {
      const teams = TEAMS.filter((t) => t.conference === conf);
      const section = this._buildConferenceSection(conf, teams);
      body.appendChild(section);
    }

    // ── Footer / confirm button ─────────────────────────────────────────
    const footer = document.createElement('div');
    footer.id = 'rig-title-footer';

    this.confirmBtn = document.createElement('button');
    this.confirmBtn.id = 'rig-confirm-btn';
    this.confirmBtn.textContent = 'ENTER THE JUMP';
    this.confirmBtn.disabled = true;
    this.confirmBtn.addEventListener('click', () => this._confirm());

    footer.appendChild(this.confirmBtn);

    // ── Keyboard navigation ─────────────────────────────────────────────
    this.overlay.addEventListener('keydown', (e) => this._onKey(e));

    // ── Assemble ────────────────────────────────────────────────────────
    this.overlay.append(header, body, footer);
    root.style.position = 'relative';
    root.appendChild(this.overlay);
  }

  // ── Public API ───────────────────────────────────────────────────────────

  show(onPick: (f: Franchise) => void): void {
    this.onPick = onPick;
    this.selected = null;
    this._clearSelection();
    this.confirmBtn.disabled = true;
    this.confirmBtn.classList.remove('ready');
    this.overlay.style.display = 'flex';
    // Let layout settle before focusing
    requestAnimationFrame(() => {
      const first = this.overlay.querySelector<HTMLElement>('.rig-card');
      first?.focus();
    });
  }

  hide(): void {
    this.overlay.style.display = 'none';
    this.onPick = null;
  }

  // ── Conference section builder ────────────────────────────────────────────

  private _buildConferenceSection(
    conf: 'sol' | 'near' | 'far',
    teams: Franchise[],
  ): HTMLElement {
    const section = document.createElement('div');
    section.className = 'rig-conf-section';

    const heading = document.createElement('div');
    heading.className = 'rig-conf-heading';
    heading.textContent = CONF_LABELS[conf];
    section.appendChild(heading);

    const grid = document.createElement('div');
    grid.className = 'rig-team-grid';

    for (const team of teams) {
      const card = this._buildCard(team);
      this.cards.set(team.id, card);
      grid.appendChild(card);
    }

    section.appendChild(grid);
    return section;
  }

  // ── Card builder ─────────────────────────────────────────────────────────

  private _buildCard(team: Franchise): HTMLElement {
    const card = document.createElement('div');
    card.className = 'rig-card';
    card.tabIndex = 0;
    card.dataset.id = team.id;
    card.setAttribute('role', 'option');
    card.setAttribute('aria-selected', 'false');
    card.setAttribute('aria-label', team.name);

    // Jump seed badge
    if (team.seed !== null) {
      const badge = document.createElement('div');
      badge.className = 'rig-seed-badge';
      badge.textContent = `#${team.seed}`;
      card.appendChild(badge);
    }

    // Colour swatches
    const swatches = document.createElement('div');
    swatches.className = 'rig-swatches';
    for (const hex of team.colors) {
      const sw = document.createElement('div');
      sw.className = 'rig-swatch';
      sw.style.background = hex;
      sw.title = hex;
      swatches.appendChild(sw);
    }
    card.appendChild(swatches);

    // Name
    const name = document.createElement('div');
    name.className = 'rig-card-name';
    name.textContent = team.name;
    card.appendChild(name);

    // Meta: faction · record
    const meta = document.createElement('div');
    meta.className = 'rig-card-meta';

    const faction = document.createElement('span');
    faction.className = 'rig-card-faction';
    faction.textContent = team.faction;

    const record = document.createElement('span');
    record.className = 'rig-card-record';
    // Exhibition teams (0–0) show a dash
    if (team.record.w === 0 && team.record.l === 0) {
      record.textContent = 'exhibition';
    } else {
      record.textContent = `${team.record.w}–${team.record.l}`;
    }

    meta.append(faction, record);
    card.appendChild(meta);

    // Interaction
    card.addEventListener('click', () => this._select(team));
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        this._select(team);
      }
    });

    return card;
  }

  // ── Selection logic ───────────────────────────────────────────────────────

  private _select(team: Franchise): void {
    this._clearSelection();
    this.selected = team;

    const card = this.cards.get(team.id);
    if (card) {
      card.classList.add('selected');
      card.setAttribute('aria-selected', 'true');
      card.focus();
    }

    this.confirmBtn.disabled = false;
    this.confirmBtn.classList.add('ready');
  }

  private _clearSelection(): void {
    for (const [, card] of this.cards) {
      card.classList.remove('selected');
      card.setAttribute('aria-selected', 'false');
    }
  }

  private _confirm(): void {
    if (!this.selected || !this.onPick) return;
    const pick = this.selected;
    const cb = this.onPick; // hide() nulls this.onPick — capture first
    this.hide();
    cb(pick);
  }

  // ── Keyboard navigation ───────────────────────────────────────────────────

  private _onKey(e: KeyboardEvent): void {
    const allCards = Array.from(this.overlay.querySelectorAll<HTMLElement>('.rig-card'));
    if (allCards.length === 0) return;

    const focused = document.activeElement as HTMLElement | null;
    const idx = focused ? allCards.indexOf(focused) : -1;

    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown': {
        e.preventDefault();
        const next = allCards[(idx + 1) % allCards.length];
        next.focus();
        break;
      }
      case 'ArrowLeft':
      case 'ArrowUp': {
        e.preventDefault();
        const prev = allCards[(idx - 1 + allCards.length) % allCards.length];
        prev.focus();
        break;
      }
      case 'Enter': {
        if (this.selected && e.target === this.confirmBtn) {
          // button's own click handler fires
          break;
        }
        if (this.selected) {
          e.preventDefault();
          this._confirm();
        }
        break;
      }
      case 'Escape': {
        // Deselect
        this.selected = null;
        this._clearSelection();
        this.confirmBtn.disabled = true;
        this.confirmBtn.classList.remove('ready');
        break;
      }
    }
  }
}

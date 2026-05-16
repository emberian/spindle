// SpectateScreen — DOM overlay for setting up an AI-vs-AI watch session.
//
// Lets the user pick two franchises (home / away) from the 32 and either:
//   • WATCH MATCH    — runs a single cinematic AI-vs-AI match
//   • WATCH THE JUMP — simulates and watches the full bracket
//   • BACK           — return to the previous screen
//
// Pure DOM/CSS; no Three.js, no gameplay logic.
// Follows the capture-callback-before-hide pattern from TitleScreen.

import type { Franchise } from '../league/teams';
import { PAL } from './palette';

// ── Style constants ────────────────────────────────────────────────────────────

const FF = `'Courier New', 'Menlo', 'Consolas', monospace, sans-serif`;

const CONF_LABELS: Record<string, string> = {
  sol:  'SOL — FIRST SPIN',
  near: 'NEAR REACH — FIRST LIGHT',
  far:  'FAR REACH & THE BELT — DEEP JUMP',
};

// ── Augmented element type for franchise back-reference ──────────────────────

type TeamRowElement = HTMLElement & { __franchise: Franchise };

// ── CSS injection (one-time, idempotent) ───────────────────────────────────────

const STYLE_ID = '__rig_spectate_css';

function injectCSS(): void {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement('style');
  s.id = STYLE_ID;
  s.textContent = `
/* ── RIG SpectateScreen ────────────────────────────────────────────── */

#rig-spectate-overlay {
  position: absolute;
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

/* ── Header ─────────────────────────────────────────────────────── */

#rig-spectate-header {
  flex: 0 0 auto;
  padding: 28px 28px 18px;
  border-bottom: 1px solid ${PAL.dimCss}44;
  display: flex;
  align-items: baseline;
  gap: 20px;
  user-select: none;
}

#rig-spectate-title {
  font-size: clamp(20px, 4vw, 28px);
  font-weight: 900;
  letter-spacing: 0.3em;
  color: ${PAL.cyanCss};
}

#rig-spectate-subtitle {
  font-size: 10px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: ${PAL.dimCss};
}

/* ── Two-pane picker body ────────────────────────────────────────── */

#rig-spectate-body {
  flex: 1 1 auto;
  display: flex;
  flex-direction: row;
  gap: 0;
  overflow: hidden;
  min-height: 0;
}

/* ── Team list pane (shared by home and away) ────────────────────── */

.rig-spec-pane {
  flex: 1 1 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid ${PAL.dimCss}33;
}

.rig-spec-pane:last-child {
  border-right: none;
}

.rig-spec-pane-header {
  flex: 0 0 auto;
  padding: 10px 16px 8px;
  border-bottom: 1px solid ${PAL.dimCss}33;
  display: flex;
  align-items: center;
  gap: 10px;
  user-select: none;
}

.rig-spec-pane-label {
  font-size: 10px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  font-weight: 700;
}

.rig-spec-pane-label.home {
  color: ${PAL.cyanCss};
}

.rig-spec-pane-label.away {
  color: ${PAL.orangeCss};
}

.rig-spec-pane-chosen {
  font-size: 11px;
  letter-spacing: 0.06em;
  color: ${PAL.dimCss};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.rig-spec-pane-chosen.filled {
  color: ${PAL.paperCss};
}

.rig-spec-pane-scroll {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 8px 8px 100px;
  scrollbar-width: thin;
  scrollbar-color: ${PAL.dimCss}55 transparent;
}

.rig-spec-pane-scroll::-webkit-scrollbar { width: 4px; }
.rig-spec-pane-scroll::-webkit-scrollbar-track { background: transparent; }
.rig-spec-pane-scroll::-webkit-scrollbar-thumb { background: ${PAL.dimCss}55; border-radius: 2px; }

/* ── Conference section within pane ─────────────────────────────── */

.rig-spec-conf-section {
  margin-bottom: 18px;
}

.rig-spec-conf-heading {
  font-size: 9px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: ${PAL.dimCss};
  padding: 4px 6px 8px;
  border-bottom: 1px solid ${PAL.dimCss}22;
  margin-bottom: 6px;
}

/* ── Team row in pane ────────────────────────────────────────────── */

.rig-spec-team-row {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 7px 8px;
  cursor: pointer;
  border: 1px solid transparent;
  border-radius: 2px;
  margin-bottom: 2px;
  transition: background 0.08s, border-color 0.08s;
  user-select: none;
  outline: none;
}

.rig-spec-team-row:hover {
  background: #191c25;
  border-color: ${PAL.dimCss}44;
}

.rig-spec-team-row:focus-visible {
  border-color: ${PAL.dimCss}88;
}

.rig-spec-team-row.selected-home {
  background: #0f1e22;
  border-color: ${PAL.cyanCss};
  box-shadow: 0 0 0 1px ${PAL.cyanCss}33;
}

.rig-spec-team-row.selected-away {
  background: #221508;
  border-color: ${PAL.orangeCss};
  box-shadow: 0 0 0 1px ${PAL.orangeCss}33;
}

.rig-spec-swatch {
  width: 10px;
  height: 10px;
  border-radius: 1px;
  flex-shrink: 0;
  border: 1px solid rgba(255,255,255,0.08);
}

.rig-spec-seed {
  font-size: 9px;
  color: ${PAL.orangeCss};
  min-width: 18px;
  text-align: right;
  flex-shrink: 0;
  letter-spacing: 0.04em;
}

.rig-spec-name {
  flex: 1;
  font-size: 11px;
  letter-spacing: 0.04em;
  color: ${PAL.paperCss};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.rig-spec-faction {
  font-size: 9px;
  color: ${PAL.dimCss}aa;
  letter-spacing: 0.06em;
  flex-shrink: 0;
}

/* ── Footer bar ──────────────────────────────────────────────────── */

#rig-spectate-footer {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  background: linear-gradient(to top, ${PAL.bgCss} 70%, transparent);
  pointer-events: none;
  z-index: 2;
}

.rig-spec-btn {
  pointer-events: auto;
  background: transparent;
  font-family: ${FF};
  font-size: 12px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  padding: 11px 22px;
  cursor: pointer;
  border-radius: 2px;
  transition: background 0.12s, color 0.12s, opacity 0.12s;
}

.rig-spec-btn.primary {
  border: 1px solid ${PAL.cyanCss};
  color: ${PAL.cyanCss};
  opacity: 0.3;
}

.rig-spec-btn.primary:not(:disabled):hover {
  background: ${PAL.cyanCss}22;
}

.rig-spec-btn.primary.ready {
  opacity: 1;
}

.rig-spec-btn.primary:disabled {
  cursor: default;
}

.rig-spec-btn.secondary {
  border: 1px solid ${PAL.orangeCss};
  color: ${PAL.orangeCss};
}

.rig-spec-btn.secondary:hover {
  background: ${PAL.orangeCss}1a;
}

.rig-spec-btn.ghost {
  border: 1px solid ${PAL.dimCss}66;
  color: ${PAL.dimCss};
}

.rig-spec-btn.ghost:hover {
  border-color: ${PAL.dimCss}aa;
  color: ${PAL.paperCss};
}

/* ── VS divider (center column, fixed width) ─────────────────────── */

#rig-spectate-vs {
  flex: 0 0 56px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-left: 1px solid ${PAL.dimCss}33;
  border-right: 1px solid ${PAL.dimCss}33;
  gap: 12px;
  user-select: none;
  pointer-events: none;
}

#rig-spectate-vs-label {
  font-size: 16px;
  font-weight: 900;
  letter-spacing: 0.2em;
  color: ${PAL.dimCss}66;
}

.rig-spec-vs-swatch {
  width: 22px;
  height: 22px;
  border-radius: 2px;
  border: 1px solid ${PAL.dimCss}44;
}
`;
  document.head.appendChild(s);
}

// ── SpectateScreen ─────────────────────────────────────────────────────────────

export class SpectateScreen {
  private overlay: HTMLElement;
  private homeScroll: HTMLElement;
  private awayScroll: HTMLElement;
  private homeChosen: HTMLElement;
  private awayChosen: HTMLElement;
  private watchBtn: HTMLButtonElement;
  private jumpBtn: HTMLButtonElement;
  private backBtn: HTMLButtonElement;
  private vsHomeSwatch: HTMLElement;
  private vsAwaySwatch: HTMLElement;

  private selectedHome: Franchise | null = null;
  private selectedAway: Franchise | null = null;

  // per-show row element maps (rebuilt each show)
  private homeRows: Map<string, TeamRowElement> = new Map();
  private awayRows: Map<string, TeamRowElement> = new Map();

  // per-show handlers (captured then nulled on hide)
  private handlers: {
    onWatch: (home: Franchise, away: Franchise) => void;
    onWatchBracket: () => void;
    onBack: () => void;
  } | null = null;

  constructor(root: HTMLElement) {
    injectCSS();

    // ── Outer overlay ──────────────────────────────────────────────────────
    this.overlay = document.createElement('div');
    this.overlay.id = 'rig-spectate-overlay';
    this.overlay.style.display = 'none';

    // ── Header ─────────────────────────────────────────────────────────────
    const header = document.createElement('div');
    header.id = 'rig-spectate-header';

    const title = document.createElement('div');
    title.id = 'rig-spectate-title';
    title.textContent = 'SPECTATE';

    const subtitle = document.createElement('div');
    subtitle.id = 'rig-spectate-subtitle';
    subtitle.textContent = 'watch an AI-vs-AI match';

    header.append(title, subtitle);

    // ── Body: home pane | vs column | away pane ────────────────────────────
    const body = document.createElement('div');
    body.id = 'rig-spectate-body';

    // Home pane
    const homePane = this._buildPane('home');
    this.homeScroll = homePane.scroll;
    this.homeChosen = homePane.chosen;

    // VS divider
    const vsDivider = document.createElement('div');
    vsDivider.id = 'rig-spectate-vs';

    const vsLabel = document.createElement('div');
    vsLabel.id = 'rig-spectate-vs-label';
    vsLabel.textContent = 'VS';

    this.vsHomeSwatch = document.createElement('div');
    this.vsHomeSwatch.className = 'rig-spec-vs-swatch';
    this.vsHomeSwatch.style.background = PAL.bgCss;
    this.vsHomeSwatch.title = 'home';

    this.vsAwaySwatch = document.createElement('div');
    this.vsAwaySwatch.className = 'rig-spec-vs-swatch';
    this.vsAwaySwatch.style.background = PAL.bgCss;
    this.vsAwaySwatch.title = 'away';

    vsDivider.append(this.vsHomeSwatch, vsLabel, this.vsAwaySwatch);

    // Away pane
    const awayPane = this._buildPane('away');
    this.awayScroll = awayPane.scroll;
    this.awayChosen = awayPane.chosen;

    body.append(homePane.el, vsDivider, awayPane.el);

    // ── Footer ─────────────────────────────────────────────────────────────
    const footer = document.createElement('div');
    footer.id = 'rig-spectate-footer';

    this.backBtn = document.createElement('button');
    this.backBtn.className = 'rig-spec-btn ghost';
    this.backBtn.textContent = 'BACK';
    this.backBtn.addEventListener('click', () => this._back());

    this.jumpBtn = document.createElement('button');
    this.jumpBtn.className = 'rig-spec-btn secondary';
    this.jumpBtn.textContent = 'WATCH THE JUMP';
    this.jumpBtn.addEventListener('click', () => this._watchBracket());

    this.watchBtn = document.createElement('button');
    this.watchBtn.className = 'rig-spec-btn primary';
    this.watchBtn.textContent = 'WATCH MATCH';
    this.watchBtn.disabled = true;
    this.watchBtn.addEventListener('click', () => this._watch());

    footer.append(this.backBtn, this.jumpBtn, this.watchBtn);

    // ── Keyboard handler ───────────────────────────────────────────────────
    this.overlay.addEventListener('keydown', (e) => this._onKey(e));

    // ── Assemble ───────────────────────────────────────────────────────────
    this.overlay.append(header, body, footer);
    root.style.position = 'relative';
    root.appendChild(this.overlay);
  }

  // ── Public API ────────────────────────────────────────────────────────────

  show(
    franchises: Franchise[],
    handlers: {
      onWatch: (home: Franchise, away: Franchise) => void;
      onWatchBracket: () => void;
      onBack: () => void;
    },
  ): void {
    this.handlers = handlers;
    this.selectedHome = null;
    this.selectedAway = null;
    this.homeRows.clear();
    this.awayRows.clear();

    // Rebuild the team lists
    this._populatePane(this.homeScroll, this.homeRows, franchises, 'home');
    this._populatePane(this.awayScroll, this.awayRows, franchises, 'away');

    // Reset VS swatches
    this.vsHomeSwatch.style.background = PAL.bgCss;
    this.vsHomeSwatch.title = 'home';
    this.vsAwaySwatch.style.background = PAL.bgCss;
    this.vsAwaySwatch.title = 'away';

    this._refreshChosenLabels();
    this._refreshWatchBtn();

    this.overlay.style.display = 'flex';
    requestAnimationFrame(() => {
      const first = this.homeScroll.querySelector<HTMLElement>('.rig-spec-team-row');
      first?.focus();
    });
  }

  hide(): void {
    this.overlay.style.display = 'none';
    this.handlers = null;
    this.selectedHome = null;
    this.selectedAway = null;
  }

  // ── Pane builder ─────────────────────────────────────────────────────────

  private _buildPane(side: 'home' | 'away'): {
    el: HTMLElement;
    scroll: HTMLElement;
    chosen: HTMLElement;
  } {
    const el = document.createElement('div');
    el.className = 'rig-spec-pane';

    const paneHeader = document.createElement('div');
    paneHeader.className = 'rig-spec-pane-header';

    const label = document.createElement('div');
    label.className = `rig-spec-pane-label ${side}`;
    label.textContent = side === 'home' ? 'HOME' : 'AWAY';

    const chosen = document.createElement('div');
    chosen.className = 'rig-spec-pane-chosen';
    chosen.textContent = '— pick a team';

    paneHeader.append(label, chosen);

    const scroll = document.createElement('div');
    scroll.className = 'rig-spec-pane-scroll';
    scroll.setAttribute('role', 'listbox');
    scroll.setAttribute('aria-label', `${side} team`);

    el.append(paneHeader, scroll);
    return { el, scroll, chosen };
  }

  // ── Populate a pane scroll area with conference-grouped rows ──────────────

  private _populatePane(
    scroll: HTMLElement,
    rowMap: Map<string, TeamRowElement>,
    franchises: Franchise[],
    side: 'home' | 'away',
  ): void {
    scroll.innerHTML = '';

    const conferences: Array<'sol' | 'near' | 'far'> = ['sol', 'near', 'far'];
    for (const conf of conferences) {
      const group = franchises.filter((f) => f.conference === conf);
      if (group.length === 0) continue;

      const section = document.createElement('div');
      section.className = 'rig-spec-conf-section';

      const heading = document.createElement('div');
      heading.className = 'rig-spec-conf-heading';
      heading.textContent = CONF_LABELS[conf];
      section.appendChild(heading);

      for (const team of group) {
        const row = this._buildTeamRow(team, side);
        rowMap.set(team.id, row);
        section.appendChild(row);
      }

      scroll.appendChild(section);
    }
  }

  // ── Individual team row ───────────────────────────────────────────────────

  private _buildTeamRow(team: Franchise, side: 'home' | 'away'): TeamRowElement {
    const row = document.createElement('div') as unknown as TeamRowElement;
    row.__franchise = team;
    row.className = 'rig-spec-team-row';
    row.tabIndex = 0;
    row.dataset.id = team.id;
    row.setAttribute('role', 'option');
    row.setAttribute('aria-selected', 'false');
    row.setAttribute('aria-label', team.name);

    // Primary colour swatch
    const swatch = document.createElement('div');
    swatch.className = 'rig-spec-swatch';
    swatch.style.background = team.colors[0];

    // Seed badge
    const seedEl = document.createElement('div');
    seedEl.className = 'rig-spec-seed';
    seedEl.textContent = team.seed !== null ? `#${team.seed}` : '';

    // Name
    const nameEl = document.createElement('div');
    nameEl.className = 'rig-spec-name';
    nameEl.textContent = team.name;

    // Faction tag
    const factionEl = document.createElement('div');
    factionEl.className = 'rig-spec-faction';
    factionEl.textContent = team.faction;

    row.append(swatch, seedEl, nameEl, factionEl);

    row.addEventListener('click', () => this._selectTeam(team, side));
    row.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        this._selectTeam(team, side);
      }
    });

    return row;
  }

  // ── Selection logic ───────────────────────────────────────────────────────

  private _selectTeam(team: Franchise, side: 'home' | 'away'): void {
    if (side === 'home') {
      this.selectedHome = team;
      this._applySelectionStyle(this.homeRows, team.id, 'selected-home');
      this.vsHomeSwatch.style.background = team.colors[0];
      this.vsHomeSwatch.title = team.name;
    } else {
      this.selectedAway = team;
      this._applySelectionStyle(this.awayRows, team.id, 'selected-away');
      this.vsAwaySwatch.style.background = team.colors[0];
      this.vsAwaySwatch.title = team.name;
    }
    this._refreshChosenLabels();
    this._refreshWatchBtn();
  }

  private _applySelectionStyle(
    rowMap: Map<string, TeamRowElement>,
    selectedId: string,
    cls: 'selected-home' | 'selected-away',
  ): void {
    for (const [id, row] of rowMap) {
      const isSelected = id === selectedId;
      row.classList.remove('selected-home', 'selected-away');
      row.setAttribute('aria-selected', isSelected ? 'true' : 'false');
      if (isSelected) row.classList.add(cls);
    }
  }

  private _refreshChosenLabels(): void {
    if (this.selectedHome) {
      this.homeChosen.textContent = this.selectedHome.name;
      this.homeChosen.classList.add('filled');
    } else {
      this.homeChosen.textContent = '— pick a team';
      this.homeChosen.classList.remove('filled');
    }

    if (this.selectedAway) {
      this.awayChosen.textContent = this.selectedAway.name;
      this.awayChosen.classList.add('filled');
    } else {
      this.awayChosen.textContent = '— pick a team';
      this.awayChosen.classList.remove('filled');
    }
  }

  private _refreshWatchBtn(): void {
    const ready = this.selectedHome !== null && this.selectedAway !== null;
    this.watchBtn.disabled = !ready;
    this.watchBtn.classList.toggle('ready', ready);
  }

  // ── Action callbacks (capture-before-hide pattern) ────────────────────────

  private _watch(): void {
    if (!this.selectedHome || !this.selectedAway || !this.handlers) return;
    const home = this.selectedHome;
    const away = this.selectedAway;
    const cb = this.handlers.onWatch; // capture before hide() nulls handlers
    this.hide();
    cb(home, away);
  }

  private _watchBracket(): void {
    if (!this.handlers) return;
    const cb = this.handlers.onWatchBracket; // capture before hide()
    this.hide();
    cb();
  }

  private _back(): void {
    if (!this.handlers) return;
    const cb = this.handlers.onBack; // capture before hide()
    this.hide();
    cb();
  }

  // ── Keyboard navigation ───────────────────────────────────────────────────

  private _onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      this._back();
      return;
    }

    // ArrowUp / ArrowDown navigate within the focused pane
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      const focused = document.activeElement as HTMLElement | null;
      if (!focused?.classList.contains('rig-spec-team-row')) return;

      const focusedId = focused.dataset.id ?? '';
      const inHome = this.homeRows.has(focusedId);
      const inAway = this.awayRows.has(focusedId);
      if (!inHome && !inAway) return;

      e.preventDefault();
      const rows = Array.from(
        (inHome ? this.homeScroll : this.awayScroll)
          .querySelectorAll<HTMLElement>('.rig-spec-team-row'),
      );
      const idx = rows.indexOf(focused);
      if (idx < 0) return;
      const next =
        e.key === 'ArrowDown'
          ? rows[(idx + 1) % rows.length]
          : rows[(idx - 1 + rows.length) % rows.length];
      next.focus();
    }
  }
}

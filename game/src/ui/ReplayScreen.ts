// ReplayScreen — DOM overlay for "the re-call": pick a recorded match to
// watch back. Pure DOM/CSS; no Three.js, no gameplay logic.
//
// Lists ReplayStore.list() as on-palette cards (home vs away colours/names
// via TEAMS lookup, date, length). The final score is HIDDEN behind a
// "no-spoiler" toggle that defaults ON — when on, the score reads "— : —"
// and a SPOILER-tagged reveal is offered. The toggle state is passed through
// onPlay(r, noSpoiler) so the orchestrator can keep the watch spoiler-free.
//
// Follows the capture-callback-before-hide pattern from SpectateScreen /
// TitleScreen. Cleans up fully on hide(); multiple show/hide is safe.

import { TEAMS } from '../league/teams';
import type { Franchise } from '../league/teams';
import { ReplayStore } from '../league/Replay';
import type { ReplayData, ReplayMeta } from '../league/Replay';
import { PAL } from './palette';

// ── Style constants ────────────────────────────────────────────────────────

const FF = `'Courier New', 'Menlo', 'Consolas', monospace, sans-serif`;

const TEAM_BY_ID: Map<string, Franchise> = new Map(TEAMS.map((t) => [t.id, t]));

function teamName(id: string): string {
  return TEAM_BY_ID.get(id)?.name ?? id;
}
function teamColor(id: string): string {
  return TEAM_BY_ID.get(id)?.colors[0] ?? PAL.dimCss;
}

function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(
    d.getHours(),
  )}:${pad(d.getMinutes())}`;
}

/** ~60 sim ticks/sec — show a friendly mm:ss length. */
function fmtLength(ticks: number): string {
  const secs = Math.round(ticks / 60);
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

// ── CSS injection (one-time, idempotent) ───────────────────────────────────

const STYLE_ID = '__rig_replay_css';

function injectCSS(): void {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement('style');
  s.id = STYLE_ID;
  s.textContent = `
/* ── RIG ReplayScreen ─────────────────────────────────────────────── */

#rig-replay-overlay {
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

#rig-replay-header {
  flex: 0 0 auto;
  padding: 28px 28px 18px;
  border-bottom: 1px solid ${PAL.dimCss}44;
  display: flex;
  align-items: baseline;
  gap: 20px;
  user-select: none;
  flex-wrap: wrap;
}

#rig-replay-title {
  font-size: clamp(20px, 4vw, 28px);
  font-weight: 900;
  letter-spacing: 0.3em;
  color: ${PAL.cyanCss};
}

#rig-replay-subtitle {
  font-size: 10px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: ${PAL.dimCss};
  flex: 1;
}

/* no-spoiler toggle */
.rig-replay-spoiler-toggle {
  pointer-events: auto;
  background: transparent;
  border: 1px solid ${PAL.dimCss}66;
  color: ${PAL.dimCss};
  font-family: ${FF};
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  padding: 7px 14px;
  cursor: pointer;
  border-radius: 2px;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
  user-select: none;
}
.rig-replay-spoiler-toggle:hover {
  border-color: ${PAL.dimCss}aa;
  color: ${PAL.paperCss};
}
.rig-replay-spoiler-toggle.off {
  border-color: ${PAL.orangeCss};
  color: ${PAL.orangeCss};
}

#rig-replay-body {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 24px 20px 100px;
  scrollbar-width: thin;
  scrollbar-color: ${PAL.dimCss}55 transparent;
}
#rig-replay-body::-webkit-scrollbar { width: 6px; }
#rig-replay-body::-webkit-scrollbar-track { background: transparent; }
#rig-replay-body::-webkit-scrollbar-thumb { background: ${PAL.dimCss}55; border-radius: 3px; }

.rig-replay-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 12px;
}

.rig-recall-card {
  background: ${PAL.bgCss};
  border: 1px solid ${PAL.dimCss}44;
  border-radius: 2px;
  padding: 14px 14px 12px;
  position: relative;
  outline: none;
  user-select: none;
  transition: border-color 0.1s, background 0.1s;
}
.rig-recall-card:hover {
  border-color: ${PAL.dimCss}99;
  background: #191c25;
}
.rig-recall-card:focus-visible {
  border-color: ${PAL.cyanCss};
  box-shadow: 0 0 0 1px ${PAL.cyanCss}44;
}

.rig-recall-matchup {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  padding-right: 24px;
}
.rig-recall-team {
  display: flex;
  align-items: center;
  gap: 7px;
  flex: 1;
  min-width: 0;
}
.rig-recall-team.away { justify-content: flex-end; }
.rig-recall-swatch {
  width: 12px;
  height: 12px;
  border-radius: 1px;
  flex-shrink: 0;
  border: 1px solid rgba(255,255,255,0.08);
}
.rig-recall-tname {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.03em;
  color: ${PAL.paperCss};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.rig-recall-vs {
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.12em;
  color: ${PAL.dimCss}88;
  flex: 0 0 auto;
}

.rig-recall-score {
  text-align: center;
  font-size: 18px;
  font-weight: 900;
  letter-spacing: 0.12em;
  color: ${PAL.paperCss};
  margin: 4px 0 8px;
}
.rig-recall-score.hidden { color: ${PAL.dimCss}88; }
.rig-recall-reveal {
  display: inline-block;
  margin-left: 10px;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: ${PAL.orangeCss};
  border: 1px solid ${PAL.orangeCss}66;
  border-radius: 2px;
  padding: 2px 7px;
  cursor: pointer;
  vertical-align: middle;
}
.rig-recall-reveal:hover { background: ${PAL.orangeCss}1a; }

.rig-recall-meta {
  font-size: 10px;
  letter-spacing: 0.06em;
  color: ${PAL.dimCss};
  display: flex;
  gap: 14px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.rig-recall-actions {
  display: flex;
  gap: 10px;
}
.rig-recall-btn {
  background: transparent;
  font-family: ${FF};
  font-size: 11px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  padding: 8px 18px;
  cursor: pointer;
  border-radius: 2px;
  transition: background 0.12s, color 0.12s;
}
.rig-recall-btn.play {
  border: 1px solid ${PAL.cyanCss};
  color: ${PAL.cyanCss};
  flex: 1;
}
.rig-recall-btn.play:hover { background: ${PAL.cyanCss}22; }
.rig-recall-btn.del {
  border: 1px solid ${PAL.dimCss}66;
  color: ${PAL.dimCss};
}
.rig-recall-btn.del:hover {
  border-color: ${PAL.orangeCss};
  color: ${PAL.orangeCss};
}

#rig-replay-empty {
  margin: 80px auto;
  text-align: center;
  color: ${PAL.dimCss};
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  user-select: none;
}
#rig-replay-empty .hint {
  margin-top: 12px;
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: none;
  color: ${PAL.dimCss}99;
}

#rig-replay-footer {
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
#rig-replay-back {
  pointer-events: auto;
  background: transparent;
  border: 1px solid ${PAL.dimCss}66;
  color: ${PAL.dimCss};
  font-family: ${FF};
  font-size: 12px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  padding: 11px 30px;
  cursor: pointer;
  border-radius: 2px;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
}
#rig-replay-back:hover {
  border-color: ${PAL.dimCss}aa;
  color: ${PAL.paperCss};
}
`;
  document.head.appendChild(s);
}

// ── ReplayScreen ───────────────────────────────────────────────────────────

type Handlers = {
  onPlay: (r: ReplayData, noSpoiler: boolean) => void;
  onDelete: (id: string) => void;
  onBack: () => void;
};

export class ReplayScreen {
  private overlay: HTMLElement;
  private body: HTMLElement;
  private spoilerToggle: HTMLButtonElement;
  private backBtn: HTMLButtonElement;

  private noSpoiler = true;
  private handlers: Handlers | null = null;

  constructor(root: HTMLElement) {
    injectCSS();

    this.overlay = document.createElement('div');
    this.overlay.id = 'rig-replay-overlay';
    this.overlay.style.display = 'none';

    // ── Header ─────────────────────────────────────────────────────────────
    const header = document.createElement('div');
    header.id = 'rig-replay-header';

    const title = document.createElement('div');
    title.id = 'rig-replay-title';
    title.textContent = 'THE RE-CALL';

    const subtitle = document.createElement('div');
    subtitle.id = 'rig-replay-subtitle';
    subtitle.textContent = 'watch a recorded match back';

    this.spoilerToggle = document.createElement('button');
    this.spoilerToggle.className = 'rig-replay-spoiler-toggle';
    this.spoilerToggle.addEventListener('click', () => this._toggleSpoiler());

    header.append(title, subtitle, this.spoilerToggle);

    // ── Body ───────────────────────────────────────────────────────────────
    this.body = document.createElement('div');
    this.body.id = 'rig-replay-body';

    // ── Footer ─────────────────────────────────────────────────────────────
    const footer = document.createElement('div');
    footer.id = 'rig-replay-footer';

    this.backBtn = document.createElement('button');
    this.backBtn.id = 'rig-replay-back';
    this.backBtn.textContent = 'BACK';
    this.backBtn.addEventListener('click', () => this._back());

    footer.appendChild(this.backBtn);

    this.overlay.addEventListener('keydown', (e) => this._onKey(e));

    this.overlay.append(header, this.body, footer);
    root.style.position = 'relative';
    root.appendChild(this.overlay);
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  show(handlers: Handlers): void {
    this.handlers = handlers;
    this.noSpoiler = true;
    this._refreshSpoilerToggle();
    this._render();
    this.overlay.style.display = 'flex';
    requestAnimationFrame(() => {
      const first =
        this.overlay.querySelector<HTMLElement>('.rig-recall-card') ?? this.backBtn;
      first?.focus();
    });
  }

  hide(): void {
    this.overlay.style.display = 'none';
    this.handlers = null;
    this.body.innerHTML = '';
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  private _render(): void {
    this.body.innerHTML = '';
    const metas = ReplayStore.list();

    if (metas.length === 0) {
      const empty = document.createElement('div');
      empty.id = 'rig-replay-empty';
      empty.textContent = 'no re-calls recorded yet';
      const hint = document.createElement('div');
      hint.className = 'hint';
      hint.textContent = 'play or watch a match — it records itself';
      empty.appendChild(hint);
      this.body.appendChild(empty);
      return;
    }

    const grid = document.createElement('div');
    grid.className = 'rig-replay-grid';
    for (const m of metas) grid.appendChild(this._buildCard(m));
    this.body.appendChild(grid);
  }

  private _buildCard(m: ReplayMeta): HTMLElement {
    const card = document.createElement('div');
    card.className = 'rig-recall-card';
    card.tabIndex = 0;
    card.dataset.id = m.id;

    // Matchup row
    const matchup = document.createElement('div');
    matchup.className = 'rig-recall-matchup';

    const homeWrap = document.createElement('div');
    homeWrap.className = 'rig-recall-team home';
    const homeSw = document.createElement('div');
    homeSw.className = 'rig-recall-swatch';
    homeSw.style.background = teamColor(m.homeId);
    const homeNm = document.createElement('div');
    homeNm.className = 'rig-recall-tname';
    homeNm.textContent = teamName(m.homeId);
    homeWrap.append(homeSw, homeNm);

    const vs = document.createElement('div');
    vs.className = 'rig-recall-vs';
    vs.textContent = 'VS';

    const awayWrap = document.createElement('div');
    awayWrap.className = 'rig-recall-team away';
    const awayNm = document.createElement('div');
    awayNm.className = 'rig-recall-tname';
    awayNm.textContent = teamName(m.awayId);
    const awaySw = document.createElement('div');
    awaySw.className = 'rig-recall-swatch';
    awaySw.style.background = teamColor(m.awayId);
    awayWrap.append(awayNm, awaySw);

    matchup.append(homeWrap, vs, awayWrap);

    // Score (spoiler-gated)
    const score = document.createElement('div');
    score.className = 'rig-recall-score';
    const renderScore = (revealed: boolean) => {
      score.innerHTML = '';
      if (this.noSpoiler && !revealed) {
        score.classList.add('hidden');
        score.append(document.createTextNode('— : —'));
        const reveal = document.createElement('span');
        reveal.className = 'rig-recall-reveal';
        reveal.textContent = 'SPOILER · reveal';
        reveal.addEventListener('click', (e) => {
          e.stopPropagation();
          renderScore(true);
        });
        score.appendChild(reveal);
      } else {
        score.classList.remove('hidden');
        score.textContent = `${m.scoreHome} : ${m.scoreAway}`;
      }
    };
    renderScore(false);

    // Meta row
    const meta = document.createElement('div');
    meta.className = 'rig-recall-meta';
    const dateEl = document.createElement('span');
    dateEl.textContent = fmtDate(m.recordedAt);
    const lenEl = document.createElement('span');
    lenEl.textContent = `${fmtLength(m.ticks)} · ${m.ticks} ticks`;
    meta.append(dateEl, lenEl);
    if (m.label) {
      const labelEl = document.createElement('span');
      labelEl.textContent = m.label;
      meta.appendChild(labelEl);
    }

    // Actions
    const actions = document.createElement('div');
    actions.className = 'rig-recall-actions';

    const playBtn = document.createElement('button');
    playBtn.className = 'rig-recall-btn play';
    playBtn.textContent = 'WATCH';
    playBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this._play(m.id);
    });

    const delBtn = document.createElement('button');
    delBtn.className = 'rig-recall-btn del';
    delBtn.textContent = 'DELETE';
    delBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this._delete(m.id);
    });

    actions.append(playBtn, delBtn);

    card.append(matchup, score, meta, actions);

    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this._play(m.id);
      }
    });

    return card;
  }

  // ── Spoiler toggle ─────────────────────────────────────────────────────────

  private _refreshSpoilerToggle(): void {
    this.spoilerToggle.textContent = this.noSpoiler
      ? 'NO-SPOILER · ON'
      : 'NO-SPOILER · OFF';
    this.spoilerToggle.classList.toggle('off', !this.noSpoiler);
  }

  private _toggleSpoiler(): void {
    this.noSpoiler = !this.noSpoiler;
    this._refreshSpoilerToggle();
    this._render();
  }

  // ── Actions (capture-callback-before-hide pattern) ─────────────────────────

  private _play(id: string): void {
    if (!this.handlers) return;
    const data = ReplayStore.load(id);
    if (!data) {
      // Stale card (deleted/corrupt) — just refresh the list.
      this._render();
      return;
    }
    const cb = this.handlers.onPlay;
    const noSpoiler = this.noSpoiler;
    this.hide();
    cb(data, noSpoiler);
  }

  private _delete(id: string): void {
    if (!this.handlers) return;
    // onDelete owns the actual removal; we re-render from the store after.
    this.handlers.onDelete(id);
    ReplayStore.remove(id);
    this._render();
  }

  private _back(): void {
    if (!this.handlers) return;
    const cb = this.handlers.onBack;
    this.hide();
    cb();
  }

  private _onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      this._back();
      return;
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      const cards = Array.from(
        this.overlay.querySelectorAll<HTMLElement>('.rig-recall-card'),
      );
      if (cards.length === 0) return;
      const focused = document.activeElement as HTMLElement | null;
      const idx = focused ? cards.indexOf(focused) : -1;
      e.preventDefault();
      const next =
        e.key === 'ArrowDown'
          ? cards[(idx + 1) % cards.length]
          : cards[(idx - 1 + cards.length) % cards.length];
      next.focus();
    }
  }
}

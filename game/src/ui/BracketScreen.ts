// BracketScreen.ts — DOM overlay for the RIG "Jump" single-elimination bracket.
//
// Renders a 16→8→4→2→champion bracket laid out in columns by round (1..4).
// Each BracketGame shows franchise name + colour swatch + seed, score if
// result available, TBD if participants null.  The player's chosen franchise
// is highlighted throughout; its live path is emphasised.
//
// No Three.js, no gameplay logic — purely a DOM/CSS overlay.

import type { BracketState, BracketGame } from '../league/Bracket';
import type { Franchise } from '../league/teams';
import { PAL } from './palette';

// ── Layout constants ───────────────────────────────────────────────────────

const ROUND_LABELS: Record<number, string> = {
  1: 'ROUND OF 16',
  2: 'QUARTERFINALS',
  3: 'SEMIFINALS',
  4: 'FINAL',
};

// ── BracketScreen ──────────────────────────────────────────────────────────

export class BracketScreen {
  private overlay: HTMLElement;
  private headerEl: HTMLElement;
  private bracketEl: HTMLElement;
  private continueBtn: HTMLButtonElement;

  private _chosenId: string = '';
  private _onContinue: (() => void) | null = null;

  constructor(root: HTMLElement) {

    // Full-viewport dark overlay
    this.overlay = document.createElement('div');
    Object.assign(this.overlay.style, {
      position:   'absolute',
      inset:      '0',
      zIndex:     '50',
      background: PAL.bgCss,
      display:    'none',
      flexDirection: 'column',
      alignItems: 'stretch',
      overflow:   'hidden',
      fontFamily: 'monospace, "Space Mono", ui-monospace, sans-serif',
      color:      PAL.paperCss,
      userSelect: 'none',
    } satisfies Partial<CSSStyleDeclaration>);

    // ── Header ──────────────────────────────────────────────────────────
    this.headerEl = document.createElement('div');
    Object.assign(this.headerEl.style, {
      flexShrink:   '0',
      padding:      '18px 28px 12px',
      borderBottom: `1px solid ${PAL.dimCss}44`,
      display:      'flex',
      alignItems:   'baseline',
      gap:          '20px',
    } satisfies Partial<CSSStyleDeclaration>);

    const title = document.createElement('span');
    title.textContent = 'THE JUMP';
    Object.assign(title.style, {
      fontSize:      '22px',
      letterSpacing: '4px',
      color:         PAL.cyanCss,
      fontWeight:    'bold',
    } satisfies Partial<CSSStyleDeclaration>);

    const subtitle = document.createElement('span');
    subtitle.textContent = 'RIG LEAGUE SINGLE-ELIMINATION PLAYOFF';
    Object.assign(subtitle.style, {
      fontSize:      '10px',
      letterSpacing: '2px',
      color:         PAL.dimCss,
    } satisfies Partial<CSSStyleDeclaration>);

    this.headerEl.appendChild(title);
    this.headerEl.appendChild(subtitle);

    // ── Scrollable bracket area ──────────────────────────────────────────
    const scrollWrap = document.createElement('div');
    Object.assign(scrollWrap.style, {
      flex:      '1',
      overflowX: 'auto',
      overflowY: 'auto',
      padding:   '24px 20px',
    } satisfies Partial<CSSStyleDeclaration>);

    this.bracketEl = document.createElement('div');
    Object.assign(this.bracketEl.style, {
      display:       'flex',
      flexDirection: 'row',
      alignItems:    'center',
      gap:           '0',
      minWidth:      '900px',
      minHeight:     '560px',
    } satisfies Partial<CSSStyleDeclaration>);

    scrollWrap.appendChild(this.bracketEl);

    // ── Continue button ──────────────────────────────────────────────────
    const footer = document.createElement('div');
    Object.assign(footer.style, {
      flexShrink:   '0',
      padding:      '14px 28px',
      borderTop:    `1px solid ${PAL.dimCss}44`,
      display:      'flex',
      justifyContent: 'flex-end',
      alignItems:   'center',
    } satisfies Partial<CSSStyleDeclaration>);

    this.continueBtn = document.createElement('button');
    this.continueBtn.textContent = 'CONTINUE ▶';
    Object.assign(this.continueBtn.style, {
      background:    'transparent',
      border:        `2px solid ${PAL.cyanCss}`,
      color:         PAL.cyanCss,
      fontFamily:    'inherit',
      fontSize:      '14px',
      letterSpacing: '3px',
      padding:       '10px 28px',
      cursor:        'pointer',
      transition:    'background 0.15s, color 0.15s',
    } satisfies Partial<CSSStyleDeclaration>);
    this.continueBtn.addEventListener('mouseenter', () => {
      this.continueBtn.style.background = PAL.cyanCss;
      this.continueBtn.style.color      = PAL.bgCss;
    });
    this.continueBtn.addEventListener('mouseleave', () => {
      this.continueBtn.style.background = 'transparent';
      this.continueBtn.style.color      = PAL.cyanCss;
    });
    this.continueBtn.addEventListener('click', () => {
      if (this._onContinue) this._onContinue();
    });

    footer.appendChild(this.continueBtn);

    this.overlay.appendChild(this.headerEl);
    this.overlay.appendChild(scrollWrap);
    this.overlay.appendChild(footer);

    root.style.position = 'relative';
    root.appendChild(this.overlay);
  }

  // ── Public API ─────────────────────────────────────────────────────────

  show(state: BracketState, chosenFranchiseId: string, onContinue: () => void): void {
    this._chosenId   = chosenFranchiseId;
    this._onContinue = onContinue;

    this.overlay.style.display = 'flex';
    this._render(state);
  }

  update(state: BracketState): void {
    this._render(state);
  }

  hide(): void {
    this.overlay.style.display = 'none';
    this._onContinue = null;
  }

  // ── Render ─────────────────────────────────────────────────────────────

  private _render(state: BracketState): void {
    // Champion banner in header
    this._updateHeader(state.champion);

    // Rebuild bracket columns
    this.bracketEl.innerHTML = '';

    const rounds = [1, 2, 3, 4];
    for (let ri = 0; ri < rounds.length; ri++) {
      const round = rounds[ri];
      const gamesInRound = state.games.filter((g) => g.round === round);

      const col = this._buildColumn(round, gamesInRound, state);
      this.bracketEl.appendChild(col);

      // Connector column between rounds (except after the final)
      if (ri < rounds.length - 1) {
        const connector = this._buildConnectors(gamesInRound, state);
        this.bracketEl.appendChild(connector);
      }
    }

    // Champion slot after final
    const champ = this._buildChampionSlot(state.champion);
    this.bracketEl.appendChild(champ);
  }

  private _updateHeader(champion: Franchise | null): void {
    // Remove any previous champion badge
    const existing = this.headerEl.querySelector('._champ-badge');
    if (existing) this.headerEl.removeChild(existing);

    if (champion) {
      const badge = document.createElement('span');
      badge.className = '_champ-badge';
      Object.assign(badge.style, {
        marginLeft:    'auto',
        fontSize:      '13px',
        letterSpacing: '2px',
        color:         PAL.orangeCss,
        borderLeft:    `2px solid ${PAL.orangeCss}`,
        paddingLeft:   '14px',
        display:       'flex',
        alignItems:    'center',
        gap:           '8px',
      } satisfies Partial<CSSStyleDeclaration>);

      const swatch = this._swatch(champion.colors[0], 10);
      badge.appendChild(swatch);
      const label = document.createElement('span');
      label.textContent = `CHAMPION: ${champion.name.toUpperCase()}`;
      badge.appendChild(label);

      this.headerEl.appendChild(badge);
    }
  }

  // ── Column builder ─────────────────────────────────────────────────────

  private _buildColumn(
    round: number,
    games: BracketGame[],
    state: BracketState,
  ): HTMLElement {
    const col = document.createElement('div');
    Object.assign(col.style, {
      display:        'flex',
      flexDirection:  'column',
      alignItems:     'stretch',
      justifyContent: 'space-around',
      flex:           '1',
      minWidth:       '180px',
      maxWidth:       '240px',
      gap:            '0',
      padding:        '0 4px',
    } satisfies Partial<CSSStyleDeclaration>);

    // Round label
    const label = document.createElement('div');
    label.textContent = ROUND_LABELS[round] ?? `ROUND ${round}`;
    Object.assign(label.style, {
      textAlign:     'center',
      fontSize:      '9px',
      letterSpacing: '2px',
      color:         PAL.dimCss,
      marginBottom:  '12px',
      paddingBottom: '6px',
      borderBottom:  `1px solid ${PAL.dimCss}33`,
    } satisfies Partial<CSSStyleDeclaration>);
    col.appendChild(label);

    // Wrapper that fills remaining space and stacks games evenly
    const gamesWrap = document.createElement('div');
    Object.assign(gamesWrap.style, {
      display:        'flex',
      flexDirection:  'column',
      flex:           '1',
      justifyContent: 'space-around',
    } satisfies Partial<CSSStyleDeclaration>);

    for (const game of games) {
      const card = this._buildGameCard(game, state);
      gamesWrap.appendChild(card);
    }

    col.appendChild(gamesWrap);
    return col;
  }

  // ── Game card ──────────────────────────────────────────────────────────

  private _buildGameCard(game: BracketGame, state: BracketState): HTMLElement {
    const playerInvolved = this._gameInvolvesPlayer(game);
    const playerEliminated = this._playerIsEliminated(state);

    const card = document.createElement('div');
    Object.assign(card.style, {
      margin:        '5px 0',
      border:        `1px solid ${playerInvolved ? PAL.cyanCss : PAL.dimCss + '66'}`,
      background:    playerInvolved ? PAL.bgCss : PAL.bgCss,
      boxShadow:     playerInvolved ? `0 0 8px ${PAL.cyanCss}44` : 'none',
      borderRadius:  '2px',
      overflow:      'hidden',
      position:      'relative',
    } satisfies Partial<CSSStyleDeclaration>);

    // "YOUR MATCH" indicator strip at top
    if (playerInvolved && !playerEliminated && !game.result) {
      const strip = document.createElement('div');
      strip.textContent = '▶ YOUR MATCH';
      Object.assign(strip.style, {
        background:    PAL.cyanCss + '22',
        borderBottom:  `1px solid ${PAL.cyanCss}66`,
        fontSize:      '8px',
        letterSpacing: '2px',
        color:         PAL.cyanCss,
        padding:       '2px 6px',
        textAlign:     'center',
      } satisfies Partial<CSSStyleDeclaration>);
      card.appendChild(strip);
    }

    // Home row
    card.appendChild(this._buildTeamRow(game, 'home', game.home, game.result));
    // Divider
    const divider = document.createElement('div');
    Object.assign(divider.style, {
      height:     '1px',
      background: PAL.dimCss + '33',
      margin:     '0 6px',
    } satisfies Partial<CSSStyleDeclaration>);
    card.appendChild(divider);
    // Away row
    card.appendChild(this._buildTeamRow(game, 'away', game.away, game.result));

    return card;
  }

  private _buildTeamRow(
    _game: BracketGame,
    side: 'home' | 'away',
    franchise: Franchise | null,
    result: import('../league/SimMatch').MatchResult | null,
  ): HTMLElement {
    const row = document.createElement('div');
    Object.assign(row.style, {
      display:    'flex',
      alignItems: 'center',
      padding:    '5px 6px',
      gap:        '5px',
      minHeight:  '28px',
      position:   'relative',
    } satisfies Partial<CSSStyleDeclaration>);

    if (!franchise) {
      // TBD slot
      const tbd = document.createElement('span');
      tbd.textContent = 'TBD';
      Object.assign(tbd.style, {
        color:         PAL.dimCss + '88',
        fontSize:      '10px',
        letterSpacing: '1px',
        fontStyle:     'italic',
      } satisfies Partial<CSSStyleDeclaration>);
      row.appendChild(tbd);
      return row;
    }

    const isPlayer     = franchise.id === this._chosenId;
    const isWinner     = result !== null && result.winner === side;
    const isLoser      = result !== null && result.winner !== side;

    // Colour swatch
    row.appendChild(this._swatch(franchise.colors[0], 8));

    // Seed badge
    if (franchise.seed !== null) {
      const seedEl = document.createElement('span');
      seedEl.textContent = String(franchise.seed);
      Object.assign(seedEl.style, {
        fontSize:   '9px',
        color:      PAL.dimCss,
        minWidth:   '14px',
        textAlign:  'right',
        flexShrink: '0',
      } satisfies Partial<CSSStyleDeclaration>);
      row.appendChild(seedEl);
    }

    // Name
    const nameEl = document.createElement('span');
    nameEl.textContent = franchise.name;
    Object.assign(nameEl.style, {
      flex:          '1',
      fontSize:      '10px',
      letterSpacing: '0.5px',
      overflow:      'hidden',
      textOverflow:  'ellipsis',
      whiteSpace:    'nowrap',
      color:         isLoser
        ? PAL.dimCss + 'aa'
        : isPlayer
          ? PAL.cyanCss
          : PAL.paperCss,
      fontWeight:    isPlayer || isWinner ? 'bold' : 'normal',
      textDecoration: isLoser ? 'line-through' : 'none',
    } satisfies Partial<CSSStyleDeclaration>);
    row.appendChild(nameEl);

    // Score
    if (result !== null) {
      const score = side === 'home' ? result.scoreHome : result.scoreAway;
      const scoreEl = document.createElement('span');
      scoreEl.textContent = String(score);
      Object.assign(scoreEl.style, {
        fontSize:   '11px',
        fontWeight: isWinner ? 'bold' : 'normal',
        color:      isWinner
          ? (isPlayer ? PAL.cyanCss : PAL.orangeCss)
          : PAL.dimCss,
        minWidth:   '20px',
        textAlign:  'right',
        flexShrink: '0',
      } satisfies Partial<CSSStyleDeclaration>);
      row.appendChild(scoreEl);

      if (isWinner) {
        const marker = document.createElement('span');
        marker.textContent = '▶';
        Object.assign(marker.style, {
          fontSize:   '8px',
          color:      isPlayer ? PAL.cyanCss : PAL.orangeCss,
          flexShrink: '0',
        } satisfies Partial<CSSStyleDeclaration>);
        row.appendChild(marker);
      }
    }

    // Player highlight pulse border
    if (isPlayer) {
      Object.assign(row.style, {
        background: PAL.cyanCss + '11',
      } satisfies Partial<CSSStyleDeclaration>);
    }

    return row;
  }

  // ── Connectors between columns ─────────────────────────────────────────
  // Draws simple horizontal line segments that pair up adjacent games via SVG.

  private _buildConnectors(
    gamesInCurrentRound: BracketGame[],
    _state: BracketState,
  ): HTMLElement {
    const wrap = document.createElement('div');
    Object.assign(wrap.style, {
      display:        'flex',
      flexDirection:  'column',
      justifyContent: 'space-around',
      width:          '32px',
      flexShrink:     '0',
      position:       'relative',
    } satisfies Partial<CSSStyleDeclaration>);

    // One connector bracket per pair of games → next round's single slot
    const pairCount = Math.floor(gamesInCurrentRound.length / 2);

    for (let i = 0; i < pairCount; i++) {
      const seg = document.createElement('div');
      Object.assign(seg.style, {
        flex:          '1',
        borderRight:   `1px solid ${PAL.dimCss}66`,
        borderTop:     `1px solid ${PAL.dimCss}66`,
        borderBottom:  `1px solid ${PAL.dimCss}66`,
        margin:        '0 0 0 8px',
        // Visual only — connects the top-pair and bottom-pair vertically
      } satisfies Partial<CSSStyleDeclaration>);
      wrap.appendChild(seg);

      // A small gap spacer between bracket pairs
      if (i < pairCount - 1) {
        const gap = document.createElement('div');
        Object.assign(gap.style, { height: '10px', flexShrink: '0' } satisfies Partial<CSSStyleDeclaration>);
        wrap.appendChild(gap);
      }
    }

    return wrap;
  }

  // ── Champion slot ──────────────────────────────────────────────────────

  private _buildChampionSlot(champion: Franchise | null): HTMLElement {
    const slot = document.createElement('div');
    Object.assign(slot.style, {
      display:        'flex',
      flexDirection:  'column',
      alignItems:     'center',
      justifyContent: 'center',
      minWidth:       '140px',
      padding:        '0 12px',
      gap:            '10px',
    } satisfies Partial<CSSStyleDeclaration>);

    const cupLabel = document.createElement('div');
    cupLabel.textContent = 'JUMP CHAMPION';
    Object.assign(cupLabel.style, {
      fontSize:      '9px',
      letterSpacing: '2px',
      color:         PAL.dimCss,
      textAlign:     'center',
    } satisfies Partial<CSSStyleDeclaration>);
    slot.appendChild(cupLabel);

    const champBox = document.createElement('div');
    Object.assign(champBox.style, {
      border:       `2px solid ${champion ? PAL.orangeCss : PAL.dimCss + '44'}`,
      padding:      '10px 14px',
      minWidth:     '120px',
      textAlign:    'center',
      boxShadow:    champion ? `0 0 16px ${PAL.orangeCss}44` : 'none',
      borderRadius: '2px',
    } satisfies Partial<CSSStyleDeclaration>);

    if (champion) {
      const isPlayer = champion.id === this._chosenId;

      const sw = this._swatch(champion.colors[0], 12);
      sw.style.margin = '0 auto 6px';
      sw.style.display = 'block';
      champBox.appendChild(sw);

      const nameEl = document.createElement('div');
      nameEl.textContent = champion.name.toUpperCase();
      Object.assign(nameEl.style, {
        fontSize:      '11px',
        letterSpacing: '1px',
        fontWeight:    'bold',
        color:         isPlayer ? PAL.cyanCss : PAL.orangeCss,
        lineHeight:    '1.3',
      } satisfies Partial<CSSStyleDeclaration>);
      champBox.appendChild(nameEl);

      if (isPlayer) {
        const glory = document.createElement('div');
        glory.textContent = '★ YOU WIN THE JUMP ★';
        Object.assign(glory.style, {
          marginTop:     '6px',
          fontSize:      '8px',
          letterSpacing: '1px',
          color:         PAL.cyanCss,
        } satisfies Partial<CSSStyleDeclaration>);
        champBox.appendChild(glory);
      }
    } else {
      const tbd = document.createElement('div');
      tbd.textContent = '?';
      Object.assign(tbd.style, {
        fontSize: '24px',
        color:    PAL.dimCss + '66',
      } satisfies Partial<CSSStyleDeclaration>);
      champBox.appendChild(tbd);
    }

    slot.appendChild(champBox);
    return slot;
  }

  // ── Helpers ────────────────────────────────────────────────────────────

  /** Colour swatch square. */
  private _swatch(hexColor: string, size: number): HTMLElement {
    const el = document.createElement('span');
    Object.assign(el.style, {
      display:         'inline-block',
      width:           `${size}px`,
      height:          `${size}px`,
      background:      hexColor,
      flexShrink:      '0',
      borderRadius:    '1px',
      verticalAlign:   'middle',
      border:          `1px solid ${PAL.dimCss}55`,
    } satisfies Partial<CSSStyleDeclaration>);
    return el;
  }

  /** True if the player's chosen franchise appears in this game. */
  private _gameInvolvesPlayer(game: BracketGame): boolean {
    return (
      game.home?.id === this._chosenId ||
      game.away?.id === this._chosenId
    );
  }

  /** True if the player's chosen franchise has been knocked out already. */
  private _playerIsEliminated(state: BracketState): boolean {
    // If the champion is set and is not the player, player is out.
    if (state.champion && state.champion.id !== this._chosenId) return true;
    // If a completed game has the player as the loser, they're out.
    return state.games.some((g) => {
      if (!g.result) return false;
      const loser = g.result.winner === 'home' ? g.away : g.home;
      return loser?.id === this._chosenId;
    });
  }
}

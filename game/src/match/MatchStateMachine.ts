// MatchStateMachine.ts — RIG match-rules layer.
//
// FROZEN contract: codes only against sim/types.ts.
// No DOM, no Date, no Math.random. Pure reducer / deterministic state machine.
//
// ─── Field geometry ──────────────────────────────────────────────────────────
// sport.md §3.1: field is 640 m long; goal rings at ±320 m.
// Three gates divide the half-field between midline and the attacked ring:
//
//   first : 0.25 × 320 =  80 m  (first water — just past midfield commitment)
//   deep  : 0.55 × 320 = 176 m  (deep water — real commitment zone)
//   mouth : 0.85 × 320 = 272 m  (mouth  — just short of the ring)
//
// The attacker (possession) moves toward their target ring:
//   home always attacks toward +x (ring at +320);
//   away always attacks toward -x (ring at -320).
// Signed gate x = GATE_ABS * direction.
//
// ─── Cast / downs logic ──────────────────────────────────────────────────────
// A cast starts at throwsLeft=3 at gate 'first'.
// bell_caught / bell_bobble / bell_clatter each consume one throw.
// If bell.p.x has passed the current gate x-plane → gate cleared →
//   gate advances, throwsLeft resets to 3.
// throwsLeft hits 0 without clearing → turnover at spotX.
//
// ─── Contest ─────────────────────────────────────────────────────────────────
// contest_started → phase 'contest'.
// The outer loop resolves the contest with Contest.resolveContest() and calls
// MatchStateMachine.resolveContest(result, sim).
// Thrower wins → cast continues (phase 'live').
// Contester wins → turnover.
//
// ─── Scoring ─────────────────────────────────────────────────────────────────
// bell_through_ring → Scoring.scoreFor; endsInning on Loop (or any score).
// player_skinned of attacker → Scoring.groundFor → +1 to defense.
//
// ─── Innings ─────────────────────────────────────────────────────────────────
// Inning ends on score OR turnover-and-clear.
// 9 innings → check tie; tie → spine (sudden death, Free end).
// First score in spine → winner. phase 'final'.
//
// ─── Return shape (MatchUpdate) ──────────────────────────────────────────────
// consume() returns MatchUpdate:
//   resets    — ordered ops caller must perform ('pushoff' | 'set')
//   scored?   — goal scored this batch
//   turnover? — possession changed, at spotX
//   inningEnd?— inning closed
//   spineStart? — spine period began
//   winner?   — match over

import type { SimEvent, SimState, MatchState, TeamSide } from '../sim/types';
import { scoreFor, groundFor, type ScoreResult } from './Scoring';

// ─── Gate geometry ────────────────────────────────────────────────────────────

/** Half-field length (goal ring absolute x). */
const GATE_X = 320;

const GATE_ABS: Record<'first' | 'deep' | 'mouth', number> = {
  first: GATE_X * 0.25, //  80 m
  deep: GATE_X * 0.55, // 176 m
  mouth: GATE_X * 0.85, // 272 m
};

const GATE_ORDER: Array<'first' | 'deep' | 'mouth'> = ['first', 'deep', 'mouth'];

/** Signed gate x: direction +1 = toward +x, -1 = toward -x. */
function gateX(gate: 'first' | 'deep' | 'mouth', dir: 1 | -1): number {
  return GATE_ABS[gate] * dir;
}

/**
 * Direction of attack for the current possession:
 *   home attacks +x (Free end for home, or Faith — immaterial here);
 *   away attacks -x.
 * The physical direction is what matters for gate-clearing geometry.
 */
function attackDir(possession: TeamSide): 1 | -1 {
  return possession === 'home' ? 1 : -1;
}

// ─── Initial states ───────────────────────────────────────────────────────────

function initialCast(possession: TeamSide): MatchState['cast'] {
  const dir = attackDir(possession);
  return {
    throwsLeft: 3,
    gate: 'first',
    spotX: gateX('first', dir),
  };
}

function makeInitialState(faithEnd: '+x' | '-x', firstPossession: TeamSide): MatchState {
  return {
    inning: 1,
    spine: false,
    possession: firstPossession,
    faithEnd,
    cast: initialCast(firstPossession),
    contest: null,
    scoreHome: 0,
    scoreAway: 0,
    phase: 'set',
    message: '',
    winner: null,
  };
}

// ─── Return type ─────────────────────────────────────────────────────────────

export interface MatchUpdate {
  /** Ordered sim-reset operations for the outer loop. */
  resets: Array<'pushoff' | 'set'>;
  /** Set if a goal was scored this batch. */
  scored?: ScoreResult;
  /** Set if possession changed via turnover. */
  turnover?: { spotX: number; team: TeamSide };
  /** Set if an inning ended. */
  inningEnd?: { inning: number };
  /** Set if the spine period just started. */
  spineStart?: true;
  /** Set if the match is over. */
  winner?: TeamSide;
}

// ─── Main class ──────────────────────────────────────────────────────────────

export class MatchStateMachine {
  private _state: MatchState;

  /**
   * @param faithEnd        — which ring is the spinward/Faith end
   * @param firstPossession — which team has the opening cast
   */
  constructor(faithEnd: '+x' | '-x', firstPossession: TeamSide) {
    this._state = makeInitialState(faithEnd, firstPossession);
  }

  get state(): Readonly<MatchState> {
    return this._state;
  }

  /**
   * If phase === 'contest', returns the frozen contest state so the outer loop
   * can call Contest.resolveContest() and then MatchStateMachine.resolveContest().
   */
  get pendingContest(): MatchState['contest'] {
    return this._state.phase === 'contest' ? this._state.contest : null;
  }

  /**
   * Consume a batch of SimEvents from one sim step.
   * Updates MatchState; returns MatchUpdate for the caller to act on.
   */
  consume(events: SimEvent[], sim: SimState): MatchUpdate {
    const update: MatchUpdate = { resets: [] };

    for (const ev of events) {
      if (this._state.winner !== null) break;
      this._handleEvent(ev, sim, update);
    }

    return update;
  }

  /**
   * Resolve a pending contest after the outer loop has called Contest.resolveContest().
   */
  resolveContest(result: { winner: 'thrower' | 'contester' }, _sim: SimState): MatchUpdate {
    const update: MatchUpdate = { resets: [] };
    if (this._state.phase !== 'contest' || this._state.contest === null) return update;

    if (result.winner === 'thrower') {
      // Cast continues from the catch point
      this._state = {
        ...this._state,
        contest: null,
        phase: 'live',
        message: 'Contest complete — cast continues',
      };
    } else {
      // Contester wins → turnover at current spotX
      this._applyTurnover(update);
    }

    return update;
  }

  // ─── Private: event dispatch ─────────────────────────────────────────────

  private _handleEvent(ev: SimEvent, sim: SimState, update: MatchUpdate): void {
    // Once the match is decided, everything is absorbed.
    if (this._state.phase === 'final' || this._state.winner !== null) return;

    // foul_garrote is phase-agnostic: it always re-arms a fresh live cast
    // (this is also how the outer loop returns to 'live' after an inning
    // break / dead ball).
    if (ev.type === 'foul_garrote') {
      this._onFoulGarrote(update);
      return;
    }

    // Other physics events are inert while the ball is dead / between innings.
    const phase = this._state.phase;
    if (phase === 'dead' || phase === 'inning_break') return;

    switch (ev.type) {
      case 'bell_through_ring':
        this._onBellThroughRing(ev, sim, update);
        break;
      case 'contest_started':
        if (phase === 'live') this._onContestStarted(ev);
        break;
      case 'player_skinned':
        this._onPlayerSkinned(ev, sim);
        break;
      case 'bell_caught':
      case 'bell_bobble':
      case 'bell_clatter':
        if (phase === 'live') this._onThrowSpent(sim, update);
        break;
      case 'bell_skin':
        // Bell bouncing off skin: no throw consumed, no score.
        break;
      default:
        break;
    }
  }

  // ─── Private: specific events ────────────────────────────────────────────

  private _onBellThroughRing(
    ev: Extract<SimEvent, { type: 'bell_through_ring' }>,
    sim: SimState,
    update: MatchUpdate,
  ): void {
    const result = scoreFor(ev, this._state, sim);
    if (result === null) {
      this._state = { ...this._state, message: 'Bell through ring — no attribution' };
      return;
    }

    // Apply points
    const newHome = result.team === 'home' ? this._state.scoreHome + result.points : this._state.scoreHome;
    const newAway = result.team === 'away' ? this._state.scoreAway + result.points : this._state.scoreAway;

    update.scored = result;
    this._state = {
      ...this._state,
      scoreHome: newHome,
      scoreAway: newAway,
      message: `${result.kind.toUpperCase()} +${result.points} (${result.team})`,
    };

    // Every score ends the inning (rules.md §5; Loop also ends immediately).
    this._endInning(update);
  }

  private _onContestStarted(ev: Extract<SimEvent, { type: 'contest_started' }>): void {
    // Direction: rough proxy from the gate the attacker is currently at.
    const gate = this._state.cast.gate;
    const direction: 'fair' | 'cross' = gate === 'mouth' ? 'cross' : 'fair';
    // Radius proxy: gateX value gives depth in metres, normalise to ~[8..27].
    const radius = GATE_ABS[gate];

    this._state = {
      ...this._state,
      phase: 'contest',
      contest: {
        thrower: ev.thrower,
        contester: ev.contester,
        count: 0,
        radius,
        direction,
      },
      message: `Contest: ${ev.thrower} vs ${ev.contester} at ${gate}`,
    };
  }

  private _onPlayerSkinned(
    ev: Extract<SimEvent, { type: 'player_skinned' }>,
    sim: SimState,
  ): void {
    const result = groundFor(ev, sim);
    if (result === null) return;

    const newHome = result.team === 'home' ? this._state.scoreHome + result.points : this._state.scoreHome;
    const newAway = result.team === 'away' ? this._state.scoreAway + result.points : this._state.scoreAway;

    this._state = {
      ...this._state,
      scoreHome: newHome,
      scoreAway: newAway,
      message: `GROUND +1 (${result.team})`,
    };
    // Ground does not end inning or change possession.
  }

  private _onThrowSpent(sim: SimState, update: MatchUpdate): void {
    const s = this._state;
    const dir = attackDir(s.possession);
    const bellX = sim.bell.p.x;

    // Furthest gate the bell's x has actually passed (gates are ordered by
    // increasing distance, so stop at the first not-yet-passed).
    let clearedIdx = -1;
    for (let i = 0; i < GATE_ORDER.length; i++) {
      const gx = gateX(GATE_ORDER[i], dir);
      const passed = dir === 1 ? bellX >= gx : bellX <= gx;
      if (passed) clearedIdx = i;
      else break;
    }
    const curIdx = GATE_ORDER.indexOf(s.cast.gate);

    if (clearedIdx >= curIdx) {
      // Advanced — jump to the gate after the furthest cleared (capped at
      // mouth) and refresh the cast.
      const newIdx = Math.min(clearedIdx + 1, GATE_ORDER.length - 1);
      const newGate = GATE_ORDER[newIdx];
      this._state = {
        ...s,
        cast: { throwsLeft: 3, gate: newGate, spotX: gateX(newGate, dir) },
        message: `Gate cleared → ${newGate}`,
      };
    } else {
      // Did not clear the current gate — burn a throw.
      const newLeft = (s.cast.throwsLeft - 1) as 0 | 1 | 2 | 3;
      this._state = {
        ...s,
        cast: { ...s.cast, throwsLeft: newLeft, spotX: bellX },
      };
      if (newLeft === 0) this._applyTurnover(update);
    }
  }

  private _onFoulGarrote(update: MatchUpdate): void {
    // Garrote: the fouled team (current possession holder, the attacker) gets
    // a fresh cast from current position.
    this._state = {
      ...this._state,
      cast: initialCast(this._state.possession),
      phase: 'live',
      message: `Foul garrote — fresh cast for ${this._state.possession}`,
    };
    update.resets.push('set');
  }

  // ─── Private: turnover & inning management ───────────────────────────────

  private _applyTurnover(update: MatchUpdate): void {
    const s = this._state;
    const spotX = s.cast.spotX;
    const newPossession: TeamSide = s.possession === 'home' ? 'away' : 'home';

    update.turnover = { spotX, team: newPossession };

    this._state = {
      ...s,
      possession: newPossession,
      contest: null,
      cast: {
        ...initialCast(newPossession),
        spotX,
      },
      phase: 'dead',
      message: `Turnover → ${newPossession} at x=${spotX.toFixed(1)}`,
    };

    // Turnover-and-clear → inning ends. Possession was just set by the
    // turnover; the inning roll must NOT alternate it again.
    this._endInning(update, true);
  }

  private _endInning(update: MatchUpdate, preservePossession = false): void {
    const s = this._state;
    update.inningEnd = { inning: s.inning };

    if (s.spine) {
      // Spine sudden-death: any score decides the match
      const winner: TeamSide = s.scoreHome > s.scoreAway ? 'home' : 'away';
      this._state = {
        ...s,
        winner,
        phase: 'final',
        message: `Spine won by ${winner}`,
      };
      update.winner = winner;
      return;
    }

    if (s.inning >= 9) {
      // End of regulation
      if (s.scoreHome === s.scoreAway) {
        // Tie → spine
        const freeEnd: '+x' | '-x' = s.faithEnd === '+x' ? '-x' : '+x';
        const spineDir: 1 | -1 = freeEnd === '+x' ? 1 : -1;
        const spineFirstPossession: TeamSide = s.possession === 'home' ? 'away' : 'home';
        this._state = {
          ...s,
          spine: true,
          phase: 'spine',
          inning: s.inning + 1,
          possession: spineFirstPossession,
          contest: null,
          cast: {
            throwsLeft: 3,
            gate: 'first',
            spotX: gateX('first', spineDir),
          },
          message: `Tie after 9 innings — spine (sudden death at Free end)`,
        };
        update.spineStart = true;
        update.resets.push('pushoff', 'set');
      } else {
        const winner: TeamSide = s.scoreHome > s.scoreAway ? 'home' : 'away';
        this._state = {
          ...s,
          winner,
          phase: 'final',
          message: `Final: home ${s.scoreHome} – away ${s.scoreAway}`,
        };
        update.winner = winner;
      }
    } else {
      // Advance inning. A score-ended inning alternates possession; a
      // turnover-ended inning keeps the possession the turnover just set.
      const newInning = s.inning + 1;
      const newPossession: TeamSide = preservePossession
        ? s.possession
        : s.possession === 'home'
          ? 'away'
          : 'home';
      this._state = {
        ...s,
        inning: newInning,
        possession: newPossession,
        contest: null,
        cast: initialCast(newPossession),
        phase: 'inning_break',
        message: `Inning ${newInning} — ${newPossession} to cast`,
      };
      update.resets.push('pushoff', 'set');
    }
  }
}

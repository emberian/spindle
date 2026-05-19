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
   * Re-cast a dead / inning-break ball back to LIVE, preserving the current
   * cast (throwsLeft / gate / spotX). The outer loop calls this right after
   * re-gripping the bell on the possessing rigger. Distinct from foul_garrote,
   * which is a real foul that resets the cast.
   */
  resumeLive(): void {
    const s = this._state;
    if (
      s.winner !== null ||
      s.phase === 'final' ||
      s.phase === 'live' ||
      s.phase === 'contest'
    ) {
      return;
    }
    this._state = { ...s, phase: 'live' };
  }

  /**
   * Resolve a pending contest after the outer loop has called Contest.resolveContest().
   */
  resolveContest(result: { winner: 'thrower' | 'contester' }, _sim: SimState): MatchUpdate {
    const update: MatchUpdate = { resets: [] };
    this._resolveContestInto(result, update);
    return update;
  }

  /**
   * Resolve a pending contest, merging effects into an existing update (so a
   * physics-driven contest resolved mid-consume still propagates its
   * turnover / inning-end to the caller). Twin of match_sm.rs.
   */
  private _resolveContestInto(
    result: { winner: 'thrower' | 'contester' },
    update: MatchUpdate,
  ): void {
    if (this._state.phase !== 'contest' || this._state.contest === null) return;
    if (result.winner === 'thrower') {
      this._state = {
        ...this._state,
        contest: null,
        phase: 'live',
        message: 'Contest complete — cast continues',
      };
    } else {
      this._applyTurnover(update);
    }
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

    // PART B — physics-resolved contest (twin of match_sm.rs). A
    // `contest_started` from real loose-bell play puts the match in
    // Contest phase; the duel resolves from the ACTUAL physics with NO
    // rng (determinism is sacred — Contest.resolveContest's dice path
    // stays for the headless statistical layer only). A catch by the
    // CONTESTER's team is a won contest → turnover; a catch by the
    // THROWER's team, or the bell dying/through-ring, means the cast
    // continues. Deterministic, id-driven.
    if (phase === 'contest' && this._state.contest !== null) {
      const c = this._state.contest;
      switch (ev.type) {
        case 'bell_caught': {
          const byTeam = this._teamOf(sim, ev.by);
          const contesterTeam = this._teamOf(sim, c.contester);
          const winner: 'thrower' | 'contester' =
            byTeam !== null && byTeam === contesterTeam ? 'contester' : 'thrower';
          this._resolveContestInto({ winner }, update);
          return;
        }
        case 'bell_missed':
          this._resolveContestInto({ winner: 'thrower' }, update);
          return;
        case 'bell_through_ring':
          this._state = { ...this._state, phase: 'live', contest: null };
          this._onBellThroughRing(ev, sim, update);
          return;
        default:
          // Bobble/clatter/skin/etc.: contest still in flight.
          return;
      }
    }

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
      case 'bell_clatter': {
        // A clean catch by a TEAMMATE (same team as the possessing/
        // attacking side) is a COMPLETED PASS, not a failed throw — it must
        // not burn a down on its own (only gate progress or a real
        // failure does). Bobble/clatter and any opponent catch are still
        // failed/contested throws and spend the down as before.
        const caughtById =
          ev.type === 'bell_caught' ? ev.by : null;
        const completedPass =
          caughtById !== null &&
          this._teamOf(sim, caughtById) === this._state.possession;
        if (phase === 'live') this._onThrowSpent(sim, update, completedPass);
        break;
      }
      case 'bell_missed':
        if (phase === 'live') this._onBellMissed(sim, update);
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

  /** Team of a player id in the sim, or null if unknown. */
  private _teamOf(sim: SimState, id: string): TeamSide | null {
    const pl = sim.players.find((p) => p.id === id);
    return pl ? pl.team : null;
  }

  private _onThrowSpent(
    sim: SimState,
    update: MatchUpdate,
    completedPass = false,
  ): void {
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
    } else if (completedPass) {
      // Completed pass that did not clear a gate: possession is retained by
      // a teammate, the cast LIVES — do NOT burn a down. Track the ball's
      // new spot so subsequent gate progress is measured from here.
      this._state = {
        ...s,
        cast: { ...s.cast, spotX: bellX },
        message: `Pass completed — cast continues`,
      };
    } else {
      // Did not clear the current gate and the throw was NOT retained by a
      // teammate (failed/contested/opponent) — burn a throw.
      const newLeft = (s.cast.throwsLeft - 1) as 0 | 1 | 2 | 3;
      this._state = {
        ...s,
        cast: { ...s.cast, throwsLeft: newLeft, spotX: bellX },
      };
      if (newLeft === 0) this._applyTurnover(update);
    }
  }

  /**
   * A free bell left play without scoring (crossed a gate plane outside the
   * ring, or drifted free far too long). Same gate/throw bookkeeping as a
   * spent throw, but it is a DEAD BALL — the outer loop re-grips it on the
   * possessing rigger and calls resumeLive() (cast preserved, so a run of
   * misses actually exhausts the cast and turns the ball over).
   */
  private _onBellMissed(sim: SimState, update: MatchUpdate): void {
    const s = this._state;
    const bellX = sim.bell.p.x;
    const spotX = Math.max(-GATE_X, Math.min(GATE_X, bellX));

    // A miss is a FAILED throw — it always burns one, no matter how far the
    // dead bell sailed. (Only a *caught* bell beyond a gate advances the
    // cast; rewarding raw distance here let possession never turn over.)
    const newLeft = (s.cast.throwsLeft - 1) as 0 | 1 | 2 | 3;
    if (newLeft <= 0) {
      // Cast exhausted on a miss → turnover (this also ends the inning).
      this._state = { ...s, cast: { ...s.cast, spotX } };
      this._applyTurnover(update);
      return;
    }
    this._state = {
      ...s,
      cast: { ...s.cast, throwsLeft: newLeft, spotX },
      phase: 'dead',
      message: `Missed throw — ${newLeft} left`,
    };
    update.resets.push('set');
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
      // SUDDEN-DEATH CORRECTNESS (twin of match_sm.rs): a spine inning ENDS
      // the match only when a team has actually pulled ahead. A turnover-
      // and-clear in a still-tied spine inning does NOT default-win for
      // Away (the old `scoreHome > scoreAway` → false bug): the spine
      // CONTINUES with the other team casting. Only a real score resolves it.
      if (s.scoreHome === s.scoreAway) {
        const next: TeamSide = s.possession === 'home' ? 'away' : 'home';
        this._state = {
          ...s,
          inning: s.inning + 1,
          possession: next,
          contest: null,
          cast: initialCast(next),
          phase: 'spine',
          message: `Spine still tied — ${next} to cast (sudden death)`,
        };
        update.resets.push('pushoff', 'set');
        return;
      }
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

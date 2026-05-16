// SimMatch.ts — Headless, deterministic AI-vs-AI match resolver for the RIG league layer.
//
// Does NOT run physics. Drives MatchStateMachine with synthesised plausible SimEvents
// weighted by each team's styleToProfile() — Rise teams attempt more rises/loops,
// Fall teams grind Falls, brokers play for the away point — until 9 innings (+ spine
// on tie). Spine has a cap (MAX_SPINE_INNINGS) + higher-seed-advances fallback to
// guarantee termination.
//
// Deterministic contract: same (home, away, seed) → identical MatchResult.

import { styleToProfile, type Franchise } from './teams';
import { MatchStateMachine } from '../match/MatchStateMachine';
import type { SimEvent, SimState, TeamSide } from '../sim/types';

// ─── Seeded RNG (mulberry32) ──────────────────────────────────────────────────

/** Mulberry32 — fast, seedable, good-enough uniform [0,1). */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 0x100000000;
  };
}

// ─── Public result types ──────────────────────────────────────────────────────

export interface BoxScore {
  falls: number;
  rises: number;
  loops: number;
  curls: number;
  grounds: number;
}

export interface HighlightEvent {
  inning: number;
  kind: 'loop' | 'rise' | 'fall' | 'curl' | 'ground' | 'spine_winner';
  team: TeamSide;
  points: number;
}

export interface MatchResult {
  home: Franchise;
  away: Franchise;
  scoreHome: number;
  scoreAway: number;
  winner: TeamSide;
  box: { home: BoxScore; away: BoxScore };
  events: HighlightEvent[];
}

// ─── Minimal SimState factory ─────────────────────────────────────────────────

const ZERO_V = { x: 0, y: 0, z: 0 };
const ZERO_Q = { x: 0, y: 0, z: 0, w: 1 };

function makeSim(bellX: number, playerId: string, team: TeamSide): SimState {
  return {
    tick: 0,
    omega: 0.32,
    bell: {
      p: { x: bellX, y: 0, z: 0 },
      v: ZERO_V,
      q: ZERO_Q,
      w: ZERO_V,
      chime: 1,
      heldBy: null,
      thrownBy: playerId,
      touchedSinceThrow: false,
      releasePos: ZERO_V,
      releaseTick: 0,
      passChain: [playerId],
    },
    players: [
      {
        id: playerId,
        team,
        role: 'spinner',
        p: ZERO_V,
        v: ZERO_V,
        q: ZERO_Q,
        line: null,
        dvBudget: 10,
        contactRef: null,
        grounded: false,
      },
    ],
    rngCursor: {},
  };
}

// ─── Play synthesiser ─────────────────────────────────────────────────────────

// Maximum spine innings before the fallback rule applies.
const MAX_SPINE_INNINGS = 10;

/**
 * Given a possession team's style profile + rng, emit 1–3 plausible SimEvents
 * that represent one "play" (one mini-possession attempt). Returns the events
 * to feed to MatchStateMachine.consume().
 *
 * Outcome weights per profile:
 *   - Loop: loopPropensity * aggression (high-risk high-reward)
 *   - Rise: freeEndBias * aggression
 *   - Fall: (1 - freeEndBias) * aggression
 *   - Ground (attacker skinned → +1 defense): (1-aggression) * 0.3
 *   - Curl: loopPropensity * 0.5 (partial arc)
 *   - Turnover (contest lost / bell_caught 3×): remainder
 *
 * Returns events for one MatchStateMachine step.
 */
function synthesisePlay(
  possession: TeamSide,
  home: Franchise,
  away: Franchise,
  rng: () => number,
  matchFaithEnd: '+x' | '-x',
): { events: SimEvent[]; sim: SimState } {
  const franchise = possession === 'home' ? home : away;
  const prof = styleToProfile(franchise.styleTag, franchise.cylinderClass);

  const pid = `${possession}-p`;
  const defPid = `${possession === 'home' ? 'away' : 'home'}-p`;

  // Roll outcome probabilities:
  // loop, rise, fall, curl, ground, turnover-via-contest, turnover-via-downs
  // Risky plays (Loop/Rise) are gated by aggression; the Fall is the bread —
  // a *patient*, low-aggression Fall team banks more of them, not fewer
  // (canon "deep is faithful"). So pFall is NOT gated by aggression.
  const pLoop = prof.loopPropensity * prof.aggression * 0.35;
  const pRise = prof.freeEndBias * prof.aggression * 0.5;
  const pFall = (1 - prof.freeEndBias) * 0.55;
  const pCurl = prof.loopPropensity * 0.15;
  const pGround = (1 - prof.aggression) * 0.2;
  const pContestWin = prof.contestAggression * (1 - prof.grappleRisk) * 0.1;
  // pContestLose + pDownsTurnover fill the rest (normalized below)

  const weights = [pLoop, pRise, pFall, pCurl, pGround, pContestWin];
  const total = weights.reduce((a, b) => a + b, 0);
  // Turnover probability is the remainder (at least 0.05 to ensure non-infinite loops)
  const pTurnover = Math.max(0.05, 1 - total);

  const allWeights = [...weights, pTurnover];
  const allTotal = allWeights.reduce((a, b) => a + b, 0);

  const roll = rng() * allTotal;
  let cumulative = 0;
  let outcome = allWeights.length - 1; // default: turnover
  for (let i = 0; i < allWeights.length; i++) {
    cumulative += allWeights[i];
    if (roll < cumulative) {
      outcome = i;
      break;
    }
  }

  // MUST match the match's actual faithEnd, else Scoring (which classifies
  // fall vs rise by end-vs-faithEnd) inverts ~half the outcomes and the
  // style divergence washes out.
  const faithEnd: '+x' | '-x' = matchFaithEnd;
  // For scoring attribution, possession team's player must be in passChain.
  // Bell x-position is set to 290 (past mouth) so that caught/bobble events
  // clear the gate and don't cause downs-turnover if we want a score.

  // outcome 0: Loop (7 pts, untouched Coriolis arc)
  if (outcome === 0) {
    const ringEnd: '+x' | '-x' =
      prof.freeEndBias > 0.5
        ? faithEnd === '+x' ? '-x' : '+x' // Rise teams prefer Free end
        : faithEnd;
    const sim = makeSim(290, pid, possession);
    return {
      events: [{ type: 'bell_through_ring', end: ringEnd, touched: false, loopTier: 'loop' }],
      sim,
    };
  }

  // outcome 1: Rise (5 pts, Free end)
  if (outcome === 1) {
    const freeEnd: '+x' | '-x' = faithEnd === '+x' ? '-x' : '+x';
    const sim = makeSim(290, pid, possession);
    return {
      events: [{ type: 'bell_through_ring', end: freeEnd, touched: true, loopTier: 'none' }],
      sim,
    };
  }

  // outcome 2: Fall (2 pts, Faith end)
  if (outcome === 2) {
    const sim = makeSim(290, pid, possession);
    return {
      events: [{ type: 'bell_through_ring', end: faithEnd, touched: true, loopTier: 'none' }],
      sim,
    };
  }

  // outcome 3: Curl (partial arc, 4–6 pts depending on end)
  if (outcome === 3) {
    const ringEnd: '+x' | '-x' =
      prof.freeEndBias > 0.5 ? (faithEnd === '+x' ? '-x' : '+x') : faithEnd;
    const sim = makeSim(290, pid, possession);
    return {
      events: [{ type: 'bell_through_ring', end: ringEnd, touched: false, loopTier: 'curl' }],
      sim,
    };
  }

  // outcome 4: Ground — attacker skinned (defender scores +1, no inning end)
  if (outcome === 4) {
    const defTeam: TeamSide = possession === 'home' ? 'away' : 'home';
    // We need attacker (possession) player in passChain, and both players in sim
    const sim: SimState = {
      tick: 0,
      omega: 0.32,
      bell: {
        p: { x: 40, y: 0, z: 0 },
        v: ZERO_V,
        q: ZERO_Q,
        w: ZERO_V,
        chime: 1,
        heldBy: null,
        thrownBy: pid,
        touchedSinceThrow: false,
        releasePos: ZERO_V,
        releaseTick: 0,
        passChain: [pid],
      },
      players: [
        {
          id: pid,
          team: possession,
          role: 'spinner',
          p: ZERO_V,
          v: ZERO_V,
          q: ZERO_Q,
          line: null,
          dvBudget: 10,
          contactRef: null,
          grounded: false,
        },
        {
          id: defPid,
          team: defTeam,
          role: 'spinner',
          p: ZERO_V,
          v: ZERO_V,
          q: ZERO_Q,
          line: null,
          dvBudget: 10,
          contactRef: null,
          grounded: false,
        },
      ],
      rngCursor: {},
    };
    // Ground does not end inning — just emit the skinned event and then follow with
    // a turnover to advance play (otherwise we'd loop forever on the same inning)
    return {
      events: [{ type: 'player_skinned', id: pid }],
      sim,
    };
  }

  // outcome 5: Contest won by possession team — cast continues; emit an
  // immediate score after (a Fall for simplicity)
  if (outcome === 5) {
    const sim = makeSim(290, pid, possession);
    return {
      events: [{ type: 'bell_through_ring', end: faithEnd, touched: true, loopTier: 'none' }],
      sim,
    };
  }

  // outcome 6 (default): Turnover via 3 failed throws
  // Bell at x=5 so it does NOT clear any gate.
  const sim = makeSim(5, pid, possession);
  return {
    events: [
      { type: 'bell_caught', by: pid },
      { type: 'bell_caught', by: pid },
      { type: 'bell_caught', by: pid },
    ],
    sim,
  };
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/**
 * Run a fully headless, deterministic, AI-vs-AI RIG match.
 *
 * @param home  — the home franchise
 * @param away  — the away franchise
 * @param seed  — master seed (uint32); same seed → identical result
 */
export function runHeadless(home: Franchise, away: Franchise, seed: number): MatchResult {
  const rng = mulberry32(seed);

  // Coin-flip faith end
  const faithEnd: '+x' | '-x' = rng() < 0.5 ? '+x' : '-x';
  // Coin-flip first possession
  const firstPossession: TeamSide = rng() < 0.5 ? 'home' : 'away';

  const msm = new MatchStateMachine(faithEnd, firstPossession);

  // Move to live phase
  msm.consume([{ type: 'foul_garrote', by: '__setup__' }], makeSim(0, 'home-p', 'home'));

  const boxHome: BoxScore = { falls: 0, rises: 0, loops: 0, curls: 0, grounds: 0 };
  const boxAway: BoxScore = { falls: 0, rises: 0, loops: 0, curls: 0, grounds: 0 };
  const highlights: HighlightEvent[] = [];

  let spineCount = 0;
  let maxSteps = 500; // hard safety cap — extremely unlikely to hit in practice

  while (msm.state.winner === null && maxSteps-- > 0) {
    const phase = msm.state.phase;

    // Advance from break/dead/set phases to live
    if (phase === 'inning_break' || phase === 'dead' || phase === 'set' || phase === 'spine') {
      msm.consume([{ type: 'foul_garrote', by: '__resume__' }], makeSim(0, 'home-p', 'home'));
      if (msm.state.spine) {
        spineCount++;
        // Spine cap: if we've been in spine too long, force the higher-seed team to win.
        if (spineCount > MAX_SPINE_INNINGS) {
          // Winner is the team with the lower seed number (better seed), or home as tiebreak.
          break;
        }
      }
      continue;
    }

    if (phase === 'final') break;

    // Generate and consume a synthesised play
    const possession = msm.state.possession;
    const { events, sim } = synthesisePlay(possession, home, away, rng, msm.state.faithEnd);

    const update = msm.consume(events, sim);

    // Track box score from scoring events
    if (update.scored) {
      const { kind, team, points } = update.scored;
      const box = team === 'home' ? boxHome : boxAway;
      if (kind === 'fall') box.falls++;
      else if (kind === 'rise') box.rises++;
      else if (kind === 'loop') box.loops++;
      else if (kind === 'curl') box.curls++;

      const inning = msm.state.inning;
      if (kind === 'loop' || kind === 'rise' || points >= 5) {
        highlights.push({ inning, kind: kind as HighlightEvent['kind'], team, points });
      }
    }

    // Track grounds from player_skinned
    // We detect this by checking if the box score would track it: ground awards +1 to defense.
    // Since Scoring.groundFor is called inside MSM, we observe it via score delta on re-read.
    // Instead, we track grounds directly from our event synthesis: if we sent a player_skinned
    // event AND the score changed (MSM awarded ground), count it.
    if (events.some((e) => e.type === 'player_skinned')) {
      // Check if a ground actually landed (diff from prev)
      // We approximate: if the defteam scored (update has no scored but scores changed),
      // we'll do a simpler tally — just count every ground play attempt as a ground.
      // More precisely: ground only lands when attackingTeam resolves correctly.
      // Our synthesisePlay always puts the attacker in passChain, so it will land.
      const defTeam: TeamSide = possession === 'home' ? 'away' : 'home';
      const defBox = defTeam === 'home' ? boxHome : boxAway;
      defBox.grounds++;
    }
  }

  // Spine cap fallback: pick winner by seed (lower seed = better; higher seed-number means lower rank)
  let winner: TeamSide;
  if (msm.state.winner !== null) {
    winner = msm.state.winner;
  } else {
    // Fallback: compare seeds; lower number wins; if both null, home wins
    const homeSeed = home.seed ?? 999;
    const awaySeed = away.seed ?? 999;
    winner = homeSeed <= awaySeed ? 'home' : 'away';
  }

  if (msm.state.spine) {
    highlights.push({
      inning: msm.state.inning,
      kind: 'spine_winner',
      team: winner,
      points: 0,
    });
  }

  return {
    home,
    away,
    scoreHome: msm.state.scoreHome,
    scoreAway: msm.state.scoreAway,
    winner,
    box: { home: boxHome, away: boxAway },
    events: highlights,
  };
}

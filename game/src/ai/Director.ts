// Director — per-team strategic AI layer, runs at ~2 Hz (every 30 sim-ticks).
//
// The Director is the team's brain. It produces a *committed* plan that the
// per-player role policies execute UNCHANGED until the next Director tick.
// This commitment is the single biggest visible-competence win: decisions are
// taken once at 2 Hz (with hysteresis), not re-rolled every 240 Hz frame, so
// players stop flickering between intents and look deliberate.
//
// Responsibilities:
//   1. Faith vs Free end EV decision (situational: score, inning, spine, posture)
//   2. Cast posture: grind / tempo / chase
//   3. Pick the loop-setter (which Spinner runs the loop play)
//   4. Contest commit / decline (committed, hysteretic)
//   5. Team coordination: designate a bell-carrier lane, assign receiver depth
//      slots, assign defensive marks (man coverage instead of all-chase)
//   6. Loose-bell recovery: designate the single closest pursuer (no scrums,
//      no deadlock — someone always fetches the bell)
//
// Parameterised by TeamProfile — different franchise philosophies emerge, and
// the deterministic style-divergence test still sees measurable Rise/Fall and
// difficulty separation because the EV math and biases are unchanged in spirit.

import type { TeamProfile } from '../league/teams';
import type { SimState, MatchState, PlayerSim, TeamSide } from '../sim/types';
import { REG, GATE_X } from '../sim/RegConstants';
import { pFall, pRise, pLoop } from './decide/ScoreEV';
import { attackRingX, attackSign, defendRingX } from './Orientation';

// ── Director output ────────────────────────────────────────────────────────────

export type CastPosture = 'grind' | 'tempo' | 'chase';

/** A committed per-player assignment, stable until the next Director tick. */
export interface PlayerAssignment {
  /** 'carry' = designated bell carrier/advancer; 'recover' = fetch loose bell;
   *  'receive' = offensive receiver slot; 'mark' = man-mark an opponent;
   *  'support' = trailing outlet; 'zone' = hold a zone. */
  job: 'carry' | 'recover' | 'receive' | 'mark' | 'support' | 'zone';
  /** Opponent id this player is marking, if job === 'mark'. */
  markId: string | null;
  /** Assigned receiver depth slot: fraction of axial distance carrier→ring
   *  ([0,1], deeper = closer to target ring), if job === 'receive'. */
  depthSlot: number;
  /** Assigned axis-radius fraction [0,1] for spacing receivers vertically. */
  radiusSlot: number;
}

export interface DirectorState {
  /** Which end the team is currently attacking toward (Faith vs Free RING —
   *  affects shot VALUE/spin, NOT physical direction). */
  attackingFree: boolean;
  /** Signed physical attack direction for THIS team: +1 → drive toward +X
   *  ring, -1 → toward -X ring. The single source of "forward". */
  attackSign: 1 | -1;
  /** World-x of the ring this team is attacking (scoring at). */
  attackRingX: number;
  /** World-x of the ring this team is defending (its own). */
  defendRingX: number;
  /** Current cast posture. */
  posture: CastPosture;
  /** PlayerId of the designated loop-setter (a Spinner), or null. */
  loopSetterId: string | null;
  /** Should the team commit to contesting the current bell in flight? */
  contestCommit: boolean;
  /**
   * E[Faith] and E[Free] values from current position — used by role policies
   * for fine-grained scoring decisions.
   */
  evFaith: number;
  evFree: number;
  /** How many ticks until the next Director update. */
  updateIn: number;

  // ── Team coordination (committed, stable for the Director window) ──────────
  /** Whether this team currently has possession (held or in-flight from us). */
  hasPossession: boolean;
  /** Is the bell loose (nobody holding it)? */
  bellLoose: boolean;
  /** Designated bell-carrier / lead-advancer player id (offense). */
  carrierId: string | null;
  /** Designated loose-bell recovery player id (the single closest pursuer). */
  recoverId: string | null;
  /** Per-player committed assignments, keyed by player id. */
  assignments: Record<string, PlayerAssignment>;
  /** A small seeded jitter in [-1,1] that is STABLE across the Director
   *  window — role policies use this for positional variety without flicker. */
  styleNoise: number;
}

// Director runs at ~2 Hz: once every 30 sim ticks.
export const DIRECTOR_TICK_INTERVAL = 30;

function axisRadius(p: { y: number; z: number }): number {
  return Math.sqrt(p.y * p.y + p.z * p.z);
}

function dist3(
  a: { x: number; y: number; z: number },
  b: { x: number; y: number; z: number },
): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Compute the EV of throwing toward the Faith (spinward) ring.
 * E[Faith] = p_fall · 2
 */
export function evFaith(
  bellPos: { x: number; y: number; z: number },
  faithX: number,
): number {
  const dist = Math.abs(bellPos.x - faithX);
  const radius = axisRadius(bellPos);
  return pFall(dist, radius) * 2;
}

/**
 * Compute the EV of throwing toward the Free (antispinward) ring.
 * E[Free] = p_rise · 5 + p_loop · 7
 */
export function evFree(
  bellPos: { x: number; y: number; z: number },
  freeX: number,
  loopPropensity: number,
): number {
  const dist = Math.abs(bellPos.x - freeX);
  const radius = axisRadius(bellPos);
  const pr = pRise(dist, radius);
  const pl = pLoop(radius, dist, loopPropensity);
  return pr * 5 + pl * 7;
}

/**
 * Choose cast posture based on score differential, inning, possession, spine.
 *
 * grind:  conservative, minimize turnovers (ahead, or late in match)
 * tempo:  balanced, normal execution
 * chase:  aggressive, take risks (behind, need big scores)
 */
function choosePosture(
  match: MatchState,
  profile: TeamProfile,
  teamSide: TeamSide,
  evFaithVal: number,
  evFreeVal: number,
): CastPosture {
  const myScore = teamSide === 'home' ? match.scoreHome : match.scoreAway;
  const oppScore = teamSide === 'home' ? match.scoreAway : match.scoreHome;
  const scoreDiff = myScore - oppScore;
  const inningsLeft = Math.max(1, 9 - match.inning + (match.spine ? 0 : 1));

  // Spine (overtime) or significantly behind → chase.
  if (scoreDiff < -6 || (match.spine && scoreDiff <= 0)) return 'chase';
  // Behind late → chase even on a smaller deficit.
  if (scoreDiff < 0 && inningsLeft <= 3) return 'chase';
  // Comfortably ahead with few innings left → grind out the clock.
  if (scoreDiff > 4 && inningsLeft <= 3) return 'grind';
  // Ahead, mid-game → mild grind for high-discipline (low-variance) teams.
  if (scoreDiff > 2 && profile.variance < 0.35) return 'grind';
  // High-variance teams chase when the Free EV is genuinely there.
  if (profile.variance > 0.65 && evFreeVal > evFaithVal * 1.2) return 'chase';
  return 'tempo';
}

/**
 * Pick which Spinner should set the loop play this cast.
 * Prefers Spinners already near the axis (high).
 */
function pickLoopSetter(players: PlayerSim[], teamSide: TeamSide): string | null {
  let best: PlayerSim | null = null;
  let bestRadius = Infinity;
  for (const p of players) {
    if (p.team !== teamSide || p.role !== 'spinner') continue;
    const r = axisRadius(p.p);
    if (r < bestRadius) {
      bestRadius = r;
      best = p;
    }
  }
  return best ? best.id : null;
}

/**
 * Decide whether to commit to a contest (one-on-one duel).
 * Pure heuristic; the *commitment* (hysteresis) lives in index.ts cache.
 */
function shouldContest(
  match: MatchState,
  profile: TeamProfile,
  teamSide: TeamSide,
  posture: CastPosture,
  contestRadius: number,
): boolean {
  let score = profile.contestAggression;

  const radiusFrac = contestRadius / REG.R;
  if (radiusFrac > 0.7) score -= 0.1; // deep is risky even for the brave
  else score += 0.05; // high (near axis) contests are reversible

  if (posture === 'chase') score += 0.15;
  if (posture === 'grind') score -= 0.1;

  // Whose contest is it? If we're the contester we're more eager; if we're the
  // thrower being contested, our "commit" means hold/fight rather than dump.
  if (match.contest) {
    const myScore = teamSide === 'home' ? match.scoreHome : match.scoreAway;
    const oppScore = teamSide === 'home' ? match.scoreAway : match.scoreHome;
    if (myScore - oppScore < 0) score += 0.08; // behind → fight for it
  }

  return score > 0.5;
}

/**
 * Assign defensive marks: each of our players man-marks the nearest unmarked
 * opponent (greedy), so defenders spread out instead of all chasing the bell.
 * The opponent holding the bell gets priority coverage.
 */
function assignMarks(
  myPlayers: PlayerSim[],
  opponents: PlayerSim[],
  bellHolderId: string | null,
  assignments: Record<string, PlayerAssignment>,
): void {
  if (opponents.length === 0) return;
  const usedOpp = new Set<string>();

  // Sort opponents by threat: bell holder first, then by axial advancement.
  const ranked = [...opponents].sort((a, b) => {
    if (a.id === bellHolderId) return -1;
    if (b.id === bellHolderId) return 1;
    return 0;
  });

  // Each defender marks the nearest still-unmarked, highest-threat opponent.
  const defenders = [...myPlayers];
  for (const opp of ranked) {
    if (defenders.length === 0) break;
    if (usedOpp.has(opp.id)) continue;
    let bestI = -1;
    let bestD = Infinity;
    for (let i = 0; i < defenders.length; i++) {
      const d = dist3(defenders[i].p, opp.p);
      if (d < bestD) {
        bestD = d;
        bestI = i;
      }
    }
    if (bestI < 0) break;
    const def = defenders.splice(bestI, 1)[0];
    assignments[def.id] = {
      job: 'mark',
      markId: opp.id,
      depthSlot: 0,
      radiusSlot: 0,
    };
    usedOpp.add(opp.id);
  }

  // Leftover defenders (more of us than them): hold a zone near our ring.
  for (const def of defenders) {
    assignments[def.id] = { job: 'zone', markId: null, depthSlot: 0, radiusSlot: 0.5 };
  }
}

/**
 * Assign offensive receiver slots: spread receivers across distinct axial
 * depths and axis-radii so the carrier always has well-separated outlets.
 */
function assignReceivers(
  receivers: PlayerSim[],
  loopSetterId: string | null,
  assignments: Record<string, PlayerAssignment>,
): void {
  // Stable ordering by id keeps slot assignment deterministic & non-flickering.
  const ordered = [...receivers].sort((a, b) => (a.id < b.id ? -1 : 1));
  const n = ordered.length;
  // GUARANTEE a safe short outlet: the first non-loop-setter receiver always
  // sits at a shallow depth (slightly AHEAD of the carrier, never deep) so the
  // carrier is never forced into a hopeless heave. Remaining receivers fan to
  // progressively deeper scoring threats.
  let nonLoopIndex = 0;
  const nonLoopCount = ordered.filter(p => p.id !== loopSetterId).length;
  for (let i = 0; i < n; i++) {
    const p = ordered[i];
    if (p.id === loopSetterId) {
      // Loop-setter sits high & near the carrier's depth (axis hand-off).
      assignments[p.id] = {
        job: 'receive',
        markId: null,
        depthSlot: 0.4,
        radiusSlot: 0.15,
      };
      continue;
    }
    const k = nonLoopIndex++;
    if (k === 0) {
      // The mandatory safe outlet: shallow, mid-radius, always reachable.
      assignments[p.id] = {
        job: 'receive',
        markId: null,
        depthSlot: 0.18,
        radiusSlot: 0.35,
      };
      continue;
    }
    // Spread the rest from a mid outlet to a deep scoring threat.
    const frac = nonLoopCount > 2 ? (k - 1) / (nonLoopCount - 2) : 0.6;
    const depthSlot = 0.4 + frac * 0.55; // 0.40 (mid) .. 0.95 (deep at ring)
    // Alternate radius bands by role so wings keep their identity.
    let radiusSlot: number;
    switch (p.role) {
      case 'freewing':
        radiusSlot = 0.22;
        break;
      case 'faithwing':
        radiusSlot = 0.6;
        break;
      case 'anchor':
        radiusSlot = 0.78;
        break;
      case 'spinner':
        radiusSlot = 0.42;
        break;
      default:
        radiusSlot = 0.4 + 0.3 * frac;
    }
    assignments[p.id] = { job: 'receive', markId: null, depthSlot, radiusSlot };
  }
}

/**
 * Run the Director for one team.
 * Call this at ~2 Hz; cache the result. The returned plan is meant to be
 * held CONSTANT by every per-player policy until the next Director tick.
 */
export function runDirector(
  state: SimState,
  match: MatchState,
  profile: TeamProfile,
  teamSide: TeamSide,
  rng: () => number,
): DirectorState {
  // ── ORIENTATION: physical attack direction is FIXED by team, NOT faithEnd.
  //   home → +X ring, away → -X ring (see Orientation.ts / match rules).
  const aSign = attackSign(teamSide);
  const aRingX = attackRingX(teamSide);
  const dRingX = defendRingX(teamSide);
  // Is the ring we physically attack the Faith (2-pt) or Free (5-pt) ring?
  const faithX = match.faithEnd === '+x' ? GATE_X : -GATE_X;
  const freeX = -faithX;
  const ourRingIsFaith = Math.sign(aRingX) === Math.sign(faithX);

  // evFaith/evFree keep their LEGACY meaning: the EV of a shot toward the
  // PHYSICAL Faith / Free ring from the bell's position, profile-parameterised.
  // These feed posture + style divergence (and the divergence CI test); they
  // are direction-agnostic by construction (abs distance to a fixed ring).
  const ef = evFaith(state.bell.p, faithX);
  const efr = evFree(state.bell.p, freeX, profile.loopPropensity);

  const posture = choosePosture(match, profile, teamSide, ef, efr);

  // `attackingFree` = the ring THIS team physically attacks is the Free
  // (5-pt) ring. This drives shot VALUE/spin choice, never direction. The
  // Rise-vs-Fall profile still tilts loop/curl willingness elsewhere, so the
  // style-divergence test (which checks evFree bias + loopSetter counts) and
  // the attackingFree-divergence test both still hold.
  const attackingFree = !ourRingIsFaith;

  const useLoop = profile.loopPropensity > 0.25 && posture !== 'grind';
  const loopSetterId = useLoop ? pickLoopSetter(state.players, teamSide) : null;

  // Possession / loose-bell determination.
  const myPlayers = state.players.filter(p => p.team === teamSide && p.id !== 'P1');
  const opponents = state.players.filter(p => p.team !== teamSide);
  const allMyPlayers = state.players.filter(p => p.team === teamSide);
  const holder = state.bell.heldBy
    ? state.players.find(p => p.id === state.bell.heldBy) ?? null
    : null;
  const bellLoose = holder === null;
  const heldByUs = holder !== null && holder.team === teamSide;
  // In-flight from us counts as possession (we keep attacking shape).
  const thrownByUs =
    bellLoose &&
    state.bell.thrownBy != null &&
    (state.players.find(p => p.id === state.bell.thrownBy)?.team === teamSide);
  const hasPossession = heldByUs || thrownByUs || match.possession === teamSide;

  // Contest decision.
  let contestCommit = false;
  if (match.contest !== null) {
    const contester = state.players.find(p => p.id === match.contest!.contester);
    const contestRadius = contester ? axisRadius(contester.p) : REG.R * 0.5;
    contestCommit = shouldContest(match, profile, teamSide, posture, contestRadius);
  }

  // ── Assignments (committed for the Director window) ───────────────────────
  const assignments: Record<string, PlayerAssignment> = {};
  let carrierId: string | null = null;
  let recoverId: string | null = null;

  if (bellLoose) {
    // ROBUSTNESS: exactly one designated recoverer — the closest of OUR
    // players to the bell — so the team never deadlocks on a loose bell and
    // never collapses everyone onto it. Tie-break by id for determinism.
    let best: PlayerSim | null = null;
    let bestD = Infinity;
    for (const p of myPlayers) {
      const d = dist3(p.p, state.bell.p);
      if (d < bestD - 1e-6 || (Math.abs(d - bestD) <= 1e-6 && best && p.id < best.id)) {
        bestD = d;
        best = p;
      }
    }
    recoverId = best ? best.id : null;
    for (const p of myPlayers) {
      if (p.id === recoverId) {
        assignments[p.id] = { job: 'recover', markId: null, depthSlot: 0, radiusSlot: 0 };
      } else if (hasPossession || thrownByUs) {
        // Our throw in flight: keep offensive receiving shape.
        assignments[p.id] = { job: 'receive', markId: null, depthSlot: 0.6, radiusSlot: 0.4 };
      } else {
        // Their loose bell about to be theirs: pre-mark.
        assignments[p.id] = { job: 'zone', markId: null, depthSlot: 0, radiusSlot: 0.5 };
      }
    }
    if (hasPossession || thrownByUs) {
      const recvs = myPlayers.filter(p => p.id !== recoverId);
      assignReceivers(recvs, loopSetterId, assignments);
      if (recoverId)
        assignments[recoverId] = { job: 'recover', markId: null, depthSlot: 0, radiusSlot: 0 };
    } else {
      assignMarks(myPlayers.filter(p => p.id !== recoverId), opponents, null, assignments);
      if (recoverId)
        assignments[recoverId] = { job: 'recover', markId: null, depthSlot: 0, radiusSlot: 0 };
    }
  } else if (heldByUs) {
    // OFFENSE: the holder is the carrier; the rest are receivers in slots.
    carrierId = holder!.id;
    const receivers = myPlayers.filter(p => p.id !== carrierId);
    assignReceivers(receivers, loopSetterId, assignments);
    // If the carrier is a non-AI/excluded player (e.g. P1 in single-player),
    // carrierId may not be in myPlayers — that's fine, slots still hold.
    if (myPlayers.some(p => p.id === carrierId)) {
      assignments[carrierId] = { job: 'carry', markId: null, depthSlot: 1, radiusSlot: 0.4 };
    }
  } else {
    // DEFENSE: the opponent holds it — man-mark, prioritising the carrier.
    assignMarks(myPlayers, opponents, holder ? holder.id : null, assignments);
  }

  // Stable style noise for the window (decorrelated, deterministic).
  const styleNoise = rng() * 2 - 1;

  // Safety net: any of our players without an assignment gets a sane default
  // so nobody ever idles.
  for (const p of allMyPlayers) {
    if (p.id === 'P1') continue;
    if (!assignments[p.id]) {
      assignments[p.id] = hasPossession
        ? { job: 'support', markId: null, depthSlot: 0.4, radiusSlot: 0.45 }
        : { job: 'zone', markId: null, depthSlot: 0, radiusSlot: 0.5 };
    }
  }

  return {
    attackingFree,
    attackSign: aSign,
    attackRingX: aRingX,
    defendRingX: dRingX,
    posture,
    loopSetterId,
    contestCommit,
    evFaith: ef,
    evFree: efr,
    updateIn: DIRECTOR_TICK_INTERVAL,
    hasPossession,
    bellLoose,
    carrierId,
    recoverId,
    assignments,
    styleNoise,
  };
}

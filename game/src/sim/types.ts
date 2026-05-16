// FROZEN sim/match/input interface. Every module codes against THIS.
// render/ audio/ ui/ input/ may import only this file + core/EventBus.
// Changing these shapes is a deliberate cross-cutting decision.

import type { Vec3, Quat } from './vec';

export type TeamSide = 'home' | 'away';
export type RiggerRole = 'anchor' | 'spinner' | 'faithwing' | 'freewing' | 'reach';
export type PlayerId = string;

export interface BellState {
  p: Vec3;
  v: Vec3;
  q: Quat; // orientation
  w: Vec3; // angular velocity (body frame)
  chime: number; // [0,1] spin-trueness (1 = rings true, 0 = clatters)
  heldBy: PlayerId | null;
  thrownBy: PlayerId | null;
  touchedSinceThrow: boolean;
  releasePos: Vec3;
  releaseTick: number;
  passChain: PlayerId[];
}

export interface GrappleState {
  anchorType: 'spar' | 'skin' | 'player' | 'ring';
  anchorRef: string | null; // spar/player id, null for free-space skin point
  anchorPos: Vec3;
  restLen: number;
  taut: boolean;
}

export interface PlayerSim {
  id: PlayerId;
  team: TeamSide;
  role: RiggerRole;
  p: Vec3;
  v: Vec3;
  q: Quat;
  line: GrappleState | null;
  dvBudget: number; // thrumbler delta-v remaining this possession
  contactRef: string | null; // spar/ring/player currently clipped to, or null
  grounded: boolean; // skin contact (out of the calm)
}

export interface SimState {
  tick: number; // sim ticks since match start — the ONLY clock
  omega: number;
  bell: BellState;
  players: PlayerSim[];
  // serialisable RNG cursor (substream -> draw count) for replay re-seeding
  rngCursor: Record<string, number>;
}

export type MatchPhase =
  | 'set'
  | 'live'
  | 'contest'
  | 'dead'
  | 'inning_break'
  | 'spine'
  | 'final';

export interface MatchState {
  inning: number; // 1..9, then spine
  spine: boolean;
  possession: TeamSide;
  faithEnd: '+x' | '-x'; // which ring is the spinward/Faith end this match
  cast: { throwsLeft: 0 | 1 | 2 | 3; gate: 'first' | 'deep' | 'mouth'; spotX: number };
  contest: null | {
    thrower: PlayerId;
    contester: PlayerId;
    count: 0 | 1 | 2 | 3;
    radius: number;
    direction: 'fair' | 'cross';
  };
  scoreHome: number;
  scoreAway: number;
  phase: MatchPhase;
  message: string;
  winner: TeamSide | null;
}

// One tick of input for one controlled player. The replay unit.
export interface PlayerInput {
  id: PlayerId;
  aim: Vec3; // 3D aim direction (unit-ish)
  fireLineAt: Vec3 | null; // anchor world point, or null
  reel: -1 | 0 | 1; // in / none / out
  release: boolean;
  pushoff: boolean;
  throwCharge: number; // [0,1]
  throwReleased: boolean; // fire this tick (when holding the bell)
  throwSpin: number; // [-1,1] spin imparted (the curve weapon)
  thrumbler: Vec3; // small delta-v request (capped by budget in sim)
}

export interface InputFrame {
  tick: number;
  players: PlayerInput[];
}

// Events the sim emits (consumed by match rules + render/audio, read-only).
export type SimEvent =
  | {
      type: 'bell_through_ring';
      end: '+x' | '-x';
      touched: boolean;
      loopTier: 'loop' | 'curl' | 'none';
    }
  | { type: 'bell_caught'; by: PlayerId }
  | { type: 'bell_bobble'; by: PlayerId }
  | { type: 'bell_clatter'; by: PlayerId | null }
  | { type: 'bell_skin' }
  | { type: 'player_skinned'; id: PlayerId }
  | { type: 'contest_started'; thrower: PlayerId; contester: PlayerId }
  | { type: 'foul_garrote'; by: PlayerId };

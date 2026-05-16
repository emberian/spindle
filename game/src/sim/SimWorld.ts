// The deterministic world. Pure given (state, inputs): one fixed sub-step of
// physics + possession + ring/loop detection. Emits SimEvents; the match
// rules (P3) consume them. No DOM, no wall-clock, no Math.random.

import type { Vec3 } from './vec';
import { vadd, vsub, vlen, vscale, vnorm } from './vec';
import type { InputFrame, PlayerInput, SimState, SimEvent } from './types';
import { REG, GATE_X } from './RegConstants';
import { stepBell, chime, type BellBody } from './Bell';
import { skinBounce, tryCatch, applyBobble, contestClatter } from './Collision';
import { stepPlayer, pushOff, thrumbler, type PlayerBody, makePlayer } from './Player';
import { type Line, TETHER_MAX } from './Grapple';
import { LoopTracker } from './LoopDetector';
import { Rng } from './rng';

const THROW_MIN = 9;
const THROW_MAX = 34;

interface World {
  id: string;
  team: 'home' | 'away';
  role: SimState['players'][number]['role'];
  body: PlayerBody;
}

export class SimWorld {
  tick = 0;
  bell: BellBody = {
    p: { x: 0, y: 0, z: 0 },
    v: { x: 0, y: 0, z: 0 },
    q: { x: 0, y: 0, z: 0, w: 1 },
    w: { x: 0, y: 0, z: 0 },
  };
  bellHeldBy: string | null = null;
  bellThrownBy: string | null = null;
  bellTouched = true;
  passChain: string[] = [];
  releasePos: Vec3 = { x: 0, y: 0, z: 0 };
  releaseTick = 0;
  players: World[] = [];
  private loop = new LoopTracker();
  private rng: Rng;
  private events: SimEvent[] = [];

  constructor(seed: number) {
    this.rng = new Rng(seed);
  }

  addPlayer(id: string, team: 'home' | 'away', role: World['role'], p: Vec3): void {
    this.players.push({ id, team, role, body: makePlayer(p) });
  }

  /** Place the bell free with a velocity & spin (used by set/throw/tests). */
  launchBell(p: Vec3, v: Vec3, w: Vec3, thrownBy: string | null): void {
    this.bell.p = { ...p };
    this.bell.v = { ...v };
    this.bell.w = { ...w };
    this.bellHeldBy = null;
    this.bellThrownBy = thrownBy;
    this.bellTouched = false;
    this.releasePos = { ...p };
    this.releaseTick = this.tick;
    if (thrownBy) this.passChain.push(thrownBy);
    this.loop.onRelease(p);
  }

  private find(id: string): World | undefined {
    return this.players.find((w) => w.id === id);
  }

  private applyInput(inp: PlayerInput): void {
    const w = this.find(inp.id);
    if (!w) return;
    const pl = w.body;

    // Grapple
    if (inp.fireLineAt) {
      const len = vlen(vsub(pl.p, inp.fireLineAt));
      pl.line = {
        anchorPos: { ...inp.fireLineAt },
        anchorBody: null,
        restLen: Math.min(TETHER_MAX, Math.max(3, len)),
        taut: false,
      } as Line;
    } else if (inp.release && pl.line) {
      pl.line = null;
    }
    if (inp.pushoff) pushOff(pl, inp.aim, 7);
    if (inp.thrumbler) thrumbler(pl, inp.thrumbler);

    // Throw (only if holding the bell)
    if (inp.throwReleased && this.bellHeldBy === inp.id) {
      const dir = vnorm(inp.aim);
      const speed = THROW_MIN + Math.max(0, Math.min(1, inp.throwCharge)) * (THROW_MAX - THROW_MIN);
      const v = vadd(pl.v, vscale(dir, speed));
      // Spin: mostly on the ring axis (rings true); throwSpin tilts it.
      const s = Math.max(-1, Math.min(1, inp.throwSpin));
      this.launchBell(pl.p, v, { x: 26, y: s * 7, z: 0 }, inp.id);
    }
  }

  private reelOf(inp: PlayerInput | undefined): -1 | 0 | 1 {
    return inp ? inp.reel : 0;
  }

  /** Advance one fixed sub-step `h`. Returns events emitted this step. */
  step(frame: InputFrame, h: number): SimEvent[] {
    this.events = [];
    const inMap = new Map(frame.players.map((p) => [p.id, p]));
    for (const pi of frame.players) this.applyInput(pi);

    // Players
    for (const w of this.players) {
      const wasGround = w.body.grounded;
      stepPlayer(w.body, h, this.reelOf(inMap.get(w.id)));
      if (w.body.grounded && !wasGround) {
        this.events.push({ type: 'player_skinned', id: w.id });
      }
    }

    // Bell
    if (this.bellHeldBy) {
      const holder = this.find(this.bellHeldBy);
      if (holder) {
        this.bell.p = { ...holder.body.p };
        this.bell.v = { ...holder.body.v };
      }
    } else {
      const prevX = this.bell.p.x;
      this.bell = stepBell(this.bell, REG.omega, h);
      if (skinBounce(this.bell)) {
        this.loop.onTouch();
        this.bellTouched = true;
        this.events.push({ type: 'bell_skin' });
      }
      this.loop.update(this.bell.v);

      // Catch / contest by proximity. A just-released bell ignores ALL
      // contact briefly, and ignores its thrower until it has clearly
      // separated — you cannot bobble your own throw.
      const sinceRelease = this.tick - this.releaseTick;
      for (const w of this.players) {
        if (w.body.grounded) continue;
        if (sinceRelease < 8) continue;
        if (
          w.id === this.bellThrownBy &&
          vlen(vsub(this.bell.p, w.body.p)) < 4
        )
          continue;
        const r = tryCatch(this.bell.p, this.bell.v, w.body.p, w.body.v);
        if (r === 'caught') {
          this.bellHeldBy = w.id;
          this.bellTouched = true;
          this.loop.onTouch();
          if (this.passChain[this.passChain.length - 1] !== w.id) this.passChain.push(w.id);
          this.events.push({ type: 'bell_caught', by: w.id });
          break;
        } else if (r === 'bobble') {
          applyBobble(this.bell, w.body.v);
          this.loop.onTouch();
          this.bellTouched = true;
          this.events.push({ type: 'bell_bobble', by: w.id });
          break;
        } else {
          // a defender brushing a fast bell clatters it
          const near = vlen(vsub(this.bell.p, w.body.p));
          if (near < 2.2 && this.bellThrownBy && w.team !== this.teamOf(this.bellThrownBy)) {
            contestClatter(this.bell, w.body.v, 1);
            this.loop.onTouch();
            this.bellTouched = true;
            this.events.push({ type: 'bell_clatter', by: w.id });
            break;
          }
        }
      }

      // Ring crossing → score event.
      this.checkRing(prevX);
    }

    this.tick++;
    return this.events;
  }

  /** Live loop status for the loop-cam / audio hush (render-only read). */
  loopInfo(): { free: boolean; untouched: boolean; turn: number } {
    return {
      free: this.bellHeldBy === null,
      untouched: this.loop.untouched,
      turn: this.loop.turn,
    };
  }

  private teamOf(id: string): 'home' | 'away' | null {
    return this.find(id)?.team ?? null;
  }

  private checkRing(prevX: number): void {
    const x = this.bell.p.x;
    const rho = Math.hypot(this.bell.p.y, this.bell.p.z);
    const through = rho <= REG.gateRadius;
    const crossed = (end: number) =>
      (prevX < end && x >= end) || (prevX > end && x <= end);
    if (crossed(GATE_X) && through) {
      this.events.push({
        type: 'bell_through_ring',
        end: '+x',
        touched: this.bellTouched,
        closedArc: this.loop.isLoop(this.bell.p),
      });
      this.bellThrownBy = null;
    } else if (crossed(-GATE_X) && through) {
      this.events.push({
        type: 'bell_through_ring',
        end: '-x',
        touched: this.bellTouched,
        closedArc: this.loop.isLoop(this.bell.p),
      });
      this.bellThrownBy = null;
    }
  }

  /** Immutable-ish snapshot for render/replay/hashing. */
  snapshot(): SimState {
    return {
      tick: this.tick,
      omega: REG.omega,
      bell: {
        p: { ...this.bell.p },
        v: { ...this.bell.v },
        q: { ...this.bell.q },
        w: { ...this.bell.w },
        chime: chime(this.bell.w),
        heldBy: this.bellHeldBy,
        thrownBy: this.bellThrownBy,
        touchedSinceThrow: this.bellTouched,
        releasePos: { ...this.releasePos },
        releaseTick: this.releaseTick,
        passChain: [...this.passChain],
      },
      players: this.players.map((w) => ({
        id: w.id,
        team: w.team,
        role: w.role,
        p: { ...w.body.p },
        v: { ...w.body.v },
        q: { x: 0, y: 0, z: 0, w: 1 },
        line: w.body.line
          ? {
              anchorType: 'spar',
              anchorRef: null,
              anchorPos: { ...w.body.line.anchorPos },
              restLen: w.body.line.restLen,
              taut: w.body.line.taut,
            }
          : null,
        dvBudget: w.body.dvBudget,
        contactRef: w.body.contact ? 'spar' : null,
        grounded: w.body.grounded,
      })),
      rngCursor: this.rng.cursor(),
    };
  }
}

// Stable hash of a snapshot for determinism tests.
export function hashSnapshot(s: SimState): string {
  const round = (n: number) => Math.round(n * 1e6) / 1e6;
  const parts: number[] = [s.tick, round(s.bell.p.x), round(s.bell.p.y), round(s.bell.p.z), round(s.bell.v.x), round(s.bell.v.y), round(s.bell.v.z), round(s.bell.w.x), round(s.bell.w.y), round(s.bell.w.z)];
  for (const p of s.players) parts.push(round(p.p.x), round(p.p.y), round(p.p.z), round(p.v.x), round(p.v.y), round(p.v.z));
  let h = 2166136261 >>> 0;
  const str = parts.join(',');
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return (h >>> 0).toString(16);
}

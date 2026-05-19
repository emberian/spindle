// The deterministic world. Pure given (state, inputs): one fixed sub-step of
// physics + possession + ring/loop detection. Emits SimEvents; the match
// rules (P3) consume them. No DOM, no wall-clock, no Math.random.

import type { Vec3 } from './vec';
import { vadd, vsub, vlen, vscale, vnorm } from './vec';
import type { InputFrame, PlayerInput, SimState, SimEvent } from './types';
import { REG, GATE_X, FEEL } from './RegConstants';
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
  // PART B (twin of sim_world.rs): latched once a `contest_started` has
  // been emitted for the current loose-bell episode (one-shot per loose
  // ball). Reset on launch/grip/catch. Not a physics quantity ⇒ not in
  // hashSnapshot; only gates a deterministic one-shot event.
  private contestEmitted = false;
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

  /** Grip the bell to a player (cast start / re-arm). Mirrors the WASM
   *  runtime's setBellHeld so the TS twin can drive full headless matches
   *  (the fast in-process skill harness). Deterministic; no rng. */
  setBellHeld(id: string): void {
    const w = this.find(id);
    if (!w) return;
    this.bellHeldBy = id;
    this.bellThrownBy = null;
    this.bellTouched = true;
    this.passChain = [];
    this.contestEmitted = false;
    this.bell.p = { ...w.body.p };
    this.bell.v = { x: 0, y: 0, z: 0 };
    this.bell.w = { x: 0, y: 0, z: 0 };
  }

  /** Place the bell free with a velocity & spin (used by set/throw/tests). */
  launchBell(p: Vec3, v: Vec3, w: Vec3, thrownBy: string | null): void {
    this.bell.p = { ...p };
    this.bell.v = { ...v };
    this.bell.w = { ...w };
    this.bellHeldBy = null;
    this.bellThrownBy = thrownBy;
    this.bellTouched = false;
    this.contestEmitted = false; // new loose-bell episode
    this.releasePos = { ...p };
    this.releaseTick = this.tick;
    if (thrownBy) this.passChain.push(thrownBy);
    this.loop.onRelease(p);
  }

  private find(id: string): World | undefined {
    return this.players.find((w) => w.id === id);
  }

  private applyInput(inp: PlayerInput, h: number): void {
    const w = this.find(inp.id);
    if (!w) return;
    const pl = w.body;

    // Grapple — GRAPPLE LATENCY (the catch fix), mirrors rig-core
    // sim_world.rs apply_input. A fire is IGNORED if a line is already
    // present (claw in flight OR attached — committed to target, no silent
    // per-tick re-anchor) OR the re-fire cooldown has not elapsed
    // (tick < refireReadyTick). Otherwise the claw launches: the line
    // exists immediately (blocks re-aim) but attaches only after
    // ceil(dist / CLAW_SPEED / h) ticks. Integer ticks, no wall clock.
    if (inp.fireLineAt) {
      const blocked = pl.line !== null || this.tick < pl.refireReadyTick;
      if (!blocked) {
        const len = vlen(vsub(pl.p, inp.fireLineAt));
        const flightTicks = Math.max(1, Math.ceil(len / FEEL.CLAW_SPEED / h));
        pl.line = {
          anchorPos: { ...inp.fireLineAt },
          anchorBody: null,
          restLen: Math.min(TETHER_MAX, Math.max(3, len)),
          taut: false,
          attached: false,
          attachTick: this.tick + flightTicks,
        } as Line;
      }
    } else if (inp.release && pl.line) {
      // Explicit release cancels the line (in flight or attached) and
      // starts the re-fire cooldown.
      pl.line = null;
      pl.refireReadyTick = this.tick + FEEL.REFIRE_COOLDOWN_TICKS;
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
    for (const pi of frame.players) this.applyInput(pi, h);

    // GRAPPLE LATENCY: flip in-flight claws to attached once the sim tick
    // reaches their landing tick — BEFORE stepping players so a claw landing
    // this tick exerts its first constraint force this same tick. Integer
    // compare, deterministic (mirrors rig-core sim_world.rs).
    for (const w of this.players) {
      const ln = w.body.line;
      if (ln && !ln.attached && this.tick >= ln.attachTick) {
        ln.attached = true;
      }
    }

    // Players
    for (const w of this.players) {
      const wasGround = w.body.grounded;
      stepPlayer(w.body, h, this.reelOf(inMap.get(w.id)));
      if (w.body.grounded && !wasGround) {
        this.events.push({ type: 'player_skinned', id: w.id });
      }
    }

    // PART A — PLAYER↔PLAYER SOFT-BODY COLLISION (twin of
    // rig-core/src/sim_world.rs). One O(n²) pairwise pass resolves
    // penetration of two PLAYER_RADIUS spheres with a momentum-conserving
    // soft spring + damper applied EQUAL-AND-OPPOSITE along the contact
    // normal. Fixed (i, j) index order ⇒ deterministic, no Math.random.
    // Symplectic; equal player mass ⇒ linear momentum conserved exactly.
    {
      const twoR = 2 * FEEL.PLAYER_RADIUS;
      const twoR2 = twoR * twoR;
      const n = this.players.length;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = this.players[i].body;
          const b = this.players[j].body;
          const dx = a.p.x - b.p.x;
          const dy = a.p.y - b.p.y;
          const dz = a.p.z - b.p.z;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 >= twoR2 || d2 < 1e-18) continue;
          const dist = Math.sqrt(d2);
          const inv = 1 / dist;
          const nx = dx * inv;
          const ny = dy * inv;
          const nz = dz * inv;
          const pen = twoR - dist;
          const rvx = a.v.x - b.v.x;
          const rvy = a.v.y - b.v.y;
          const rvz = a.v.z - b.v.z;
          const vRelN = rvx * nx + rvy * ny + rvz * nz;
          const f = FEEL.COLLIDE_K * pen - FEEL.COLLIDE_C * vRelN;
          const ka = f * a.invMass * h;
          const kb = f * b.invMass * h;
          a.v = { x: a.v.x + nx * ka, y: a.v.y + ny * ka, z: a.v.z + nz * ka };
          b.v = { x: b.v.x - nx * kb, y: b.v.y - ny * kb, z: b.v.z - nz * kb };
        }
      }
    }

    // PART B — GARROTE foul detection (twin of sim_world.rs). A fired line
    // whose taut, attached segment (hand → effective anchor) sweeps within
    // GARROTE_RADIUS of an OPPOSING rigger's body centre is a foul —
    // independent of bell state. Per (line, victim) pair, fixed id order;
    // first hit wins ⇒ deterministic, no Math.random.
    {
      const gr2 = FEEL.GARROTE_RADIUS * FEEL.GARROTE_RADIUS;
      let garroteBy: string | null = null;
      outer: for (let li = 0; li < this.players.length; li++) {
        const owner = this.players[li];
        const ln = owner.body.line;
        if (!ln || !ln.attached || !ln.taut) continue;
        const a = owner.body.p;
        const b = ln.anchorBody ? ln.anchorBody.p : ln.anchorPos;
        for (let vi = 0; vi < this.players.length; vi++) {
          if (vi === li) continue;
          const victim = this.players[vi];
          if (victim.team === owner.team || victim.body.grounded) continue;
          if (segPointDist2(a, b, victim.body.p) < gr2) {
            garroteBy = owner.id;
            break outer;
          }
        }
      }
      if (garroteBy !== null) {
        this.events.push({ type: 'foul_garrote', by: garroteBy });
      }
    }

    // Bell
    if (this.bellHeldBy) {
      const holder = this.find(this.bellHeldBy);
      if (holder) {
        // Held bell follows the holder's hand via a stiff spring (continuous
        // tracking, no raw p/v copy). Unit mass; semi-implicit Euler
        // (ω_n·h ≈ 0.118 ≪ 2 ⇒ stable, tight). The throw/release velocity
        // math in applyInput is unchanged — it reads the PLAYER's v.
        const hp = holder.body.p;
        const hv = holder.body.v;
        const dx = this.bell.p.x - hp.x;
        const dy = this.bell.p.y - hp.y;
        const dz = this.bell.p.z - hp.z;
        const fx = -FEEL.HOLD_K * dx - FEEL.HOLD_C * (this.bell.v.x - hv.x);
        const fy = -FEEL.HOLD_K * dy - FEEL.HOLD_C * (this.bell.v.y - hv.y);
        const fz = -FEEL.HOLD_K * dz - FEEL.HOLD_C * (this.bell.v.z - hv.z);
        this.bell.v = {
          x: this.bell.v.x + fx * h,
          y: this.bell.v.y + fy * h,
          z: this.bell.v.z + fz * h,
        };
        this.bell.p = {
          x: this.bell.p.x + this.bell.v.x * h,
          y: this.bell.p.y + this.bell.v.y * h,
          z: this.bell.p.z + this.bell.v.z * h,
        };
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

      // PART B — CONTEST detection (twin of sim_world.rs). Loose bell: if
      // two opposing non-grounded riggers are BOTH within CONTEST_RADIUS
      // of it, emit `contest_started` ONCE per loose-bell episode so the
      // match SM's Contest path runs (it resolves from the physics, no
      // rng). Thrower = attacking side's closest rigger; contester =
      // opposing side's closest. Fixed order, lowest index breaks ties.
      if (!this.contestEmitted && sinceRelease >= 8 && this.bellThrownBy) {
        const attTeam = this.teamOf(this.bellThrownBy);
        if (attTeam) {
          const cr = FEEL.CONTEST_RADIUS;
          let atk: { i: number; d: number } | null = null;
          let def: { i: number; d: number } | null = null;
          for (let k = 0; k < this.players.length; k++) {
            const w = this.players[k];
            if (w.body.grounded) continue;
            const d = vlen(vsub(w.body.p, this.bell.p));
            if (d > cr) continue;
            if (w.team === attTeam) {
              if (atk === null || d < atk.d) atk = { i: k, d };
            } else if (def === null || d < def.d) {
              def = { i: k, d };
            }
          }
          if (atk !== null && def !== null) {
            this.events.push({
              type: 'contest_started',
              thrower: this.players[atk.i].id,
              contester: this.players[def.i].id,
            });
            this.contestEmitted = true;
          }
        }
      }

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
          this.contestEmitted = false;
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
        loopTier: this.loop.tier(this.bell.p),
      });
      this.bellThrownBy = null;
    } else if (crossed(-GATE_X) && through) {
      this.events.push({
        type: 'bell_through_ring',
        end: '-x',
        touched: this.bellTouched,
        loopTier: this.loop.tier(this.bell.p),
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
              attached: w.body.line.attached,
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

// Squared distance from point `p` to segment `a`→`b` (clamped projection).
// Pure; byte-identical to rig-core/src/sim_world.rs seg_point_dist2.
function segPointDist2(a: Vec3, b: Vec3, p: Vec3): number {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const abz = b.z - a.z;
  const ab2 = abx * abx + aby * aby + abz * abz;
  let t = 0;
  if (ab2 >= 1e-18) {
    t = ((p.x - a.x) * abx + (p.y - a.y) * aby + (p.z - a.z) * abz) / ab2;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
  }
  const cx = a.x + abx * t;
  const cy = a.y + aby * t;
  const cz = a.z + abz * t;
  const dx = p.x - cx;
  const dy = p.y - cy;
  const dz = p.z - cz;
  return dx * dx + dy * dy + dz * dz;
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

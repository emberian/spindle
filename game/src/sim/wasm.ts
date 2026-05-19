// TS ↔ WASM binding. Thin wrapper over the Rust `RigSim` facade that
// decodes the documented flat snapshot into the frozen SimState shape so
// render/UI consume the deterministic Rust core exactly like the TS sim.
//
// The Rust core is the source of truth; the TS sim/ modules remain the
// parity oracle (cargo known-answer test holds RNG bit-identical).

import init, { RigSim, RigAi, RigPolicy } from '../../rig-core/pkg/rig_core.js';
import type {
  SimState,
  InputFrame,
  TeamSide,
  RiggerRole,
  PlayerSim,
  SimEvent,
} from './types';

let ready: Promise<void> | null = null;
export function wasmReady(): Promise<void> {
  if (!ready) ready = init().then(() => undefined);
  return ready;
}

// ── Rust AI boundary (RigAi) ──────────────────────────────────────────────────
// The AI is the ported Rust `AiSystem` behind the wasm `RigAi` struct.
// One persistent instance (it owns the director/commit caches across
// ticks, exactly like the old TS AiSystem). Created lazily after init();
// the sim is always constructed first (createWasmSim awaits wasmReady),
// so the wasm module is initialised by the time the first AI tick runs.
let aiInst: RigAi | null = null;
function rigAi(): RigAi {
  if (!aiInst) aiInst = new RigAi();
  return aiInst;
}
/** One AI tick. JSON in (SimState/MatchState/TeamConfig[]) → InputFrame JSON. */
export function aiTick(
  simJson: string,
  matchJson: string,
  configsJson: string,
  seed: number,
): string {
  return rigAi().tick(simJson, matchJson, configsJson, seed >>> 0);
}
/** Clear the AI's director + commitment caches (new inning). */
export function aiReset(): void {
  aiInst?.reset();
}

// ── RENDER-ONLY legibility seam ───────────────────────────────────────────────
// One per-rigger record, PARALLEL to `snapshot_meta.playerIds` / the emitted
// InputFrame players. It is a pure deterministic function of the committed AI
// state, crosses its OWN JSON method (never the InputFrame), and is EXCLUDED
// from `hash_snapshot` / `sim.step` — exactly the `lineAnchors` rigor, only it
// lives in the AI layer. It exists ONLY to make the swarm's coordination
// human-watchable in spectate; nothing in the sim or replay reads it.
export type ControlledBy = 'baseline' | 'rl' | 'human';
export interface AiDebugRec {
  id: string;
  /** Structural role: anchor/spinner/faithwing/freewing/reach. */
  role: string;
  /** Committed intent verb: carry/recover/receive/mark/support/zone, or
   *  'learned' for an RL-driven rigger (no Director Job). */
  job: string;
  /** The committed world point the rigger is acting on, or null when it
   *  is holding/throwing in place (the bell IS the rigger then). */
  intentTargetPos: { x: number; y: number; z: number } | null;
  isDiver: boolean;
  isContester: boolean;
  isPrimary: boolean;
  isShadow: boolean;
  isOutlet: boolean;
  controlledBy: ControlledBy;
}

/** The baseline AI's render-only legibility records from the last `aiTick`. */
export function aiDebug(): AiDebugRec[] {
  if (!aiInst) return [];
  return JSON.parse(aiInst.ai_debug_json()) as AiDebugRec[];
}

// ── Trained RL policy boundary (RigPolicy) ────────────────────────────────────
// The browser's seam to the learned shared-parameter MLP. One `RigPolicy`
// per policy-driven team (it owns the internal baseline `AiSystem` that
// fills riggers NOT in the policy's controlled set). Same JSON contract as
// `RigAi` plus a controlled-ids array. Pure f64 forward ⇒ deterministic;
// the recorded `InputFrame`s capture its emitted inputs ⇒ replay-stable.
export class PolicyAi {
  private pol: RigPolicy;
  constructor(weightsJson: string) {
    this.pol = new RigPolicy(weightsJson);
  }
  /** One policy tick. The riggers in `controlledIds` get the learned
   *  policy; every other rigger falls to the internal baseline AI. */
  tick(
    simJson: string,
    matchJson: string,
    controlledIdsJson: string,
    seed: number,
  ): string {
    return this.pol.tick(simJson, matchJson, controlledIdsJson, seed >>> 0);
  }
  /** RENDER-ONLY: merged legibility records from the last `tick`, in the
   *  same order as the emitted players (policy-driven first, then the
   *  baseline half). Spectate-overlay channel only — never sim/replay. */
  debug(): AiDebugRec[] {
    return JSON.parse(this.pol.ai_debug_json()) as AiDebugRec[];
  }

  reset(): void {
    this.pol.reset();
  }
}

/** Construct a `PolicyAi` from a trained-weights artifact JSON blob. */
export async function createPolicyAi(weightsJson: string): Promise<PolicyAi> {
  await wasmReady();
  return new PolicyAi(weightsJson);
}

// snapshot_flat layout (see rig-core/src/wasm.rs header).
const HDR = 18;
// GRAPPLE LATENCY: stride is 11 (added line_attached at offset +10).
const PSTRIDE = 11;

export interface SimMeta {
  playerIds: string[];
  bellHeldBy: string | null;
  bellThrownBy: string | null;
  bellTouched: boolean;
  passChain: string[];
  /** Render-only: per-player bound grapple target id (parallel to
   *  `playerIds`); `null` ⇒ static-anchor line or no line. Excluded from
   *  the determinism hash. */
  lineAnchors: (string | null)[];
  loopTier: 'loop' | 'curl' | 'none';
}

function v3a(a: number[]): { x: number; y: number; z: number } {
  return { x: a[0], y: a[1], z: a[2] };
}

export class WasmSim {
  private sim: RigSim;
  private roster: { id: string; team: TeamSide; role: RiggerRole }[] = [];

  constructor(seed: number) {
    this.sim = new RigSim(seed >>> 0);
  }

  addPlayer(id: string, team: TeamSide, role: RiggerRole, p: { x: number; y: number; z: number }): void {
    this.roster.push({ id, team, role });
    const t = team === 'home' ? 0 : 1;
    const r = ['anchor', 'spinner', 'faithwing', 'freewing', 'reach'].indexOf(role);
    this.sim.add_player(id, t, r < 0 ? 4 : r, p.x, p.y, p.z);
  }
  setBellHeld(id: string): void {
    this.sim.set_bell_held(id);
  }
  launchBell(
    p: { x: number; y: number; z: number },
    v: { x: number; y: number; z: number },
    w: { x: number; y: number; z: number },
    thrownBy: string,
  ): void {
    this.sim.launch_bell(p.x, p.y, p.z, v.x, v.y, v.z, w.x, w.y, w.z, thrownBy);
  }

  /** Advance one fixed sub-step. `frame === null` → idle. Returns the
   *  SimEvents the Rust core emitted this step (for the match rules). */
  step(frame: InputFrame | null): SimEvent[] {
    if (frame === null) {
      return JSON.parse(this.sim.step_idle()) as SimEvent[];
    }
    // Match the Rust parser: vec3 fields as [x,y,z] arrays.
    const json = JSON.stringify({
      tick: frame.tick,
      players: frame.players.map((pl) => ({
        id: pl.id,
        aim: [pl.aim.x, pl.aim.y, pl.aim.z],
        fireLineAt: pl.fireLineAt
          ? [pl.fireLineAt.x, pl.fireLineAt.y, pl.fireLineAt.z]
          : null,
        reel: pl.reel,
        release: pl.release,
        pushoff: pl.pushoff,
        throwCharge: pl.throwCharge,
        throwReleased: pl.throwReleased,
        throwSpin: pl.throwSpin,
        thrumbler: [pl.thrumbler.x, pl.thrumbler.y, pl.thrumbler.z],
        catchIntent: pl.catchIntent ?? false,
      })),
    });
    return JSON.parse(this.sim.step_json(json)) as SimEvent[];
  }

  meta(): SimMeta {
    return JSON.parse(this.sim.snapshot_meta()) as SimMeta;
  }

  /** Decode the flat snapshot into the frozen SimState shape. */
  snapshot(): SimState & { loopTurn: number; loopUntouched: boolean; loopTier: SimMeta['loopTier'] } {
    const f = this.sim.snapshot_flat() as unknown as Float64Array;
    const m = this.meta();
    const heldIdx = f[15];
    const players: PlayerSim[] = this.roster.map((r, i) => {
      const o = HDR + i * PSTRIDE;
      return {
        id: r.id,
        team: r.team,
        role: r.role,
        p: { x: f[o], y: f[o + 1], z: f[o + 2] },
        v: { x: f[o + 3], y: f[o + 4], z: f[o + 5] },
        q: { x: 0, y: 0, z: 0, w: 1 },
        line:
          f[o + 9] > 0
            ? {
                // Render-only anchor classification from the meta seam:
                // a non-null bound id ⇒ player↔player line (draw to the
                // LIVE target rigger); null ⇒ static anchor (unchanged).
                anchorType: m.lineAnchors[i] != null ? 'player' : 'spar',
                anchorRef: m.lineAnchors[i] ?? null,
                anchorPos: { x: 0, y: 0, z: 0 },
                restLen: f[o + 9],
                taut: f[o + 8] > 0.5,
                // GRAPPLE LATENCY: 1.0 ⇒ claw landed/line live; 0.0 ⇒ claw
                // still in flight toward anchorPos (no constraint force yet).
                attached: f[o + 10] > 0.5,
              }
            : null,
        dvBudget: f[o + 7],
        contactRef: null,
        grounded: f[o + 6] > 0.5,
      };
    });
    return {
      tick: f[0],
      omega: 0.32,
      bell: {
        p: v3a([f[1], f[2], f[3]]),
        v: v3a([f[4], f[5], f[6]]),
        q: { x: f[7], y: f[8], z: f[9], w: f[10] },
        w: v3a([f[11], f[12], f[13]]),
        chime: f[14],
        heldBy: heldIdx >= 0 ? this.roster[heldIdx]?.id ?? null : null,
        thrownBy: m.bellThrownBy,
        touchedSinceThrow: m.bellTouched,
        releasePos: { x: 0, y: 0, z: 0 },
        releaseTick: 0,
        passChain: m.passChain,
      },
      players,
      rngCursor: {},
      loopTurn: f[16],
      loopUntouched: f[17] > 0.5,
      loopTier: m.loopTier,
    };
  }
}

export async function createWasmSim(seed: number): Promise<WasmSim> {
  await wasmReady();
  return new WasmSim(seed);
}

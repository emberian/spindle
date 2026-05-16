// TS ↔ WASM binding. Thin wrapper over the Rust `RigSim` facade that
// decodes the documented flat snapshot into the frozen SimState shape so
// render/UI consume the deterministic Rust core exactly like the TS sim.
//
// The Rust core is the source of truth; the TS sim/ modules remain the
// parity oracle (cargo known-answer test holds RNG bit-identical).

import init, { RigSim } from '../../rig-core/pkg/rig_core.js';
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

// snapshot_flat layout (see rig-core/src/wasm.rs header).
const HDR = 18;
const PSTRIDE = 10;

export interface SimMeta {
  playerIds: string[];
  bellHeldBy: string | null;
  bellThrownBy: string | null;
  bellTouched: boolean;
  passChain: string[];
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
                anchorType: 'spar',
                anchorRef: null,
                anchorPos: { x: 0, y: 0, z: 0 },
                restLen: f[o + 9],
                taut: f[o + 8] > 0.5,
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

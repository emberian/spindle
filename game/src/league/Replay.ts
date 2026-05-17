// Replay.ts — "the re-call": deterministic match recording for RIG.
//
// The sim core is deterministic: same seed + same per-tick InputFrame stream
// ⇒ bit-identical match.  So a re-call is just:
//   (seed, roster, faithEnd, firstPossession, the InputFrame per sim step).
//
// This module owns the DATA MODEL, (de)serialisation, and localStorage only.
// The playback LOOP lives in main.ts (the orchestrator wires this API).
//
// localStorage conventions mirror Persistence.ts:
//   - versioned key, in-memory fallback when no window (tests/headless),
//   - robust to quota errors, LRU-style cap (keep newest ~12 re-calls).
//
// ── Frame encoding (why it round-trips bit-exactly) ────────────────────────
//
// The bulk of a re-call is `frames: InputFrame[]`.  Each InputFrame is
// { tick, players: PlayerInput[] } and the player set / order is FIXED for a
// match (it is exactly `roster`, in order).  So we do NOT store ids per frame:
// we store frames as a flat binary blob, base64'd.
//
// Per PlayerInput we serialise, in order:
//   aim.x aim.y aim.z                         3 × float64
//   fireFlag (1.0 if fireLineAt!=null else 0)  1 × float64
//   fire.x fire.y fire.z                       3 × float64  (0 when null)
//   reel                                       1 × float64  (-1|0|1, exact)
//   release                                    1 × float64  (0|1)
//   pushoff                                    1 × float64  (0|1)
//   throwCharge                                1 × float64
//   throwReleased                              1 × float64  (0|1)
//   throwSpin                                  1 × float64
//   thrumbler.x thrumbler.y thrumbler.z        3 × float64
// = 16 float64 per player.  Per frame: 1 float64 `tick` + N*16 player floats.
//
// float64 stores every JS number with its EXACT IEEE-754 bits (JS numbers
// ARE float64), so aim/thrumbler/throwCharge/throwSpin round-trip with zero
// loss.  `fireLineAt` null-vs-vector is preserved by the explicit fireFlag
// (decode reconstructs `null` iff fireFlag===0, regardless of the stored
// fire xyz).  `reel` is one of -1/0/1 which are exact integers in float64,
// and we Math.round on decode then narrow to the -1|0|1 literal so the
// reconstructed object is byte-identical to the original.  bool fields are
// stored as 0/1 and decoded with `!== 0`.
//
// Self-check (round-trip is identity):
//   const f: InputFrame = { tick: 7, players: [{
//     id: 'h0', aim: {x:0.1,y:-0.2,z:0.33333333}, fireLineAt: null,
//     reel: -1, release: true, pushoff: false, throwCharge: 0.5,
//     throwReleased: false, throwSpin: -0.75,
//     thrumbler: {x:1e-9,y:0,z:-2.5} }] };
//   const r: ReplayData = { meta, roster:[{id:'h0',team:'home',role:'anchor',x:0}], frames:[f] };
//   const back = decodeReplay(encodeReplay(r));
//   // JSON.stringify(back.frames) === JSON.stringify(r.frames)  // true
//   // back.frames[0].players[0].fireLineAt === null              // true
//   // Object.is(back.frames[0].players[0].thrumbler.x, 1e-9)     // true

import type { InputFrame, PlayerInput, RiggerRole, TeamSide } from '../sim/types';

// ── Public data model (FROZEN — orchestrator wires exactly these) ──────────

export interface ReplayMeta {
  id: string;
  homeId: string;
  awayId: string;
  seed: number;
  faithEnd: '+x' | '-x';
  firstPossession: TeamSide;
  scoreHome: number;
  scoreAway: number;
  winner: TeamSide;
  ticks: number;
  recordedAt: string;
  label: string;
}

export interface ReplayData {
  meta: ReplayMeta;
  roster: { id: string; team: TeamSide; role: RiggerRole; x: number }[];
  /** frames[i] = the InputFrame fed to sim step i, in order. */
  frames: InputFrame[];
}

// ── Recorder ───────────────────────────────────────────────────────────────

export class ReplayRecorder {
  private readonly meta: Omit<
    ReplayMeta,
    'scoreHome' | 'scoreAway' | 'winner' | 'ticks' | 'recordedAt'
  >;
  private readonly roster: ReplayData['roster'];
  private readonly frames: InputFrame[] = [];

  constructor(
    meta: Omit<ReplayMeta, 'scoreHome' | 'scoreAway' | 'winner' | 'ticks' | 'recordedAt'>,
    roster: ReplayData['roster'],
  ) {
    this.meta = meta;
    // Defensive copy of the roster so later mutation can't corrupt the re-call.
    this.roster = roster.map((r) => ({ id: r.id, team: r.team, role: r.role, x: r.x }));
  }

  /**
   * Record one tick of input. MUST be called with the SAME InputFrame object
   * passed to sim.step() this tick, every sim tick, in order.
   */
  push(frame: InputFrame): void {
    // Deep-copy so the recorded frame is immune to the caller reusing /
    // mutating the InputFrame object between ticks.
    this.frames.push(cloneFrame(frame));
  }

  /** Finalise the re-call with the match result. */
  finish(result: { scoreHome: number; scoreAway: number; winner: TeamSide }): ReplayData {
    const meta: ReplayMeta = {
      ...this.meta,
      scoreHome: result.scoreHome,
      scoreAway: result.scoreAway,
      winner: result.winner,
      ticks: this.frames.length,
      recordedAt: new Date().toISOString(),
    };
    return {
      meta,
      roster: this.roster.map((r) => ({ id: r.id, team: r.team, role: r.role, x: r.x })),
      frames: this.frames,
    };
  }
}

// ── Frame (de)serialisation helpers ────────────────────────────────────────

const FLOATS_PER_PLAYER = 16;

function cloneVec(v: { x: number; y: number; z: number }): { x: number; y: number; z: number } {
  return { x: v.x, y: v.y, z: v.z };
}

function clonePlayerInput(p: PlayerInput): PlayerInput {
  return {
    id: p.id,
    aim: cloneVec(p.aim),
    fireLineAt: p.fireLineAt === null ? null : cloneVec(p.fireLineAt),
    reel: p.reel,
    release: p.release,
    pushoff: p.pushoff,
    throwCharge: p.throwCharge,
    throwReleased: p.throwReleased,
    throwSpin: p.throwSpin,
    thrumbler: cloneVec(p.thrumbler),
  };
}

function cloneFrame(f: InputFrame): InputFrame {
  return { tick: f.tick, players: f.players.map(clonePlayerInput) };
}

/** Bytes ⇆ base64 without relying on Node Buffer (browser + jsdom safe). */
function bytesToBase64(bytes: Uint8Array): string {
  let bin = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  if (typeof btoa === 'function') return btoa(bin);
  // Node fallback
  return Buffer.from(bin, 'binary').toString('base64');
}

function base64ToBytes(b64: string): Uint8Array {
  let bin: string;
  if (typeof atob === 'function') bin = atob(b64);
  else bin = Buffer.from(b64, 'base64').toString('binary');
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * Encode just the frames as a compact base64 float64 blob.
 * playerCount is fixed (= roster.length); ids are NOT stored per frame.
 */
function encodeFrames(frames: InputFrame[], playerCount: number): string {
  const perFrame = 1 + playerCount * FLOATS_PER_PLAYER;
  const buf = new Float64Array(frames.length * perFrame);
  let o = 0;
  for (const f of frames) {
    buf[o++] = f.tick;
    for (let pi = 0; pi < playerCount; pi++) {
      const p = f.players[pi];
      buf[o++] = p.aim.x;
      buf[o++] = p.aim.y;
      buf[o++] = p.aim.z;
      const hasFire = p.fireLineAt !== null;
      buf[o++] = hasFire ? 1 : 0;
      buf[o++] = hasFire ? p.fireLineAt!.x : 0;
      buf[o++] = hasFire ? p.fireLineAt!.y : 0;
      buf[o++] = hasFire ? p.fireLineAt!.z : 0;
      buf[o++] = p.reel;
      buf[o++] = p.release ? 1 : 0;
      buf[o++] = p.pushoff ? 1 : 0;
      buf[o++] = p.throwCharge;
      buf[o++] = p.throwReleased ? 1 : 0;
      buf[o++] = p.throwSpin;
      buf[o++] = p.thrumbler.x;
      buf[o++] = p.thrumbler.y;
      buf[o++] = p.thrumbler.z;
    }
  }
  return bytesToBase64(new Uint8Array(buf.buffer, 0, buf.byteLength));
}

function decodeFrames(
  b64: string,
  roster: ReplayData['roster'],
  frameCount: number,
): InputFrame[] {
  const playerCount = roster.length;
  const bytes = base64ToBytes(b64);
  // Copy into an aligned buffer (subarray may be unaligned for Float64Array).
  const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const buf = new Float64Array(ab);
  const frames: InputFrame[] = [];
  let o = 0;
  for (let fi = 0; fi < frameCount; fi++) {
    const tick = buf[o++];
    const players: PlayerInput[] = [];
    for (let pi = 0; pi < playerCount; pi++) {
      const aim = { x: buf[o++], y: buf[o++], z: buf[o++] };
      const hasFire = buf[o++] !== 0;
      const fx = buf[o++];
      const fy = buf[o++];
      const fz = buf[o++];
      const reelRaw = Math.round(buf[o++]);
      const reel: -1 | 0 | 1 = reelRaw < 0 ? -1 : reelRaw > 0 ? 1 : 0;
      const release = buf[o++] !== 0;
      const pushoff = buf[o++] !== 0;
      const throwCharge = buf[o++];
      const throwReleased = buf[o++] !== 0;
      const throwSpin = buf[o++];
      const thrumbler = { x: buf[o++], y: buf[o++], z: buf[o++] };
      players.push({
        id: roster[pi].id,
        aim,
        fireLineAt: hasFire ? { x: fx, y: fy, z: fz } : null,
        reel,
        release,
        pushoff,
        throwCharge,
        throwReleased,
        throwSpin,
        thrumbler,
      });
    }
    frames.push({ tick, players });
  }
  return frames;
}

// ── Compact (de)serialiser (FROZEN API) ────────────────────────────────────

interface EncodedReplay {
  v: 1;
  meta: ReplayMeta;
  roster: ReplayData['roster'];
  frameCount: number;
  /** base64 float64 blob of all frames. */
  frames: string;
}

export function encodeReplay(r: ReplayData): string {
  const payload: EncodedReplay = {
    v: 1,
    meta: r.meta,
    roster: r.roster,
    frameCount: r.frames.length,
    frames: encodeFrames(r.frames, r.roster.length),
  };
  return JSON.stringify(payload);
}

export function decodeReplay(s: string): ReplayData {
  const payload = JSON.parse(s) as EncodedReplay;
  const roster = payload.roster.map((r) => ({
    id: r.id,
    team: r.team,
    role: r.role,
    x: r.x,
  }));
  return {
    meta: payload.meta,
    roster,
    frames: decodeFrames(payload.frames, roster, payload.frameCount),
  };
}

// ── Storage (localStorage, mirroring Persistence.ts conventions) ───────────

const STORAGE_KEY = 'rig_recalls_v1';
/** Keep only the newest N re-calls (frames are bulky — evict oldest). */
const MAX_RECALLS = 12;

interface RecallStore {
  schema: 1;
  /** Newest LAST. Each entry is an encodeReplay() string. */
  entries: { id: string; data: string }[];
}

let _inMemory: RecallStore | null = null;

function _hasStorage(): boolean {
  try {
    return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
  } catch {
    return false;
  }
}

function _rawGet(): string | null {
  if (!_hasStorage()) return null;
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function _rawSet(value: string): boolean {
  if (!_hasStorage()) return false;
  try {
    window.localStorage.setItem(STORAGE_KEY, value);
    return true;
  } catch {
    return false;
  }
}

function freshStore(): RecallStore {
  return { schema: 1, entries: [] };
}

function readStore(): RecallStore {
  if (!_hasStorage()) {
    return _inMemory ?? freshStore();
  }
  const raw = _rawGet();
  if (raw === null) return freshStore();
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (
      parsed !== null &&
      typeof parsed === 'object' &&
      (parsed as RecallStore).schema === 1 &&
      Array.isArray((parsed as RecallStore).entries)
    ) {
      return parsed as RecallStore;
    }
  } catch {
    /* fall through */
  }
  return freshStore();
}

/**
 * Persist the store. On quota error, drop oldest entries one at a time and
 * retry; if still failing, degrade to in-memory (mirrors Persistence.ts).
 */
function writeStore(store: RecallStore): void {
  // Enforce cap (newest last → keep the tail).
  if (store.entries.length > MAX_RECALLS) {
    store.entries = store.entries.slice(store.entries.length - MAX_RECALLS);
  }

  if (!_hasStorage()) {
    _inMemory = store;
    return;
  }

  let attempt: RecallStore = store;
  while (attempt.entries.length > 0) {
    if (_rawSet(JSON.stringify(attempt))) return;
    // Quota — evict the oldest re-call and retry.
    attempt = { schema: 1, entries: attempt.entries.slice(1) };
  }
  // Even an empty store won't fit (or no storage) — degrade to in-memory.
  if (!_rawSet(JSON.stringify(freshStore()))) {
    _inMemory = store;
  }
}

export class ReplayStore {
  /** List re-call metadata, newest first. */
  static list(): ReplayMeta[] {
    const store = readStore();
    const metas: ReplayMeta[] = [];
    for (const e of store.entries) {
      try {
        const payload = JSON.parse(e.data) as EncodedReplay;
        if (payload && payload.meta) metas.push(payload.meta);
      } catch {
        /* skip corrupt entry */
      }
    }
    return metas.reverse();
  }

  /** Save a re-call. Caps stored count at MAX_RECALLS (evicts oldest). */
  static save(r: ReplayData): void {
    const store = readStore();
    const encoded = encodeReplay(r);
    const next = store.entries.filter((e) => e.id !== r.meta.id);
    next.push({ id: r.meta.id, data: encoded });
    writeStore({ schema: 1, entries: next });
  }

  static load(id: string): ReplayData | null {
    const store = readStore();
    const entry = store.entries.find((e) => e.id === id);
    if (!entry) return null;
    try {
      return decodeReplay(entry.data);
    } catch {
      return null;
    }
  }

  static remove(id: string): void {
    const store = readStore();
    const next = store.entries.filter((e) => e.id !== id);
    writeStore({ schema: 1, entries: next });
  }
}

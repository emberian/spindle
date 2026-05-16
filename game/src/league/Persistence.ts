// Persistence.ts — localStorage schema v2 for the RIG league layer.
//
// Schema v2: { schema: 2, chosenFranchiseId, bracket, seed, difficulty,
//              replays: Record<id, seed>, createdAt, updatedAt }
//
// - save / load / migrate (absent/v0 → fresh, v1 → v2)
// - Guards against quota errors (LRU-cap replays)
// - Works with no window (degrades to in-memory) so tests/headless run

import type { BracketState } from './Bracket';

// ─── Schema types ─────────────────────────────────────────────────────────────

export type Difficulty = 'exhibition' | 'league' | 'playoff';

/** The v2 schema stored in localStorage. */
export interface SaveDataV2 {
  schema: 2;
  chosenFranchiseId: string | null;
  bracket: BracketState | null;
  seed: number;
  difficulty: Difficulty;
  /** Map from game id → replay seed. LRU-capped at MAX_REPLAYS. */
  replays: Record<string, number>;
  /** ISO date string */
  createdAt: string;
  /** ISO date string */
  updatedAt: string;
}

/** Legacy v1 schema (pre-release; bracket stored flat). */
interface SaveDataV1 {
  schema: 1;
  franchiseId?: string | null;
  bracketData?: unknown;
  seed?: number;
  difficulty?: string;
  replays?: Record<string, number>;
  createdAt?: string;
  updatedAt?: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const STORAGE_KEY = 'rig_league_save_v2';
const MAX_REPLAYS = 50;

// ─── In-memory fallback (no window / no localStorage) ─────────────────────────

let _inMemory: SaveDataV2 | null = null;

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

// ─── Fresh save ───────────────────────────────────────────────────────────────

export function freshSave(seed?: number): SaveDataV2 {
  const now = new Date().toISOString();
  return {
    schema: 2,
    chosenFranchiseId: null,
    bracket: null,
    seed: seed ?? Math.floor(Math.random() * 0xffffffff),
    difficulty: 'league',
    replays: {},
    createdAt: now,
    updatedAt: now,
  };
}

// ─── Migration ────────────────────────────────────────────────────────────────

/**
 * Migrate any version of saved data to v2.
 * - absent / null / v0 → fresh
 * - v1 → v2 (field remapping)
 * - v2 → pass-through (validate + repair)
 */
export function migrate(raw: unknown): SaveDataV2 {
  if (raw === null || raw === undefined || typeof raw !== 'object') {
    return freshSave();
  }

  const obj = raw as Record<string, unknown>;

  // Absent / v0 (no schema field)
  if (!('schema' in obj)) {
    return freshSave();
  }

  const schema = (obj as { schema?: unknown }).schema;

  // v1 → v2
  if (schema === 1) {
    const v1 = obj as unknown as SaveDataV1;
    const now = new Date().toISOString();
    return {
      schema: 2,
      chosenFranchiseId: v1.franchiseId ?? null,
      bracket: null, // v1 bracket format incompatible — reset
      seed: typeof v1.seed === 'number' ? v1.seed : Math.floor(Math.random() * 0xffffffff),
      difficulty: (v1.difficulty as Difficulty) ?? 'league',
      replays: typeof v1.replays === 'object' && v1.replays !== null ? v1.replays : {},
      createdAt: v1.createdAt ?? now,
      updatedAt: now,
    };
  }

  // v2 — validate and repair missing fields
  if (schema === 2) {
    const v2 = obj as Partial<SaveDataV2>;
    const now = new Date().toISOString();
    return {
      schema: 2,
      chosenFranchiseId: v2.chosenFranchiseId ?? null,
      bracket: v2.bracket ?? null,
      seed: typeof v2.seed === 'number' ? v2.seed : Math.floor(Math.random() * 0xffffffff),
      difficulty: (['exhibition', 'league', 'playoff'] as Difficulty[]).includes(
        v2.difficulty as Difficulty,
      )
        ? (v2.difficulty as Difficulty)
        : 'league',
      replays: typeof v2.replays === 'object' && v2.replays !== null ? (v2.replays as Record<string, number>) : {},
      createdAt: v2.createdAt ?? now,
      updatedAt: v2.updatedAt ?? now,
    };
  }

  // Unknown schema version → fresh
  return freshSave();
}

// ─── LRU replay cap ───────────────────────────────────────────────────────────

/**
 * Enforce MAX_REPLAYS cap by removing oldest entries when over limit.
 * Since plain objects don't track insertion order robustly, we use the
 * first-inserted entries from Object.keys() which V8 preserves in insertion order.
 */
function capReplays(replays: Record<string, number>): Record<string, number> {
  const keys = Object.keys(replays);
  if (keys.length <= MAX_REPLAYS) return replays;

  // Remove oldest entries (front of key list)
  const trimmed: Record<string, number> = {};
  for (const key of keys.slice(keys.length - MAX_REPLAYS)) {
    trimmed[key] = replays[key];
  }
  return trimmed;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Load the current save. Returns a migrated v2 save (or fresh if absent/corrupt).
 * Always succeeds.
 */
export function load(): SaveDataV2 {
  if (!_hasStorage()) {
    return _inMemory ?? freshSave();
  }

  const raw = _rawGet();
  if (raw === null) {
    return freshSave();
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return freshSave();
  }

  return migrate(parsed);
}

/**
 * Save the given v2 data. Updates updatedAt automatically.
 * If localStorage quota is exceeded, trims replays and retries once.
 * If headless (no window), stores in-memory.
 *
 * @returns true if saved successfully to persistent storage, false if in-memory only.
 */
export function save(data: SaveDataV2): boolean {
  const toStore: SaveDataV2 = {
    ...data,
    replays: capReplays(data.replays),
    updatedAt: new Date().toISOString(),
  };

  if (!_hasStorage()) {
    _inMemory = toStore;
    return false;
  }

  const serialized = JSON.stringify(toStore);
  const ok = _rawSet(serialized);

  if (!ok) {
    // Quota error — trim replays aggressively and retry
    const trimmed: SaveDataV2 = { ...toStore, replays: {} };
    const retryOk = _rawSet(JSON.stringify(trimmed));
    if (!retryOk) {
      // Still failing — degrade to in-memory
      _inMemory = trimmed;
      return false;
    }
    return true;
  }

  return true;
}

/**
 * Clear all saved data (resets to fresh). Returns a fresh SaveDataV2.
 */
export function clearSave(): SaveDataV2 {
  _inMemory = null;
  if (_hasStorage()) {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }
  return freshSave();
}

/**
 * Add or update a replay seed for a given game id.
 * Returns the updated save data (caller must call save() to persist).
 */
export function addReplay(data: SaveDataV2, gameId: string, replaySeed: number): SaveDataV2 {
  return {
    ...data,
    replays: capReplays({ ...data.replays, [gameId]: replaySeed }),
  };
}

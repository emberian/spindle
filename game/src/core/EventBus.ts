// Typed pub/sub (pattern ported from frisqueendom, retyped for RIG).
// The one-way channel sim/match -> render/audio/ui. No payload may carry a
// sim object by reference; consumers read interpolated snapshots separately.

import type { SimEvent, MatchPhase, TeamSide } from '../sim/types';

export interface RigEvents {
  sim: SimEvent;
  match_phase: { from: MatchPhase; to: MatchPhase };
  score: { team: TeamSide; kind: 'fall' | 'rise' | 'loop' | 'ground'; points: number };
  match_end: { winner: TeamSide; home: number; away: number };
  screen: { to: 'title' | 'game' | 'end' | 'bracket' | 'replay' };
  resize: { width: number; height: number };
  audio_started: Record<string, never>;
}

export type RigEventName = keyof RigEvents;
type Listener<T> = (payload: T) => void;

export class EventBus {
  private listeners = new Map<string, Set<Listener<unknown>>>();

  on<K extends RigEventName>(event: K, fn: Listener<RigEvents[K]>): () => void {
    let set = this.listeners.get(event);
    if (!set) {
      set = new Set();
      this.listeners.set(event, set);
    }
    set.add(fn as Listener<unknown>);
    return () => set!.delete(fn as Listener<unknown>);
  }

  emit<K extends RigEventName>(event: K, payload: RigEvents[K]): void {
    const set = this.listeners.get(event);
    if (!set) return;
    for (const fn of set) (fn as Listener<RigEvents[K]>)(payload);
  }
}

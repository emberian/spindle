// Tiny render-only handoff between Rigger.ts (owns the figure + its animated
// grapple hand) and RigLine.ts (draws the line from that hand). Rigger writes
// the current world-space grapple-hand position per player id each frame;
// RigLine reads it so the line visually originates from the posed hand instead
// of a fixed body offset. Pure cosmetic, no sim impact, no per-frame alloc
// (Vector3 instances are created once per player and reused).

import * as THREE from 'three';

const _hands = new Map<string, THREE.Vector3>();

/** Rigger publishes the animated grapple-hand world position for a player. */
export function publishGrappleHand(id: string, world: THREE.Vector3): void {
  let v = _hands.get(id);
  if (!v) {
    v = new THREE.Vector3();
    _hands.set(id, v);
  }
  v.copy(world);
}

/** RigLine reads it; returns null if the figure hasn't posed this id yet. */
export function grappleHand(id: string): THREE.Vector3 | null {
  return _hands.get(id) ?? null;
}

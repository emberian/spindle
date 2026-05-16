// Detects the 7-point Loop: a thrown bell, touched by no one, that traces a
// CLOSED Coriolis arc and falls through a ring. We track the cumulative
// turning of the cross-section (y,z) velocity heading since release; a real
// loop winds ≥ ~π and the path comes back near the launch line — distinct
// from a lazy banana (which barely turns).

import type { Vec3 } from './vec';

// Loop = an *approximate* closed orbit: winds enough heading to read as
// "coming back around" and lands near where it was thrown. Curl = a big
// untouched curve that doesn't quite close — scored as a lesser highlight.
export const LOOP_TURN = 2.2; // rad — approximate coming-around
export const CURL_TURN = 1.1; // rad — a notable curve, short of a loop
const CLOSE_RADIUS = 22; // m — looser "approximately back where it came"

export type LoopTier = 'loop' | 'curl' | 'none';

export class LoopTracker {
  private active = false;
  private touched = true;
  private releaseY = 0;
  private releaseZ = 0;
  private prevHeading = 0;
  private cumTurn = 0;
  private hasPrev = false;

  /** Call when the bell is released (thrown, leaves a hand). */
  onRelease(p: Vec3): void {
    this.active = true;
    this.touched = false;
    this.releaseY = p.y;
    this.releaseZ = p.z;
    this.cumTurn = 0;
    this.hasPrev = false;
  }

  /** Call when anyone/anything contacts the bell in flight. */
  onTouch(): void {
    this.touched = true;
    this.active = false;
  }

  /** Call every sim step while the bell is free. */
  update(v: Vec3): void {
    if (!this.active) return;
    const speed = Math.hypot(v.y, v.z);
    if (speed < 1e-4) return;
    const heading = Math.atan2(v.z, v.y);
    if (this.hasPrev) {
      let d = heading - this.prevHeading;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      this.cumTurn += Math.abs(d);
    }
    this.prevHeading = heading;
    this.hasPrev = true;
  }

  /** At a ring crossing: Loop (approx-closed), Curl (big curve), or none. */
  tier(p: Vec3): LoopTier {
    if (this.touched) return 'none';
    if (this.cumTurn >= LOOP_TURN) {
      const back = Math.hypot(p.y - this.releaseY, p.z - this.releaseZ);
      if (back <= CLOSE_RADIUS) return 'loop';
    }
    if (this.cumTurn >= CURL_TURN) return 'curl';
    return 'none';
  }

  /** Back-compat: a strict-ish loop check (now = approximate). */
  isLoop(p: Vec3): boolean {
    return this.tier(p) === 'loop';
  }

  get untouched(): boolean {
    return !this.touched;
  }
  get turn(): number {
    return this.cumTurn;
  }
}

// Detects the 7-point Loop: a thrown bell, touched by no one, that traces a
// CLOSED Coriolis arc and falls through a ring. We track the cumulative
// turning of the cross-section (y,z) velocity heading since release; a real
// loop winds ≥ ~π and the path comes back near the launch line — distinct
// from a lazy banana (which barely turns).

import type { Vec3 } from './vec';

const TURN_THRESHOLD = Math.PI * 0.85; // radians of accumulated heading change
const CLOSE_RADIUS = 14; // m — must return near the release line in (y,z)

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

  /** At a ring crossing: was this a legal closed-arc Loop? */
  isLoop(p: Vec3): boolean {
    if (!this.active || this.touched) return false;
    if (this.cumTurn < TURN_THRESHOLD) return false;
    const back = Math.hypot(p.y - this.releaseY, p.z - this.releaseZ);
    return back <= CLOSE_RADIUS;
  }

  get untouched(): boolean {
    return !this.touched;
  }
  get turn(): number {
    return this.cumTurn;
  }
}

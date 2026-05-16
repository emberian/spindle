// Owns the rAF loop. dt clamped to avoid the spiral of death; timeScale
// enables slow-mo without touching the sim's fixed step (pattern from
// frisqueendom/src/core/GameRuntime.ts, simplified to a callback pair).

export class GameRuntime {
  private rAF: number | null = null;
  private last = 0;
  private running = false;
  public timeScale = 1;

  constructor(
    private update: (dt: number) => void,
    private render: (dt: number) => void,
  ) {}

  start(): void {
    if (this.running) return;
    this.running = true;
    this.last = 0;
    this.rAF = requestAnimationFrame((t) => this.loop(t));
  }

  stop(): void {
    this.running = false;
    if (this.rAF !== null) cancelAnimationFrame(this.rAF);
    this.rAF = null;
  }

  private loop(now: number): void {
    if (!this.running) return;
    this.rAF = requestAnimationFrame((t) => this.loop(t));
    if (this.last === 0) {
      this.last = now;
      return; // no valid dt on the first frame
    }
    const rawDt = Math.min((now - this.last) / 1000, 0.05);
    this.last = now;
    const dt = rawDt * this.timeScale;
    this.update(dt);
    this.render(dt);
  }
}

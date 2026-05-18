// Single debug/viz toggle — ONE flag, ONE entry point.
//
// All rich developer-only visualization / telemetry in the game is gated
// behind this single `debugViz` flag, which DEFAULTS OFF. When OFF the
// expensive debug payloads are never constructed (lazy — zero cost, not
// just hidden); when ON they all appear together.
//
// Currently gated:
//   • window.__rigai — rich per-player AI telemetry (per-frame object +
//     per-player array allocation in the watch loop).
//   • window.__rigp  — rich human-play diagnostic (per-frame object in the
//     play loop).
//   • an on-screen "DEBUG VIZ ON" indicator.
//
// NOT gated (deliberately): window.__rig — a tiny (~13 scalar) match-state
// object the AI-vs-AI progression e2e gate polls. It is gameplay-truth
// telemetry the test depends on, not a rich developer overlay, and carries
// no per-element allocation. It stays always-on.
//
// Toggle: the backtick / backquote key (`) flips the flag at runtime.
// Render-only. No sim / AI / match coupling. No determinism impact.

let _on = false;
let _indicator: HTMLDivElement | null = null;

function ensureIndicator(host: HTMLElement): HTMLDivElement {
  if (_indicator) return _indicator;
  const el = document.createElement('div');
  el.textContent = 'DEBUG VIZ ON  ·  ` to toggle';
  el.style.cssText =
    'position:absolute;right:10px;bottom:8px;z-index:60;' +
    'font-family:ui-monospace,"Space Mono",monospace;font-size:11px;' +
    'letter-spacing:.14em;color:#6fe9ff;text-shadow:0 0 8px #1aa6b7cc;' +
    'pointer-events:none;display:none;text-transform:uppercase;';
  host.appendChild(el);
  _indicator = el;
  return el;
}

/** Wire the single toggle key (backtick) and the on-screen indicator. */
export function initDebugViz(host: HTMLElement): void {
  ensureIndicator(host);
  addEventListener('keydown', (e: KeyboardEvent) => {
    // Backquote / backtick. Ignore when typing in a field.
    if (e.code !== 'Backquote' && e.key !== '`') return;
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    e.preventDefault();
    setDebugViz(!_on);
  });
}

/** The single source of truth: is rich debug viz active? */
export function debugViz(): boolean {
  return _on;
}

/** Programmatic toggle (also driven by the backtick key). */
export function setDebugViz(on: boolean): void {
  _on = on;
  if (_indicator) _indicator.style.display = on ? 'block' : 'none';
}

// GrapplePlanner — AI movement via grapple-only locomotion.
// In the calm there is no walking. Players change vector only by:
//   1. Firing the line at an anchor (spar, skin, teammate) and reeling.
//   2. Pushing off something solid.
//
// Strategy: greedy ≤3-ply lookahead over candidate anchors.
// Each candidate is scored on how much closer to the target it gets the
// player after one swing+reel cycle. The best anchor is selected each tick.
//
// IMPORTANT: the skin is a VALID anchor for opponents (and used by AI to
// swing off), but the planner must NEVER target the skin as a self-anchor
// when the player is already outside the safe-zone or when the trajectory
// would push them through the skin (grounding).
//
// Candidate anchors:
//   - Spars along the spin axis (every SPAR_SPACING meters axially, at y=z=0)
//   - The skin (at radius R) — only if not already near skin and target is
//     axis-side of the player
//   - Teammates (their current positions, invMass ≠ 0)
//
// Output: one PlayerInput fragment (only the grapple fields are set).

import type { Vec3 } from '../../sim/vec';
import { vsub, vadd, vscale, vlen, vnorm, vdot, v3 } from '../../sim/vec';
import type { PlayerInput, PlayerSim, SimState } from '../../sim/types';
import type { PointState } from '../../sim/trajectory';
import { rk4Step } from '../../sim/trajectory';
import { REG } from '../../sim/RegConstants';

// ── Spar geometry ─────────────────────────────────────────────────────────────
const SPAR_SPACING = 40; // m along X axis
const SPAR_COUNT = Math.ceil(REG.L / SPAR_SPACING) + 1;

// Off-axis clip lattice — MUST mirror render/Calm.ts makeSpars() exactly
// (16 rings × 3, radius R·0.62) so the AI hooks the very clip-points the
// player can SEE. #3 affordance: a sport you could only rig on the spine
// left the whole high-radius volume unreachable (the bell floats at the
// skin, nobody can get there). In-universe the league bolts a clip-lattice
// into the calm so riggers can traverse and contest the entire volume.
const SPAR_RINGS = 16;
const SPAR_AROUND = 3;
const SPAR_RING_R = REG.R * 0.62;

/** All spar positions: the canon axis spine PLUS the off-axis clip lattice
 *  (identical to the rendered spars) so high-radius play is reachable. */
export function sparPositions(): Vec3[] {
  const out: Vec3[] = [];
  const start = -REG.L / 2;
  for (let i = 0; i < SPAR_COUNT; i++) {
    out.push({ x: start + i * SPAR_SPACING, y: 0, z: 0 });
  }
  for (let i = 0; i < SPAR_RINGS; i++) {
    const x = -REG.L / 2 + ((i + 0.5) / SPAR_RINGS) * REG.L;
    for (let a = 0; a < SPAR_AROUND; a++) {
      const ang = (a / SPAR_AROUND) * Math.PI * 2;
      out.push({
        x,
        y: Math.cos(ang) * SPAR_RING_R,
        z: Math.sin(ang) * SPAR_RING_R,
      });
    }
  }
  return out;
}

const SPARS = sparPositions();

// ── Grapple GRAPH (the planner half of the co-design) ────────────────────────
// The greedy 1-cycle planner can't route across a 640×90 m volume — it just
// hooks whatever's locally closest and yo-yos. So for FAR targets we route:
// spars are graph nodes, an edge joins two spars within one realistic
// fire+swing+reel reach, and A* finds the path. The rigger fires at the
// FIRST hop and re-routes each tick → it hops spine→lattice→skin-zone to
// actually reach a bell loitering at high radius. Greedy is kept for the
// close/final approach (where it's reliable and all current behaviour lives).

/** Max single fire+swing+reel reach (m). REEL_RATE·3 s ≈ 42 m of reel plus
 *  swing arc; 75 m connects the lattice well without unrealistic edges. */
const HOP_MAX = 75;
/** If one greedy cycle already gets within this of the target, just do it
 *  (close / final approach) — don't bother routing. */
const DIRECT_REACH = 12;

// Static adjacency over the fixed spar lattice — precomputed once.
const SPAR_ADJ: number[][] = (() => {
  const adj: number[][] = SPARS.map(() => []);
  for (let i = 0; i < SPARS.length; i++) {
    for (let j = i + 1; j < SPARS.length; j++) {
      const dx = SPARS[i].x - SPARS[j].x;
      const dy = SPARS[i].y - SPARS[j].y;
      const dz = SPARS[i].z - SPARS[j].z;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > 1e-6 && d <= HOP_MAX) { adj[i].push(j); adj[j].push(i); }
    }
  }
  return adj;
})();

function nearestSparIdx(p: Vec3): number {
  let bi = 0, bd = Infinity;
  for (let i = 0; i < SPARS.length; i++) {
    const dx = SPARS[i].x - p.x, dy = SPARS[i].y - p.y, dz = SPARS[i].z - p.z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bd) { bd = d; bi = i; }
  }
  return bi;
}

function sparDist(i: number, j: number): number {
  const dx = SPARS[i].x - SPARS[j].x;
  const dy = SPARS[i].y - SPARS[j].y;
  const dz = SPARS[i].z - SPARS[j].z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** A* over the static spar graph. Returns the node-index path start→goal
 *  (inclusive), or [goal] if unreachable. Deterministic: index tie-break. */
function routeSpars(start: number, goal: number): number[] {
  if (start === goal) return [start];
  const n = SPARS.length;
  const g = new Float64Array(n).fill(Infinity);
  const came = new Int32Array(n).fill(-1);
  const closed = new Uint8Array(n);
  g[start] = 0;
  // Tiny graph (~65 nodes) → a linear-scan open set is plenty fast.
  const open = new Set<number>([start]);
  while (open.size) {
    let cur = -1, best = Infinity;
    for (const k of open) {
      const f = g[k] + sparDist(k, goal);
      if (f < best - 1e-9 || (Math.abs(f - best) <= 1e-9 && (cur < 0 || k < cur))) {
        best = f; cur = k;
      }
    }
    if (cur === goal) break;
    open.delete(cur);
    closed[cur] = 1;
    for (const nb of SPAR_ADJ[cur]) {
      if (closed[nb]) continue;
      const ng = g[cur] + sparDist(cur, nb);
      if (ng < g[nb] - 1e-9) { g[nb] = ng; came[nb] = cur; open.add(nb); }
    }
  }
  if (came[goal] === -1 && start !== goal) return [goal];
  const path: number[] = [];
  for (let c = goal; c !== -1; c = came[c]) { path.push(c); if (c === start) break; }
  path.reverse();
  return path;
}

// ── Simulation helpers ────────────────────────────────────────────────────────
const PLAN_H = 1 / 15; // planning integration step
const PLAN_STEPS = 45;  // simulate 3 s of flight
const REEL_RATE = 14;   // m/s (mirrors Grapple.ts canon)
const TETHER_MIN = 3;
const SKIN_BUFFER = 4;  // m inside skin — don't fire at skin if within this

// Nav-anchor hysteresis: a freshly-considered anchor must beat the currently
// committed ("sticky") anchor's cost by this margin before we switch to it.
// Mirrors the throw-target SWITCH_MARGIN pattern in RiggerAI. Without this,
// two near-equal-cost anchors flip frame-to-frame and the rigger is yanked
// (fireLineAt + reel toggling). Tuned: ~6 m of projected-distance cost — large
// enough to kill dithering between adjacent spars, small enough that a
// genuinely better anchor (closer approach, defender avoidance) still wins.
const ANCHOR_SWITCH_MARGIN = 6;

// ── Swoop (the Xonotic-style swing→release) ──────────────────────────────────
// A free-swing (reel=0) only pays off if you LET GO at the right moment and
// fly with the carried momentum. The AI never released, so swinging always
// scored worse than winching and was never chosen. These gate the release:
// once a taut swing has built real speed roughly toward the goal, drop the
// line and soar. Exported so the executor (RiggerAI) releases on the same
// condition the planner's cost simulated → the plan and the act agree.
// ── Live behaviour knobs (playful exploration, not locked in) ────────────────
// These read window.__rigtune at call time so the behaviour space can be
// roamed interactively from the watch view + telemetry, no rebuild/deploy
// per experiment. ABSENT (tests, replay, normal play) ⇒ the documented
// default, so determinism / replay / the gate are untouched — only an
// explicitly-set dev override changes anything.
//   window.__rigtune = { wMom: 0.6, swoopMinV: 8, swoopAlign: 0.6 }
function tune(k: string, d: number): number {
  const g = (globalThis as { __rigtune?: Record<string, unknown> }).__rigtune;
  const v = g && g[k];
  return typeof v === 'number' && Number.isFinite(v) ? v : d;
}

// Defaults. Swoop release gate: a free-swing only pays off if you LET GO
// once it's fast and pointed ~at the goal (the slingshot). Exported so the
// RiggerAI executor releases on the SAME condition the planner simulated.
const SWOOP_MIN_V_D = 8;    // m/s — a swing worth releasing into
const SWOOP_ALIGN_D = 0.6;  // cos: velocity must point ~at the target
export const SWOOP_MIN_V = SWOOP_MIN_V_D;   // back-compat export (default)
export const SWOOP_ALIGN = SWOOP_ALIGN_D;
/** Live-resolved swoop gate (used by the executor too). */
export function swoopMinV(): number { return tune('swoopMinV', SWOOP_MIN_V_D); }
export function swoopAlign(): number { return tune('swoopAlign', SWOOP_ALIGN_D); }

// Momentum reward weight — THE paradigm parameter. Old cost minimised
// "distance to a point in 3 s while tethered" (myopic, momentum-blind, so
// swooping never paid). New cost rewards ending fast & pointed at the goal
// (the slingshot payoff). ~0.6 ≈ "10 m/s of useful speed ≈ 6 m closer".
// Higher ⇒ more swoop/dynamism (bell stays central) but riggers blow past
// catches; lower ⇒ more possession but tamer. Roam it live via __rigtune.
const W_MOM_D = 0.6;
function wMom(): number { return tune('wMom', W_MOM_D); }

export interface GrapplePlan {
  /** The world-space anchor point to fire the line at. */
  anchorPos: Vec3;
  /** Whether to reel in (-1) or let swing (0) after firing. */
  reel: -1 | 0;
  /** Estimated closest approach to target (m) after PLAN_STEPS ticks. */
  projectedDist: number;
  /** true if the anchor is a spar, false = skin / player */
  isSpar: boolean;
}

/**
 * Defender-aware swing simulation: like simulateGrappleSwing but also returns
 * the closest the swung path comes to any opponent (so the planner can avoid
 * stranding the player on top of a defender or swinging through coverage).
 */
function simulateGrappleSwingDef(
  pos: Vec3,
  vel: Vec3,
  anchorPos: Vec3,
  omega: number,
  reel: -1 | 0,
  target: Vec3,
  steps: number,
  h: number,
  opponents: { x: number; y: number; z: number }[],
): { closestDist: number; minOppDist: number; term: number } {
  let p = { ...pos };
  let v = { ...vel };
  let restLen = vlen(vsub(p, anchorPos));
  let closestDist = vlen(vsub(p, target));
  let minOppDist = Infinity;
  let released = false;

  for (let i = 0; i < steps; i++) {
    const s: PointState = rk4Step({ p, v }, omega, h);
    p = s.p;
    v = s.v;

    if (reel === 0 && !released && i >= 4) {
      const sp = vlen(v);
      const toT = vsub(target, p);
      const dl = vlen(toT);
      if (sp > swoopMinV() && dl > 1e-6 && vdot(v, toT) / (sp * dl) > swoopAlign()) {
        released = true;
      }
    }

    if (!released) {
      const d = vsub(p, anchorPos);
      const len = vlen(d);
      if (len > 1e-6) {
        const n = vscale(d, 1 / len);
        const vRad = vdot(v, n);
        if (len >= restLen && vRad > 0) v = vsub(v, vscale(n, vRad));
        if (reel === -1 && len >= TETHER_MIN) {
          const target_len = Math.max(TETHER_MIN, restLen - REEL_RATE * h);
          if (target_len < restLen) {
            const vRadCurrent = vdot(v, n);
            const scale = restLen > 1e-6 ? restLen / target_len : 1;
            const vTan = vsub(v, vscale(n, vRadCurrent));
            v = vadd(vscale(vTan, scale), vscale(n, vRadCurrent));
            restLen = target_len;
          }
        }
      }
    }

    const dist = vlen(vsub(p, target));
    if (dist < closestDist) closestDist = dist;

    // Sample opponent proximity every few steps (cheap).
    if ((i & 3) === 0) {
      for (const o of opponents) {
        const od = vlen(vsub(p, o));
        if (od < minOppDist) minOppDist = od;
      }
    }
  }

  // Terminal momentum toward the target — the slingshot payoff the OLD
  // objective ignored. A swoop that ends fast & aimed scores well even if
  // its closest pass wasn't tiny (next replan continues the arc).
  const _toT = vsub(target, p);
  const _dl = vlen(_toT);
  const term = _dl > 1e-6 ? Math.max(0, vdot(v, vscale(_toT, 1 / _dl))) : 0;
  return { closestDist, minOppDist, term };
}

/**
 * Simulate one player grapple cycle: fire line at anchor, reel in, return
 * the closest distance to `target` achieved over `steps` ticks.
 */
function simulateGrappleSwing(
  pos: Vec3,
  vel: Vec3,
  anchorPos: Vec3,
  omega: number,
  reel: -1 | 0,
  target: Vec3,
  steps: number,
  h: number,
): { closestDist: number; term: number } {
  let p = { ...pos };
  let v = { ...vel };
  let restLen = vlen(vsub(p, anchorPos));
  let closestDist = vlen(vsub(p, target));
  let released = false;

  for (let i = 0; i < steps; i++) {
    // Coriolis free-flight step.
    const s: PointState = rk4Step({ p, v }, omega, h);
    p = s.p;
    v = s.v;

    // SWOOP: a reel=0 swing builds tangential speed; the moment it's fast
    // and pointed ~at the target, RELEASE — fly free with the carried
    // momentum (the slingshot). The cost then reflects the real swoop
    // payoff, so the planner will actually choose to swing-and-soar.
    if (reel === 0 && !released && i >= 4) {
      const sp = vlen(v);
      const toT = vsub(target, p);
      const dl = vlen(toT);
      if (sp > swoopMinV() && dl > 1e-6 && vdot(v, toT) / (sp * dl) > swoopAlign()) {
        released = true;
      }
    }

    // Grapple constraint: only while still attached (not after release).
    if (!released) {
      const d = vsub(p, anchorPos);
      const len = vlen(d);
      if (len > 1e-6) {
        const n = vscale(d, 1 / len);
        const vRad = vdot(v, n);
        if (len >= restLen && vRad > 0) {
          // Taut — cancel outward radial component.
          v = vsub(v, vscale(n, vRad));
        }
        // Reel: shorten restLen, conserve angular momentum.
        if (reel === -1 && len >= TETHER_MIN) {
          const target_len = Math.max(TETHER_MIN, restLen - REEL_RATE * h);
          if (target_len < restLen) {
            const vRadCurrent = vdot(v, n);
            const scale = restLen > 1e-6 ? restLen / target_len : 1;
            // Scale tangential component.
            const vTan = vsub(v, vscale(n, vRadCurrent));
            v = vadd(vscale(vTan, scale), vscale(n, vRadCurrent));
            restLen = target_len;
          }
        }
      }
    }

    const dist = vlen(vsub(p, target));
    if (dist < closestDist) closestDist = dist;
  }

  const _toT = vsub(target, p);
  const _dl = vlen(_toT);
  const term = _dl > 1e-6 ? Math.max(0, vdot(v, vscale(_toT, 1 / _dl))) : 0;
  return { closestDist, term };
}

// ── Deterministic hash RNG (no signature change, replay-safe) ────────────────
// Seeded from player id + tick (both deterministic) → reproducible sampling
// for the RRT without threading rng or breaking the frozen planGrapple sig.
function hashRng(id: string, tick: number): () => number {
  let s = (2166136261 ^ tick) >>> 0;
  for (let i = 0; i < id.length; i++) s = Math.imul(s ^ id.charCodeAt(i), 16777619) >>> 0;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

// One grapple PRIMITIVE rolled out → END state (for chaining in the RRT) plus
// the branch's closest pass & terminal momentum. Same physics + swoop-release
// as simulateGrappleSwing; returns where you END so primitives can compose
// into multi-hop swoop chains.
function rolloutPrimitive(
  pos: Vec3, vel: Vec3, anchorPos: Vec3, omega: number,
  reel: -1 | 0, target: Vec3, steps: number, h: number,
): { p: Vec3; v: Vec3; minDist: number; term: number } {
  let p = { ...pos };
  let v = { ...vel };
  let restLen = vlen(vsub(p, anchorPos));
  let minDist = vlen(vsub(p, target));
  let released = false;
  for (let i = 0; i < steps; i++) {
    const s: PointState = rk4Step({ p, v }, omega, h);
    p = s.p; v = s.v;
    if (reel === 0 && !released && i >= 4) {
      const sp = vlen(v); const toT = vsub(target, p); const dl = vlen(toT);
      if (sp > swoopMinV() && dl > 1e-6 && vdot(v, toT) / (sp * dl) > swoopAlign()) released = true;
    }
    if (!released) {
      const d = vsub(p, anchorPos); const len = vlen(d);
      if (len > 1e-6) {
        const n = vscale(d, 1 / len);
        const vRad = vdot(v, n);
        if (len >= restLen && vRad > 0) v = vsub(v, vscale(n, vRad));
        if (reel === -1 && len >= TETHER_MIN) {
          const tl = Math.max(TETHER_MIN, restLen - REEL_RATE * h);
          if (tl < restLen) {
            const vrc = vdot(v, n);
            const sc = restLen > 1e-6 ? restLen / tl : 1;
            const vt = vsub(v, vscale(n, vrc));
            v = vadd(vscale(vt, sc), vscale(n, vrc));
            restLen = tl;
          }
        }
      }
    }
    const dd = vlen(vsub(p, target));
    if (dd < minDist) minDist = dd;
  }
  const toT = vsub(target, p); const dl = vlen(toT);
  const term = dl > 1e-6 ? Math.max(0, vdot(v, vscale(toT, 1 / dl))) : 0;
  return { p, v, minDist, term };
}

// ── RRT-style kinodynamic planner (knob: __rigtune.planner >= 1) ─────────────
// A *tree of reachable states*: each edge is a real grapple primitive (fire
// an anchor, reel/swing, swoop-release) rolled out through the true physics.
// Goal-biased growth finds multi-hop swoop CHAINS (slingshot off one spar to
// set up the next) the 1-cycle planner can't see; we execute the first
// primitive of the best branch and re-grow next decision (RRT-with-replan).
// Deterministic (hash-rng). Returns null → caller falls back to the MPC.
interface RrtNode { p: Vec3; v: Vec3; parent: number; aPos: Vec3; reel: -1 | 0; isSpar: boolean; best: number; }
const RRT_ITERS = 28;
const RRT_PRIM_STEPS = 18; // ~1.2 s primitive → chains form over the horizon

function planRRT(
  player: PlayerSim, target: Vec3, state: SimState,
  opponents: Vec3[], teammates: Vec3[],
): GrapplePlan | null {
  const pos = player.p, omega = state.omega;
  const toTargetDir = (() => { const d = vsub(target, pos); const l = vlen(d); return l > 1e-6 ? vscale(d, 1 / l) : v3(1, 0, 0); })();

  // Candidate anchors: lattice spars in range + teammates, forward-ish,
  // ordered by nearness to the target (good next hop first).
  const anchors: { pos: Vec3; isSpar: boolean }[] = [];
  for (const sp of SPARS) {
    const dd = vlen(vsub(pos, sp));
    if (dd < 2 || dd > 170) continue;
    if (vdot(vnorm(vsub(sp, pos)), toTargetDir) < -0.7) continue;
    anchors.push({ pos: sp, isSpar: true });
  }
  for (const pl of state.players) {
    if (pl.id === player.id || pl.team !== player.team) continue;
    const dd = vlen(vsub(pos, pl.p));
    if (dd < 3 || dd > 70) continue;
    anchors.push({ pos: pl.p, isSpar: false });
  }
  if (anchors.length === 0) return null;
  anchors.sort((a, b) =>
    (vlen(vsub(a.pos, target)) - vlen(vsub(b.pos, target))) || (a.pos.x - b.pos.x));

  const wMomV = wMom();
  const wSpace = tune('wSpace', 0);
  const DANGER = 6;
  const cost = (n: { p: Vec3; v: Vec3; minDist: number; term: number }): number => {
    let c = n.minDist - wMomV * n.term;
    for (const o of opponents) { const od = vlen(vsub(n.p, o)); if (od < DANGER) c += (DANGER - od) * 3.5; }
    if (wSpace > 0) {
      // Spacing: penalise ending crowded onto a teammate → riggers spread
      // and run distinct lanes; coordinated structure EMERGES from this.
      let nearTm = Infinity;
      for (const t of teammates) { const td = vlen(vsub(n.p, t)); if (td < nearTm) nearTm = td; }
      if (nearTm < 18) c += (18 - nearTm) * wSpace;
    }
    return c;
  };

  const rng = hashRng(player.id, state.tick);
  const nodes: RrtNode[] = [{
    p: player.p, v: player.v, parent: -1,
    aPos: pos, reel: -1, isSpar: true, best: vlen(vsub(pos, target)),
  }];
  let bestIdx = 0, bestCost = nodes[0].best;

  for (let it = 0; it < RRT_ITERS; it++) {
    // Expand the best-so-far node most of the time; explore a random node
    // sometimes (the RRT exploration/exploitation balance).
    const fromIdx = rng() < 0.7 ? bestIdx : (Math.floor(rng() * nodes.length) | 0);
    const fn = nodes[fromIdx];
    // Goal-biased anchor pick: u² favours the low-index (toward-target) ones.
    const u = rng();
    const ai = Math.min(anchors.length - 1, (u * u * anchors.length) | 0);
    const a = anchors[ai];
    const reel: -1 | 0 = rng() < 0.5 ? -1 : 0;
    const r = rolloutPrimitive(fn.p, fn.v, a.pos, omega, reel, target, RRT_PRIM_STEPS, PLAN_H);
    const branchBest = Math.min(fn.best, r.minDist);
    const nd: RrtNode = {
      p: r.p, v: r.v, parent: fromIdx, aPos: a.pos, reel, isSpar: a.isSpar, best: branchBest,
    };
    nodes.push(nd);
    const c = cost({ p: r.p, v: r.v, minDist: branchBest, term: r.term });
    if (c < bestCost - 1e-9) { bestCost = c; bestIdx = nodes.length - 1; }
  }

  if (bestIdx === 0) return null; // tree found nothing better → use MPC
  // Backtrack to the FIRST primitive off the root.
  let cur = bestIdx;
  while (nodes[cur].parent > 0) cur = nodes[cur].parent;
  const first = nodes[cur];
  return {
    anchorPos: first.aPos,
    reel: first.reel,
    projectedDist: nodes[bestIdx].best,
    isSpar: first.isSpar,
  };
}

/**
 * Main planner entry: choose the best grapple anchor to approach `target`.
 *
 * Returns a GrapplePlan (anchorPos + reel), or null if already close enough.
 */
export function planGrapple(
  player: PlayerSim,
  target: Vec3,
  state: SimState,
  avoidDefenders = true,
  sticky?: { pos: Vec3; reel: -1 | 0 } | null,
): GrapplePlan | null {
  const pos = player.p;
  const vel = player.v;
  const omega = state.omega;

  // Already close enough — no grapple needed.
  const directDist = vlen(vsub(pos, target));
  if (directDist < 3.0) return null;

  // Opponent positions for defender-aware path scoring. We avoid swings whose
  // path strands us inside DEFENDER_DANGER of an opponent.
  const opponents: Vec3[] = avoidDefenders
    ? state.players.filter(p => p.team !== player.team).map(p => p.p)
    : [];
  const DEFENDER_DANGER = 6; // m — within this is "swung into coverage"
  const teammates: Vec3[] = state.players
    .filter(p => p.id !== player.id && p.team === player.team)
    .map(p => p.p);

  // STRATEGY SWITCH (playful exploration, knob-gated, MPC default). 1+ =
  // RRT kinodynamic tree (multi-hop swoop chains); else the momentum-MPC.
  // RRT may return null (found nothing better) → fall through to MPC.
  if (tune('planner', 0) >= 1) {
    const r = planRRT(player, target, state, opponents, teammates);
    if (r) return r;
  }

  // Teammate-spacing weight: penalise a trajectory that ends crowded onto a
  // teammate so riggers spread & run distinct lanes — coordinated team
  // structure EMERGES from individual planning (knob: __rigtune.wSpace).
  const wSpaceV = tune('wSpace', 0);
  const spacingPen = (endRef: Vec3): number => {
    if (wSpaceV <= 0) return 0;
    let nearTm = Infinity;
    for (const t of teammates) { const td = vlen(vsub(endRef, t)); if (td < nearTm) nearTm = td; }
    return nearTm < 18 ? (18 - nearTm) * wSpaceV : 0;
  };

  // Cost wrapper: projected distance to target + penalty for grazing a
  // defender. Keeps the planner from sailing the player into a mark.
  const scorePlan = (
    anchor: Vec3,
    reel: -1 | 0,
  ): { projectedDist: number; cost: number } => {
    // THE PARADIGM SHIFT. Old cost = "how close did I get to a point in 3 s
    // while tethered" — myopic, momentum-blind, so a dead-stop winch always
    // beat a swoop. New cost = trajectory quality: still want to get near,
    // but REWARD ending with speed carrying toward the goal (the slingshot
    // / swoop payoff that the receding-horizon replan then continues). This
    // single objective makes swooping, slingshots and committed arcs emerge
    // instead of inching — no special-casing.
    if (opponents.length === 0) {
      const r = simulateGrappleSwing(
        pos, vel, anchor, omega, reel, target, PLAN_STEPS, PLAN_H,
      );
      return { projectedDist: r.closestDist, cost: r.closestDist - wMom() * r.term + spacingPen(anchor) };
    }
    const r = simulateGrappleSwingDef(
      pos, vel, anchor, omega, reel, target, PLAN_STEPS, PLAN_H, opponents,
    );
    const danger =
      r.minOppDist < DEFENDER_DANGER
        ? (DEFENDER_DANGER - r.minOppDist) * 3.5
        : 0;
    return { projectedDist: r.closestDist, cost: r.closestDist - wMom() * r.term + danger + spacingPen(anchor) };
  };

  const candidates: (GrapplePlan & { cost: number })[] = [];

  // ── Candidate 1: Spar anchors ──────────────────────────────────────────────
  for (const spar of SPARS) {
    // Only consider spars within useful range.
    const sparDist = vlen(vsub(pos, spar));
    if (sparDist < 2 || sparDist > 80) continue;

    // Don't fire back the way we came (would reduce progress).
    const toSpar = vnorm(vsub(spar, pos));
    const toTarget = vnorm(vsub(target, pos));
    if (vdot(toSpar, toTarget) < -0.7) continue;

    // BOTH actions on every spar (the Xonotic-swoop fix): reel=-1 winches
    // straight in (slow, dead-stop); reel=0 keeps the tether long and lets
    // the taut swing + Coriolis WHIP the rigger around — momentum is
    // preserved, so this builds & carries speed (a slingshot arc), exactly
    // the swooping the AI never did (it was 100% winch). The cost oracle
    // (closest approach over 3 s) naturally picks the swing when arcing
    // gets there faster, the winch when it doesn't — so the AI now
    // *discovers* swoops instead of inching.
    const scReel = scorePlan(spar, -1);
    candidates.push({ anchorPos: spar, reel: -1, projectedDist: scReel.projectedDist, isSpar: true, cost: scReel.cost });
    const scSwing = scorePlan(spar, 0);
    candidates.push({ anchorPos: spar, reel: 0, projectedDist: scSwing.projectedDist, isSpar: true, cost: scSwing.cost });
  }

  // ── Candidate 2: Skin anchor ───────────────────────────────────────────────
  // Use the skin point closest to the player (at radius R in the (y,z) plane).
  const playerRadius = Math.sqrt(pos.y * pos.y + pos.z * pos.z);
  const isTooCloseSkin = playerRadius > REG.R - SKIN_BUFFER;
  const targetRadius = Math.sqrt(target.y * target.y + target.z * target.z);
  // Only use skin if target is axis-side of the player (we'd swing inward).
  const skinUseful = !isTooCloseSkin && targetRadius < playerRadius;

  if (skinUseful) {
    const yzLen = playerRadius > 1e-6 ? playerRadius : 1;
    const skinPoint: Vec3 = {
      x: pos.x,
      y: (pos.y / yzLen) * REG.R,
      z: (pos.z / yzLen) * REG.R,
    };
    const sc = scorePlan(skinPoint, 0);
    candidates.push({ anchorPos: skinPoint, reel: 0, projectedDist: sc.projectedDist, isSpar: false, cost: sc.cost });
  }

  // ── Candidate 3: Teammate anchors ─────────────────────────────────────────
  for (const p of state.players) {
    if (p.id === player.id) continue;
    if (p.team !== player.team) continue;
    const tmDist = vlen(vsub(pos, p.p));
    if (tmDist < 3 || tmDist > 60) continue;

    const toTm = vnorm(vsub(p.p, pos));
    const toTarget = vnorm(vsub(target, pos));
    if (vdot(toTm, toTarget) < -0.5) continue;

    const sc = scorePlan(p.p, -1);
    candidates.push({ anchorPos: p.p, reel: -1, projectedDist: sc.projectedDist, isSpar: false, cost: sc.cost });
  }

  if (candidates.length === 0) {
    // Fallback: nearest spar.
    let nearestSpar = SPARS[0];
    let nearestDist = Infinity;
    for (const spar of SPARS) {
      const d = vlen(vsub(pos, spar));
      if (d < nearestDist) { nearestDist = d; nearestSpar = spar; }
    }
    return { anchorPos: nearestSpar, reel: -1, projectedDist: directDist, isSpar: true };
  }

  // Pick the candidate with the smallest *cost* (projected distance plus a
  // penalty for swinging through / stranding near a defender). Tie-break on
  // raw projected distance, then anchor x for full determinism.
  candidates.sort((a, b) => {
    if (a.cost !== b.cost) return a.cost - b.cost;
    if (a.projectedDist !== b.projectedDist) return a.projectedDist - b.projectedDist;
    return a.anchorPos.x - b.anchorPos.x;
  });
  let best = candidates[0];

  // ── Graph route (FAR targets) ─────────────────────────────────────────────
  // If no single greedy cycle gets us near the target, the target is across
  // the volume — greedy would just yo-yo on a local spar (the regression).
  // Instead A* over the spar lattice and fire at the FIRST hop toward the
  // goal; re-routed every tick the rigger walks spine→lattice→skin-zone.
  if (best.projectedDist > DIRECT_REACH) {
    const goalN = nearestSparIdx(target);
    const dToGoal = vlen(vsub(pos, SPARS[goalN]));
    let hop: number;
    if (dToGoal <= HOP_MAX) {
      hop = goalN; // the goal-side lattice node is one reach away — take it
    } else {
      const startN = nearestSparIdx(pos);
      const path = routeSpars(startN, goalN);
      // First node that is meaningfully away from us (don't fire at the spar
      // we're already sitting on); fall back along the path / to the goal.
      hop = goalN;
      for (const node of path) {
        if (vlen(vsub(pos, SPARS[node])) > 4) { hop = node; break; }
      }
    }
    const hopPos = SPARS[hop];
    // Only override greedy if this hop is a sane forward fire (not behind us).
    const toHop = vnorm(vsub(hopPos, pos));
    const toTgt = vnorm(vsub(target, pos));
    if (vlen(vsub(pos, hopPos)) > 2 && vdot(toHop, toTgt) > -0.6) {
      const sc = scorePlan(hopPos, -1);
      best = {
        anchorPos: hopPos,
        reel: -1,
        projectedDist: sc.projectedDist,
        isSpar: true,
        cost: sc.cost,
      };
    }
  }

  // ── Nav-anchor hysteresis ─────────────────────────────────────────────────
  // If the caller passed the previously-committed anchor and it is STILL a
  // valid candidate under the SAME filters used to admit fresh candidates,
  // re-score it with the SAME scorer and KEEP it unless the new best beats it
  // by ANCHOR_SWITCH_MARGIN. This mirrors the throw-target hysteresis in
  // RiggerAI (decideThrow ~649-667) and is deterministic: no rng, pure
  // geometry / replayed-physics scoring of the injected state.
  if (sticky) {
    const sPos = sticky.pos;
    const stickyDist = vlen(vsub(pos, sPos));
    // Determine whether the sticky anchor is the skin point (high radius in
    // the y,z plane and matching the player's current skin projection) vs a
    // spar/teammate-style point — so we apply the matching admission filter.
    const sRadius = Math.sqrt(sPos.y * sPos.y + sPos.z * sPos.z);
    const isSkinSticky = sRadius > REG.R * 0.85;

    let stickyValid = false;
    if (isSkinSticky) {
      // Same gate as the skin candidate above.
      stickyValid = skinUseful && sticky.reel === 0;
    } else {
      // Same gate as spar / teammate candidates: in useful range and not
      // fired back the way we came relative to the target.
      const toAnchor = vnorm(vsub(sPos, pos));
      const toTarget = vnorm(vsub(target, pos));
      stickyValid =
        stickyDist >= 2 &&
        stickyDist <= 80 &&
        vdot(toAnchor, toTarget) >= -0.7 &&
        sticky.reel === -1;
    }

    if (stickyValid) {
      const sc = scorePlan(sPos, sticky.reel);
      // Keep sticky unless the fresh best is clearly better.
      if (best.cost >= sc.cost - ANCHOR_SWITCH_MARGIN) {
        return {
          anchorPos: sPos,
          reel: sticky.reel,
          projectedDist: sc.projectedDist,
          // isSpar only affects callers' cosmetics; reflect geometry.
          isSpar: !isSkinSticky && sRadius < 1,
        };
      }
    }
  }

  return {
    anchorPos: best.anchorPos,
    reel: best.reel,
    projectedDist: best.projectedDist,
    isSpar: best.isSpar,
  };
}

/**
 * Convert a GrapplePlan into PlayerInput fields.
 * The caller merges other fields (aim, throw*, etc.) around this.
 */
export function planToInput(plan: GrapplePlan | null, aim: Vec3): Partial<PlayerInput> {
  if (!plan) {
    return {
      aim,
      fireLineAt: null,
      reel: 0,
      release: false,
      pushoff: false,
      throwCharge: 0,
      throwReleased: false,
      throwSpin: 0,
      thrumbler: v3(),
    };
  }

  return {
    aim,
    fireLineAt: plan.anchorPos,
    reel: plan.reel,
    release: false,
    pushoff: false,
    throwCharge: 0,
    throwReleased: false,
    throwSpin: 0,
    thrumbler: v3(),
  };
}

// Legibility overlay — the watchable payoff.
//
// RIG's whole point is a SPECTATABLE multi-agent sandbox: you should be
// able to SEE the swarm coordinate, not just see the bell move. The rich
// agent state (role, Director job, committed intent target, who is the
// single diver/contester, which swarm is the learned one) already crosses
// a RENDER-ONLY wasm seam (`RigAi.ai_debug_json` / `RigPolicy`), parallel
// to the player list and EXCLUDED from the determinism hash. This module
// turns that channel into a tasteful, recessive, ON-by-default overlay.
//
// What it draws (canon palette: bg #11131a, cyan #1aa6b7 = home, orange
// #d4602a = away, paper #f4f1ea, dim #6b7079):
//   • 3D world-space, per rigger:
//       - a thin DASHED intent line from the figure to its committed
//         target (where it's diving / receiving / marking / contesting).
//         Dashed + thin + team-tinted so it never reads as the solid
//         grapple RigLine (different idiom on purpose — no mystery
//         vectors: it always points at a real committed point).
//       - a small team-tinted control-source ring UNDER the figure:
//         a clean ring = baseline AI, a doubled/“learned” ring = the RL
//         policy, a paper ring = the human. You can SEE which swarm is
//         the learned one at a glance.
//       - a brief expanding PULSE the moment a rigger (re)commits a
//         decision / becomes the diver / contester — the "something just
//         happened here" cue.
//   • 2D, recessive:
//       - a compact label tag tracking each figure: "ANCHOR · recover"
//         (role · job), the committed-role flag in caps when notable
//         (DIVE / CONTEST / OUTLET), drawn at the projected head.
//       - a small corner panel: possession, live pass-chain depth,
//         per-team posture (control source), and the legend.
//
// Detail levels (one key cycles them — see main.ts): full → labels → off.
// Default = full in watch/spectate. Render-only; never feeds sim.step;
// reads the same per-tick deterministic AI-debug channel so it is fully
// replay-safe (replay records inputs; this is a pure function of state).

import * as THREE from 'three';
import type { AiDebugRec, ControlledBy } from '../sim/wasm';
import type { PlayerSim } from '../sim/types';
import { PAL } from '../ui/palette';

export type LegibilityLevel = 'full' | 'labels' | 'off';

// Recessive world-space sizes (m, pre figure-scale-independent: these are
// absolute world units so they read the same at any cinematic distance).
const RING_R = 2.6;          // control-source ring radius under the figure
const DASH = 2.2;            // intent-line dash length
const GAP = 1.8;             // intent-line gap length
const PULSE_MS = 480;        // commit pulse lifetime
const PULSE_MAX_R = 6.5;     // commit pulse max radius

// Hot-path scratch — no per-frame allocation (RigLine.ts discipline: this
// runs every render frame for ≤ 24 rings/pulses).
const _right = new THREE.Vector3();
const _up = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _proj = new THREE.Vector3();

function teamHex(team: 'home' | 'away'): number {
  return team === 'home' ? PAL.home : PAL.away;
}
function teamCss(team: 'home' | 'away'): string {
  return team === 'home' ? PAL.cyanCss : PAL.orangeCss;
}

interface Pulse {
  x: number; y: number; z: number;
  color: number;
  start: number;
}

export class LegibilityOverlay {
  private group = new THREE.Group();
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private level: LegibilityLevel = 'full';

  // Pooled 3D primitives (≤ 8 riggers; rebuilt per frame, trivial off the
  // 240 Hz sim path — same discipline as RigLine).
  private intentLines: THREE.LineSegments[] = [];
  private rings: THREE.Line[] = [];
  private rings2: THREE.Line[] = [];   // doubled ring for the RL source
  private pulses: THREE.Line[] = [];
  private pulseState: Pulse[] = [];

  // Commit-edge detection: a rigger's "what am I doing" key. A change
  // (job / diver / contester / control source) fires a transient pulse.
  private lastKey = new Map<string, string>();

  constructor(scene: THREE.Scene, host: HTMLElement) {
    scene.add(this.group);

    this.canvas = document.createElement('canvas');
    Object.assign(this.canvas.style, {
      position: 'absolute',
      inset: '0',
      pointerEvents: 'none',
      // Above the WebGL canvas, below the HUD chrome (HUD canvas z=10).
      zIndex: '8',
    });
    this.ctx = this.canvas.getContext('2d')!;
    host.appendChild(this.canvas);
    const ro = new ResizeObserver(() => this.resize());
    ro.observe(host);
    this.resize();

    const mkGeomMat = (
      n: number,
      color: number,
      opacity: number,
    ): [THREE.BufferGeometry, THREE.LineBasicMaterial] => {
      const g = new THREE.BufferGeometry();
      g.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(3 * n), 3),
      );
      const m = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
      });
      return [g, m];
    };
    const mkLine = (n: number, color: number, op: number): THREE.Line => {
      const [g, m] = mkGeomMat(n, color, op);
      const l = new THREE.Line(g, m);
      l.frustumCulled = false;
      l.renderOrder = 9980; // under the bold assisted arc (9990), over scene
      l.visible = false;
      this.group.add(l);
      return l;
    };
    for (let i = 0; i < 8; i++) {
      // Intent line: LineSegments so dash/gap pairs DON'T bridge (a Line
      // would connect every vertex into one polyline).
      const [ig, im] = mkGeomMat(80, PAL.dim, 0.0);
      const seg = new THREE.LineSegments(ig, im);
      seg.frustumCulled = false;
      seg.renderOrder = 9980;
      seg.visible = false;
      this.group.add(seg);
      this.intentLines.push(seg);
      // Control rings: 33-vertex loops.
      this.rings.push(mkLine(33, PAL.dim, 0.0));
      this.rings2.push(mkLine(33, PAL.dim, 0.0));
      this.pulses.push(mkLine(33, PAL.paper, 0.0));
      this.pulseState.push({ x: 0, y: 0, z: 0, color: PAL.paper, start: -1 });
    }
  }

  private resize(): void {
    const dpr = Math.min(devicePixelRatio, 2);
    this.canvas.width = Math.floor(innerWidth * dpr);
    this.canvas.height = Math.floor(innerHeight * dpr);
    this.canvas.style.width = innerWidth + 'px';
    this.canvas.style.height = innerHeight + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /** Cycle full → labels → off. Returns the new level (for a toast). */
  cycle(): LegibilityLevel {
    this.level =
      this.level === 'full' ? 'labels' : this.level === 'labels' ? 'off' : 'full';
    if (this.level !== 'full') this.hide3D();
    if (this.level === 'off') this.ctx.clearRect(0, 0, innerWidth, innerHeight);
    return this.level;
  }

  getLevel(): LegibilityLevel {
    return this.level;
  }

  /** Restore the default (ON, full) — called when a watch/spectate match
   *  starts so each session begins legible regardless of prior cycling. */
  reset(): void {
    this.level = 'full';
    this.lastKey.clear();
    for (const p of this.pulseState) p.start = -1;
  }

  private hide3D(): void {
    for (const l of this.intentLines) l.visible = false;
    for (const l of this.rings) l.visible = false;
    for (const l of this.rings2) l.visible = false;
    for (const l of this.pulses) l.visible = false;
  }

  /** Hard hide (leaving spectate). */
  hide(): void {
    this.hide3D();
    this.ctx.clearRect(0, 0, innerWidth, innerHeight);
    for (const p of this.pulseState) p.start = -1;
    this.lastKey.clear();
  }

  // Write a flat ring (XY-ish billboard-free; ring lies roughly facing the
  // camera by using camera right/up) into a Line geometry.
  private writeRing(
    line: THREE.Line,
    cx: number, cy: number, cz: number,
    r: number,
    camera: THREE.Camera,
  ): void {
    camera.matrixWorld.extractBasis(_right, _up, _fwd);
    const a = line.geometry.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i <= 32; i++) {
      const t = (i / 32) * Math.PI * 2;
      const ox = (Math.cos(t) * _right.x + Math.sin(t) * _up.x) * r;
      const oy = (Math.cos(t) * _right.y + Math.sin(t) * _up.y) * r;
      const oz = (Math.cos(t) * _right.z + Math.sin(t) * _up.z) * r;
      a.setXYZ(i, cx + ox, cy + oy, cz + oz);
    }
    a.needsUpdate = true;
    line.geometry.setDrawRange(0, 33);
  }

  private writeDashedLine(
    line: THREE.LineSegments,
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
  ): void {
    const dx = bx - ax, dy = by - ay, dz = bz - az;
    const len = Math.hypot(dx, dy, dz);
    const a = line.geometry.getAttribute('position') as THREE.BufferAttribute;
    if (len < 1e-3) {
      line.geometry.setDrawRange(0, 0);
      return;
    }
    const ux = dx / len, uy = dy / len, uz = dz / len;
    const seg = DASH + GAP;
    let d = 0;
    let v = 0;
    const maxV = 80;
    while (d < len && v + 2 <= maxV) {
      const d0 = d;
      const d1 = Math.min(d + DASH, len);
      a.setXYZ(v++, ax + ux * d0, ay + uy * d0, az + uz * d0);
      a.setXYZ(v++, ax + ux * d1, ay + uy * d1, az + uz * d1);
      d += seg;
    }
    a.needsUpdate = true;
    // Paired vertices consumed as LineSegments → each (d0,d1) is a drawn
    // dash and the gap between pairs is NOT bridged.
    line.geometry.setDrawRange(0, v);
  }

  /**
   * Update + draw. Called every render frame from the watch/replay loop.
   * @param players  interpolated render players (world positions)
   * @param recs     the render-only AI-debug records (parallel-by-id)
   * @param camera   the live camera (projection for 2D + ring facing)
   * @param ctx2     possession/passChain context for the corner panel
   * @param now      performance.now()
   */
  render(
    players: PlayerSim[],
    recs: AiDebugRec[],
    camera: THREE.PerspectiveCamera,
    ctx2: {
      possession: 'home' | 'away';
      passDepth: number;
      homeSrc: ControlledBy;
      awaySrc: ControlledBy;
    },
    now: number,
  ): void {
    if (this.level === 'off') return;
    const byId = new Map<string, AiDebugRec>();
    for (const r of recs) byId.set(r.id, r);

    const ctx = this.ctx;
    ctx.clearRect(0, 0, innerWidth, innerHeight);

    // ── 3D layer (full only) ────────────────────────────────────────────
    if (this.level === 'full') {
      let i = 0;
      for (const p of players) {
        const rec = byId.get(p.id);
        const il = this.intentLines[i];
        const rg = this.rings[i];
        const rg2 = this.rings2[i];
        if (!rec) {
          il.visible = false; rg.visible = false; rg2.visible = false;
          i++; continue;
        }
        const team = p.team;
        const tHex = teamHex(team);

        // Control-source ring under the figure.
        this.writeRing(rg, p.p.x, p.p.y, p.p.z, RING_R, camera);
        const src = rec.controlledBy;
        const ringColor =
          src === 'human' ? PAL.paper : tHex;
        (rg.material as THREE.LineBasicMaterial).color.setHex(ringColor);
        (rg.material as THREE.LineBasicMaterial).opacity =
          src === 'rl' ? 0.55 : 0.32;
        rg.visible = true;
        if (src === 'rl') {
          // Doubled concentric ring = the LEARNED swarm (unmistakable).
          this.writeRing(rg2, p.p.x, p.p.y, p.p.z, RING_R * 1.5, camera);
          (rg2.material as THREE.LineBasicMaterial).color.setHex(tHex);
          (rg2.material as THREE.LineBasicMaterial).opacity = 0.32;
          rg2.visible = true;
        } else {
          rg2.visible = false;
        }

        // Intent line to the committed target.
        if (rec.intentTargetPos) {
          const t = rec.intentTargetPos;
          this.writeDashedLine(
            il,
            p.p.x, p.p.y, p.p.z,
            t.x, t.y, t.z,
          );
          // A committed diver/contester reads HOT (team-bright); a routine
          // nav intent stays dim/recessive so the eye finds the action.
          const hot = rec.isDiver || rec.isContester;
          (il.material as THREE.LineBasicMaterial).color.setHex(
            hot ? tHex : PAL.dim,
          );
          (il.material as THREE.LineBasicMaterial).opacity = hot ? 0.7 : 0.34;
          il.visible = true;
        } else {
          il.visible = false;
        }

        // Commit-edge → transient pulse.
        const key =
          `${rec.job}|${rec.isDiver ? 'D' : ''}${rec.isContester ? 'C' : ''}` +
          `${rec.isPrimary ? 'P' : ''}|${rec.controlledBy}`;
        if (this.lastKey.get(p.id) !== key) {
          if (this.lastKey.has(p.id)) {
            // not the first frame for this id → genuine (re)commit
            const ps = this.pulseState[i];
            ps.x = p.p.x; ps.y = p.p.y; ps.z = p.p.z;
            ps.color = rec.isDiver || rec.isContester ? tHex : PAL.paper;
            ps.start = now;
          }
          this.lastKey.set(p.id, key);
        }
        i++;
      }
      for (; i < 8; i++) {
        this.intentLines[i].visible = false;
        this.rings[i].visible = false;
        this.rings2[i].visible = false;
      }

      // Pulses (decoupled from rigger index; world-anchored).
      for (let k = 0; k < this.pulseState.length; k++) {
        const ps = this.pulseState[k];
        const pl = this.pulses[k];
        if (ps.start < 0) { pl.visible = false; continue; }
        const age = (now - ps.start) / PULSE_MS;
        if (age >= 1) { ps.start = -1; pl.visible = false; continue; }
        const r = 0.6 + age * PULSE_MAX_R;
        this.writeRing(pl, ps.x, ps.y, ps.z, r, camera);
        (pl.material as THREE.LineBasicMaterial).color.setHex(ps.color);
        (pl.material as THREE.LineBasicMaterial).opacity = 0.6 * (1 - age);
        pl.visible = true;
      }
    }

    // ── 2D label tags (full + labels) ───────────────────────────────────
    ctx.save();
    ctx.font =
      '11px ui-monospace, "Space Mono", monospace';
    ctx.textBaseline = 'middle';
    const v = _proj;
    for (const p of players) {
      const rec = byId.get(p.id);
      if (!rec) continue;
      // Project a point slightly ABOVE the figure head.
      v.set(p.p.x, p.p.y + 4.0, p.p.z);
      v.project(camera);
      if (v.z > 1) continue; // behind camera
      const sx = (v.x * 0.5 + 0.5) * innerWidth;
      const sy = (-v.y * 0.5 + 0.5) * innerHeight;
      if (sx < -120 || sx > innerWidth + 120 || sy < -40 || sy > innerHeight + 40)
        continue;

      const accent = teamCss(p.team);
      // The committed-role flag in caps when notable; else the job verb.
      let flag = rec.job.toUpperCase();
      if (rec.isDiver) flag = 'DIVE';
      else if (rec.isContester) flag = 'CONTEST';
      else if (rec.isOutlet) flag = 'OUTLET';
      else if (rec.isShadow) flag = 'SHADOW';
      const label = `${rec.role.toUpperCase()} · ${flag}`;
      const w = ctx.measureText(label).width;
      const padX = 6;
      const boxW = w + padX * 2 + 14;
      const boxH = 17;
      const bx = sx - boxW / 2;
      const by = sy - boxH / 2;

      // Recessive plate: faint dark fill + a 1px team-tint left bar.
      ctx.globalAlpha = 0.82;
      ctx.fillStyle = 'rgba(17,19,26,0.66)';
      ctx.fillRect(bx, by, boxW, boxH);
      ctx.fillStyle = accent;
      ctx.fillRect(bx, by, 2.5, boxH);
      // Control-source dot: filled = RL, hollow = baseline, paper = human.
      const cdx = bx + 9;
      const cdy = sy;
      ctx.beginPath();
      ctx.arc(cdx, cdy, 3, 0, Math.PI * 2);
      if (rec.controlledBy === 'rl') {
        ctx.fillStyle = accent;
        ctx.fill();
      } else if (rec.controlledBy === 'human') {
        ctx.fillStyle = PAL.paperCss;
        ctx.fill();
      } else {
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      // Text.
      ctx.globalAlpha = 0.95;
      ctx.fillStyle =
        rec.isDiver || rec.isContester ? accent : PAL.paperCss;
      ctx.fillText(label, bx + 14 + padX, sy + 0.5);
    }
    ctx.restore();

    // ── Corner panel (full + labels) ────────────────────────────────────
    this.drawPanel(ctx, ctx2);
  }

  private drawPanel(
    ctx: CanvasRenderingContext2D,
    c: {
      possession: 'home' | 'away';
      passDepth: number;
      homeSrc: ControlledBy;
      awaySrc: ControlledBy;
    },
  ): void {
    const x = 14;
    const y = innerHeight - 104;
    const W = 234;
    const H = 90;
    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = 'rgba(17,19,26,0.72)';
    ctx.fillRect(x, y, W, H);
    ctx.strokeStyle = 'rgba(107,112,121,0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, W, H);

    ctx.font = '10px ui-monospace, "Space Mono", monospace';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = PAL.dimCss;
    ctx.fillText('COORDINATION', x + 10, y + 16);

    const srcLabel = (s: ControlledBy): string =>
      s === 'rl' ? 'LEARNED·RL' : s === 'human' ? 'HUMAN' : 'BASELINE';

    // Possession + attack.
    ctx.font = '11px ui-monospace, "Space Mono", monospace';
    ctx.fillStyle = c.possession === 'home' ? PAL.cyanCss : PAL.orangeCss;
    ctx.fillText(
      `BELL · ${c.possession.toUpperCase()}`,
      x + 10,
      y + 36,
    );
    ctx.fillStyle = PAL.paperCss;
    ctx.fillText(`PASS CHAIN · ${c.passDepth}`, x + 122, y + 36);

    // Per-team control source rows.
    ctx.fillStyle = PAL.cyanCss;
    ctx.fillText('HOME', x + 10, y + 56);
    ctx.fillStyle = PAL.paperCss;
    ctx.fillText(srcLabel(c.homeSrc), x + 64, y + 56);

    ctx.fillStyle = PAL.orangeCss;
    ctx.fillText('AWAY', x + 10, y + 74);
    ctx.fillStyle = PAL.paperCss;
    ctx.fillText(srcLabel(c.awaySrc), x + 64, y + 74);

    ctx.restore();
  }
}

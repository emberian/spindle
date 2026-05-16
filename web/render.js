// render.js — AX.Render
// World: x in [-250,+250], y in [-40,+40] (y=0 axis, y=±40 skin)
// Canvas maps world to screen via camera transform.
window.AX = window.AX || {};

AX.Render = {};
(function () {
  'use strict';

  // ── palette ──────────────────────────────────────────────────────────────
  const C = {
    bg:       '#11131a',
    cyan:     '#1aa6b7',
    orange:   '#d4602a',
    paper:    '#f4f1ea',
    dim:      '#6b7079',
    dimmer:   '#2a2d38',
    home:     '#1aa6b7',
    away:     '#d4602a',
    stone:    '#e8e0c8',
    skin:     '#3a4060',
    axis:     '#1e2230',
    midline:  '#2a3050',
    deep:     '#252a3a',
    tether:   '#4af0c0',
    p1ring:   '#ffffff',
  };

  // ── camera state ─────────────────────────────────────────────────────────
  const cam = {
    x: 0,       // world-space center
    y: 0,
    scale: 8,   // pixels per world unit
    tw: 0,      // canvas pixel dims
    th: 0,
    // smooth targets
    tx: 0,
    ty: 0,
    tscale: 8,
  };

  // ── stone trail ──────────────────────────────────────────────────────────
  const TRAIL_LEN = 48;
  const trail = [];

  // ── canvas / context ─────────────────────────────────────────────────────
  let canvas, ctx;

  function init(c) {
    canvas = c;
    ctx = c.getContext('2d');
    resize();
    window.addEventListener('resize', resize);
  }

  function resize() {
    canvas.width  = canvas.offsetWidth  || window.innerWidth;
    canvas.height = canvas.offsetHeight || window.innerHeight;
    cam.tw = canvas.width;
    cam.th = canvas.height;
  }

  // ── world ↔ screen ────────────────────────────────────────────────────────
  function wx(worldX) { return (worldX - cam.x) * cam.scale + cam.tw / 2; }
  function wy(worldY) { return (worldY - cam.y) * cam.scale + cam.th / 2; }
  function ws(worldLen) { return worldLen * cam.scale; }

  // screen → world (for input module use)
  AX.Render.screenToWorld = function (sx, sy) {
    return {
      x: (sx - cam.tw / 2) / cam.scale + cam.x,
      y: (sy - cam.th / 2) / cam.scale + cam.y,
    };
  };

  // ── update camera ─────────────────────────────────────────────────────────
  function updateCamera(engineState) {
    if (!engineState) return;

    const { stone, players, lane } = engineState;
    const p1 = players && players.find(p => p.id === 'P1');

    // Build a bounding box of stone + P1 with some lookahead
    let focusX = 0, focusY = 0;
    let count = 0;

    if (stone) {
      focusX += stone.x + stone.vx * 0.4;
      focusY += stone.y + stone.vy * 0.15;
      count++;
    }
    if (p1) {
      focusX += p1.x;
      focusY += p1.y;
      count++;
    }
    if (count > 0) { focusX /= count; focusY /= count; }

    // Clamp focus within lane
    const hw = lane ? lane.length / 2 : 250;
    const hr = lane ? lane.radius : 40;
    focusX = Math.max(-hw + 30, Math.min(hw - 30, focusX));
    focusY = Math.max(-hr + 10, Math.min(hr - 10, focusY));

    // Scale: fit lane vertically with padding
    const desiredScale = (cam.th * 0.7) / (hr * 2);
    cam.tx = focusX;
    cam.ty = focusY;
    cam.tscale = Math.max(4, Math.min(14, desiredScale));

    // Smooth
    const k = 0.08;
    cam.x += (cam.tx - cam.x) * k;
    cam.y += (cam.ty - cam.y) * k;
    cam.scale += (cam.tscale - cam.scale) * k;
  }

  // ── draw ──────────────────────────────────────────────────────────────────
  function draw(engineState, gameState) {
    // Guard: engine or game may not exist yet
    const lane = engineState && engineState.lane
      ? engineState.lane
      : { length: 500, radius: 40, gateRadius: 10, deeplineX: 170, midline: 0 };

    updateCamera(engineState);

    const W = cam.tw, H = cam.th;
    ctx.clearRect(0, 0, W, H);

    drawBackground(W, H, lane);
    drawLane(lane);
    drawStone(engineState, lane);
    if (engineState && engineState.players) {
      drawPlayers(engineState.players, engineState.stone);
    }
    drawHUD(engineState, gameState, W, H);
  }

  // ── background ────────────────────────────────────────────────────────────
  function drawBackground(W, H, lane) {
    // Deep space fill
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    // Archimedean spiral motif (faint, centered on canvas)
    ctx.save();
    ctx.translate(W / 2, H / 2);
    ctx.strokeStyle = 'rgba(26,166,183,0.04)';
    ctx.lineWidth = 1;
    const spiralScale = Math.min(W, H) * 0.45;
    ctx.beginPath();
    for (let t = 0; t < Math.PI * 14; t += 0.05) {
      const r = (t / (Math.PI * 14)) * spiralScale;
      const sx = Math.cos(t) * r;
      const sy = Math.sin(t) * r;
      if (t === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    ctx.restore();

    // Radial gradient — heavier toward top/bottom (skin direction)
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0,   'rgba(40,30,20,0.55)');
    grad.addColorStop(0.2, 'rgba(17,19,26,0)');
    grad.addColorStop(0.8, 'rgba(17,19,26,0)');
    grad.addColorStop(1,   'rgba(40,30,20,0.55)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);
  }

  // ── lane structure ────────────────────────────────────────────────────────
  function drawLane(lane) {
    const hw = lane.length / 2;   // 250
    const r  = lane.radius;       // 40

    // Lane interior subtle fill
    const laneLeft   = wx(-hw);
    const laneRight  = wx(hw);
    const laneTop    = wy(-r);
    const laneBottom = wy(r);
    const laneW = laneRight - laneLeft;
    const laneH = laneBottom - laneTop;

    ctx.fillStyle = 'rgba(20,25,40,0.6)';
    ctx.fillRect(laneLeft, laneTop, laneW, laneH);

    // Axis centerline
    ctx.save();
    ctx.strokeStyle = 'rgba(26,166,183,0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([8, 12]);
    ctx.beginPath();
    ctx.moveTo(laneLeft, wy(0));
    ctx.lineTo(laneRight, wy(0));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Midline
    ctx.save();
    ctx.strokeStyle = 'rgba(107,112,121,0.5)';
    ctx.lineWidth = ws(0.5);
    ctx.setLineDash([4, 8]);
    ctx.beginPath();
    ctx.moveTo(wx(0), laneTop);
    ctx.lineTo(wx(0), laneBottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Deeplines at ±170
    for (const dx of [-lane.deeplineX, lane.deeplineX]) {
      ctx.save();
      ctx.strokeStyle = 'rgba(107,112,121,0.3)';
      ctx.lineWidth = ws(0.35);
      ctx.setLineDash([3, 6]);
      ctx.beginPath();
      ctx.moveTo(wx(dx), laneTop);
      ctx.lineTo(wx(dx), laneBottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // Skin / hull walls — bright boundary lines
    ctx.save();
    ctx.strokeStyle = '#3a4060';
    ctx.lineWidth = ws(1.2);
    // top skin
    ctx.beginPath();
    ctx.moveTo(laneLeft, wy(-r));
    ctx.lineTo(laneRight, wy(-r));
    ctx.stroke();
    // bottom skin
    ctx.beginPath();
    ctx.moveTo(laneLeft, wy(r));
    ctx.lineTo(laneRight, wy(r));
    ctx.stroke();
    // bright inner edge
    ctx.strokeStyle = '#5060a0';
    ctx.lineWidth = ws(0.4);
    ctx.beginPath();
    ctx.moveTo(laneLeft, wy(-r));
    ctx.lineTo(laneRight, wy(-r));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(laneLeft, wy(r));
    ctx.lineTo(laneRight, wy(r));
    ctx.stroke();
    ctx.restore();

    // Texture dots on skin
    ctx.save();
    ctx.fillStyle = 'rgba(80,100,160,0.4)';
    for (let sx = -hw + 10; sx <= hw - 10; sx += 20) {
      ctx.beginPath(); ctx.arc(wx(sx), wy(-r), ws(0.5), 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(wx(sx), wy(r),  ws(0.5), 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();

    // Gates — glowing rings
    drawGate(hw,  lane.gateRadius, C.cyan,   'HOME');   // +x
    drawGate(-hw, lane.gateRadius, C.orange,  'AWAY');  // -x
  }

  function drawGate(gx, gr, color, label) {
    const cx = wx(gx);
    const top    = wy(-gr);
    const bottom = wy(gr);
    const rad    = ws(gr);

    // Glow
    const g = ctx.createRadialGradient(cx, wy(0), 0, cx, wy(0), rad * 2);
    g.addColorStop(0,   color + '44');
    g.addColorStop(0.5, color + '18');
    g.addColorStop(1,   'transparent');
    ctx.fillStyle = g;
    ctx.fillRect(cx - rad * 2, top - rad, rad * 4, rad * 2 + wy(gr) - wy(-gr));

    // Ring arc — only the part inside lane radius
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = ws(0.8);
    ctx.shadowColor = color;
    ctx.shadowBlur = ws(3);
    ctx.beginPath();
    ctx.moveTo(cx, top);
    ctx.lineTo(cx, bottom);
    ctx.stroke();

    // Tick marks across the gate
    ctx.lineWidth = ws(1.5);
    ctx.beginPath();
    ctx.moveTo(cx - ws(1.5), wy(-gr));
    ctx.lineTo(cx + ws(1.5), wy(-gr));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - ws(1.5), wy(gr));
    ctx.lineTo(cx + ws(1.5), wy(gr));
    ctx.stroke();
    ctx.restore();

    // Label
    ctx.save();
    ctx.font = `bold ${Math.max(9, ws(2))}px -apple-system,Arial,sans-serif`;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.8;
    ctx.textAlign = gx > 0 ? 'right' : 'left';
    ctx.textBaseline = 'top';
    const pad = ws(2.5);
    ctx.fillText(label, cx + (gx > 0 ? -pad : pad), wy(-gr) - ws(4));
    ctx.restore();
  }

  // ── stone + trail ─────────────────────────────────────────────────────────
  function drawStone(engineState, lane) {
    if (!engineState || !engineState.stone) return;
    const st = engineState.stone;

    // Trail
    trail.push({ x: st.x, y: st.y });
    if (trail.length > TRAIL_LEN) trail.shift();

    if (trail.length > 2) {
      for (let i = 1; i < trail.length; i++) {
        const alpha = (i / trail.length) * 0.75;
        const width = ws(0.5 + (i / trail.length) * 1.2);
        ctx.save();
        ctx.strokeStyle = `rgba(232,224,200,${alpha})`;
        ctx.lineWidth = width;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(wx(trail[i-1].x), wy(trail[i-1].y));
        ctx.lineTo(wx(trail[i].x),   wy(trail[i].y));
        ctx.stroke();
        ctx.restore();
      }

      // Color tint in the last 1/3 of trail — Coriolis glow
      for (let i = Math.floor(trail.length * 0.65); i < trail.length; i++) {
        const alpha = ((i - trail.length * 0.65) / (trail.length * 0.35)) * 0.4;
        ctx.save();
        ctx.strokeStyle = `rgba(26,166,183,${alpha})`;
        ctx.lineWidth = ws(1.2);
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(wx(trail[i-1].x), wy(trail[i-1].y));
        ctx.lineTo(wx(trail[i].x),   wy(trail[i].y));
        ctx.stroke();
        ctx.restore();
      }
    }

    // Stone sphere
    const sx = wx(st.x), sy = wy(st.y);
    const sr = ws(2.2);

    // Glow
    const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr * 2.5);
    g.addColorStop(0,   'rgba(232,224,200,0.6)');
    g.addColorStop(0.5, 'rgba(232,224,200,0.15)');
    g.addColorStop(1,   'transparent');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(sx, sy, sr * 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Core
    const core = ctx.createRadialGradient(sx - sr * 0.3, sy - sr * 0.3, sr * 0.1, sx, sy, sr);
    core.addColorStop(0, '#fff8e8');
    core.addColorStop(0.5, C.stone);
    core.addColorStop(1, '#8a8070');
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.arc(sx, sy, sr, 0, Math.PI * 2);
    ctx.fill();

    // Spin indicator ring if spinning
    if (st.spinTrue) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,200,0.5)';
      ctx.lineWidth = ws(0.3);
      ctx.beginPath();
      ctx.arc(sx, sy, sr * 1.4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // Held indicator
    if (st.heldBy) {
      ctx.save();
      ctx.strokeStyle = st.heldBy === 'P1' ? C.cyan : C.orange;
      ctx.lineWidth = ws(0.5);
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(sx, sy, sr * 1.8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }

  // ── players ───────────────────────────────────────────────────────────────
  const ROLE_COLORS = {
    runner: 1.0,
    anchor: 0.7,
    launch: 0.85,
    rover:  0.6,
  };

  function drawPlayers(players, stone) {
    // Draw tethers first (below players)
    for (const p of players) {
      if (p.tether && p.tether.anchored) {
        drawTether(p);
      }
    }
    for (const p of players) {
      drawPlayer(p, stone);
    }
  }

  function drawTether(p) {
    const { ax, ay } = p.tether;
    ctx.save();
    ctx.strokeStyle = p.team === 'home' ? 'rgba(26,166,183,0.7)' : 'rgba(212,96,42,0.7)';
    ctx.lineWidth = ws(0.4);
    ctx.setLineDash([ws(1), ws(2)]);
    ctx.beginPath();
    ctx.moveTo(wx(p.x), wy(p.y));
    ctx.lineTo(wx(ax), wy(ay));
    ctx.stroke();

    // Anchor dot
    ctx.fillStyle = p.team === 'home' ? C.cyan : C.orange;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.arc(wx(ax), wy(ay), ws(1), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function drawPlayer(p, stone) {
    const px = wx(p.x), py = wy(p.y);
    const isP1 = p.id === 'P1';
    const isHome = p.team === 'home';
    const teamColor = isHome ? C.home : C.away;
    const roleAlpha = ROLE_COLORS[p.role] || 1.0;

    // Capsule body
    const capW = ws(3.5), capH = ws(2.2);

    ctx.save();
    ctx.globalAlpha = roleAlpha;

    // Glow for P1
    if (isP1) {
      ctx.shadowColor = teamColor;
      ctx.shadowBlur = ws(4);
    }

    // Body fill
    const bodyGrad = ctx.createLinearGradient(px - capW, py, px + capW, py);
    bodyGrad.addColorStop(0, teamColor + 'aa');
    bodyGrad.addColorStop(0.5, teamColor + 'ff');
    bodyGrad.addColorStop(1, teamColor + 'aa');
    ctx.fillStyle = bodyGrad;

    ctx.beginPath();
    ctx.ellipse(px, py, capW, capH, 0, 0, Math.PI * 2);
    ctx.fill();

    // Edge
    ctx.strokeStyle = isP1 ? '#ffffff' : teamColor;
    ctx.lineWidth = ws(isP1 ? 0.5 : 0.3);
    ctx.stroke();
    ctx.restore();

    // Velocity indicator (facing direction)
    const speed = Math.hypot(p.vx || 0, p.vy || 0);
    if (speed > 0.5) {
      const nx = (p.vx || 0) / speed, ny = (p.vy || 0) / speed;
      ctx.save();
      ctx.strokeStyle = teamColor;
      ctx.lineWidth = ws(0.5);
      ctx.globalAlpha = Math.min(1, speed / 15);
      ctx.beginPath();
      ctx.moveTo(px + nx * capW, py + ny * capH);
      ctx.lineTo(px + nx * (capW + ws(3)), py + ny * (capH + ws(2)));
      ctx.stroke();
      ctx.restore();
    }

    // onSkin indicator
    if (p.onSkin) {
      ctx.save();
      ctx.strokeStyle = '#ff8040';
      ctx.lineWidth = ws(0.4);
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.ellipse(px, py, capW * 1.3, capH * 1.3, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // P1 highlight ring
    if (isP1) {
      ctx.save();
      ctx.strokeStyle = C.p1ring;
      ctx.lineWidth = ws(0.3);
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.ellipse(px, py, capW * 1.5, capH * 1.5, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // Role label
    ctx.save();
    ctx.font = `bold ${Math.max(8, ws(1.5))}px -apple-system,Arial,sans-serif`;
    ctx.fillStyle = isP1 ? '#ffffff' : teamColor;
    ctx.globalAlpha = 0.85;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const roleShort = { runner: 'RUN', anchor: 'ANC', launch: 'LCH', rover: 'RVR' }[p.role] || p.role.substring(0, 3).toUpperCase();
    ctx.fillText(roleShort, px, py);
    ctx.restore();

    // dvBudget bar (small, below player)
    if (p.dvBudget !== undefined && p.dvBudget !== null) {
      const barW = capW * 2;
      const barH = ws(0.6);
      const barX = px - barW / 2;
      const barY = py + capH + ws(0.8);
      const frac = Math.max(0, Math.min(1, p.dvBudget / 60));
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillRect(barX, barY, barW, barH);
      ctx.fillStyle = frac > 0.3 ? teamColor : '#ff4040';
      ctx.fillRect(barX, barY, barW * frac, barH);
      ctx.restore();
    }
  }

  // ── HUD ───────────────────────────────────────────────────────────────────
  function drawHUD(engineState, gameState, W, H) {
    if (!gameState) return;
    const gs = gameState;

    const hudFont = `bold ${Math.round(H * 0.022)}px -apple-system,'Helvetica Neue',Arial,sans-serif`;
    const smallFont = `${Math.round(H * 0.016)}px -apple-system,'Helvetica Neue',Arial,sans-serif`;
    const bigFont = `bold ${Math.round(H * 0.032)}px -apple-system,'Helvetica Neue',Arial,sans-serif`;

    // Top bar background
    ctx.save();
    ctx.fillStyle = 'rgba(17,19,26,0.75)';
    ctx.fillRect(0, 0, W, H * 0.072);
    ctx.restore();

    const cy = H * 0.036;
    const pad = W * 0.018;

    // Score — HOME  vs  AWAY
    const scoreStr = `${gs.score ? gs.score.home : 0}  —  ${gs.score ? gs.score.away : 0}`;
    ctx.save();
    ctx.font = bigFont;
    ctx.fillStyle = '#f4f1ea';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(scoreStr, W / 2, cy);
    ctx.restore();

    // HOME label (left of center)
    ctx.save();
    ctx.font = smallFont;
    ctx.fillStyle = C.cyan;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.letterSpacing = '2px';
    ctx.fillText('HOME', W / 2 - W * 0.085, cy);
    ctx.restore();

    // AWAY label (right of center)
    ctx.save();
    ctx.font = smallFont;
    ctx.fillStyle = C.orange;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('AWAY', W / 2 + W * 0.085, cy);
    ctx.restore();

    // Frame # + HIGH FRAME
    ctx.save();
    ctx.font = hudFont;
    ctx.fillStyle = '#f4f1ea';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const isRig = gs.mode === 'rig';
    const frameStr = isRig
      ? `RIG · INNING ${gs.frame || 1}/9`
      : `FRAME ${gs.frame || 1}/6`;
    ctx.fillText(frameStr, pad, cy);
    if (!isRig && gs.highFrame) {
      ctx.fillStyle = '#ffcc00';
      ctx.fillText('  ★ HIGH', pad + ctx.measureText(frameStr).width, cy);
    }
    ctx.restore();

    // Frame clock (top right)
    ctx.save();
    ctx.font = hudFont;
    ctx.fillStyle = '#f4f1ea';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const fc = gs.frameClock != null ? gs.frameClock : 0;
    if (gs.mode === 'rig' || fc < 0) {
      ctx.fillStyle = C.cyan;
      ctx.fillText('NO CLOCK', W - pad, cy);
    } else {
      const mins = Math.floor(fc / 60), secs = Math.floor(fc % 60);
      ctx.fillText(`${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}`, W - pad, cy);
    }
    ctx.restore();

    // Shot clock + possession (bottom-right corner)
    {
      const scVal = gs.shotClock != null ? gs.shotClock : 5;
      const urgent = scVal <= 2;
      ctx.save();
      ctx.font = `bold ${Math.round(H * 0.045)}px -apple-system,Arial,sans-serif`;
      ctx.fillStyle = urgent ? C.orange : '#f4f1ea';
      if (urgent) { ctx.shadowColor = C.orange; ctx.shadowBlur = 8; }
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(Math.ceil(scVal), W - pad, H - pad * 0.5);
      ctx.restore();

      ctx.save();
      ctx.font = smallFont;
      ctx.fillStyle = C.dim;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText('SHOT CLK', W - pad, H - pad * 0.5 - Math.round(H * 0.045) - 2);
      ctx.restore();

      // Possession strip
      if (gs.possession) {
        const posColor = gs.possession === 'home' ? C.cyan : C.orange;
        ctx.save();
        ctx.fillStyle = posColor;
        ctx.globalAlpha = 0.85;
        ctx.font = smallFont;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'bottom';
        ctx.fillText(`${gs.possession.toUpperCase()} POSS`, W - pad, H - pad * 0.5 - Math.round(H * 0.045) - Math.round(H * 0.022) - 4);
        ctx.restore();
      }
    }

    // Phase badge (bottom left)
    if (gs.phase) {
      const phaseColors = {
        pushoff: C.cyan,
        live: '#f4f1ea',
        score: '#ffcc00',
        turnover: C.orange,
        freespin: '#cc44ff',
        highframe: '#ffcc00',
        over: C.orange,
      };
      ctx.save();
      ctx.font = hudFont;
      ctx.fillStyle = phaseColors[gs.phase] || C.paper;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(gs.phase.toUpperCase(), pad, H - pad * 0.5);
      ctx.restore();
    }

    // Center toast message
    if (gs.message && gs.message.trim() !== '') {
      const msgFont = `bold ${Math.round(H * 0.038)}px -apple-system,'Helvetica Neue',Arial,sans-serif`;
      const msgW = Math.min(W * 0.6, 600);
      const msgH = H * 0.07;
      const msgX = W / 2;
      const msgY = H * 0.38;

      ctx.save();
      ctx.fillStyle = 'rgba(17,19,26,0.85)';
      roundRect(ctx, msgX - msgW / 2, msgY - msgH / 2, msgW, msgH, 6);
      ctx.fill();

      ctx.strokeStyle = C.cyan + '80';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.font = msgFont;
      ctx.fillStyle = '#f4f1ea';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(gs.message.toUpperCase(), msgX, msgY);
      ctx.restore();
    }
  }

  // ── roundRect helper ─────────────────────────────────────────────────────
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ── screens (called from main.js) ────────────────────────────────────────
  AX.Render.drawTitleScreen = function (W, H) {
    canvas.width  = canvas.offsetWidth  || window.innerWidth;
    canvas.height = canvas.offsetHeight || window.innerHeight;
    const w = canvas.width, h = canvas.height;

    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);

    // Spiral bg
    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.strokeStyle = 'rgba(26,166,183,0.05)';
    ctx.lineWidth = 1;
    const ss = Math.min(w, h) * 0.48;
    ctx.beginPath();
    for (let t = 0; t < Math.PI * 16; t += 0.04) {
      const r = (t / (Math.PI * 16)) * ss;
      const sx2 = Math.cos(t) * r, sy2 = Math.sin(t) * r;
      if (t === 0) ctx.moveTo(sx2, sy2); else ctx.lineTo(sx2, sy2);
    }
    ctx.stroke();
    ctx.restore();

    // Wordmark: AXBALL
    ctx.save();
    ctx.font = `900 ${Math.round(h * 0.14)}px -apple-system,'Helvetica Neue',Arial,sans-serif`;
    ctx.letterSpacing = '0.12em';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Gradient fill: cyan → orange
    const wGrad = ctx.createLinearGradient(w * 0.2, 0, w * 0.8, 0);
    wGrad.addColorStop(0, C.cyan);
    wGrad.addColorStop(1, C.orange);
    ctx.fillStyle = wGrad;
    ctx.shadowColor = C.cyan;
    ctx.shadowBlur = Math.round(h * 0.04);
    ctx.fillText('AXBALL', w / 2, h * 0.34);
    ctx.restore();

    // Tagline
    ctx.save();
    ctx.font = `${Math.round(h * 0.025)}px -apple-system,'Helvetica Neue',Arial,sans-serif`;
    ctx.fillStyle = C.dim;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.letterSpacing = '0.25em';
    ctx.fillText('THE GAME AT THE CENTER OF THE WORLD', w / 2, h * 0.48);
    ctx.restore();

    // Controls box
    const controls = [
      ['MOUSE',       'Aim'],
      ['LMB HOLD',    'Grapple / tether toward aim'],
      ['SPACE HOLD',  'Charge throw (hold Stone)'],
      ['SPACE REL',   'Release throw toward aim'],
      ['SHIFT',       'Push off surface toward aim'],
      ['W / S',       'Reel tether in / out'],
      ['SCROLL',      'Reel tether in / out'],
    ];

    const boxW = Math.min(w * 0.55, 520);
    const lineH = h * 0.038;
    const boxH = controls.length * lineH + lineH * 1.2;
    const boxX = w / 2 - boxW / 2;
    const boxY = h * 0.56;

    ctx.save();
    ctx.fillStyle = 'rgba(20,24,36,0.7)';
    roundRect(ctx, boxX, boxY, boxW, boxH, 8);
    ctx.fill();
    ctx.strokeStyle = 'rgba(26,166,183,0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.textBaseline = 'middle';
    for (let i = 0; i < controls.length; i++) {
      const [key, desc] = controls[i];
      const ly = boxY + lineH * 0.6 + i * lineH;
      ctx.font = `bold ${Math.round(h * 0.018)}px -apple-system,Arial,sans-serif`;
      ctx.fillStyle = C.cyan;
      ctx.textAlign = 'right';
      ctx.fillText(key, w / 2 - boxW * 0.04, ly);

      ctx.font = `${Math.round(h * 0.018)}px -apple-system,Arial,sans-serif`;
      ctx.fillStyle = C.paper;
      ctx.textAlign = 'left';
      ctx.fillText(desc, w / 2 + boxW * 0.04, ly);
    }
    ctx.restore();

    // START button
    const btnW = Math.min(w * 0.25, 240);
    const btnH = h * 0.065;
    const btnX = w / 2 - btnW / 2;
    const btnY = h * 0.56 + boxH + h * 0.035;

    const btnGrad = ctx.createLinearGradient(btnX, 0, btnX + btnW, 0);
    btnGrad.addColorStop(0, C.cyan);
    btnGrad.addColorStop(1, C.orange);
    ctx.save();
    ctx.fillStyle = btnGrad;
    ctx.shadowColor = C.cyan;
    ctx.shadowBlur = 12;
    roundRect(ctx, btnX, btnY, btnW, btnH, 6);
    ctx.fill();

    ctx.font = `bold ${Math.round(h * 0.028)}px -apple-system,Arial,sans-serif`;
    ctx.fillStyle = '#11131a';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowBlur = 0;
    ctx.fillText('START', w / 2, btnY + btnH / 2);
    ctx.restore();

    return { btnX, btnY, btnW, btnH };
  };

  AX.Render.drawEndScreen = function (gameState) {
    const w = canvas.width  = canvas.offsetWidth  || window.innerWidth;
    const h = canvas.height = canvas.offsetHeight || window.innerHeight;

    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);

    // Spiral
    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.strokeStyle = 'rgba(26,166,183,0.04)';
    ctx.lineWidth = 1;
    const ss = Math.min(w, h) * 0.46;
    ctx.beginPath();
    for (let t = 0; t < Math.PI * 16; t += 0.04) {
      const r2 = (t / (Math.PI * 16)) * ss;
      const sx2 = Math.cos(t) * r2, sy2 = Math.sin(t) * r2;
      if (t === 0) ctx.moveTo(sx2, sy2); else ctx.lineTo(sx2, sy2);
    }
    ctx.stroke();
    ctx.restore();

    const winner = gameState && gameState.winner;
    const score  = gameState && gameState.score;
    const wColor = winner === 'home' ? C.cyan : winner === 'away' ? C.orange : C.paper;
    const wLabel = winner ? winner.toUpperCase() + ' WINS' : 'FINAL';

    ctx.save();
    ctx.font = `900 ${Math.round(h * 0.1)}px -apple-system,'Helvetica Neue',Arial,sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = wColor;
    ctx.shadowColor = wColor;
    ctx.shadowBlur = Math.round(h * 0.05);
    ctx.fillText(wLabel, w / 2, h * 0.3);
    ctx.restore();

    if (score) {
      ctx.save();
      ctx.font = `bold ${Math.round(h * 0.07)}px -apple-system,Arial,sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = C.paper;
      ctx.shadowBlur = 0;
      ctx.fillText(`${score.home}  —  ${score.away}`, w / 2, h * 0.46);

      ctx.font = `${Math.round(h * 0.022)}px -apple-system,Arial,sans-serif`;
      ctx.fillStyle = C.dim;
      ctx.fillText('HOME              AWAY', w / 2, h * 0.55);
      ctx.restore();
    }

    // REMATCH button
    const btnW = Math.min(w * 0.28, 260);
    const btnH = h * 0.065;
    const btnX = w / 2 - btnW / 2;
    const btnY = h * 0.68;

    const btnGrad = ctx.createLinearGradient(btnX, 0, btnX + btnW, 0);
    btnGrad.addColorStop(0, C.cyan);
    btnGrad.addColorStop(1, C.orange);
    ctx.save();
    ctx.fillStyle = btnGrad;
    ctx.shadowColor = wColor;
    ctx.shadowBlur = 14;
    roundRect(ctx, btnX, btnY, btnW, btnH, 6);
    ctx.fill();

    ctx.font = `bold ${Math.round(h * 0.028)}px -apple-system,Arial,sans-serif`;
    ctx.fillStyle = '#11131a';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowBlur = 0;
    ctx.fillText('REMATCH', w / 2, btnY + btnH / 2);
    ctx.restore();

    return { btnX, btnY, btnW, btnH };
  };

  AX.Render.drawLoadingScreen = function () {
    const w = canvas.width  = canvas.offsetWidth  || window.innerWidth;
    const h = canvas.height = canvas.offsetHeight || window.innerHeight;
    ctx.fillStyle = '#11131a';
    ctx.fillRect(0, 0, w, h);
    ctx.save();
    ctx.font = `bold ${Math.round(h * 0.035)}px -apple-system,Arial,sans-serif`;
    ctx.fillStyle = '#6b7079';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('LOADING MODULES…', w / 2, h / 2);
    ctx.restore();
  };

  AX.Render.init = init;
  AX.Render.draw = draw;
}());

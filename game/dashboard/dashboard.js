/**
 * Spindle MAPPO Training Dashboard
 *
 * File-based protocol:
 *   training_output/metrics.jsonl   — one JSON object per line per generation
 *   training_output/matches/*.json  — one file per eval match
 *
 * Polls every 2s for new data. No build step, no framework.
 */

'use strict';

// ── Config ─────────────────────────────────────────────────────────────────────
const POLL_INTERVAL = 2000;
const METRICS_URL = '/training_output/metrics.jsonl';
const MATCHES_INDEX_URL = '/training_output/matches/index.json';
const MATCHES_DIR = '/training_output/matches/';

const COLORS = {
  cyan:   '#1aa6b7',
  orange: '#d4602a',
  paper:  '#f4f1ea',
  dim:    '#6b7079',
  green:  '#4caf50',
  red:    '#e57373',
  bg:     '#11131a',
  bg2:    '#1a1d26',
};

// ── State ──────────────────────────────────────────────────────────────────────
let metrics = [];        // Array of generation metric objects
let matches = [];        // Array of match metadata objects
let matchDataCache = {}; // id -> full match data (with frames)
let replayState = null;  // { matchData, currentTick, playing, animFrame }

// ── DOM refs ───────────────────────────────────────────────────────────────────
const $liveGen = document.getElementById('live-gen');
const $liveThroughput = document.getElementById('live-throughput');
const $liveEta = document.getElementById('live-eta');
const $liveDot = document.querySelector('#live-indicator .dot');
const $matchGrid = document.getElementById('match-grid');
const $matchCount = document.getElementById('match-count');
const $overlay = document.getElementById('match-overlay');
const $overlayContent = document.getElementById('match-overlay-content');
const $overlayClose = document.getElementById('match-overlay-close');
const $detailHeader = document.getElementById('match-detail-header');
const $detailJson = document.getElementById('match-detail-json');
const $replaySvg = document.getElementById('match-replay-svg');
const $replayPlay = document.getElementById('replay-play');
const $replayPause = document.getElementById('replay-pause');
const $replayScrub = document.getElementById('replay-scrub');
const $replayTick = document.getElementById('replay-tick');

// ── Charts (Canvas 2D line charts) ────────────────────────────────────────────

class LineChart {
  constructor(canvasId, color, label, yFormat) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.color = color;
    this.label = label;
    this.yFormat = yFormat || (v => v.toFixed(3));
    this.data = [];
    this._resize();
    window.addEventListener('resize', () => this._resize());
  }

  _resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = 160 * dpr;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = '160px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = rect.width;
    this.h = 160;
    this.draw();
  }

  update(data) {
    this.data = data;
    this.draw();
  }

  draw() {
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    const pad = { t: 10, r: 10, b: 24, l: 50 };

    ctx.clearRect(0, 0, w, h);

    if (this.data.length < 2) {
      ctx.fillStyle = COLORS.dim;
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for data...', w / 2, h / 2);
      return;
    }

    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    // Compute Y range
    let yMin = Infinity, yMax = -Infinity;
    for (const v of this.data) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const yRange = yMax - yMin;
    const yPad = yRange * 0.1;
    yMin -= yPad;
    yMax += yPad;

    // Grid lines
    ctx.strokeStyle = COLORS.dim + '30';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (plotH * i / 4);
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.stroke();
    }

    // Y axis labels
    ctx.fillStyle = COLORS.dim;
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (plotH * i / 4);
      const val = yMax - (i / 4) * (yMax - yMin);
      ctx.fillText(this.yFormat(val), pad.l - 4, y + 3);
    }

    // X axis labels
    ctx.textAlign = 'center';
    const n = this.data.length;
    const xStep = Math.max(1, Math.floor(n / 6));
    for (let i = 0; i < n; i += xStep) {
      const x = pad.l + (i / (n - 1)) * plotW;
      ctx.fillText(i.toString(), x, h - 4);
    }

    // Line
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = pad.l + (i / (n - 1)) * plotW;
      const y = pad.t + plotH - ((this.data[i] - yMin) / (yMax - yMin)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under
    const lastX = pad.l + plotW;
    const lastY = pad.t + plotH - ((this.data[n - 1] - yMin) / (yMax - yMin)) * plotH;
    ctx.lineTo(lastX, pad.t + plotH);
    ctx.lineTo(pad.l, pad.t + plotH);
    ctx.closePath();
    ctx.fillStyle = this.color + '15';
    ctx.fill();

    // Current value indicator
    ctx.fillStyle = this.color;
    ctx.font = '11px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(this.yFormat(this.data[n - 1]), w - pad.r, pad.t + 12);
  }
}

const chartReward = new LineChart('chart-reward', COLORS.cyan, 'Reward', v => v.toFixed(2));
const chartWinRate = new LineChart('chart-winrate', COLORS.green, 'Win Rate', v => (v * 100).toFixed(1) + '%');
const chartEntropy = new LineChart('chart-entropy', COLORS.orange, 'Entropy', v => v.toFixed(3));
const chartVLoss = new LineChart('chart-vloss', COLORS.red, 'V-Loss', v => v.toFixed(4));

// ── Data fetching ──────────────────────────────────────────────────────────────

async function fetchMetrics() {
  try {
    const resp = await fetch(METRICS_URL + '?t=' + Date.now());
    if (!resp.ok) return;
    const text = await resp.text();
    const lines = text.trim().split('\n').filter(l => l.length > 0);
    metrics = lines.map(l => {
      try { return JSON.parse(l); }
      catch { return null; }
    }).filter(Boolean);
    updateCharts();
    updateLiveIndicator();
  } catch (e) {
    // Silently retry on next poll
  }
}

async function fetchMatches() {
  try {
    const resp = await fetch(MATCHES_INDEX_URL + '?t=' + Date.now());
    if (!resp.ok) return;
    const data = await resp.json();
    if (Array.isArray(data)) {
      matches = data;
      renderMatchGrid();
    }
  } catch (e) {
    // Try loading individual match files from a listing
  }
}

async function fetchMatchData(id) {
  if (matchDataCache[id]) return matchDataCache[id];
  try {
    const resp = await fetch(MATCHES_DIR + id + '.json?t=' + Date.now());
    if (!resp.ok) return null;
    const data = await resp.json();
    matchDataCache[id] = data;
    return data;
  } catch (e) {
    return null;
  }
}

// ── Chart updates ──────────────────────────────────────────────────────────────

function updateCharts() {
  if (metrics.length === 0) return;

  chartReward.update(metrics.map(m => m.reward ?? m.best_fitness ?? 0));
  chartWinRate.update(metrics.map(m => m.win_rate ?? m.vs_population_win_rate ?? 0));
  chartEntropy.update(metrics.map(m => m.entropy ?? m.diversity ?? 0));
  chartVLoss.update(metrics.map(m => m.value_loss ?? m.mean_fitness ?? 0));
}

// ── Live indicator ─────────────────────────────────────────────────────────────

let lastMetricTime = 0;

function updateLiveIndicator() {
  if (metrics.length === 0) return;

  const latest = metrics[metrics.length - 1];
  const gen = latest.gen ?? metrics.length - 1;
  $liveGen.textContent = `Gen ${gen}`;

  // Estimate throughput from timestamps if available
  if (latest.timestamp) {
    const now = Date.now() / 1000;
    const elapsed = now - (metrics[0].timestamp || now);
    if (elapsed > 0 && metrics.length > 1) {
      const genPerSec = metrics.length / elapsed;
      $liveThroughput.textContent = `${genPerSec.toFixed(2)} gen/s`;
    }
  } else if (latest.transitions_per_sec) {
    $liveThroughput.textContent = `${latest.transitions_per_sec.toFixed(0)} trans/s`;
  }

  if (latest.total_generations) {
    const remaining = latest.total_generations - gen;
    $liveEta.textContent = `${remaining} gen remaining`;
  }

  // Mark as active if data updated recently
  $liveDot.classList.add('active');
  clearTimeout(lastMetricTime);
  lastMetricTime = setTimeout(() => $liveDot.classList.remove('active'), 5000);
}

// ── Match grid ─────────────────────────────────────────────────────────────────

function renderMatchGrid() {
  $matchCount.textContent = `(${matches.length} matches)`;
  $matchGrid.innerHTML = '';

  // Show most recent first
  const sorted = [...matches].reverse();

  for (const m of sorted) {
    const card = document.createElement('div');
    card.className = 'match-card';
    card.dataset.matchId = m.id;

    const winnerClass = m.winner === 'home' ? 'home' :
                        m.winner === 'away' ? 'away' : 'draw';
    const winnerLabel = m.winner === 'home' ? 'HOME' :
                        m.winner === 'away' ? 'AWAY' : 'DRAW';

    card.innerHTML = `
      <div class="mc-header">
        <span class="mc-gen">Gen ${m.gen ?? '?'}</span>
        <span class="mc-winner ${winnerClass}">${winnerLabel}</span>
      </div>
      <div class="mc-score">
        <span class="home-score">${m.score_home ?? 0}</span>
        <span class="dim"> - </span>
        <span class="away-score">${m.score_away ?? 0}</span>
      </div>
      <svg class="mc-mini-replay" viewBox="-110 -110 220 220" data-match-id="${m.id}"></svg>
      <div class="mc-meta">seed ${m.seed ?? '?'} | ${m.ticks ?? '?'} ticks</div>
    `;

    card.addEventListener('click', () => openMatchOverlay(m));
    $matchGrid.appendChild(card);

    // Render mini replay if we have frame data in the match
    if (m.snapshot_positions) {
      renderMiniReplay(card.querySelector('svg'), m.snapshot_positions);
    }
  }
}

function renderMiniReplay(svg, positions) {
  // positions: array of frames, each frame: { home: [[x,z],...], away: [[x,z],...], bell: [x,z] }
  // For the mini replay, show the final frame as a static 2D top-down view
  if (!positions || positions.length === 0) return;

  const frame = positions[positions.length - 1]; // show final state
  svg.innerHTML = '';

  // Cylinder boundary (circle)
  const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  circle.setAttribute('cx', '0');
  circle.setAttribute('cy', '0');
  circle.setAttribute('r', '100');
  circle.setAttribute('fill', 'none');
  circle.setAttribute('stroke', COLORS.dim + '40');
  circle.setAttribute('stroke-width', '1.5');
  svg.appendChild(circle);

  // Players
  if (frame.home) {
    for (const [x, z] of frame.home) {
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', x);
      dot.setAttribute('cy', z);
      dot.setAttribute('r', '6');
      dot.setAttribute('fill', COLORS.cyan);
      svg.appendChild(dot);
    }
  }
  if (frame.away) {
    for (const [x, z] of frame.away) {
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', x);
      dot.setAttribute('cy', z);
      dot.setAttribute('r', '6');
      dot.setAttribute('fill', COLORS.orange);
      svg.appendChild(dot);
    }
  }
  // Bell
  if (frame.bell) {
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('cx', frame.bell[0]);
    dot.setAttribute('cy', frame.bell[1]);
    dot.setAttribute('r', '5');
    dot.setAttribute('fill', COLORS.paper);
    dot.setAttribute('stroke', COLORS.paper);
    dot.setAttribute('stroke-width', '2');
    svg.appendChild(dot);
  }
}

// ── Match overlay / expanded view ──────────────────────────────────────────────

async function openMatchOverlay(matchMeta) {
  $overlay.classList.remove('hidden');

  $detailHeader.innerHTML = `
    <h3>Gen ${matchMeta.gen ?? '?'} — <span class="cyan">${matchMeta.home_policy ?? 'trained'}</span>
    vs <span class="orange">${matchMeta.away_policy ?? 'opponent'}</span></h3>
    <p class="dim" style="font-size:12px; margin-top:4px;">
      Score: ${matchMeta.score_home ?? 0} - ${matchMeta.score_away ?? 0} |
      Winner: ${matchMeta.winner ?? 'none'} |
      Ticks: ${matchMeta.ticks ?? '?'} |
      Seed: ${matchMeta.seed ?? '?'}
    </p>
  `;

  // Try to load full match data with frame log
  const fullData = await fetchMatchData(matchMeta.id);
  if (fullData && fullData.frames) {
    startReplay(fullData);
    $detailJson.textContent = JSON.stringify({
      id: fullData.id,
      gen: fullData.gen,
      score_home: fullData.score_home,
      score_away: fullData.score_away,
      winner: fullData.winner,
      ticks: fullData.ticks,
      frame_count: fullData.frames.length,
    }, null, 2);
  } else {
    $detailJson.textContent = JSON.stringify(matchMeta, null, 2);
    $replaySvg.innerHTML = `<text x="0" y="0" fill="${COLORS.dim}" text-anchor="middle" font-size="12">No frame data available</text>`;
  }
}

function closeMatchOverlay() {
  $overlay.classList.add('hidden');
  stopReplay();
}

$overlayClose.addEventListener('click', closeMatchOverlay);
$overlay.addEventListener('click', (e) => {
  if (e.target === $overlay) closeMatchOverlay();
});

// ── 2D SVG replay (expanded view) ─────────────────────────────────────────────

function startReplay(matchData) {
  stopReplay();
  if (!matchData.frames || matchData.frames.length === 0) return;

  replayState = {
    matchData,
    currentTick: 0,
    playing: true,
    animFrame: null,
    speed: 8, // 8x speed
    lastFrameTime: performance.now(),
  };

  $replayScrub.max = matchData.frames.length - 1;
  $replayScrub.value = 0;
  $replayTick.textContent = `0 / ${matchData.frames.length}`;

  renderReplayFrame(0);
  animateReplay();
}

function stopReplay() {
  if (replayState && replayState.animFrame) {
    cancelAnimationFrame(replayState.animFrame);
  }
  replayState = null;
}

function animateReplay() {
  if (!replayState || !replayState.playing) return;

  const now = performance.now();
  const dt = now - replayState.lastFrameTime;

  // Advance at speed * 60fps equivalent
  if (dt > (1000 / 60)) {
    replayState.lastFrameTime = now;
    replayState.currentTick += replayState.speed;
    if (replayState.currentTick >= replayState.matchData.frames.length) {
      replayState.currentTick = 0; // loop
    }
    renderReplayFrame(replayState.currentTick);
    $replayScrub.value = replayState.currentTick;
    $replayTick.textContent = `${replayState.currentTick} / ${replayState.matchData.frames.length}`;
  }

  replayState.animFrame = requestAnimationFrame(animateReplay);
}

function renderReplayFrame(tickIdx) {
  if (!replayState) return;
  const frames = replayState.matchData.frames;
  if (tickIdx >= frames.length) return;

  const frame = frames[tickIdx];
  $replaySvg.innerHTML = '';

  // Cylinder boundary
  const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  circle.setAttribute('cx', '0');
  circle.setAttribute('cy', '0');
  circle.setAttribute('r', '100');
  circle.setAttribute('fill', 'none');
  circle.setAttribute('stroke', COLORS.dim + '50');
  circle.setAttribute('stroke-width', '1');
  $replaySvg.appendChild(circle);

  // Center line
  const centerLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  centerLine.setAttribute('x1', '-100');
  centerLine.setAttribute('y1', '0');
  centerLine.setAttribute('x2', '100');
  centerLine.setAttribute('y2', '0');
  centerLine.setAttribute('stroke', COLORS.dim + '30');
  centerLine.setAttribute('stroke-width', '0.5');
  $replaySvg.appendChild(centerLine);

  // Players
  if (frame.home) {
    for (const [x, z] of frame.home) {
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', x);
      dot.setAttribute('cy', z);
      dot.setAttribute('r', '5');
      dot.setAttribute('fill', COLORS.cyan);
      dot.setAttribute('opacity', '0.9');
      $replaySvg.appendChild(dot);
    }
  }
  if (frame.away) {
    for (const [x, z] of frame.away) {
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', x);
      dot.setAttribute('cy', z);
      dot.setAttribute('r', '5');
      dot.setAttribute('fill', COLORS.orange);
      dot.setAttribute('opacity', '0.9');
      $replaySvg.appendChild(dot);
    }
  }
  // Bell (distinct shape — diamond)
  if (frame.bell) {
    const [bx, bz] = frame.bell;
    const bell = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    const s = 6;
    bell.setAttribute('points', `${bx},${bz-s} ${bx+s},${bz} ${bx},${bz+s} ${bx-s},${bz}`);
    bell.setAttribute('fill', COLORS.paper);
    bell.setAttribute('opacity', '0.95');
    $replaySvg.appendChild(bell);
  }

  // Tick counter inside SVG
  const tickText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  tickText.setAttribute('x', '-105');
  tickText.setAttribute('y', '-105');
  tickText.setAttribute('fill', COLORS.dim);
  tickText.setAttribute('font-size', '8');
  tickText.setAttribute('font-family', 'monospace');
  tickText.textContent = `t=${tickIdx}`;
  $replaySvg.appendChild(tickText);
}

// Replay controls
$replayPlay.addEventListener('click', () => {
  if (replayState) {
    replayState.playing = true;
    replayState.lastFrameTime = performance.now();
    animateReplay();
  }
});

$replayPause.addEventListener('click', () => {
  if (replayState) {
    replayState.playing = false;
    if (replayState.animFrame) cancelAnimationFrame(replayState.animFrame);
  }
});

$replayScrub.addEventListener('input', (e) => {
  if (replayState) {
    replayState.currentTick = parseInt(e.target.value);
    renderReplayFrame(replayState.currentTick);
    $replayTick.textContent = `${replayState.currentTick} / ${replayState.matchData.frames.length}`;
  }
});

// ── Polling loop ───────────────────────────────────────────────────────────────

async function poll() {
  await Promise.all([fetchMetrics(), fetchMatches()]);
}

poll();
setInterval(poll, POLL_INTERVAL);

// ── Keyboard shortcuts ─────────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeMatchOverlay();
});

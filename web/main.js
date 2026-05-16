// main.js — AX loop + screen manager
// Owns requestAnimationFrame, screen transitions, and audio event driving.
window.AX = window.AX || {};

(function () {
  'use strict';

  // ── DOM setup ─────────────────────────────────────────────────────────────
  const canvas = document.getElementById('axball-canvas');
  AX.Render.init(canvas);
  AX.Input.init(canvas);

  // ── screen FSM ───────────────────────────────────────────────────────────
  // 'loading' | 'title' | 'game' | 'end'
  let screen = 'title';

  // Buttons returned by draw functions (hit-test on click)
  let titleBtn = null;
  let endBtn   = null;

  // ── engine + game instances ───────────────────────────────────────────────
  let engine = null;
  let game   = null;

  function modulesAvailable() {
    return typeof AX.Engine === 'function' && typeof AX.Game === 'function';
  }

  function createGameInstances() {
    try {
      engine = new AX.Engine();
      game   = new AX.Game(engine);
      return true;
    } catch (e) {
      console.error('AX: failed to create Engine/Game instances', e);
      engine = null;
      game   = null;
      return false;
    }
  }

  // ── audio event tracking ─────────────────────────────────────────────────
  // We watch game.state changes each frame to fire audio.
  const prev = {
    scoreHome:  0,
    scoreAway:  0,
    phase:      null,
    shotClock:  5,
    highFrame:  false,
    winner:     null,
  };

  function driveAudio(gs) {
    if (!gs) return;

    const score = gs.score || { home: 0, away: 0 };

    // Score events
    const scoreDelta = (score.home - prev.scoreHome) + (score.away - prev.scoreAway);
    if (scoreDelta > 0) {
      if (gs.phase === 'long' || (gs.message && gs.message.toLowerCase().includes('long'))) {
        AX.Audio.event('long');
      } else if (scoreDelta >= 2) {
        AX.Audio.event('score2');
      } else {
        AX.Audio.event('score1');
      }
    }
    prev.scoreHome = score.home;
    prev.scoreAway = score.away;

    // Phase change events
    if (gs.phase !== prev.phase) {
      switch (gs.phase) {
        case 'pushoff':   AX.Audio.event('pushoff');   break;
        case 'turnover':  AX.Audio.event('turnover');  break;
        case 'freespin':  AX.Audio.event('freespin');  break;
      }
      prev.phase = gs.phase;
    }

    // High frame
    if (gs.highFrame && !prev.highFrame) {
      AX.Audio.event('highframe');
    }
    prev.highFrame = gs.highFrame;

    // Shot clock tick (each whole second crossing below 3)
    if (gs.shotClock != null && gs.shotClock <= 3) {
      const prevFloor = Math.floor(prev.shotClock || 5);
      const curFloor  = Math.floor(gs.shotClock);
      if (curFloor < prevFloor) AX.Audio.event('tick');
    }
    prev.shotClock = gs.shotClock;

    // Win / lose
    if (gs.winner && gs.winner !== prev.winner) {
      const isP1Home = true; // P1 is always home team
      AX.Audio.event(
        (gs.winner === 'home' && isP1Home) || (gs.winner === 'away' && !isP1Home)
          ? 'win'
          : 'lose'
      );
      prev.winner = gs.winner;
    }
  }

  // Also drive audio from engine events if available
  function driveEngineAudio() {
    if (!engine || typeof engine.pollEvents !== 'function') return;
    const events = engine.pollEvents();
    if (!events) return;
    for (const ev of events) {
      switch (ev.type) {
        case 'catch':   AX.Audio.event('catch');  break;
        case 'throw':   AX.Audio.event('throw', ev.charge || 0.5); break;
        case 'skin':    AX.Audio.event('skin');   break;
        case 'pushoff': AX.Audio.event('pushoff'); break;
      }
    }
  }

  // ── click handler ─────────────────────────────────────────────────────────
  canvas.addEventListener('click', function (e) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    if (screen === 'title' && titleBtn) {
      if (cx >= titleBtn.btnX && cx <= titleBtn.btnX + titleBtn.btnW &&
          cy >= titleBtn.btnY && cy <= titleBtn.btnY + titleBtn.btnH) {
        startGame();
      }
    } else if (screen === 'end' && endBtn) {
      if (cx >= endBtn.btnX && cx <= endBtn.btnX + endBtn.btnW &&
          cy >= endBtn.btnY && cy <= endBtn.btnY + endBtn.btnH) {
        resetGame();
      }
    }
  });

  function startGame() {
    AX.Audio.init(); // first user gesture — safe to init

    if (!modulesAvailable()) {
      console.warn('AX: engine/game modules not yet available');
      return;
    }
    if (!createGameInstances()) return;

    // Reset audio tracking
    prev.scoreHome = 0;
    prev.scoreAway = 0;
    prev.phase     = null;
    prev.shotClock = 5;
    prev.highFrame = false;
    prev.winner    = null;

    screen = 'game';
  }

  function resetGame() {
    engine = null;
    game   = null;
    screen = 'title';
    titleBtn = null;
    endBtn   = null;
  }

  // ── main loop ─────────────────────────────────────────────────────────────
  let lastTime = null;

  function loop(ts) {
    requestAnimationFrame(loop);

    const now = ts / 1000;
    const rawDt = lastTime !== null ? now - lastTime : 0.016;
    lastTime = now;
    const dt = Math.min(rawDt, 1 / 30); // cap at 30fps worth

    switch (screen) {

      case 'title': {
        // Check if engine/game loaded while on title
        if (modulesAvailable()) {
          titleBtn = AX.Render.drawTitleScreen();
        } else {
          AX.Render.drawLoadingScreen();
        }
        break;
      }

      case 'game': {
        if (!engine || !game) { screen = 'title'; break; }

        // Read input
        const input = AX.Input.get(dt);

        // Step game
        try {
          game.update(dt, input);
        } catch (e) {
          console.error('AX: game.update error', e);
        }

        // Audio
        driveAudio(game.state);
        driveEngineAudio();

        // Render
        AX.Render.draw(engine.state, game.state);

        // Transition to end screen
        if (game.state && game.state.phase === 'over') {
          setTimeout(() => { screen = 'end'; }, 2200);
        }
        break;
      }

      case 'end': {
        const gs = game ? game.state : null;
        endBtn = AX.Render.drawEndScreen(gs);
        break;
      }

      default: {
        AX.Render.drawLoadingScreen();
        screen = 'title';
      }
    }
  }

  // ── kick off ──────────────────────────────────────────────────────────────
  requestAnimationFrame(loop);

}());

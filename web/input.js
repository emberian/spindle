// input.js — AX.Input
// Converts raw browser events into a normalized input snapshot each tick.
// aim is in WORLD coordinates (via AX.Render.screenToWorld).
window.AX = window.AX || {};

AX.Input = (function () {
  'use strict';

  // ── state ─────────────────────────────────────────────────────────────────
  const raw = {
    mouseX: 0,
    mouseY: 0,
    lmbDown: false,
    spaceDown: false,
    shiftDown: false,
    wDown: false,
    sDown: false,
    scrollDelta: 0,    // accumulated since last get()
    spaceReleased: false,  // true the tick it was released
  };

  let _spaceWasDown = false;

  // Derived / smoothed
  let throwCharge = 0;           // 0..1, builds while space held
  let throwReleased = false;     // one-tick true on space release
  const CHARGE_RATE = 1 / 1.2;  // full charge in 1.2 seconds

  let canvas = null;

  // ── init ──────────────────────────────────────────────────────────────────
  function init(c) {
    canvas = c;

    c.addEventListener('mousemove', onMouseMove);
    c.addEventListener('mousedown', onMouseDown);
    c.addEventListener('mouseup',   onMouseUp);
    c.addEventListener('wheel',     onWheel, { passive: true });
    c.addEventListener('contextmenu', e => e.preventDefault());

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup',   onKeyUp);

    // Prevent space from scrolling the page
    window.addEventListener('keydown', e => {
      if (e.code === 'Space') e.preventDefault();
    }, { passive: false });
  }

  // ── event handlers ────────────────────────────────────────────────────────
  function onMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    raw.mouseX = e.clientX - rect.left;
    raw.mouseY = e.clientY - rect.top;
  }

  function onMouseDown(e) {
    if (e.button === 0) raw.lmbDown = true;
  }

  function onMouseUp(e) {
    if (e.button === 0) raw.lmbDown = false;
  }

  function onWheel(e) {
    // positive delta = scroll down = reel in (approach anchor)
    raw.scrollDelta += e.deltaY;
  }

  function onKeyDown(e) {
    switch (e.code) {
      case 'Space':  raw.spaceDown = true; break;
      case 'ShiftLeft': case 'ShiftRight': raw.shiftDown = true; break;
      case 'KeyW': raw.wDown = true; break;
      case 'KeyS': raw.sDown = true; break;
    }
  }

  function onKeyUp(e) {
    switch (e.code) {
      case 'Space':
        raw.spaceDown = false;
        raw.spaceReleased = true;
        break;
      case 'ShiftLeft': case 'ShiftRight': raw.shiftDown = false; break;
      case 'KeyW': raw.wDown = false; break;
      case 'KeyS': raw.sDown = false; break;
    }
  }

  // ── get (called once per frame by main loop) ──────────────────────────────
  function get(dt) {
    dt = dt || 0.016;

    // Convert mouse position to world coords via render
    let aim = { x: 0, y: 0 };
    if (AX.Render && AX.Render.screenToWorld) {
      aim = AX.Render.screenToWorld(raw.mouseX, raw.mouseY);
    }

    // Throw charge
    throwReleased = false;
    if (raw.spaceDown) {
      throwCharge = Math.min(1, throwCharge + dt * CHARGE_RATE);
    } else if (_spaceWasDown || raw.spaceReleased) {
      // Space was just released
      throwReleased = throwCharge > 0.05;
      throwCharge = 0;
      raw.spaceReleased = false;
    }
    _spaceWasDown = raw.spaceDown;

    // Reel: W = reel in (-1), S = reel out (+1), scroll maps similarly
    let reel = 0;
    if (raw.wDown) reel = -1;
    else if (raw.sDown) reel = 1;
    // Override with scroll if active (sign: scroll up = reel in)
    if (Math.abs(raw.scrollDelta) > 5) {
      reel = raw.scrollDelta > 0 ? 1 : -1;
    }
    raw.scrollDelta = 0;  // consume

    const snapshot = {
      aim,
      grapple:       raw.lmbDown,
      throwCharge,
      throwReleased,
      reel,
      pushKey:       raw.shiftDown,
    };

    return snapshot;
  }

  return { init, get };
}());

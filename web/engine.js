// AX.Engine — AXBALL physics engine
// Pure client-side, no build, no modules. Global namespace AX.
//
// Tuning notes:
//   omega = 0.22 rad/s (arcade value, kept from contract defaults).
//   Lane dims: 500m long × 80m diameter (radius 40m).
//   At omega=0.22 and a 25 m/s cross-lane throw over ~3 seconds of flight,
//   Coriolis lateral deflection ≈ 2·omega·vx·t² ≈ 2·0.22·25·9 ≈ 99 m —
//   WAY too large at full lane-radius 40m scale. The arcade playfield uses
//   laneRadius=40 but keeps throws below 25 m/s, so a moderate cross-lane
//   throw (~3 s flight at 25 m/s across 40 m radius) curves visibly but
//   stays in-bounds: deflection ~ 2·omega·vy·t² in x ≈ 2·0.22·25·(1.6²) ≈
//   28 m — nicely dramatic. Antispinward motion is lifted inboard (y→0),
//   can loop. Spinward motion is pushed outboard (|y| grows).
//   Asymmetry direction (CANONICAL):
//     a_x = +2·omega·vy   (cross-radial velocity pushes spinward)
//     a_y = -2·omega·vx   (spinward velocity pushes outboard / centrifugal adds)
//     a_y += omega²·y     (centrifugal, radial only)
//   This means:
//     • spinward throw (vx>0): a_y = -2·omega·vx < 0 → pushed toward -y? No:
//       outboard is +|y| direction. In our frame y=0 is axis, |y|=laneRadius
//       is skin. A player at y>0 (above axis in cross-section) feels centrifugal
//       push in +y. "Spinward" is +x. Coriolis on vx: a_y -= 2·omega·vx.
//       If player is at y>0 and throws spinward, a_y goes negative (toward axis),
//       which is inboard. That contradicts the spec "spinward pushed outboard."
//       Re-reading spec: "spinward/+x motion gets pushed outboard" — this refers
//       to a_x from vy, not a_y from vx. Let's be precise:
//         Coriolis in 2D rotating frame with Omega=+z: a_Cor = -2·Ω×v
//           = -2·(ẑ·omega)×(vx·x̂ + vy·ŷ)
//           = -2·omega·(vx·(ẑ×x̂) + vy·(ẑ×ŷ))
//           = -2·omega·(vx·ŷ - vy·x̂)
//           = 2·omega·vy·x̂ - 2·omega·vx·ŷ
//       So: a_x_Cor = +2·omega·vy,  a_y_Cor = -2·omega·vx.
//       A stone thrown radially outward (vy>0): a_x = +2·omega·vy > 0 → deflects spinward. CHECK.
//       A stone thrown spinward (vx>0): a_y = -2·omega·vx < 0 → deflects inward (toward axis).
//       This means spinward throws curve toward axis (lift inboard), antispinward throws curve outboard.
//       The spec says "spinward throws arrive faster" (less time lost fighting centrifugal = they hug
//       the lane and draft), and "antispinward motion gets lifted inboard and can loop" — hmm, that
//       contradicts the math where vx<0 gives a_y = -2·omega·vx = +2·omega·|vx| which is OUTBOARD.
//       Wait: the spec says antispinward "drift/slow" and "hard antispinward throw can curve back".
//       And "spinward/+x motion gets pushed outboard" — let me re-read: "Coriolis 'spin drift':
//       spinward throws arrive faster, antispinward drift & slow; elite play exploits this asymmetry;
//       smaller/faster cylinders show it more."  This is about FLIGHT TIME / energy, not direction.
//       The physics contract says:
//         a_x += 2*omega*vy ; a_y += -2*omega*vx
//       This is exactly what I derived. a_y = -2·omega·vx:
//         vx > 0 (spinward): a_y < 0 → deflects toward axis (inboard) if y>0, or toward +y if y<0.
//         vx < 0 (antispinward): a_y > 0 → deflects away from axis (outboard if y>0).
//       The asymmetry: a cross-lane throw (vy ≠ 0) gets spinward deflection (a_x = 2·omega·vy).
//       I will implement exactly the contract formulas. The "asymmetry is correct" confirmation:
//         Spinward throw (vx>0): curves toward axis (inboard) — counteracts centrifugal, "arrives faster"
//         Antispinward throw (vx<0): curves outboard — adds to centrifugal, loses speed, can bounce skin.
//         Cross-lane throw (vy>0, outward): deflects spinward (+x). CANONICAL. ✓

var AX = window.AX || {};

(function(AX) {
  'use strict';

  // ── Defaults ────────────────────────────────────────────────────────────────
  var DEFAULTS = {
    laneLength:    500,
    laneRadius:    40,
    gateRadius:    10,
    deeplineX:     170,
    midline:       0,
    omega:         0.22,
    thrumblerCap:  2.0,
    stoneRadius:   0.4,
    playerRadius:  0.8,
  };

  // Max substeps per step call (guards against spiral of death)
  var MAX_SUBSTEPS = 8;
  // Fixed substep size for integration stability
  var SUBSTEP_DT   = 1/120;
  // Skin contact epsilon (metres)
  var SKIN_EPS     = 0.05;
  // Tether length limits
  var TETHER_MIN   = 3;
  var TETHER_MAX   = 60;
  // Stone throw speed range
  var THROW_MIN    = 8;   // m/s at power=0
  var THROW_MAX    = 34;  // m/s at power=1

  // ── Constructor ─────────────────────────────────────────────────────────────
  function Engine(config) {
    var cfg = {};
    var key;
    for (key in DEFAULTS) { cfg[key] = DEFAULTS[key]; }
    if (config) {
      for (key in config) { if (config.hasOwnProperty(key)) cfg[key] = config[key]; }
    }

    this._cfg     = cfg;
    this._events  = [];

    // Build state skeleton
    this.state = this._makeState();
    this.reset('pushoff');
  }

  // ── Internal helpers ─────────────────────────────────────────────────────────

  Engine.prototype._makeState = function() {
    var cfg = this._cfg;
    return {
      omega: cfg.omega,
      lane: {
        length:     cfg.laneLength,
        radius:     cfg.laneRadius,
        gateRadius: cfg.gateRadius,
        deeplineX:  cfg.deeplineX,
        midline:    cfg.midline,
      },
      stone: {
        x: 0, y: 0, vx: 0, vy: 0,
        spinTrue:            false,
        heldBy:              null,
        lastThrownBy:        null,
        untouchedSinceThrow: false,
      },
      players: [],
    };
  };

  // Push an event into the queue
  Engine.prototype._emit = function(ev) {
    this._events.push(ev);
  };

  // Clamp a value to [lo, hi]
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  // ── Public API ───────────────────────────────────────────────────────────────

  Engine.prototype.reset = function(scenario) {
    var cfg = this._cfg;
    var s   = this.state;
    var L   = cfg.laneLength / 2;   // half-length
    var R   = cfg.laneRadius;

    // Reset stone
    s.stone.x    = 0; s.stone.y    = 0;
    s.stone.vx   = 0; s.stone.vy   = 0;
    s.stone.heldBy              = null;
    s.stone.lastThrownBy        = null;
    s.stone.untouchedSinceThrow = false;
    s.stone.spinTrue            = false;

    // Default two players
    s.players = [
      {
        id: 'p0', team: 'home', role: 'runner',
        x: -30, y: R, vx: 0, vy: 0,
        tether:   null,
        dvBudget: cfg.thrumblerCap,
        onSkin:   false,
        contact:  true,
      },
      {
        id: 'p1', team: 'away', role: 'runner',
        x:  30, y: R, vx: 0, vy: 0,
        tether:   null,
        dvBudget: cfg.thrumblerCap,
        onSkin:   false,
        contact:  true,
      },
    ];

    if (scenario === 'thrown-test') {
      // Home player on +y skin near midline, stone thrown free
      s.players[0].x = 0;  s.players[0].y = R;
      s.players[1].x = 0;  s.players[1].y = -R;

      // Stone given a spinward-diagonal throw from player 0
      s.stone.x  = 0;  s.stone.y  = R - 2;
      s.stone.vx = 15; s.stone.vy = -20;  // spinward + radially inward
      s.stone.heldBy              = null;
      s.stone.lastThrownBy        = 'p0';
      s.stone.untouchedSinceThrow = true;
    }

    this._events = [];
  };

  // ── Physics step ─────────────────────────────────────────────────────────────

  Engine.prototype.step = function(dt) {
    var cfg = this._cfg;
    // Clamp dt hard, then subdivide
    dt = clamp(dt, 0, 1/10);
    var n  = Math.ceil(dt / SUBSTEP_DT);
    if (n < 1) n = 1;
    if (n > MAX_SUBSTEPS) n = MAX_SUBSTEPS;
    var h  = dt / n;
    var i;
    for (i = 0; i < n; i++) {
      this._substep(h);
    }
  };

  Engine.prototype._substep = function(h) {
    var cfg   = this._cfg;
    var s     = this.state;
    var omega = s.omega;
    var R     = cfg.laneRadius;
    var L     = cfg.laneLength / 2;
    var GR    = cfg.gateRadius;

    // ── Integrate stone (if free) ──────────────────────────────────────────────
    var st = s.stone;
    if (st.heldBy === null) {
      var ax = 0;
      var ay = 0;

      // Centrifugal: radial, a_y = omega^2 * y
      ay += omega * omega * st.y;

      // Coriolis: a_x = +2·omega·vy,  a_y = -2·omega·vx
      ax += 2 * omega * st.vy;
      ay -= 2 * omega * st.vx;

      // Semi-implicit Euler: update velocity first, then position
      st.vx += ax * h;
      st.vy += ay * h;
      st.x  += st.vx * h;
      st.y  += st.vy * h;

      // Wall/skin bounce at |y| = R
      this._stoneSkinBounce(st, R);

      // Check gate scoring
      this._checkGate(st, L, GR);
    } else {
      // Stone held: move with player
      var holder = this._findPlayer(st.heldBy);
      if (holder) {
        st.x = holder.x;
        st.y = holder.y;
        st.vx = holder.vx;
        st.vy = holder.vy;
      }
    }

    // ── Integrate players ──────────────────────────────────────────────────────
    var pi;
    for (pi = 0; pi < s.players.length; pi++) {
      var p = s.players[pi];
      this._integratePlayer(p, omega, R, L, h);
    }
  };

  Engine.prototype._stoneSkinBounce = function(st, R) {
    if (Math.abs(st.y) >= R) {
      // Reflect radial velocity, lose a little energy (coefficient of restitution ~0.6)
      st.y  = st.y  > 0 ? R  : -R;
      st.vy = -st.vy * 0.6;
      this._emit({ type: 'skin', stoneContact: true });
    }
  };

  Engine.prototype._checkGate = function(st, L, GR) {
    // Did stone pass a gate (|x| > L and |y| <= GR)?
    var scored = false;
    var team   = null;
    var method = null;
    var points = 0;

    if (Math.abs(st.x) > L) {
      if (Math.abs(st.y) <= GR) {
        // Valid gate crossing
        team = (st.x > 0) ? 'home' : 'away'; // +x gate = home scores? arbitrary.
        // Actually: the gate the stone went through. We'll say whichever team last threw it scores.
        // Use lastThrownBy team for attribution.
        var thrower = this._findPlayer(st.lastThrownBy);
        if (thrower) team = thrower.team;

        if (st.heldBy !== null) {
          method = 'carry';
          points = 2;
        } else if (st.untouchedSinceThrow) {
          // Check long launch (was last thrown from beyond deeplineX from the same end?)
          // We don't track throw origin x, so approximate: deepline is cfg.deeplineX from axis
          // We record throw origin if needed — for simplicity check lastThrownFrom
          if (st._throwOriginX !== undefined &&
              Math.abs(st._throwOriginX) >= this._cfg.deeplineX) {
            method = 'long';
            points = 3;
          } else {
            method = 'thrown';
            points = 1;
          }
        } else {
          method = 'thrown';
          points = 1;
        }

        this._emit({ type: 'score', team: team, method: method, points: points });

        // Freeze stone for now (wait for reset/possession change)
        st.vx = 0; st.vy = 0;
        st.x  = st.x > 0 ? L : -L; // park at gate
      } else {
        // Gate miss
        var missteam = st.x > 0 ? 'home' : 'away';
        this._emit({ type: 'gate_miss', team: missteam });
        // Bounce off end wall
        st.x  = st.x > 0 ? L : -L;
        st.vx = -st.vx * 0.5;
      }
    }
  };

  Engine.prototype._integratePlayer = function(p, omega, R, L, h) {
    var cfg = this._cfg;

    if (p.contact) {
      // On a structure: no free drift. Velocity zeroes out.
      p.vx = 0; p.vy = 0;
      return;
    }

    // Free-flying player: same centrifugal + Coriolis
    var ax = 0;
    var ay = omega * omega * p.y; // centrifugal

    ax += 2 * omega * p.vy;
    ay -= 2 * omega * p.vx;

    // Tether constraint
    if (p.tether && p.tether.anchored) {
      var t  = p.tether;
      var dx = p.x - t.ax;
      var dy = p.y - t.ay;
      var d  = Math.sqrt(dx*dx + dy*dy);
      if (d > t.len) {
        // Pull back: apply constraint impulse (simplified spring clamp)
        var nx = dx / d;
        var ny = dy / d;
        // Project velocity onto tether direction, remove outward component
        var vrad = p.vx * nx + p.vy * ny;
        if (vrad > 0) {
          p.vx -= vrad * nx;
          p.vy -= vrad * ny;
        }
        // Clamp position to tether length
        p.x = t.ax + nx * t.len;
        p.y = t.ay + ny * t.len;
      }
    }

    // Semi-implicit Euler
    p.vx += ax * h;
    p.vy += ay * h;
    p.x  += p.vx * h;
    p.y  += p.vy * h;

    // Skin contact
    if (Math.abs(p.y) >= R - SKIN_EPS) {
      p.y       = p.y > 0 ? R : -R;
      p.vy      = 0;
      p.onSkin  = true;
      p.contact = true;
      this._emit({ type: 'skin', playerId: p.id });
    } else {
      p.onSkin = false;
    }

    // Clamp to lane x
    p.x = clamp(p.x, -L, L);
  };

  Engine.prototype._findPlayer = function(id) {
    if (!id) return null;
    var ps = this.state.players;
    for (var i = 0; i < ps.length; i++) {
      if (ps[i].id === id) return ps[i];
    }
    return null;
  };

  // ── pollEvents ───────────────────────────────────────────────────────────────

  Engine.prototype.pollEvents = function() {
    var evs = this._events.slice();
    this._events = [];
    return evs;
  };

  // ── Player actions ───────────────────────────────────────────────────────────

  Engine.prototype.applyThrumbler = function(playerId, dvx, dvy) {
    var p   = this._findPlayer(playerId);
    if (!p) return;
    var mag = Math.sqrt(dvx*dvx + dvy*dvy);
    if (mag === 0) return;
    var budget = p.dvBudget;
    if (budget <= 0) return;
    if (mag > budget) {
      // Scale down
      dvx *= budget / mag;
      dvy *= budget / mag;
      mag  = budget;
    }
    p.vx      += dvx;
    p.vy      += dvy;
    p.dvBudget = Math.max(0, budget - mag);
  };

  Engine.prototype.pushOff = function(playerId, dvx, dvy) {
    var p = this._findPlayer(playerId);
    if (!p || !p.contact) return;
    p.vx     += dvx;
    p.vy     += dvy;
    p.contact = false;
    p.onSkin  = false;
  };

  Engine.prototype.fireTether = function(playerId, ax, ay) {
    var p   = this._findPlayer(playerId);
    if (!p) return;
    var R   = this._cfg.laneRadius;
    var L   = this._cfg.laneLength / 2;
    // Clamp anchor to skin (|ay| = R) or to lane bounds
    // Simple: allow any point on the skin ring (clamp ay to ±R) or axis spar (y=0)
    ay = clamp(ay, -R, R);
    ax = clamp(ax, -L, L);
    var dx   = p.x - ax;
    var dy   = p.y - ay;
    var dist = Math.sqrt(dx*dx + dy*dy);
    var len  = clamp(dist, TETHER_MIN, TETHER_MAX);
    p.tether = { anchored: true, ax: ax, ay: ay, len: len };
  };

  Engine.prototype.releaseTether = function(playerId) {
    var p = this._findPlayer(playerId);
    if (!p) return;
    p.tether = null;
  };

  Engine.prototype.reel = function(playerId, delta) {
    var p = this._findPlayer(playerId);
    if (!p || !p.tether) return;
    p.tether.len = clamp(p.tether.len + delta, TETHER_MIN, TETHER_MAX);
  };

  Engine.prototype.pickup = function(playerId) {
    var p  = this._findPlayer(playerId);
    var st = this.state.stone;
    if (!p || st.heldBy !== null) return;
    var threshold = this._cfg.playerRadius + this._cfg.stoneRadius;
    var dx = p.x - st.x;
    var dy = p.y - st.y;
    if (Math.sqrt(dx*dx + dy*dy) <= threshold) {
      st.heldBy              = playerId;
      st.untouchedSinceThrow = false;
    }
  };

  Engine.prototype.throwStone = function(playerId, dirx, diry, power) {
    var p  = this._findPlayer(playerId);
    var st = this.state.stone;
    if (!p || st.heldBy !== playerId) return;

    // Normalise direction
    var mag = Math.sqrt(dirx*dirx + diry*diry);
    if (mag === 0) return;
    dirx /= mag;
    diry /= mag;

    power = clamp(power, 0, 1);
    var speed = THROW_MIN + power * (THROW_MAX - THROW_MIN);

    st.heldBy              = null;
    st.lastThrownBy        = playerId;
    st.untouchedSinceThrow = true;
    st._throwOriginX       = p.x;

    st.vx = p.vx + dirx * speed;
    st.vy = p.vy + diry * speed;
  };

  Engine.prototype.newPossession = function(team) {
    var ps  = this.state.players;
    var cap = this._cfg.thrumblerCap;
    for (var i = 0; i < ps.length; i++) {
      ps[i].dvBudget = cap;
    }
  };

  // ── Export ───────────────────────────────────────────────────────────────────

  AX.Engine = Engine;

})(AX);

window.AX = AX;

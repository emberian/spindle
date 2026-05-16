/**
 * AXBALL game.js — Ruleset state machine + AI opponent
 * Namespace: AX.Game
 * No build, no modules, no npm. Runs via file://.
 *
 * CONFIGURATION (arcade-compressed):
 *   Frame length: 90 seconds (6 frames = ~9 min + freespin)
 *   Players per side: 4 (1 runner, 1 anchor, 1 launch, 1 rover)
 *   Human: P1 (home, runner)
 *   AI controls: home anchor/launch/rover + all 4 away players
 */

(function (AX) {
  "use strict";

  // ─── Constants ────────────────────────────────────────────────────────────

  var FRAME_DURATION  = 90;       // seconds per frame (arcade-compressed)
  var NUM_FRAMES      = 6;
  var SHOT_CLOCK_MAX  = 5;        // seconds to pass/shoot/release
  var HIGH_FRAME_START = 5;       // frames 5 & 6 are High Frames
  var SCORE_THROW     = 1;
  var SCORE_CARRY     = 2;
  var SCORE_LONG      = 3;

  var DEEPLINE_X      = 170;      // |x| > 170 = behind own deepline
  var GATE_X_HOME     = 250;      // home attacks +x gate
  var GATE_X_AWAY     = -250;     // away attacks -x gate

  // AI tuning
  var AI_THRUMBLER_POWER = 0.4;   // fraction of max dv per tick
  var AI_SHOOT_RANGE     = 80;    // start shooting within this x-distance of gate
  var AI_LONG_LAUNCH_X   = 160;   // ideal x for long launch attempt
  var AI_PASS_PROB       = 0.004; // per-tick probability of AI passing to P1
  var AI_BLOCK_RANGE     = 30;    // defender triggers block attempt within this dist
  var AI_BLOCK_SPEED     = 0.3;   // push speed for block attempt
  var AI_TETHER_INTERVAL = 120;   // ticks between tether re-fires for AI swings
  var DISLODGE_CHANCE    = 0.015; // per-tick probability when in range & closing

  // ─── Utility ──────────────────────────────────────────────────────────────

  function safe(v, fallback) {
    return (v === undefined || v === null || isNaN(v)) ? (fallback || 0) : v;
  }

  function dist2(ax, ay, bx, by) {
    var dx = safe(ax) - safe(bx);
    var dy = safe(ay) - safe(by);
    return dx * dx + dy * dy;
  }

  function dist(ax, ay, bx, by) {
    return Math.sqrt(dist2(ax, ay, bx, by));
  }

  function norm(x, y) {
    var m = Math.sqrt(x * x + y * y);
    if (m < 1e-9) return { x: 1, y: 0 };
    return { x: x / m, y: y / m };
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  // ─── AX.Game ──────────────────────────────────────────────────────────────

  function Game(engine) {
    this._engine = engine;

    // Public state (read by render/main)
    this.state = {
      frame:      1,
      frameClock: FRAME_DURATION,
      highFrame:  false,
      score:      { home: 0, away: 0 },
      possession: 'home',
      phase:      'pushoff',  // 'pushoff'|'live'|'score'|'turnover'|'freespin'|'over'
      shotClock:  SHOT_CLOCK_MAX,
      message:    'AXBALL — Get ready!',
      winner:     null
    };

    // Internal bookkeeping
    this._shotHolder    = null;   // player id currently holding & clocked
    this._shotElapsed   = 0;
    this._phaseTimer    = 0;      // seconds in current phase (for score/turnover pause)
    this._pushoffReady  = false;
    this._frameStartPossession = 'home';
    this._aiTick        = {};     // per-player AI state
    this._tetherTick    = {};     // per-player tether cooldown tick counter

    // Kick off
    this._startFrame(1, 'home');
  }

  // ─── Public: called each tick ─────────────────────────────────────────────

  Game.prototype.update = function (dt, input) {
    if (!dt || isNaN(dt) || dt <= 0) dt = 1 / 60;
    dt = Math.min(dt, 0.1); // guard big spikes

    var st   = this.state;
    var eng  = this._engine;
    var eSt  = eng.state;

    // Bail early if game is over
    if (st.phase === 'over') return;

    // ── 1. Human (P1) input ────────────────────────────────────────────────
    this._handleHumanInput(input, eSt);

    // ── 2. AI for all non-P1 players ──────────────────────────────────────
    if (st.phase === 'live' || st.phase === 'pushoff') {
      this._runAI(dt, eSt);
    }

    // ── 3. Physics step ───────────────────────────────────────────────────
    eng.step(dt);

    // ── 4. Consume engine events ──────────────────────────────────────────
    var events = eng.pollEvents();
    for (var i = 0; i < events.length; i++) {
      this._handleEvent(events[i]);
    }

    // ── 5. Rule enforcement ───────────────────────────────────────────────
    this._enforceRules(dt, eSt);
  };

  // ─── Human input handling ─────────────────────────────────────────────────

  Game.prototype._handleHumanInput = function (input, eSt) {
    if (!input) return;

    var eng = this._engine;
    var p1  = this._findPlayer(eSt, 'P1');
    if (!p1) return;

    var aim = input.aim || { x: 0, y: 0 };

    // Tether / grapple
    if (input.grapple) {
      // Fire tether toward aim (or nearest structure — use aim directly)
      var td = norm(aim.x - safe(p1.x), aim.y - safe(p1.y));
      eng.fireTether('P1', td.x, td.y);
    } else {
      eng.releaseTether('P1');
    }

    // Reel
    if (input.reel && input.reel !== 0) {
      eng.reel('P1', input.reel);
    }

    // Throw
    if (input.throwReleased && eSt.stone && eSt.stone.heldBy === 'P1') {
      var dd = norm(aim.x - safe(p1.x), aim.y - safe(p1.y));
      eng.throwStone('P1', dd.x, dd.y, clamp(safe(input.throwCharge, 0.5), 0, 1));
    }

    // Pickup if stone is nearby and not held
    if (!eSt.stone.heldBy) {
      var dstone = dist(p1.x, p1.y, eSt.stone.x, eSt.stone.y);
      if (dstone < 15) {
        eng.pickup('P1');
      }
    }

    // Push-off
    if (input.pushKey && p1.contact) {
      var pd = norm(aim.x - safe(p1.x), aim.y - safe(p1.y));
      eng.pushOff('P1', pd.x * 0.6, pd.y * 0.6);
    }
  };

  // ─── Rule enforcement ─────────────────────────────────────────────────────

  Game.prototype._enforceRules = function (dt, eSt) {
    var st  = this.state;

    // ── Phase timer for pause phases ─────────────────────────────────────
    if (st.phase === 'score' || st.phase === 'turnover') {
      this._phaseTimer -= dt;
      if (this._phaseTimer <= 0) {
        this._beginLive();
      }
      return; // skip rest of rule logic during pause
    }

    if (st.phase === 'over' || st.phase === 'freespin') {
      // freespin: clock doesn't count, just wait for a score event
      return;
    }

    // ── Frame clock ───────────────────────────────────────────────────────
    if (st.phase === 'live' || st.phase === 'pushoff') {
      st.frameClock -= dt;
      if (st.frameClock <= 0) {
        st.frameClock = 0;
        this._endFrame();
        return;
      }
    }

    // ── Shot clock (5s rule) ──────────────────────────────────────────────
    if (st.phase === 'live' && eSt.stone) {
      var holder = eSt.stone.heldBy;
      if (holder) {
        if (holder !== this._shotHolder) {
          // New holder — reset clock
          this._shotHolder  = holder;
          this._shotElapsed = 0;
        }
        this._shotElapsed += dt;
        st.shotClock = Math.max(0, SHOT_CLOCK_MAX - this._shotElapsed);
        if (st.shotClock <= 0) {
          // Turnover!
          st.message = 'SHOT CLOCK — TURNOVER!';
          this._triggerTurnover(eSt);
          return;
        }
      } else {
        // Stone in flight or loose — reset shot clock
        this._shotHolder  = null;
        this._shotElapsed = 0;
        st.shotClock = SHOT_CLOCK_MAX;
      }
    }

    // ── Defender dislodge check ───────────────────────────────────────────
    if (st.phase === 'live' && eSt.stone && eSt.stone.heldBy) {
      this._checkDislodge(eSt);
    }

    // ── Possession tracking from stone state ──────────────────────────────
    if (st.phase === 'live' && eSt.stone && eSt.stone.heldBy) {
      var hp = this._findPlayer(eSt, eSt.stone.heldBy);
      if (hp) {
        st.possession = hp.team;
      }
    }
  };

  Game.prototype._checkDislodge = function (eSt) {
    var st      = this.state;
    var carrier = this._findPlayer(eSt, eSt.stone.heldBy);
    if (!carrier) return;

    var defenders = eSt.players.filter(function (p) {
      return p.team !== carrier.team;
    });

    for (var i = 0; i < defenders.length; i++) {
      var def = defenders[i];
      var d   = dist(carrier.x, carrier.y, def.x, def.y);
      if (d < AI_BLOCK_RANGE) {
        // Closing speed check (relative velocity toward carrier)
        var dvx = safe(def.vx) - safe(carrier.vx);
        var dvy = safe(def.vy) - safe(carrier.vy);
        var dx  = safe(carrier.x) - safe(def.x);
        var dy  = safe(carrier.y) - safe(def.y);
        var closing = dvx * dx + dvy * dy; // positive = moving toward carrier
        if (closing > 1 && Math.random() < DISLODGE_CHANCE) {
          // Stone dislodged — becomes loose at carrier position
          st.message = 'STONE DISLODGED!';
          // We can't directly free the stone, but we can have the carrier
          // immediately throw it to themselves with zero velocity (loose stone)
          // Best we can do: engine doesn't expose "drop stone". Simulate by
          // throwing at very low power in a random direction.
          this._engine.throwStone(
            eSt.stone.heldBy,
            (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.1,
            0.05
          );
          this._shotHolder  = null;
          this._shotElapsed = 0;
          break;
        }
      }
    }
  };

  // ─── Event handling ───────────────────────────────────────────────────────

  Game.prototype._handleEvent = function (ev) {
    if (!ev || !ev.type) return;

    var st = this.state;

    if (ev.type === 'score') {
      var pts = safe(ev.points, 1);
      if (st.highFrame) pts *= 2;

      if (ev.team === 'home') {
        st.score.home += pts;
      } else if (ev.team === 'away') {
        st.score.away += pts;
      }

      var methodLabel = '';
      if (ev.method === 'carry')    methodLabel = 'CARRY (+' + pts + ')';
      else if (ev.method === 'long') methodLabel = 'LONG LAUNCH (+' + pts + ')';
      else                           methodLabel = 'GOAL (+' + pts + ')';

      st.message = (ev.team === 'home' ? 'HOME' : 'AWAY') + ' — ' + methodLabel;
      st.phase   = 'score';
      this._phaseTimer = 2.5; // 2.5s pause before restart

      // In freespin, any score ends the game
      if (st.phase === 'freespin' || this._wasFreespin) {
        st.winner = ev.team;
        st.phase  = 'over';
        st.message = (ev.team === 'home' ? 'HOME' : 'AWAY') + ' WINS — FREESPIN GOAL!';
      }

      // Switch possession
      st.possession = ev.team === 'home' ? 'away' : 'home';

    } else if (ev.type === 'gate_miss') {
      st.message = 'Gate miss — ' + (ev.team || '') + ' turnover';
      this._triggerTurnover(this._engine.state);

    } else if (ev.type === 'stone_out') {
      st.message = 'Stone out of bounds — turnover';
      this._triggerTurnover(this._engine.state);

    } else if (ev.type === 'skin') {
      // Player hit the skin — informational
      st.message = 'Ouch! Skin contact (Player ' + ev.playerId + ')';
    }
  };

  // ─── Phase transitions ────────────────────────────────────────────────────

  Game.prototype._startFrame = function (frameNum, firstPossession) {
    var st = this.state;
    st.frame      = frameNum;
    st.frameClock = FRAME_DURATION;
    st.highFrame  = frameNum >= HIGH_FRAME_START;
    st.possession = firstPossession;
    st.shotClock  = SHOT_CLOCK_MAX;
    st.message    = 'FRAME ' + frameNum + (st.highFrame ? ' — HIGH FRAME!' : '') + ' — PUSH OFF!';
    st.phase      = 'pushoff';
    this._shotHolder  = null;
    this._shotElapsed = 0;
    this._wasFreespin = false;

    try {
      this._engine.reset('pushoff');
    } catch (e) {
      // Guard if engine not ready
    }
    try {
      this._engine.newPossession(firstPossession);
    } catch (e) {}
  };

  Game.prototype._beginLive = function () {
    var st = this.state;
    st.phase      = 'live';
    st.shotClock  = SHOT_CLOCK_MAX;
    this._shotHolder  = null;
    this._shotElapsed = 0;
    st.message    = 'LIVE — ' + (st.possession === 'home' ? 'HOME' : 'AWAY') + ' POSSESSION';

    try {
      this._engine.newPossession(st.possession);
    } catch (e) {}
  };

  Game.prototype._triggerTurnover = function (eSt) {
    var st        = this.state;
    st.possession = st.possession === 'home' ? 'away' : 'home';
    st.phase      = 'turnover';
    this._phaseTimer  = 2.0;
    this._shotHolder  = null;
    this._shotElapsed = 0;
    st.shotClock  = SHOT_CLOCK_MAX;
  };

  Game.prototype._endFrame = function () {
    var st        = this.state;
    st.frameClock = 0;

    if (st.frame >= NUM_FRAMES) {
      // End of final frame
      if (st.score.home === st.score.away) {
        // Tied — freespin
        st.phase         = 'freespin';
        this._wasFreespin = true;
        st.message       = 'TIED! FREESPIN — Sudden death!';
        try { this._engine.reset('pushoff'); } catch (e) {}
        try { this._engine.newPossession('home'); } catch (e) {}
      } else {
        st.winner  = st.score.home > st.score.away ? 'home' : 'away';
        st.phase   = 'over';
        st.message = (st.winner === 'home' ? 'HOME' : 'AWAY') + ' WINS!';
      }
      return;
    }

    // Advance frame — alternate first-possession
    var nextFrame = st.frame + 1;
    var nextPoss  = this._frameStartPossession === 'home' ? 'away' : 'home';
    this._frameStartPossession = nextPoss;
    this._startFrame(nextFrame, nextPoss);
  };

  // ─── AI System ────────────────────────────────────────────────────────────
  //
  // AI controls:
  //   home: anchor (H2), launch (H3), rover (H4)   [P1 is human]
  //   away: runner (A1), anchor (A2), launch (A3), rover (A4)
  //
  // Each player gets a per-tick behavior call based on role + team.

  Game.prototype._runAI = function (dt, eSt) {
    if (!eSt || !eSt.players) return;
    var self = this;
    eSt.players.forEach(function (p) {
      if (!p || p.id === 'P1') return;
      try {
        self._aiPlayer(p, dt, eSt);
      } catch (e) {
        // Swallow AI errors — never crash the game loop
      }
    });
  };

  Game.prototype._aiPlayer = function (p, dt, eSt) {
    var role = p.role || 'rover';
    var team = p.team || 'away';
    if (role === 'runner')  this._aiRunner(p, dt, eSt, team);
    else if (role === 'anchor') this._aiAnchor(p, dt, eSt, team);
    else if (role === 'launch') this._aiLaunch(p, dt, eSt, team);
    else                        this._aiRover(p, dt, eSt, team);
  };

  // Returns the x-coordinate of the gate this team attacks
  function attackGateX(team) {
    return team === 'home' ? GATE_X_HOME : GATE_X_AWAY;
  }

  // Returns the defending gate x
  function defendGateX(team) {
    return team === 'home' ? GATE_X_AWAY : GATE_X_HOME;
  }

  // Sign for advancing direction (+1 home, -1 away)
  function advanceSign(team) {
    return team === 'home' ? 1 : -1;
  }

  // ── RUNNER AI ─────────────────────────────────────────────────────────────
  // If team has stone: drive toward gate, shoot when clear or close
  // If stone loose/enemy has it: pursue stone or get open

  Game.prototype._aiRunner = function (p, dt, eSt, team) {
    var eng  = this._engine;
    var st   = this.state;
    var stone = eSt.stone;

    if (!stone) return;

    var myGateX  = attackGateX(team);
    var sign     = advanceSign(team);
    var px       = safe(p.x), py = safe(p.y);
    var sx       = safe(stone.x), sy = safe(stone.y);

    if (stone.heldBy === p.id) {
      // I have the stone
      // Home runner: offer to pass to P1 sometimes (cooperative)
      if (team === 'home') {
        var p1 = this._findPlayer(eSt, 'P1');
        if (p1 && Math.random() < AI_PASS_PROB) {
          var p1dx = norm(safe(p1.x) - px, safe(p1.y) - py);
          eng.throwStone(p.id, p1dx.x, p1dx.y, 0.4);
          return;
        }
      }

      // Shoot if close to gate or clear
      var distToGate = Math.abs(px - myGateX);
      if (distToGate < AI_SHOOT_RANGE) {
        var dir = norm(myGateX - px, -py * 0.5); // aim for gate center (y=0)
        eng.throwStone(p.id, dir.x, dir.y, 0.7 + Math.random() * 0.3);
        return;
      }

      // Otherwise drive toward gate using thrumbler
      var driveDir = norm(sign * 1, -py * 0.1);
      this._aiThrumbler(p, driveDir.x * AI_THRUMBLER_POWER, driveDir.y * AI_THRUMBLER_POWER);

    } else if (!stone.heldBy) {
      // Stone is loose — chase it
      var toStone = norm(sx - px, sy - py);
      this._aiThrumbler(p, toStone.x * AI_THRUMBLER_POWER, toStone.y * AI_THRUMBLER_POWER);

      // Pick up if close
      if (dist(px, py, sx, sy) < 15) {
        eng.pickup(p.id);
      }

    } else {
      // Enemy has stone — get into a good position (between enemy and my gate)
      // or rush the carrier if defending
      var carrier = this._findPlayer(eSt, stone.heldBy);
      if (carrier && carrier.team !== team) {
        // Rush carrier as blocker / secondary defender
        var toDef = norm(safe(carrier.x) - px, safe(carrier.y) - py);
        this._aiThrumbler(p, toDef.x * AI_THRUMBLER_POWER * 0.6, toDef.y * AI_THRUMBLER_POWER * 0.6);
      } else {
        // Teammate has it — get open toward gate
        var openDir = norm(sign * 1, -py * 0.1);
        this._aiThrumbler(p, openDir.x * AI_THRUMBLER_POWER * 0.5, openDir.y * AI_THRUMBLER_POWER * 0.5);
      }
    }
  };

  // ── ANCHOR AI ─────────────────────────────────────────────────────────────
  // Defensive: position between the stone and own gate. Attempt blocks.

  Game.prototype._aiAnchor = function (p, dt, eSt, team) {
    var eng   = this._engine;
    var stone = eSt.stone;
    if (!stone) return;

    var myDefGateX = defendGateX(team);
    var px = safe(p.x), py = safe(p.y);
    var sx = safe(stone.x), sy = safe(stone.y);

    // Ideal position: halfway between stone and own defend gate, y near 0
    var targetX = (sx + myDefGateX) * 0.5;
    var targetY = sy * 0.3;

    // If enemy carrier is close enough, attempt a block
    if (stone.heldBy) {
      var carrier = this._findPlayer(eSt, stone.heldBy);
      if (carrier && carrier.team !== team) {
        var d = dist(px, py, safe(carrier.x), safe(carrier.y));
        if (d < AI_BLOCK_RANGE) {
          // Push off toward carrier if we have contact
          if (p.contact) {
            var bd = norm(safe(carrier.x) - px, safe(carrier.y) - py);
            eng.pushOff(p.id, bd.x * AI_BLOCK_SPEED, bd.y * AI_BLOCK_SPEED);
            return;
          }
        }
        // Otherwise rush toward carrier
        targetX = safe(carrier.x);
        targetY = safe(carrier.y);
      }
    }

    // Move toward target
    var toTarget = norm(targetX - px, targetY - py);
    var d2 = dist(px, py, targetX, targetY);
    if (d2 > 10) {
      this._aiThrumbler(p, toTarget.x * AI_THRUMBLER_POWER * 0.8, toTarget.y * AI_THRUMBLER_POWER * 0.8);
    }
  };

  // ── LAUNCH AI ─────────────────────────────────────────────────────────────
  // Tries to get behind own deepline for Long Launch.
  // If not holding stone and own team has it, request a pass by getting open.
  // If holding stone and behind deepline, attempt Long Launch.

  Game.prototype._aiLaunch = function (p, dt, eSt, team) {
    var eng    = this._engine;
    var stone  = eSt.stone;
    if (!stone) return;

    var myGateX = attackGateX(team);
    var sign    = advanceSign(team);
    var px = safe(p.x), py = safe(p.y);

    // Ideal position: behind own deepline, y near 0
    var idealX = sign * (-AI_LONG_LAUNCH_X); // antispinward side of midline (own half)
    var idealY = 5;

    if (stone.heldBy === p.id) {
      // Holding stone: if behind deepline, fire the long launch
      var behindDeepline = (team === 'home') ? (px < -DEEPLINE_X) : (px > DEEPLINE_X);
      if (behindDeepline) {
        var launchDir = norm(myGateX - px, -py * 0.3);
        eng.throwStone(p.id, launchDir.x, launchDir.y, 1.0);
      } else {
        // Move back to deepline first
        var toIdeal = norm(idealX - px, idealY - py);
        this._aiThrumbler(p, toIdeal.x * AI_THRUMBLER_POWER, toIdeal.y * AI_THRUMBLER_POWER);
      }
      return;
    }

    // Stone loose — if close, pick it up
    if (!stone.heldBy) {
      var dstone = dist(px, py, safe(stone.x), safe(stone.y));
      if (dstone < 15) {
        eng.pickup(p.id);
        return;
      }
    }

    // Get to ideal launch position
    var d2 = dist(px, py, idealX, idealY);
    if (d2 > 15) {
      var dir = norm(idealX - px, idealY - py);
      this._aiThrumbler(p, dir.x * AI_THRUMBLER_POWER * 0.7, dir.y * AI_THRUMBLER_POWER * 0.7);
    }

    // Defensive help: if enemy has stone and they're near our gate
    if (stone.heldBy) {
      var carrier = this._findPlayer(eSt, stone.heldBy);
      if (carrier && carrier.team !== team) {
        var myDefX = defendGateX(team);
        if (Math.abs(safe(carrier.x) - myDefX) < 100) {
          // Rush to intercept
          var toDef = norm(safe(carrier.x) - px, safe(carrier.y) - py);
          this._aiThrumbler(p, toDef.x * AI_THRUMBLER_POWER, toDef.y * AI_THRUMBLER_POWER);
        }
      }
    }
  };

  // ── ROVER AI ──────────────────────────────────────────────────────────────
  // Dynamic — attack if team has stone, defend if not.

  Game.prototype._aiRover = function (p, dt, eSt, team) {
    var eng   = this._engine;
    var stone = eSt.stone;
    if (!stone) return;

    var myGateX = attackGateX(team);
    var px = safe(p.x), py = safe(p.y);
    var sign = advanceSign(team);

    if (stone.heldBy === p.id) {
      // I have the stone — drive and shoot
      var distToGate = Math.abs(px - myGateX);
      if (distToGate < AI_SHOOT_RANGE) {
        var dir = norm(myGateX - px, -py * 0.4);
        eng.throwStone(p.id, dir.x, dir.y, 0.8);
      } else {
        var driveDir = norm(sign * 1, -py * 0.08);
        this._aiThrumbler(p, driveDir.x * AI_THRUMBLER_POWER, driveDir.y * AI_THRUMBLER_POWER);
      }
    } else if (!stone.heldBy) {
      // Loose stone — chase
      var toStone = norm(safe(stone.x) - px, safe(stone.y) - py);
      this._aiThrumbler(p, toStone.x * AI_THRUMBLER_POWER, toStone.y * AI_THRUMBLER_POWER);
      if (dist(px, py, safe(stone.x), safe(stone.y)) < 15) {
        eng.pickup(p.id);
      }
    } else {
      // Someone else holds stone
      var carrier = this._findPlayer(eSt, stone.heldBy);
      if (carrier) {
        if (carrier.team === team) {
          // Teammate — get open toward gate
          var openX = px + sign * 30;
          var openDir = norm(openX - px, -py * 0.1);
          this._aiThrumbler(p, openDir.x * AI_THRUMBLER_POWER * 0.5, openDir.y * AI_THRUMBLER_POWER * 0.5);
        } else {
          // Enemy — defend (get between carrier and own gate)
          var defGateX = defendGateX(team);
          var midX = (safe(carrier.x) + defGateX) * 0.5;
          var midY = safe(carrier.y) * 0.4;
          var toMid = norm(midX - px, midY - py);
          this._aiThrumbler(p, toMid.x * AI_THRUMBLER_POWER * 0.7, toMid.y * AI_THRUMBLER_POWER * 0.7);
        }
      }
    }
  };

  // ── AI helpers ────────────────────────────────────────────────────────────

  // Apply thrumbler delta-v, guarding dvBudget
  Game.prototype._aiThrumbler = function (p, dvx, dvy) {
    if (!p || !p.id) return;
    if (isNaN(dvx) || isNaN(dvy)) return;
    if (safe(p.dvBudget, 1) <= 0) return;
    try {
      this._engine.applyThrumbler(p.id, dvx, dvy);
    } catch (e) {}
  };

  // ── Utility ───────────────────────────────────────────────────────────────

  Game.prototype._findPlayer = function (eSt, id) {
    if (!eSt || !eSt.players) return null;
    for (var i = 0; i < eSt.players.length; i++) {
      if (eSt.players[i] && eSt.players[i].id === id) return eSt.players[i];
    }
    return null;
  };

  // ─── Export ───────────────────────────────────────────────────────────────

  AX.Game = Game;

}(window.AX = window.AX || {}));

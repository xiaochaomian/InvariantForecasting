/*
 * DIVE high-score bot.
 *
 * Usage:
 *   1. Open https://alexfink.github.io/dive/
 *   2. Open DevTools Console.
 *   3. Paste this whole file and press Enter.
 *   4. Run one of:
 *      DiveBot.start()
 *      DiveBot.start({ preset: "strong" })
 *      DiveBot.start({ preset: "fast", delay: 20 })
 *
 * Controls:
 *   DiveBot.stop()
 *   DiveBot.step()
 *   DiveBot.suggest()
 *   DiveBot.status()
 *
 * The bot does not edit the score. It calls game.move(direction), so the page
 * still performs the real random spawns and scoring.
 */
(function () {
  "use strict";

  var DIR_NAMES = ["up", "right", "down", "left"];
  var VECTORS = [
    { x: 0, y: -1 },
    { x: 1, y: 0 },
    { x: 0, y: 1 },
    { x: -1, y: 0 }
  ];

  var PRESETS = {
    fast: { depth: 1, samples: 8, delay: 15 },
    normal: { depth: 2, samples: 12, delay: 35 },
    strong: { depth: 3, samples: 14, delay: 60 }
  };

  var config = extend({}, PRESETS.normal);
  var timer = null;
  var running = false;
  var lastDecision = null;

  function extend(target, source) {
    for (var key in source) {
      if (Object.prototype.hasOwnProperty.call(source, key)) {
        target[key] = source[key];
      }
    }
    return target;
  }

  function getGame() {
    if (!window.game || !window.game.grid) {
      throw new Error("DIVE game object not found. Open https://alexfink.github.io/dive/ first.");
    }
    return window.game;
  }

  function captureState() {
    var game = getGame();
    var board = [];

    for (var x = 0; x < game.size; x++) {
      board[x] = [];
      for (var y = 0; y < game.size; y++) {
        var tile = game.grid.cells[x][y];
        board[x][y] = tile ? tile.value : 0;
      }
    }

    return {
      size: game.size,
      board: board,
      score: game.score || 0,
      gameMode: game.gameMode || 0,
      tileTypes: (game.tileTypes || [2, 3, 5, 7]).slice(),
      tilesSeen: (game.tilesSeen || []).slice()
    };
  }

  function cloneState(state) {
    var board = [];
    for (var x = 0; x < state.size; x++) {
      board[x] = state.board[x].slice();
    }

    return {
      size: state.size,
      board: board,
      score: state.score,
      gameMode: state.gameMode,
      tileTypes: state.tileTypes.slice(),
      tilesSeen: state.tilesSeen.slice()
    };
  }

  function boardToCells(board, size) {
    var cells = [];
    for (var x = 0; x < size; x++) {
      cells[x] = [];
      for (var y = 0; y < size; y++) {
        cells[x][y] = board[x][y] ? { value: board[x][y], merged: false } : null;
      }
    }
    return cells;
  }

  function cellsToBoard(cells, size) {
    var board = [];
    for (var x = 0; x < size; x++) {
      board[x] = [];
      for (var y = 0; y < size; y++) {
        board[x][y] = cells[x][y] ? cells[x][y].value : 0;
      }
    }
    return board;
  }

  function within(size, cell) {
    return cell.x >= 0 && cell.x < size && cell.y >= 0 && cell.y < size;
  }

  function traversals(size, vector) {
    var xs = [];
    var ys = [];
    for (var i = 0; i < size; i++) {
      xs.push(i);
      ys.push(i);
    }
    if (vector.x === 1) xs.reverse();
    if (vector.y === 1) ys.reverse();
    return { x: xs, y: ys };
  }

  function farthestPosition(cells, size, cell, vector) {
    var previous;
    do {
      previous = cell;
      cell = { x: previous.x + vector.x, y: previous.y + vector.y };
    } while (within(size, cell) && !cells[cell.x][cell.y]);

    return {
      farthest: previous,
      next: cell
    };
  }

  function divMergeValue(a, b) {
    if (!a || !b) return 0;
    if (a % b === 0 || b % a === 0) return a + b;
    return 0;
  }

  function extractPrimesFrom(state, n, i) {
    if (i >= state.tileTypes.length) return n;

    var min = extractPrimesFrom(state, n, i + 1);
    while (n % state.tileTypes[i] === 0) {
      n /= state.tileTypes[i];
      var candidate = extractPrimesFrom(state, n, i + 1);
      if (candidate < min) min = candidate;
    }
    return min;
  }

  function extractNewPrimes(state, n) {
    n = extractPrimesFrom(state, n, 0);
    return n > 1 ? [n] : [];
  }

  function dedupeSorted(numbers) {
    if (numbers.length < 2) return numbers;
    numbers.sort(function (a, b) { return a - b; });
    var out = [numbers[0]];
    for (var i = 1; i < numbers.length; i++) {
      if (numbers[i] !== out[out.length - 1]) out.push(numbers[i]);
    }
    return out;
  }

  function simulateMoveNoSpawn(inputState, direction) {
    var state = cloneState(inputState);
    var size = state.size;
    var cells = boardToCells(state.board, size);
    var vector = VECTORS[direction];
    var order = traversals(size, vector);
    var moved = false;
    var newPrimes = [];
    var oldScore = state.score;

    for (var ix = 0; ix < order.x.length; ix++) {
      for (var iy = 0; iy < order.y.length; iy++) {
        var x = order.x[ix];
        var y = order.y[iy];
        var tile = cells[x][y];
        if (!tile) continue;

        var positions = farthestPosition(cells, size, { x: x, y: y }, vector);
        var next = within(size, positions.next) ? cells[positions.next.x][positions.next.y] : null;
        var mergedValue = next && !next.merged ? divMergeValue(next.value, tile.value) : 0;

        if (mergedValue) {
          cells[positions.next.x][positions.next.y] = { value: mergedValue, merged: true };
          cells[x][y] = null;
          state.score += Math.min(next.value, tile.value);
          moved = moved || positions.next.x !== x || positions.next.y !== y;

          if (state.gameMode & 1) {
            newPrimes = newPrimes.concat(extractNewPrimes(state, mergedValue));
          }
        } else {
          if (positions.farthest.x !== x || positions.farthest.y !== y) {
            cells[positions.farthest.x][positions.farthest.y] = tile;
            cells[x][y] = null;
            moved = true;
          }
        }
      }
    }

    state.board = cellsToBoard(cells, size);

    if (state.gameMode & 1) {
      newPrimes = dedupeSorted(newPrimes);
      state.tileTypes = state.tileTypes.concat(newPrimes);
    }

    if (moved && (state.gameMode & 1) && newPrimes.length) {
      if ((state.gameMode & 3) === 1) {
        state.score += sum(newPrimes);
      }
      state.tilesSeen = state.tilesSeen.concat(newPrimes);
    }

    if (moved && (state.gameMode & 3) === 3) {
      eliminateAbsentFactors(state);
    }

    return {
      state: state,
      moved: moved,
      gained: state.score - oldScore
    };
  }

  function eliminateAbsentFactors(state) {
    var eliminated = [];

    for (var i = 0; i < state.tileTypes.length; i++) {
      var seed = state.tileTypes[i];
      var present = false;

      for (var x = 0; x < state.size && !present; x++) {
        for (var y = 0; y < state.size; y++) {
          var value = state.board[x][y];
          if (value && value % seed === 0) {
            present = true;
            break;
          }
        }
      }

      if (!present) eliminated.push(i);
    }

    if (eliminated.length) {
      var eliminatedValues = eliminated.map(function (idx) { return state.tileTypes[idx]; });
      state.score += sum(eliminatedValues);

      for (var j = eliminated.length - 1; j >= 0; j--) {
        state.tileTypes.splice(eliminated[j], 1);
      }
    }
  }

  function sum(values) {
    return values.reduce(function (acc, value) { return acc + value; }, 0);
  }

  function availableCells(state) {
    var cells = [];
    for (var x = 0; x < state.size; x++) {
      for (var y = 0; y < state.size; y++) {
        if (!state.board[x][y]) cells.push({ x: x, y: y });
      }
    }
    return cells;
  }

  function hasMoves(state) {
    if (availableCells(state).length) return true;
    for (var direction = 0; direction < 4; direction++) {
      if (simulateMoveNoSpawn(state, direction).moved) return true;
    }
    return false;
  }

  function boardHash(state) {
    var hash = 2166136261;
    for (var y = 0; y < state.size; y++) {
      for (var x = 0; x < state.size; x++) {
        hash ^= state.board[x][y] + 31 * x + 131 * y;
        hash = Math.imul(hash, 16777619);
      }
    }
    hash ^= state.score;
    hash = Math.imul(hash, 16777619);
    return hash >>> 0;
  }

  function stateKey(state) {
    var parts = [state.score, state.gameMode, state.tileTypes.join(".")];
    for (var y = 0; y < state.size; y++) {
      for (var x = 0; x < state.size; x++) {
        parts.push(state.board[x][y]);
      }
    }
    return parts.join("|");
  }

  function makeRng(seed) {
    var value = seed >>> 0;
    return function () {
      value = Math.imul(1664525, value) + 1013904223;
      return (value >>> 0) / 4294967296;
    };
  }

  function spawnChildren(state, samples, seed) {
    var empty = availableCells(state);
    if (!empty.length) return [{ state: state, weight: 1 }];

    var types = state.tileTypes.length ? state.tileTypes : [2];
    var total = empty.length * types.length;
    var children = [];

    function childAt(cellIndex, typeIndex, weight) {
      var child = cloneState(state);
      var cell = empty[cellIndex];
      child.board[cell.x][cell.y] = types[typeIndex];
      children.push({ state: child, weight: weight });
    }

    if (total <= samples) {
      for (var i = 0; i < empty.length; i++) {
        for (var j = 0; j < types.length; j++) {
          childAt(i, j, 1 / total);
        }
      }
      return children;
    }

    var rng = makeRng(seed);
    for (var k = 0; k < samples; k++) {
      childAt(Math.floor(rng() * empty.length), Math.floor(rng() * types.length), 1 / samples);
    }
    return children;
  }

  function mergePotential(state) {
    var total = 0;
    var seen = Object.create(null);

    for (var x = 0; x < state.size; x++) {
      for (var y = 0; y < state.size; y++) {
        var a = state.board[x][y];
        if (!a) continue;

        for (var direction = 0; direction < 4; direction++) {
          var nx = x + VECTORS[direction].x;
          var ny = y + VECTORS[direction].y;
          if (!within(state.size, { x: nx, y: ny })) continue;

          var key = x + "," + y + "|" + nx + "," + ny;
          var reverse = nx + "," + ny + "|" + x + "," + y;
          if (seen[reverse]) continue;
          seen[key] = true;

          var b = state.board[nx][ny];
          if (!b || !divMergeValue(a, b)) continue;
          total += Math.min(a, b) * 4 + Math.log(a + b + 1) * 35;
        }
      }
    }

    return total;
  }

  function islandPenalty(state) {
    var penalty = 0;

    for (var x = 0; x < state.size; x++) {
      for (var y = 0; y < state.size; y++) {
        var value = state.board[x][y];
        if (!value) continue;

        var friendly = 0;
        for (var direction = 0; direction < 4; direction++) {
          var nx = x + VECTORS[direction].x;
          var ny = y + VECTORS[direction].y;
          if (!within(state.size, { x: nx, y: ny })) continue;

          var neighbor = state.board[nx][ny];
          if (!neighbor || divMergeValue(value, neighbor)) friendly++;
        }
        if (!friendly) penalty += Math.log(value + 1) * 90;
      }
    }

    return penalty;
  }

  function cornerBonus(state) {
    var max = 0;
    var maxX = 0;
    var maxY = 0;
    for (var x = 0; x < state.size; x++) {
      for (var y = 0; y < state.size; y++) {
        if (state.board[x][y] > max) {
          max = state.board[x][y];
          maxX = x;
          maxY = y;
        }
      }
    }
    if (!max) return 0;

    var inCorner = (maxX === 0 || maxX === state.size - 1) && (maxY === 0 || maxY === state.size - 1);
    return inCorner ? Math.log(max + 1) * 140 : 0;
  }

  function evaluate(state) {
    var empty = availableCells(state).length;
    var mobility = 0;
    var maxTile = 0;
    var boardSum = 0;

    for (var direction = 0; direction < 4; direction++) {
      if (simulateMoveNoSpawn(state, direction).moved) mobility++;
    }

    for (var x = 0; x < state.size; x++) {
      for (var y = 0; y < state.size; y++) {
        var value = state.board[x][y];
        maxTile = Math.max(maxTile, value);
        boardSum += value;
      }
    }

    if (!mobility && !empty) return state.score - 1000000;

    return state.score * 1.4 +
      empty * empty * 135 +
      mobility * 320 +
      mergePotential(state) +
      cornerBonus(state) +
      Math.log(maxTile + 1) * 120 +
      Math.log(boardSum + 1) * 80 -
      islandPenalty(state);
  }

  function search(state, depth, samples, seed, cache) {
    if (depth <= 0) return evaluate(state);

    var key = depth + ":" + samples + ":" + stateKey(state);
    if (cache[key] != null) return cache[key];

    var best = -Infinity;
    for (var direction = 0; direction < 4; direction++) {
      var value = moveValue(state, direction, depth, samples, seed + direction * 8191, cache);
      if (value > best) best = value;
    }

    cache[key] = best === -Infinity ? evaluate(state) - 500000 : best;
    return cache[key];
  }

  function moveValue(state, direction, depth, samples, seed, cache) {
    var moved = simulateMoveNoSpawn(state, direction);
    if (!moved.moved) return -Infinity;

    var children = spawnChildren(moved.state, samples, seed ^ boardHash(moved.state));
    var expected = 0;

    for (var i = 0; i < children.length; i++) {
      var child = children[i].state;
      var childValue = hasMoves(child)
        ? search(child, depth - 1, Math.max(4, Math.floor(samples * 0.65)), seed + i * 104729, cache)
        : evaluate(child) - 500000;

      expected += children[i].weight * childValue;
    }

    return expected + moved.gained * 0.35;
  }

  function chooseMove(options) {
    var state = captureState();
    var depth = options.depth;
    var samples = options.samples;
    var seed = boardHash(state) ^ Date.now();
    var cache = Object.create(null);
    var scores = [];
    var bestDirection = -1;
    var bestScore = -Infinity;

    for (var direction = 0; direction < 4; direction++) {
      var score = moveValue(state, direction, depth, samples, seed + direction * 65537, cache);
      scores[direction] = score;
      if (score > bestScore) {
        bestScore = score;
        bestDirection = direction;
      }
    }

    return {
      direction: bestDirection,
      name: bestDirection >= 0 ? DIR_NAMES[bestDirection] : "none",
      score: bestScore,
      scores: scores.map(function (score, direction) {
        return {
          direction: direction,
          name: DIR_NAMES[direction],
          value: score
        };
      })
    };
  }

  function applyOptions(options) {
    options = options || {};
    if (options.preset) {
      if (!PRESETS[options.preset]) {
        throw new Error("Unknown preset: " + options.preset + ". Use fast, normal, or strong.");
      }
      config = extend({}, PRESETS[options.preset]);
    }

    ["depth", "samples", "delay"].forEach(function (key) {
      if (options[key] != null) config[key] = +options[key];
    });

    config.depth = Math.max(1, Math.floor(config.depth));
    config.samples = Math.max(1, Math.floor(config.samples));
    config.delay = Math.max(0, Math.floor(config.delay));
  }

  function schedule() {
    if (!running) return;
    timer = window.setTimeout(function () {
      try {
        step();
      } catch (error) {
        stop();
        throw error;
      }
      schedule();
    }, config.delay);
  }

  function step() {
    var game = getGame();
    if (game.over || game.won) {
      stop();
      console.log("[DiveBot] stopped: game over");
      return null;
    }

    var decision = chooseMove(config);
    lastDecision = decision;
    if (decision.direction < 0) {
      stop();
      console.log("[DiveBot] stopped: no legal moves");
      return decision;
    }

    game.move(decision.direction);
    return decision;
  }

  function start(options) {
    applyOptions(options);
    if (running) stop();

    running = true;
    console.log("[DiveBot] started", extend({}, config));
    schedule();
  }

  function configure(options) {
    applyOptions(options);
    console.log("[DiveBot] configured", extend({}, config));
    return extend({}, config);
  }

  function stop() {
    running = false;
    if (timer != null) {
      window.clearTimeout(timer);
      timer = null;
    }
    console.log("[DiveBot] stopped");
  }

  function suggest(options) {
    var merged = extend({}, config);
    if (options) extend(merged, options);
    var decision = chooseMove(merged);
    console.table(decision.scores);
    console.log("[DiveBot] suggested move:", decision.name);
    return decision;
  }

  function status() {
    var game = getGame();
    var state = captureState();
    return {
      running: running,
      config: extend({}, config),
      score: game.score,
      best: game.scoreManager && game.scoreManager.get ? game.scoreManager.get() : undefined,
      tileTypes: state.tileTypes.slice(),
      emptyCells: availableCells(state).length,
      lastDecision: lastDecision
    };
  }

  window.DiveBot = {
    start: start,
    stop: stop,
    step: step,
    suggest: suggest,
    status: status,
    configure: configure,
    presets: PRESETS
  };

  console.log("[DiveBot] ready. Try DiveBot.start({ preset: \"strong\" })");
}());

"""Procedural, chunk-based level generator.

A level is built left to right from parameterised chunks (flat, step, stairs,
gap, spikes, platform hop). `difficulty` in [0, 1] widens hazards and unlocks
the stage-2 chunks. Gap widths and height changes are capped by what the jump
physics can clear, so every level is beatable (checked by tests/test_levels.py
with procgen_rl/solver.py).

The generator uses `random.Random`, whose output for a given seed is stable
across Python versions and platforms, so a seed means the same level on Colab
and on the laptop.
"""

import random

import numpy as np

from . import config as C

H = C.LEVEL_HEIGHT
MIN_H = 2                   # ground surface height range, in tiles from bottom
MAX_H = 9
START_H = 3
START_FLAT = 8
END_FLAT = 10
TARGET_WIDTH = 140          # columns of chunks before the finish area
MAX_WIDTH = TARGET_WIDTH + 80

# Chunks and the difficulty at which they start appearing.
STAGE1_CHUNKS = ("flat", "step", "stairs", "gap")
STAGE2_CHUNKS = ("spikes", "platforms")
SPIKES_MIN_DIFFICULTY = 0.25
PLATFORMS_MIN_DIFFICULTY = 0.35


class Level:
    def __init__(self, grid, seed, difficulty, spawn, goal_x, chunks):
        self.grid = grid                       # uint8 [height, width]
        self.rows = [bytes(r) for r in grid]   # fast per-tile lookup
        self.height, self.width = grid.shape
        self.seed = seed
        self.difficulty = difficulty
        self.spawn = spawn                     # (x, y) of the player box
        self.goal_x = goal_x                   # reaching this x wins
        self.chunks = chunks                   # [(name, start_col, end_col)]

    def surface_row(self, col):
        """Topmost solid row in a column, or None for a pit."""
        column = self.grid[:, col]
        idx = np.flatnonzero(column == C.SOLID)
        return int(idx[0]) if idx.size else None


def max_gap(dh):
    """Widest safe gap (tiles) for a jump that changes surface height by dh."""
    if dh >= 2:
        return 2
    if dh == 1:
        return 3
    return 4


class _Rng:
    def __init__(self, seed):
        self._r = random.Random(seed)

    def random(self):
        return self._r.random()

    def randint(self, a, b):
        return a + int(self._r.random() * (b - a + 1))

    def choice(self, items, weights):
        total = sum(weights)
        t = self._r.random() * total
        for item, w in zip(items, weights):
            t -= w
            if t < 0:
                return item
        return items[-1]


class _Builder:
    def __init__(self, rng, difficulty):
        self.rng = rng
        self.d = difficulty
        self.grid = np.zeros((H, MAX_WIDTH), dtype=np.uint8)
        self.x = 0
        self.h = START_H
        self.chunks = []

    def ground(self, n, h=None):
        h = self.h if h is None else h
        self.grid[H - h:, self.x:self.x + n] = C.SOLID
        self.x += n
        self.h = h

    def pit(self, n):
        self.x += n

    def block(self, col, surface_h, width):
        self.grid[H - surface_h, col:col + width] = C.SOLID

    def spike(self, col, surface_h):
        self.grid[H - surface_h - 1, col] = C.SPIKE

    def gap_limit(self, dh):
        # difficulty 0 -> 1 tile gaps, difficulty 1 -> physics limit
        return max(1, min(max_gap(dh), 1 + round(3 * self.d)))

    def clamp_h(self, h):
        return max(MIN_H, min(MAX_H, h))

    # ------------------------------------------------------------ chunks
    def c_flat(self):
        self.ground(self.rng.randint(3, 8 - round(4 * self.d)))

    def c_step(self):
        mag = self.rng.randint(1, 1 + round(2 * self.d))
        sign = 1 if self.rng.random() < 0.5 else -1
        new_h = self.clamp_h(self.h + sign * mag)
        if new_h == self.h:
            new_h = self.clamp_h(self.h - sign * mag)
        self.ground(self.rng.randint(3, 6), new_h)

    def c_stairs(self):
        k = self.rng.randint(2, 4)
        sign = 1 if self.rng.random() < 0.5 else -1
        if not MIN_H <= self.h + sign * k <= MAX_H:
            sign = -sign
        for _ in range(k):
            self.ground(self.rng.randint(1, 2), self.clamp_h(self.h + sign))
        self.ground(2)

    def c_gap(self):
        self.ground(2)
        up = 1 if self.d > 0.5 else 0
        dh = self.rng.randint(-2, up)
        new_h = self.clamp_h(self.h + dh)
        dh = new_h - self.h
        self.pit(self.rng.randint(1, self.gap_limit(dh)))
        self.ground(self.rng.randint(2, 5), new_h)

    def c_spikes(self):
        self.ground(2)
        w = self.rng.randint(1, min(3, 1 + round(2 * self.d)))
        for i in range(w):
            self.spike(self.x + i, self.h)
        self.ground(w)
        self.ground(self.rng.randint(2, 4))

    def c_platforms(self):
        self.ground(2)
        k = self.rng.randint(1, 1 + round(2 * self.d))
        min_w = 1 if self.d > 0.8 else 2
        h = self.h
        for _ in range(k):
            ph = max(MIN_H + 1, min(MAX_H + 1, h + self.rng.randint(-2, 1)))
            self.pit(self.rng.randint(1, self.gap_limit(ph - h)))
            w = self.rng.randint(min_w, 3)
            self.block(self.x, ph, w)
            self.x += w
            h = ph
        new_h = self.clamp_h(h + self.rng.randint(-2, 1))
        self.pit(self.rng.randint(1, self.gap_limit(new_h - h)))
        self.ground(self.rng.randint(3, 5), new_h)

    def weights(self):
        d = self.d
        w = {
            "flat": 3.0 - 2.0 * d,
            "step": 2.0,
            "stairs": 1.0 if d > 0.2 else 0.0,
            "gap": 1.0 + 2.0 * d,
            "spikes": 2.0 * d if d >= SPIKES_MIN_DIFFICULTY else 0.0,
            "platforms": 2.0 * d if d >= PLATFORMS_MIN_DIFFICULTY else 0.0,
        }
        names = [k for k, v in w.items() if v > 0]
        return names, [w[k] for k in names]


def generate_level(seed, difficulty=0.5):
    difficulty = float(max(0.0, min(1.0, difficulty)))
    rng = _Rng(seed)
    b = _Builder(rng, difficulty)

    b.ground(START_FLAT, START_H)
    b.chunks.append(("start", 0, b.x))
    names, weights = b.weights()
    while b.x < START_FLAT + TARGET_WIDTH:
        name = rng.choice(names, weights)
        start = b.x
        getattr(b, "c_" + name)()
        b.chunks.append((name, start, b.x))
    end_start = b.x
    b.ground(END_FLAT)
    b.chunks.append(("finish", end_start, b.x))

    grid = b.grid[:, :b.x].copy()
    spawn = (2.0, float(H - START_H) - C.PLAYER_H)
    goal_x = float(end_start + 3)
    return Level(grid, seed, difficulty, spawn, goal_x, b.chunks)

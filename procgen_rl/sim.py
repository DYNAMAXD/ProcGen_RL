"""Observation building and the per-character runner.

`Runner` is the single place where a character is advanced through a level.
The Gymnasium env, the evaluator and every ghost in the demo use it, so an
agent behaves exactly the same in training and in the demo.
"""

from math import floor

import numpy as np

from . import config as C
from .physics import ALIVE, DIED, GOAL, PlayerState, step_player

RUNNING = "running"
DEAD = "dead"
FINISHED = "finished"

_PAD_X0 = C.VIEW_BEHIND + 2
_PAD_X1 = C.VIEW_AHEAD + 2
_PAD_Y0 = C.VIEW_UP + 3
_PAD_Y1 = C.VIEW_DOWN + 3


class ObservationBuilder:
    """Precomputes padded one-hot tile layers so each observation is a slice."""

    def __init__(self, level):
        self.level = level
        h, w = level.grid.shape
        layers = np.zeros((C.OBS_CHANNELS, h + _PAD_Y0 + _PAD_Y1, w + _PAD_X0 + _PAD_X1),
                          dtype=np.float32)
        inner = (slice(_PAD_Y0, _PAD_Y0 + h), slice(_PAD_X0, _PAD_X0 + w))
        layers[0][inner] = level.grid == C.SOLID
        layers[1][inner] = level.grid == C.SPIKE
        layers[0, :, :_PAD_X0] = 1.0           # left wall
        layers[0, :, _PAD_X0 + w:] = 1.0       # right wall
        goal_col = _PAD_X0 + int(level.goal_x)
        layers[2, :, goal_col:] = 1.0          # goal region
        self.layers = layers
        self._row_lo = -(C.VIEW_UP + 2)
        self._row_hi = h + _PAD_Y1 - C.VIEW_H

    def build(self, s):
        col = floor(s.cx)
        row = floor(s.cy)
        r = min(max(row - C.VIEW_UP, self._row_lo), self._row_hi) + _PAD_Y0
        c = col - C.VIEW_BEHIND + _PAD_X0
        window = self.layers[:, r:r + C.VIEW_H, c:c + C.VIEW_W]
        scalars = np.array((
            s.vx / C.RUN_SPEED,
            s.vy / C.MAX_FALL,
            1.0 if s.on_ground else 0.0,
            1.0 if s.coyote > 0 else 0.0,
            1.0 if s.jumping else 0.0,
            s.cx - col,
            s.cy - row,
        ), dtype=np.float32)
        return np.concatenate((window.ravel(), scalars))


class Runner:
    """One character (agent, ghost or human) playing one level."""

    def __init__(self, level):
        self.level = level
        self.obs_builder = ObservationBuilder(level)
        self.reset()

    def reset(self):
        self.state = PlayerState(*self.level.spawn)
        self.status = RUNNING
        self.frames = 0
        self.max_x = self.state.x
        self.death_cause = None

    @property
    def progress(self):
        """Fraction of the course covered, by furthest x reached."""
        span = self.level.goal_x - self.level.spawn[0]
        return max(0.0, min(1.0, (self.max_x + C.PLAYER_W - self.level.spawn[0]) / span))

    def observe(self):
        return self.obs_builder.build(self.state)

    def frame(self, move, jump):
        """Advance one physics frame. Returns the physics event."""
        if self.status != RUNNING:
            return ALIVE
        ev = step_player(self.state, self.level, move, jump)
        self.frames += 1
        if self.state.x > self.max_x:
            self.max_x = self.state.x
        if ev == DIED:
            self.status = DEAD
            self.death_cause = "fell" if self.state.y > self.level.height else "spike"
        elif ev == GOAL:
            self.status = FINISHED
            self.max_x = max(self.max_x, self.level.goal_x - C.PLAYER_W)
        return ev

    def act(self, action_index):
        """Apply one agent action for FRAME_SKIP frames (stops early on death/goal)."""
        move, jump = C.ACTIONS[action_index]
        ev = ALIVE
        for _ in range(C.FRAME_SKIP):
            ev = self.frame(move, jump)
            if ev != ALIVE:
                break
        return ev

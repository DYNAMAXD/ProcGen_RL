"""Deterministic tile-based platformer physics.

One call to `step_player` advances one physics frame (1/60 s). Coordinates are
in tiles with y growing downward; (x, y) is the top-left of the player box.
Plain Python floats are used on purpose: they are fast for scalar code and give
identical results on every machine, which keeps ghosts reproducible.
"""

from math import floor

from . import config as C

# step_player return values
ALIVE = 0
DIED = 1
GOAL = 2

_EPS = 1e-6


class PlayerState:
    __slots__ = ("x", "y", "vx", "vy", "on_ground", "coyote", "jumping", "facing")

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.coyote = 0
        self.jumping = False
        self.facing = 1

    def copy(self):
        s = PlayerState(self.x, self.y)
        s.vx, s.vy = self.vx, self.vy
        s.on_ground, s.coyote = self.on_ground, self.coyote
        s.jumping, s.facing = self.jumping, self.facing
        return s

    @property
    def cx(self):
        return self.x + C.PLAYER_W * 0.5

    @property
    def cy(self):
        return self.y + C.PLAYER_H * 0.5


def is_solid(level, col, row):
    """Left/right outside the level are walls; above and below are open."""
    if col < 0 or col >= level.width:
        return True
    if row < 0 or row >= level.height:
        return False
    return level.rows[row][col] == C.SOLID


def _solid_in_rows(level, col, r0, r1):
    for r in range(r0, r1 + 1):
        if is_solid(level, col, r):
            return True
    return False


def _solid_in_cols(level, row, c0, c1):
    for c in range(c0, c1 + 1):
        if is_solid(level, c, row):
            return True
    return False


def _touches_spike(level, s):
    m = C.SPIKE_MARGIN
    left, right = s.x + m, s.x + C.PLAYER_W - m
    top, bottom = s.y + m, s.y + C.PLAYER_H - m
    for r in range(max(0, floor(top)), min(level.height - 1, floor(bottom)) + 1):
        row = level.rows[r]
        for c in range(max(0, floor(left)), min(level.width - 1, floor(right)) + 1):
            # spike hitbox is the lower half of the tile
            if row[c] == C.SPIKE and bottom > r + 0.5:
                return True
    return False


def step_player(s, level, move, jump):
    """Advance `s` by one frame. Returns ALIVE, DIED or GOAL."""
    # horizontal velocity
    if move != 0:
        s.facing = move
        accel = C.GROUND_ACCEL if s.on_ground else C.AIR_ACCEL
        target = move * C.RUN_SPEED
        if s.vx < target:
            s.vx = min(target, s.vx + accel)
        elif s.vx > target:
            s.vx = max(target, s.vx - accel)
    else:
        fr = C.GROUND_FRICTION if s.on_ground else C.AIR_FRICTION
        if s.vx > 0.0:
            s.vx = max(0.0, s.vx - fr)
        elif s.vx < 0.0:
            s.vx = min(0.0, s.vx + fr)

    # jumping (holding jump re-jumps on landing; releasing early cuts height)
    if jump and s.coyote > 0:
        s.vy = -C.JUMP_VELOCITY
        s.coyote = 0
        s.jumping = True
        s.on_ground = False
    if s.jumping and (not jump) and s.vy < 0.0:
        s.vy *= C.JUMP_CUT
        s.jumping = False
    if s.vy >= 0.0:
        s.jumping = False

    s.vy = min(C.MAX_FALL, s.vy + C.GRAVITY)

    # move x, resolve against walls
    s.x += s.vx
    r0, r1 = floor(s.y + _EPS), floor(s.y + C.PLAYER_H - _EPS)
    if s.vx > 0.0:
        col = floor(s.x + C.PLAYER_W - _EPS)
        if _solid_in_rows(level, col, r0, r1):
            s.x = col - C.PLAYER_W
            s.vx = 0.0
    elif s.vx < 0.0:
        col = floor(s.x + _EPS)
        if _solid_in_rows(level, col, r0, r1):
            s.x = col + 1.0
            s.vx = 0.0

    # move y, resolve against floors and ceilings
    s.y += s.vy
    c0, c1 = floor(s.x + _EPS), floor(s.x + C.PLAYER_W - _EPS)
    s.on_ground = False
    if s.vy > 0.0:
        row = floor(s.y + C.PLAYER_H - _EPS)
        if _solid_in_cols(level, row, c0, c1):
            s.y = row - C.PLAYER_H
            s.vy = 0.0
            s.on_ground = True
    elif s.vy < 0.0:
        row = floor(s.y + _EPS)
        if _solid_in_cols(level, row, c0, c1):
            s.y = row + 1.0
            s.vy = 0.0
            s.jumping = False

    if s.on_ground:
        s.coyote = C.COYOTE_FRAMES
    elif s.coyote > 0:
        s.coyote -= 1

    if s.y > level.height:
        return DIED
    if _touches_spike(level, s):
        return DIED
    if s.x + C.PLAYER_W >= level.goal_x:
        return GOAL
    return ALIVE

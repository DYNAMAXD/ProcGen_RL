"""Gymnasium environment wrapping the platformer for training."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config as C
from .level import generate_level
from .physics import DIED, GOAL
from .sim import FINISHED, Runner


class PlatformerEnv(gym.Env):
    """Procedurally generated left-to-right platformer.

    Each reset generates a new level. With `level_seeds=None` seeds are drawn
    from TRAIN_SEED_RANGE (effectively never repeating); pass a list to train
    on a fixed set of levels (used for the generalisation experiment).

    Difficulty: `difficulty` is the current curriculum ceiling. With
    `sample_difficulty=True` each level uses the ceiling 70% of the time and a
    uniform value below it otherwise, so easier skills are not forgotten.
    """

    metadata = {"render_modes": []}

    def __init__(self, difficulty=0.1, sample_difficulty=True, level_seeds=None,
                 max_steps=C.MAX_EPISODE_STEPS, stall_steps=C.STALL_STEPS):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(C.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(C.ACTIONS))
        self.difficulty = float(difficulty)
        self.sample_difficulty = sample_difficulty
        self.level_seeds = list(level_seeds) if level_seeds is not None else None
        self.max_steps = max_steps
        self.stall_steps = stall_steps
        self.runner = None

    def set_difficulty(self, difficulty):
        self.difficulty = float(max(0.0, min(1.0, difficulty)))

    def _pick_level(self, options):
        options = options or {}
        if "level_seed" in options:
            seed = int(options["level_seed"])
        elif self.level_seeds is not None:
            seed = int(self.level_seeds[self.np_random.integers(len(self.level_seeds))])
        else:
            seed = int(self.np_random.integers(*C.TRAIN_SEED_RANGE))
        if "difficulty" in options:
            d = float(options["difficulty"])
        elif self.sample_difficulty and self.np_random.random() < 0.3:
            d = float(self.np_random.uniform(0.0, self.difficulty))
        else:
            d = self.difficulty
        return seed, d

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        level_seed, d = self._pick_level(options)
        self.runner = Runner(generate_level(level_seed, d))
        self.steps = 0
        self.stall = 0
        return self.runner.observe(), self._info()

    def step(self, action):
        r = self.runner
        prev_max = r.max_x
        ev = r.act(int(action))
        self.steps += 1

        gained = r.max_x - prev_max
        reward = C.PROGRESS_REWARD * gained - C.TIME_PENALTY
        self.stall = 0 if gained > 1e-6 else self.stall + 1

        terminated = False
        if ev == DIED:
            reward -= C.DEATH_PENALTY
            terminated = True
        elif ev == GOAL:
            reward += C.GOAL_REWARD
            terminated = True
        truncated = (not terminated) and (self.steps >= self.max_steps or self.stall >= self.stall_steps)
        return r.observe(), float(reward), terminated, truncated, self._info()

    def _info(self):
        r = self.runner
        return {
            "level_seed": r.level.seed,
            "difficulty": r.level.difficulty,
            "progress": r.progress,
            "is_success": r.status == FINISHED,
            "death_cause": r.death_cause,
            "sim_frames": r.frames,
        }

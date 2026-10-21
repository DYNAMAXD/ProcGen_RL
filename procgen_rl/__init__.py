"""ProcGen RL: a procedurally generated platformer for reinforcement learning.

Importing the package does not import gymnasium or torch, so the demo runs
with only numpy and pygame. Import `procgen_rl.env` for the training env.
"""

from .level import Level, generate_level
from .physics import PlayerState, step_player
from .sim import Runner

__all__ = ["Level", "generate_level", "PlayerState", "step_player", "Runner"]

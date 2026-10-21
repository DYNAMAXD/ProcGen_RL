"""Measure raw environment speed: python -m procgen_rl.bench"""

import time

import numpy as np

from . import config as C
from .env import PlatformerEnv


def main(steps=20_000):
    env = PlatformerEnv(difficulty=1.0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    for _ in range(steps):
        _, _, term, trunc, _ = env.step(int(rng.integers(len(C.ACTIONS))))
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t
    sps = steps / dt
    print(f"{sps:,.0f} agent steps/s per env (random actions, incl. level generation)")
    print(f"= {sps / C.AGENT_STEPS_PER_SIM_HOUR * 3600:,.0f} simulated hours per real hour per env "
          f"(PPO overhead will reduce this)")


if __name__ == "__main__":
    main()

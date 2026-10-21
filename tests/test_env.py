import numpy as np
from gymnasium.utils.env_checker import check_env

from procgen_rl import config as C
from procgen_rl.env import PlatformerEnv
from procgen_rl.level import generate_level
from procgen_rl.sim import Runner
from procgen_rl.solver import solve


def test_gymnasium_api():
    check_env(PlatformerEnv(), skip_render_check=True)


def test_observation_shape_and_range():
    env = PlatformerEnv(difficulty=1.0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (C.OBS_DIM,) and obs.dtype == np.float32
    for _ in range(200):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all()
        if term or trunc:
            obs, _ = env.reset()


def test_solver_path_wins_in_env():
    """The env and the solver share physics: replaying a solution must succeed."""
    for seed, d in ((C.EVAL_SEED_START, 0.3), (C.EVAL_SEED_START + 1, 1.0)):
        path = solve(generate_level(seed, d))
        env = PlatformerEnv(max_steps=10_000, stall_steps=10_000)
        env.reset(seed=0, options={"level_seed": seed, "difficulty": d})
        total = 0.0
        for a in path:
            _, r, term, trunc, info = env.step(a)
            total += r
        assert term and info["is_success"] and info["progress"] == 1.0
        assert total > C.GOAL_REWARD


def test_falling_into_pit_is_penalised():
    env = PlatformerEnv()
    # find a level that starts with a gap and run right without jumping
    for seed in range(200):
        env.reset(seed=0, options={"level_seed": seed, "difficulty": 0.5})
        if env.runner.level.chunks[1][0] == "gap":
            break
    for _ in range(300):
        _, r, term, trunc, info = env.step(2)  # right
        if term or trunc:
            break
    assert term and not info["is_success"] and r < 0


def test_runner_matches_env_step():
    """Ghost-style frame stepping must match env stepping exactly."""
    env = PlatformerEnv()
    env.reset(seed=0, options={"level_seed": 42, "difficulty": 0.8})
    ghost = Runner(generate_level(42, 0.8))
    rng = np.random.default_rng(1)
    for _ in range(150):
        a = int(rng.integers(len(C.ACTIONS)))
        assert np.array_equal(env.runner.observe(), ghost.observe())
        env.step(a)
        move, jump = C.ACTIONS[a]
        for _ in range(C.FRAME_SKIP):
            if ghost.status == "running":
                ghost.frame(move, jump)
        s1, s2 = env.runner.state, ghost.state
        assert (s1.x, s1.y, s1.vx, s1.vy) == (s2.x, s2.y, s2.vx, s2.vy)
        if env.runner.status != "running":
            break

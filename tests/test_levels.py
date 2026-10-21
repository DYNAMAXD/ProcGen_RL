import numpy as np
import pytest

from procgen_rl import config as C
from procgen_rl.level import generate_level
from procgen_rl.physics import ALIVE, PlayerState, step_player
from procgen_rl.solver import solve


def test_same_seed_same_level():
    a = generate_level(1234, 0.6)
    b = generate_level(1234, 0.6)
    assert np.array_equal(a.grid, b.grid)
    assert a.goal_x == b.goal_x


def test_different_seeds_differ():
    grids = {generate_level(s, 0.6).grid.tobytes() for s in range(20)}
    assert len(grids) == 20


def test_jump_height_matches_design():
    lv = generate_level(0, 0.0)
    s = PlayerState(*lv.spawn)
    for _ in range(5):
        step_player(s, lv, 0, 0)
    assert s.on_ground
    y0, top = s.y, s.y
    for _ in range(60):
        step_player(s, lv, 0, 1)
        top = min(top, s.y)
    assert 3.5 < y0 - top < 4.0


def test_spawn_is_on_ground():
    for d in (0.0, 1.0):
        lv = generate_level(7, d)
        s = PlayerState(*lv.spawn)
        assert step_player(s, lv, 0, 0) == ALIVE
        assert s.on_ground


@pytest.mark.parametrize("difficulty", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_levels_are_solvable(difficulty):
    for seed in range(40):
        lv = generate_level(C.EVAL_SEED_START + seed, difficulty)
        assert solve(lv) is not None, f"unsolvable level seed={lv.seed} difficulty={difficulty}"


def test_stage2_chunks_only_at_higher_difficulty():
    easy = {c[0] for s in range(30) for c in generate_level(s, 0.1).chunks}
    hard = {c[0] for s in range(30) for c in generate_level(s, 1.0).chunks}
    assert not easy & {"spikes", "platforms"}
    assert {"spikes", "platforms"} <= hard

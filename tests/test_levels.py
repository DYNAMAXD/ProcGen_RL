import hashlib

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


def test_training_range_levels_unchanged():
    # Guards the trained ghosts and the published evaluation: levels at difficulty <= 1 must stay
    # byte-identical (the expert range must not draw extra random numbers below 1).
    h = hashlib.sha256()
    for d in (0.0, 0.3, 0.6, 1.0):
        for s in range(C.EVAL_SEED_START, C.EVAL_SEED_START + 25):
            h.update(generate_level(s, d).grid.tobytes())
    assert h.hexdigest() == "6c18f5f75c4800ae7167d49a6254c22b0fa99dc19815977227842a8d10ed94a7"


@pytest.mark.parametrize("difficulty", [1.25, 1.5, 2.0])
def test_expert_levels_are_solvable(difficulty):
    for seed in range(25):
        lv = generate_level(C.EVAL_SEED_START + seed, difficulty)
        assert solve(lv, max_expansions=200_000) is not None, \
            f"unsolvable level seed={lv.seed} difficulty={difficulty}"


def test_expert_chunks_only_above_one():
    normal = {c[0] for s in range(30) for c in generate_level(s, 1.0).chunks}
    expert = {c[0] for s in range(30) for c in generate_level(s, 2.0).chunks}
    assert not normal & {"spikegap", "pillars"}
    assert {"spikegap", "pillars"} <= expert

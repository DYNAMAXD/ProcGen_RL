"""Search-based level solver used to verify that generated levels are beatable.

Greedy best-first search over agent actions (same action set and frame skip
as the RL agent), always expanding the state that is furthest right. States are
deduplicated on a coarse rounding of position and velocity. Returns the action
sequence that reaches the goal, or None.
"""

import heapq

from . import config as C
from .physics import ALIVE, DIED, GOAL, PlayerState, step_player


def _key(s):
    return (round(s.x * 8), round(s.y * 8), round(s.vx * 50), round(s.vy * 25),
            s.coyote > 0, s.jumping)


def solve(level, max_expansions=100_000):
    start = PlayerState(*level.spawn)
    heap = [(-start.x, 0, start, None)]
    parents = {0: None}
    visited = {_key(start)}
    counter = 0
    expansions = 0
    while heap and expansions < max_expansions:
        _, node_id, s, _ = heapq.heappop(heap)
        expansions += 1
        for a, (move, jump) in enumerate(C.ACTIONS):
            s2 = s.copy()
            ev = ALIVE
            for _ in range(C.FRAME_SKIP):
                ev = step_player(s2, level, move, jump)
                if ev != ALIVE:
                    break
            if ev == DIED:
                continue
            counter += 1
            parents[counter] = (node_id, a)
            if ev == GOAL:
                return _path(parents, counter)
            k = _key(s2)
            if k in visited:
                continue
            visited.add(k)
            heapq.heappush(heap, (-s2.x, counter, s2, None))
    return None


def _path(parents, node_id):
    actions = []
    while parents[node_id] is not None:
        node_id, a = parents[node_id]
        actions.append(a)
    return actions[::-1]

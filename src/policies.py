"""Rule-based baselines.

GreedyRule(hold_when_degraded=True)  -> mirrors the Senior Design rule system:
    greedy step toward the target; if the comm link is degraded, force Hold.
GreedyRule(hold_when_degraded=False) -> ablation: same greedy rule, but keep
    driving when the link is down (just without any ally information).

"Greedy" = step to a neighbouring cell that is one step closer to the target
along a shortest path on the (known, static) obstacle map. While the link is
up the robots also apply a right-of-way rule so they never collide: the robot
that is closer to the target goes first; the other one never enters the
ally's cell or any cell the ally could step into (it yields with Hold).

Both rules use exactly the same information as the learned agents.
"""
from __future__ import annotations

import random

from .env import DELTAS, HOLD, MOVE_ACTIONS


def has_priority_over_me(my_dist: int, ally_dist: int, my_id: int) -> bool:
    """Right-of-way: the robot closer to the target goes first, ties -> lower id."""
    return ally_dist < my_dist or (ally_dist == my_dist and (1 - my_id) < my_id)


class GreedyRule:
    def __init__(self, hold_when_degraded: bool = True, seed: int = 0,
                 memory_yield: bool = False, yield_age: int = 3):
        """memory_yield=True (only meaningful with hold_when_degraded=False):
        while the link is down, Hold only if the ally was last seen within 2
        cells of us, had right-of-way at that moment, and that sighting is at
        most `yield_age` steps old; otherwise keep driving."""
        self.hold_when_degraded = hold_when_degraded
        self.memory_yield = memory_yield
        self.yield_age = yield_age
        self.rng = random.Random(seed)

    def act(self, obs) -> int:
        if not obs.comm_up and self.hold_when_degraded:
            return HOLD
        r, c = obs.pos
        if not obs.comm_up and self.memory_yield and obs.mem_pos is not None and obs.mem_age <= self.yield_age:
            mr, mc = obs.mem_pos
            if abs(mr - r) + abs(mc - c) <= 2 and has_priority_over_me(obs.dist, obs.field[obs.mem_pos], obs.agent_id):
                return HOLD

        # cells we must not enter this step (only knowable while the link is up)
        forbidden = set()
        if obs.ally_pos is not None:
            ar, ac = obs.ally_pos
            forbidden.add((ar, ac))
            if has_priority_over_me(obs.dist, obs.field[obs.ally_pos], obs.agent_id):
                for dr, dc in DELTAS[1:]:
                    forbidden.add((ar + dr, ac + dc))
            forbidden.discard(obs.target)  # the target cell can hold both robots

        downhill = [a for a in MOVE_ACTIONS if obs.downhill & (1 << (a - 1))]
        ok = [a for a in downhill if (r + DELTAS[a][0], c + DELTAS[a][1]) not in forbidden]
        if ok:
            return ok[0] if len(ok) == 1 else self.rng.choice(ok)
        return HOLD  # yield right-of-way to the ally

    def act_batch(self, obs_list):
        return [self.act(o) for o in obs_list]

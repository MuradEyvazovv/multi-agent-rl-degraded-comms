"""Observation encoders for the learned policies.

tabular_state(obs)          -> int in [0, N_TABULAR_STATES)
dqn_features(obs, memory)   -> float32 vector of length N_FEATURES

Both encoders see the same sensors as the rule baseline: own cell, target
offset, shortest-path distance, which moves go downhill, bumper (obstacle)
mask, link state, and the ally's cell only while the link is up. The
"memory" DQN variant additionally sees the ally's last-known cell and how
many steps old that information is.
"""
from __future__ import annotations

import numpy as np

from .env import ROWS, COLS, AGE_UNKNOWN
from .policies import has_priority_over_me

# ---------------------------------------------------------------- tabular
# ally offsets within Manhattan distance 2 of us (12 cells)
_NEAR = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if 0 < abs(dr) + abs(dc) <= 2]
_NEAR_IDX = {d: k for k, d in enumerate(_NEAR)}
N_ALLY_CODES = 1 + 1 + 2 * len(_NEAR) + 1   # link down, ally docked, near x right-of-way, far
DIST_CAP = 12
N_TABULAR_STATES = 16 * 16 * (DIST_CAP + 1) * N_ALLY_CODES


def tabular_state(obs) -> int:
    d = obs.dist
    dist_b = min(d, DIST_CAP)
    if not obs.comm_up:
        ally = 0
    elif obs.ally_docked:
        ally = 1
    else:
        r, c = obs.pos
        ar, ac = obs.ally_pos
        k = _NEAR_IDX.get((ar - r, ac - c))
        if k is None:
            ally = N_ALLY_CODES - 1
        else:
            prio = has_priority_over_me(d, obs.field[obs.ally_pos], obs.agent_id)
            ally = 2 + k + len(_NEAR) * int(prio)
    return ((obs.downhill * 16 + obs.walls) * (DIST_CAP + 1) + dist_b) * N_ALLY_CODES + ally


# -------------------------------------------------------------------- DQN
N_FEATURES = 22 + len(_NEAR)
_RS, _CS, _DS = ROWS - 1, COLS - 1, 20.0
MAX_AGE = 10


def dqn_features(obs, memory: bool) -> np.ndarray:
    """Without memory the ally block is filled only while the link is up.
    With memory it holds the last-known ally cell (relative to where we are
    now) plus how stale that information is."""
    r, c = obs.pos
    tr, tc = obs.target
    w, dh = obs.walls, obs.downhill
    f = [r / _RS, c / _CS, (tr - r) / _RS, (tc - c) / _CS, obs.dist / _DS,
         dh & 1, (dh >> 1) & 1, (dh >> 2) & 1, (dh >> 3) & 1,
         w & 1, (w >> 1) & 1, (w >> 2) & 1, (w >> 3) & 1,
         float(obs.comm_up), float(obs.agent_id)]
    if memory:
        known = obs.mem_age != AGE_UNKNOWN
        a_pos, a_docked = obs.mem_pos, obs.mem_docked
        age = min(obs.mem_age, MAX_AGE) / MAX_AGE if known else 1.0
    else:
        known = obs.comm_up
        a_pos, a_docked = obs.ally_pos, bool(obs.ally_docked)
        age = 0.0 if known else 1.0
    near = [0.0] * len(_NEAR)   # one-hot of the ally's offset if within 2 cells (same code as tabular)
    if known and a_pos is not None:
        ar, ac = a_pos
        ally_d = obs.field[a_pos]
        f += [1.0, (ar - r) / _RS, (ac - c) / _CS, 0.0,
              float(has_priority_over_me(obs.dist, ally_d, obs.agent_id)), ally_d / _DS, age]
        k = _NEAR_IDX.get((ar - r, ac - c))
        if k is not None:
            near[k] = 1.0
    elif known and a_docked:
        f += [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, age]
    else:
        f += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return np.asarray(f + near, dtype=np.float32)

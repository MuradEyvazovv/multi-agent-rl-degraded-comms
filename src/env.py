"""Two-robot gridworld with an unreliable communication link.

Pure Python / NumPy, no gym dependency.

Grid: 7 rows x 9 columns, static obstacles, 2 agents, 1 shared target.

The static obstacle map is known to both robots (it never changes), so each
robot can compute its shortest-path distance to the target and which of its
four moves go "downhill" (one step closer along a shortest path).

Each step:
  * both agents act simultaneously (Hold, N, E, S, W);
  * the comm link is "up" with probability 1 - p_drop (drawn i.i.d. per step);
  * an agent sees the ally's current position ONLY while the link is up.
    Every agent also keeps the ally's last-known position and how many steps
    old it is (policies that do not use memory simply ignore those fields).

Rewards (per agent):
  -0.1 per step while the agent is still driving,
  +10 when it reaches the target (it then "docks" and leaves the grid),
  -10 to both agents on a collision, which also ends the episode (a crash is
      a mission failure for physical robots).

Collision = both agents try to enter the same non-target cell, they swap
cells, or one drives into the other while the other stays put. The target
cell can hold both robots, so arriving together is fine.

Team success = both agents docked before the step cap, with no collision.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

ROWS, COLS = 7, 9
MAX_STEPS = 40

HOLD, NORTH, EAST, SOUTH, WEST = range(5)
ACTION_NAMES = ["Hold", "N", "E", "S", "W"]
DELTAS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
MOVE_ACTIONS = (NORTH, EAST, SOUTH, WEST)

# never-seen ally: age is reported as this sentinel
AGE_UNKNOWN = 10**6


@dataclass(frozen=True)
class Layout:
    rows: int
    cols: int
    obstacles: frozenset          # set of (r, c)
    free: tuple                   # free cells, row-major order
    wallmask: dict                # (r, c) -> 4-bit mask, bit k set if move k+1 (N,E,S,W) is blocked

    def blocked(self, r: int, c: int) -> bool:
        return r < 0 or c < 0 or r >= self.rows or c >= self.cols or (r, c) in self.obstacles

    def key(self) -> tuple:
        return tuple(sorted(self.obstacles))


def _connected(rows, cols, obstacles, free):
    if not free:
        return False
    seen = {free[0]}
    q = deque([free[0]])
    while q:
        r, c = q.popleft()
        for dr, dc in DELTAS[1:]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obstacles and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(free)


def build_layout(rows: int, cols: int, obstacles) -> Layout:
    obstacles = frozenset(obstacles)
    free = tuple((r, c) for r in range(rows) for c in range(cols) if (r, c) not in obstacles)
    wallmask = {}
    for (r, c) in free:
        m = 0
        for k, (dr, dc) in enumerate(DELTAS[1:]):
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols or (nr, nc) in obstacles:
                m |= 1 << k
        wallmask[(r, c)] = m
    return Layout(rows, cols, obstacles, free, wallmask)


def random_layout(rng: np.random.Generator, rows=ROWS, cols=COLS, n_obs_range=(6, 10)) -> Layout:
    """Scatter obstacles uniformly; reject layouts whose free cells are not all connected."""
    while True:
        k = int(rng.integers(n_obs_range[0], n_obs_range[1] + 1))
        cells = rng.choice(rows * cols, size=k, replace=False)
        obstacles = frozenset((int(x) // cols, int(x) % cols) for x in cells)
        free = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in obstacles]
        if _connected(rows, cols, obstacles, free):
            return build_layout(rows, cols, obstacles)


def make_layout_pool(seed: int, n: int, exclude_keys=frozenset(), rows=ROWS, cols=COLS):
    """n distinct random layouts; any layout whose obstacle set is in exclude_keys is skipped."""
    rng = np.random.default_rng(seed)
    pool, keys = [], set()
    while len(pool) < n:
        lay = random_layout(rng, rows, cols)
        k = lay.key()
        if k in keys or k in exclude_keys:
            continue
        keys.add(k)
        pool.append(lay)
    return pool


def bfs_dist(layout: Layout, src, dst) -> int:
    if src == dst:
        return 0
    seen = {src}
    q = deque([(src, 0)])
    while q:
        (r, c), d = q.popleft()
        for dr, dc in DELTAS[1:]:
            nxt = (r + dr, c + dc)
            if nxt == dst:
                return d + 1
            if not layout.blocked(*nxt) and nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return -1


def distance_field(layout: Layout, target) -> dict:
    """Shortest-path (BFS) distance from every free cell to the target."""
    field = {target: 0}
    q = deque([target])
    while q:
        r, c = q.popleft()
        d = field[(r, c)] + 1
        for dr, dc in DELTAS[1:]:
            nxt = (r + dr, c + dc)
            if nxt not in field and not layout.blocked(*nxt):
                field[nxt] = d
                q.append(nxt)
    return field


@dataclass
class EpisodeSpec:
    layout: Layout
    starts: tuple        # ((r, c), (r, c))
    target: tuple        # (r, c)
    comm_u: np.ndarray   # uniform draws; link is up at step t iff comm_u[t] >= p_drop


CLOSE_SEP = 2       # close-quarters: robots start at most this many cells apart (Manhattan)
CLOSE_MIN_DIST = 4  # ... and both at least this many steps from the target


def sample_episode(rng: np.random.Generator, pool, max_steps=MAX_STEPS, mode: str = "open") -> EpisodeSpec:
    """mode="open":  target and both starts uniformly random over free cells.
    mode="close": close-quarters stress test - the robots start within
                  CLOSE_SEP cells of each other and must travel >= CLOSE_MIN_DIST
                  steps, so their paths overlap and collisions are a real risk.
    mode="mix":   50/50 mixture of the two (used for training)."""
    if mode == "mix":
        mode = "open" if rng.random() < 0.5 else "close"
    if mode == "open":
        layout = pool[int(rng.integers(len(pool)))]
        idx = rng.choice(len(layout.free), size=3, replace=False)
        target = layout.free[int(idx[0])]
        starts = (layout.free[int(idx[1])], layout.free[int(idx[2])])
    elif mode == "close":
        while True:
            layout = pool[int(rng.integers(len(pool)))]
            target = layout.free[int(rng.integers(len(layout.free)))]
            field = distance_field(layout, target)
            a = layout.free[int(rng.integers(len(layout.free)))]
            if field[a] < CLOSE_MIN_DIST:
                continue
            cands = [b for b in layout.free if b != a and field[b] >= CLOSE_MIN_DIST
                     and abs(b[0] - a[0]) + abs(b[1] - a[1]) <= CLOSE_SEP]
            if cands:
                b = cands[int(rng.integers(len(cands)))]
                starts = (a, b) if rng.random() < 0.5 else (b, a)
                break
    else:
        raise ValueError(mode)
    comm_u = rng.random(max_steps + 1)
    return EpisodeSpec(layout, starts, target, comm_u)


class Obs:
    """What one robot knows at the start of a step."""
    __slots__ = ("agent_id", "pos", "target", "walls", "dist", "downhill", "field", "comm_up",
                 "ally_pos", "ally_docked", "mem_pos", "mem_docked", "mem_age")

    def __init__(self, agent_id, pos, target, walls, dist, downhill, field, comm_up,
                 ally_pos, ally_docked, mem_pos, mem_docked, mem_age):
        self.agent_id = agent_id
        self.pos = pos
        self.target = target
        self.walls = walls                # 4-bit bumper mask (N,E,S,W blocked)
        self.dist = dist                  # own shortest-path distance to target
        self.downhill = downhill          # 4-bit mask of moves that reduce that distance
        self.field = field                # static-map distance field (lets a robot rate any cell)
        self.comm_up = comm_up
        self.ally_pos = ally_pos          # current ally cell if link up and ally still driving, else None
        self.ally_docked = ally_docked    # True/False if link up, None if unknown
        self.mem_pos = mem_pos            # last-known ally cell (None if never seen or last seen docked)
        self.mem_docked = mem_docked      # last-known docked flag (False if never seen)
        self.mem_age = mem_age            # steps since last contact (AGE_UNKNOWN if never)


class CommGridEnv:
    def __init__(self, p_drop: float, max_steps: int = MAX_STEPS, step_penalty: float = 0.1,
                 goal_reward: float = 10.0, collision_penalty: float = 10.0):
        self.p_drop = float(p_drop)
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty

    # ------------------------------------------------------------------ core
    def reset(self, spec: EpisodeSpec):
        self.spec = spec
        self.layout = spec.layout
        self.target = spec.target
        self.field = distance_field(spec.layout, spec.target)
        self.pos = [spec.starts[0], spec.starts[1]]
        self.docked = [False, False]
        self.t = 0
        self.done = False
        self.collided = False
        self.finish_step = None
        # Both robots are deployed together, so each knows where the other
        # started (age 0) even if the link is down at t=0. Only the "memory"
        # policies read these fields.
        self.mem_pos = [self.pos[1], self.pos[0]]
        self.mem_docked = [False, False]
        self.mem_age = [0, 0]
        self._update_link()
        return [self._obs(0), self._obs(1)]

    def active(self, i: int) -> bool:
        return not self.done and not self.docked[i]

    def step(self, actions):
        """actions: sequence of 2 ints (ignored for docked agents).

        Returns (obs, rewards, terminal, done, info). terminal[i] is True when
        agent i's own episode ended this step for a reason that should not be
        bootstrapped (docked or crashed); a timeout is a truncation.
        """
        assert not self.done
        self.t += 1
        rewards = [0.0, 0.0]
        terminal = [False, False]
        was_active = [not self.docked[0], not self.docked[1]]
        prop = list(self.pos)
        lay = self.layout
        for i in (0, 1):
            if not was_active[i]:
                continue
            rewards[i] -= self.step_penalty
            dr, dc = DELTAS[actions[i]]
            r, c = self.pos[i]
            nr, nc = r + dr, c + dc
            if not lay.blocked(nr, nc):
                prop[i] = (nr, nc)

        collision = False
        if was_active[0] and was_active[1]:
            a, b = prop
            pa, pb = self.pos
            if a == b and a != self.target:
                collision = True                       # same destination
            elif a == pb and b == pa:
                collision = True                       # head-on swap
            elif a == pb and b == pb:
                collision = True                       # 0 drives into stationary 1
            elif b == pa and a == pa:
                collision = True                       # 1 drives into stationary 0

        if collision:
            self.collided = True
            for i in (0, 1):
                rewards[i] -= self.collision_penalty
                terminal[i] = True
            self.done = True
        else:
            for i in (0, 1):
                if not was_active[i]:
                    continue
                self.pos[i] = prop[i]
                if prop[i] == self.target:
                    self.docked[i] = True
                    rewards[i] += self.goal_reward
                    terminal[i] = True
            if self.docked[0] and self.docked[1]:
                self.done = True
                self.finish_step = self.t

        timeout = False
        if not self.done and self.t >= self.max_steps:
            self.done = True
            timeout = True

        self._update_link()
        info = {"collision": collision, "timeout": timeout,
                "success": self.done and self.docked[0] and self.docked[1] and not self.collided}
        return [self._obs(0), self._obs(1)], rewards, terminal, self.done, info

    # --------------------------------------------------------------- helpers
    def _update_link(self):
        t = min(self.t, len(self.spec.comm_u) - 1)
        self.comm_up = bool(self.spec.comm_u[t] >= self.p_drop)
        for i in (0, 1):
            j = 1 - i
            if self.comm_up:
                self.mem_age[i] = 0
                self.mem_docked[i] = self.docked[j]
                self.mem_pos[i] = None if self.docked[j] else self.pos[j]
            elif self.mem_age[i] != AGE_UNKNOWN:
                self.mem_age[i] += 1

    def _obs(self, i: int) -> Obs:
        j = 1 - i
        if self.comm_up:
            ally_docked = self.docked[j]
            ally_pos = None if ally_docked else self.pos[j]
        else:
            ally_docked = None
            ally_pos = None
        pos = self.pos[i]
        walls = self.layout.wallmask.get(pos, 0)
        field = self.field
        d = field[pos]
        r, c = pos
        downhill = 0
        for k, (dr, dc) in enumerate(DELTAS[1:]):
            if field.get((r + dr, c + dc), -9) == d - 1:
                downhill |= 1 << k
        return Obs(i, pos, self.target, walls, d, downhill, field, self.comm_up, ally_pos, ally_docked,
                   self.mem_pos[i], self.mem_docked[i], self.mem_age[i])

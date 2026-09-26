"""Lock-step evaluation of a policy on a fixed, held-out set of episodes.

Every policy at a given seed sees exactly the same episodes: same obstacle
layouts (from a pool never used in training), same start/target cells and
the same per-step link-drop draws (common random numbers).
"""
from __future__ import annotations

import numpy as np

from .env import CommGridEnv, DELTAS, HOLD, MAX_STEPS, bfs_dist, sample_episode


def make_eval_specs(seed: int, n: int, pool, mode: str = "open", max_steps=MAX_STEPS):
    rng = np.random.default_rng(seed)
    return [sample_episode(rng, pool, max_steps, mode) for _ in range(n)]


def run_episodes(policy, p_drop: float, specs, max_steps=MAX_STEPS, keep_traj=False):
    envs = [CommGridEnv(p_drop, max_steps) for _ in specs]
    obs = [env.reset(s) for env, s in zip(envs, specs)]
    n = len(specs)
    ret = np.zeros(n)
    trajs = [[tuple(env.pos)] for env in envs] if keep_traj else None
    links = [[env.comm_up] for env in envs] if keep_traj else None
    hold_down = move_down = hold_up = move_up = 0
    near = {"hold": 0, "downhill": 0, "other_move": 0, "bump": 0}
    steer = {"away": 0, "toward": 0}
    live = list(range(n))
    while live:
        batch_obs, where = [], []
        for e in live:
            for i in (0, 1):
                if envs[e].active(i):
                    batch_obs.append(obs[e][i])
                    where.append((e, i))
        acts = policy.act_batch(batch_obs)
        actions = {e: [HOLD, HOLD] for e in live}
        for (e, i), a, o in zip(where, acts, batch_obs):
            a = int(a)
            actions[e][i] = a
            if o.comm_up:
                hold_up += a == HOLD
                move_up += a != HOLD
            else:
                hold_down += a == HOLD
                move_down += a != HOLD
                # risky moment: link down and the ally was last seen within 2 cells
                if o.mem_pos is not None and abs(o.mem_pos[0] - o.pos[0]) + abs(o.mem_pos[1] - o.pos[1]) <= 2:
                    bit = 1 << (a - 1) if a != HOLD else 0
                    kind = ("hold" if a == HOLD else "downhill" if o.downhill & bit
                            else "bump" if o.walls & bit else "other_move")
                    near[kind] += 1
                    # with two downhill options, does it take the one leading AWAY from the ally's last-known cell?
                    if kind == "downhill":
                        (r, c), (mr, mc) = o.pos, o.mem_pos
                        opts = [k for k in (1, 2, 3, 4) if o.downhill & (1 << (k - 1))]
                        if len(opts) == 2:
                            dd = {k: abs(r + DELTAS[k][0] - mr) + abs(c + DELTAS[k][1] - mc) for k in opts}
                            if dd[opts[0]] != dd[opts[1]]:
                                steer["away" if dd[a] == max(dd.values()) else "toward"] += 1
        still = []
        for e in live:
            obs[e], rew, _, done, _ = envs[e].step(actions[e])
            ret[e] += rew[0] + rew[1]
            if keep_traj:
                trajs[e].append(tuple(envs[e].pos))
                links[e].append(envs[e].comm_up)
            if not done:
                still.append(e)
        live = still

    success = np.array([env.docked[0] and env.docked[1] and not env.collided for env in envs])
    collision = np.array([env.collided for env in envs])
    timeout = ~success & ~collision
    steps = np.array([env.finish_step if env.finish_step is not None else env.t for env in envs])
    out = {
        "success": success, "collision": collision, "timeout": timeout,
        "steps": steps, "team_return": ret,
        "hold_frac_link_down": hold_down / max(1, hold_down + move_down),
        "hold_frac_link_up": hold_up / max(1, hold_up + move_up),
        "near_down_decisions": sum(near.values()),
        **{f"near_down_{k}": v / max(1, sum(near.values())) for k, v in near.items()},
        "near_down_steer_choices": steer["away"] + steer["toward"],
        "near_down_steer_away": steer["away"] / max(1, steer["away"] + steer["toward"]),
    }
    if keep_traj:
        out["trajs"] = trajs
        out["links"] = links
    return out


def summarize(ep) -> dict:
    s = ep["success"]
    return {
        "success_rate": float(s.mean()),
        "collision_rate": float(ep["collision"].mean()),
        "timeout_rate": float(ep["timeout"].mean()),
        "steps_success": float(ep["steps"][s].mean()) if s.any() else float("nan"),
        "team_return": float(ep["team_return"].mean()),
        "hold_frac_link_down": float(ep["hold_frac_link_down"]),
        "hold_frac_link_up": float(ep["hold_frac_link_up"]),
        **{k: float(v) for k, v in ep.items() if k.startswith("near_down_")},
    }


def makespan_lower_bound(specs) -> float:
    """Mean over episodes of max(BFS distance of each robot to the target):
    the fewest steps any policy could need with a perfect link and no conflicts."""
    return float(np.mean([max(bfs_dist(s.layout, s.starts[0], s.target),
                              bfs_dist(s.layout, s.starts[1], s.target)) for s in specs]))

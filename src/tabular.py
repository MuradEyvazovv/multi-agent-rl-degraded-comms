"""Tabular Q-learning with a Q-table shared by both robots (parameter sharing).

Each robot acts on its own local state (decentralised execution); both
robots' transitions update the same table.
"""
from __future__ import annotations

import random
import time

import numpy as np

from .env import CommGridEnv, sample_episode, MAX_STEPS
from .features import tabular_state, N_TABULAR_STATES


class TabularPolicy:
    def __init__(self, Q: np.ndarray):
        self.Q = Q

    def act_batch(self, obs_list):
        return [int(np.argmax(self.Q[tabular_state(o)])) for o in obs_list]


def train_tabular(p_drop: float, seed: int, pool, n_episodes: int = 40_000,
                  alpha_power: float = 0.6, alpha_min: float = 0.01, gamma: float = 0.95, eps_start: float = 1.0,
                  eps_end: float = 0.05, eps_frac: float = 0.7, log_every: int = 1000,
                  max_steps: int = MAX_STEPS, train_mode: str = "mix"):
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    Q = np.zeros((N_TABULAR_STATES, 5), dtype=np.float64)
    visits = np.zeros((N_TABULAR_STATES, 5), dtype=np.int64)
    env = CommGridEnv(p_drop, max_steps)
    decay_eps = max(1, int(eps_frac * n_episodes))
    curve, window = [], []
    t0 = time.time()
    for ep in range(n_episodes):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / decay_eps)
        obs = env.reset(sample_episode(rng, pool, max_steps, train_mode))
        states = [tabular_state(obs[0]), tabular_state(obs[1])]
        done = False
        while not done:
            acts = [0, 0]
            act_mask = [env.active(0), env.active(1)]
            for i in (0, 1):
                if act_mask[i]:
                    acts[i] = prng.randrange(5) if prng.random() < eps else int(np.argmax(Q[states[i]]))
            obs, rew, term, done, info = env.step(acts)
            for i in (0, 1):
                if not act_mask[i]:
                    continue
                s, a = states[i], acts[i]
                if term[i]:
                    target = rew[i]
                else:
                    s2 = tabular_state(obs[i])
                    states[i] = s2
                    target = rew[i] + gamma * Q[s2].max()
                visits[s, a] += 1
                # visit-count step size: large for rarely seen (s, a), small once well estimated
                alpha = max(alpha_min, visits[s, a] ** -alpha_power)
                Q[s, a] += alpha * (target - Q[s, a])
        window.append(float(info["success"]))
        if (ep + 1) % log_every == 0:
            curve.append({"episode": ep + 1, "success_rate": float(np.mean(window)), "epsilon": eps})
            window = []
    visited = int((visits.sum(1) > 0).sum())
    return Q, {"curve": curve, "train_seconds": time.time() - t0, "visited_states": visited,
               "n_states": N_TABULAR_STATES, "episodes": n_episodes}

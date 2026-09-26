"""Double DQN with a network shared by both robots (parameter sharing).

memory=False : the ally block of the input is filled only while the link is up.
memory=True  : the ally block holds the last-known ally cell + its age.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import CommGridEnv, sample_episode, MAX_STEPS, HOLD
from .features import dqn_features, N_FEATURES


class QNet(nn.Module):
    def __init__(self, n_in=N_FEATURES, hidden=128, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class DQNPolicy:
    def __init__(self, net: QNet, memory: bool, device="cpu"):
        self.net = net.to(device).eval()
        self.memory = memory
        self.device = device

    @torch.no_grad()
    def act_batch(self, obs_list):
        if not obs_list:
            return []
        x = torch.from_numpy(np.stack([dqn_features(o, self.memory) for o in obs_list])).to(self.device)
        return self.net(x).argmax(1).cpu().tolist()


class Replay:
    def __init__(self, capacity, n_in):
        self.s = np.zeros((capacity, n_in), np.float32)
        self.s2 = np.zeros((capacity, n_in), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.cap, self.n, self.i = capacity, 0, 0

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, rng, k):
        idx = rng.integers(0, self.n, size=k)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


def train_dqn(p_drop: float, seed: int, pool, memory: bool, total_env_steps: int = 300_000,
              n_envs: int = 16, gamma: float = 0.95, lr: float = 5e-4, batch: int = 128,
              buffer: int = 100_000, warmup: int = 5_000, updates_per_iter: int = 2,
              target_every: int = 500, eps_start: float = 1.0, eps_end: float = 0.05,
              eps_frac: float = 0.6, hidden: int = 128, device: str = "cpu",
              loss_type: str = "huber", log_every_eps: int = 1000, max_steps: int = MAX_STEPS, train_mode: str = "mix"):
    """total_env_steps counts environment steps summed over the parallel envs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    net = QNet(hidden=hidden).to(device)
    tgt = QNet(hidden=hidden).to(device)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rb = Replay(buffer, N_FEATURES)

    envs = [CommGridEnv(p_drop, max_steps) for _ in range(n_envs)]
    obs = [env.reset(sample_episode(rng, pool, max_steps, train_mode)) for env in envs]
    n_iters = total_env_steps // n_envs
    decay_iters = max(1, int(eps_frac * n_iters))
    curve, window, n_updates, n_eps = [], [], 0, 0
    losses = []
    t0 = time.time()
    for it in range(n_iters):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - it / decay_iters)
        where, feats = [], []
        for e, env in enumerate(envs):
            for i in (0, 1):
                if env.active(i):
                    where.append((e, i))
                    feats.append(dqn_features(obs[e][i], memory))
        X = np.stack(feats)
        with torch.no_grad():
            greedy = net(torch.from_numpy(X).to(device)).argmax(1).cpu().numpy()
        actions = [[HOLD, HOLD] for _ in envs]
        chosen = []
        for k, (e, i) in enumerate(where):
            a = prng.randrange(5) if prng.random() < eps else int(greedy[k])
            actions[e][i] = a
            chosen.append(a)
        results = []
        for e, env in enumerate(envs):
            results.append(env.step(actions[e]))
        for k, (e, i) in enumerate(where):
            nobs, rew, term, done, info = results[e]
            s2 = X[k] if term[i] else dqn_features(nobs[i], memory)
            rb.add(X[k], chosen[k], rew[i], s2, float(term[i]))
        for e, env in enumerate(envs):
            nobs, rew, term, done, info = results[e]
            if done:
                window.append(float(info["success"]))
                n_eps += 1
                if n_eps % log_every_eps == 0:
                    curve.append({"episode": n_eps, "env_steps": (it + 1) * n_envs,
                                  "success_rate": float(np.mean(window)), "epsilon": eps})
                    window = []
                obs[e] = env.reset(sample_episode(rng, pool, max_steps, train_mode))
            else:
                obs[e] = nobs

        if rb.n >= warmup:
            for _ in range(updates_per_iter):
                s, a, r, s2, d = (torch.from_numpy(x).to(device) for x in rb.sample(rng, batch))
                q = net(s).gather(1, a[:, None]).squeeze(1)
                with torch.no_grad():
                    a2 = net(s2).argmax(1, keepdim=True)
                    y = r + gamma * (1.0 - d) * tgt(s2).gather(1, a2).squeeze(1)
                loss = F.smooth_l1_loss(q, y) if loss_type == "huber" else F.mse_loss(q, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                n_updates += 1
                if n_updates % target_every == 0:
                    tgt.load_state_dict(net.state_dict())
            if it % 200 == 0:
                losses.append(float(loss.item()))
    net = net.cpu()
    return net, {"curve": curve, "train_seconds": time.time() - t0, "episodes": n_eps,
                 "env_steps": n_iters * n_envs, "grad_updates": n_updates,
                 "final_loss": losses[-1] if losses else None}

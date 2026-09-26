"""One experiment job = (policy, train drop-rate p, seed).

Learned policies are trained at p and then evaluated at every p in the sweep
(the diagonal train_p == eval_p is the main result; the rest measures
robustness to a link that is better/worse than the one trained on).
Rule policies need no training and are just evaluated.
"""
from __future__ import annotations

import os
import time

import numpy as np

P_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9]
MODES = ["open", "close"]
TRAIN_POOL_SEED, TRAIN_POOL_SIZE = 0, 400
TEST_POOL_SEED, TEST_POOL_SIZE = 1, 200
EVAL_SEED_BASE = {"open": 10_000, "close": 20_000}
VAL_SEED = 77

POLICIES = ["rule_hold", "rule_nohold", "rule_memyield", "tabular_q", "dqn", "dqn_mem"]
LABELS = {
    "rule_hold": "Rule: hold when degraded (original)",
    "rule_nohold": "Rule: keep driving (ablation)",
    "rule_memyield": "Rule: last-known-ally yield",
    "tabular_q": "Tabular Q-learning",
    "dqn": "DQN",
    "dqn_mem": "DQN + last-known-ally memory",
}

_POOLS = None


def pools():
    """Training layouts and a disjoint pool of held-out test layouts."""
    global _POOLS
    if _POOLS is None:
        from .env import make_layout_pool
        train = make_layout_pool(TRAIN_POOL_SEED, TRAIN_POOL_SIZE)
        test = make_layout_pool(TEST_POOL_SEED, TEST_POOL_SIZE, exclude_keys={l.key() for l in train})
        assert not ({l.key() for l in train} & {l.key() for l in test})
        _POOLS = (train, test)
    return _POOLS


def eval_specs(seed: int, mode: str, n: int):
    from .evaluate import make_eval_specs
    return make_eval_specs(EVAL_SEED_BASE[mode] + seed, n, pools()[1], mode)


def make_rule(name: str, seed: int, yield_age: int = 2):
    from .policies import GreedyRule
    if name == "rule_hold":
        return GreedyRule(hold_when_degraded=True, seed=seed)
    if name == "rule_nohold":
        return GreedyRule(hold_when_degraded=False, seed=seed)
    if name == "rule_memyield":
        return GreedyRule(hold_when_degraded=False, seed=seed, memory_yield=True, yield_age=yield_age)
    raise ValueError(name)


def run_job(job: dict) -> dict:
    """Executed in a worker process."""
    import torch
    torch.set_num_threads(1)
    from .evaluate import run_episodes, summarize
    from .tabular import train_tabular, TabularPolicy
    from .dqn import train_dqn, DQNPolicy

    policy, p, seed = job["policy"], job["train_p"], job["seed"]
    train_pool, _ = pools()
    t0 = time.time()
    train_info = {}
    if policy == "tabular_q":
        Q, train_info = train_tabular(p, seed, train_pool, n_episodes=job["tab_episodes"])
        np.save(os.path.join(job["model_dir"], f"tabular_q_p{p:.2f}_s{seed}.npy"), Q.astype(np.float32))
        pol = TabularPolicy(Q)
        eval_ps = P_VALUES
    elif policy in ("dqn", "dqn_mem"):
        mem = policy == "dqn_mem"
        net, train_info = train_dqn(p, seed, train_pool, memory=mem, total_env_steps=job["dqn_steps"],
                                    loss_type="mse", device=job.get("device", "cpu"))
        torch.save(net.state_dict(), os.path.join(job["model_dir"], f"{policy}_p{p:.2f}_s{seed}.pt"))
        pol = DQNPolicy(net, mem)
        eval_ps = P_VALUES
    else:
        pol = make_rule(policy, seed, job.get("yield_age", 2))
        eval_ps = [p]
    train_seconds = time.time() - t0

    evals = []
    for ep in eval_ps:
        for mode in MODES:
            m = summarize(run_episodes(pol, ep, eval_specs(seed, mode, job["n_eval"])))
            evals.append({"eval_p": ep, "mode": mode, **m})
    return {"policy": policy, "train_p": p, "seed": seed, "train_seconds": train_seconds,
            "eval_seconds": time.time() - t0 - train_seconds,
            "train_info": train_info, "evals": evals}


def select_yield_age(candidates=(1, 2, 3, 5), n=500):
    """Pick the memory-yield rule's only parameter on TRAINING layouts
    (validation episodes), never on the held-out test layouts."""
    from .evaluate import make_eval_specs, run_episodes, summarize
    train_pool, _ = pools()
    table = {}
    for k in candidates:
        scores = []
        for mode in MODES:
            specs = make_eval_specs(VAL_SEED, n, train_pool, mode)
            for p in P_VALUES[1:]:
                scores.append(summarize(run_episodes(make_rule("rule_memyield", 0, k), p, specs))["success_rate"])
        table[k] = float(np.mean(scores))
    best = max(table, key=lambda k: (table[k], -k))
    return best, table

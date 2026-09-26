#!/usr/bin/env python
"""Entry point: train every learned policy, evaluate everything, write results/ and figures/.

    python run_experiments.py            # full run (5 seeds, ~10-15 min on an M3 Max)
    python run_experiments.py --quick    # smoke test, writes to _quick/
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

from src import plots
from src.bench import bench_devices
from src.evaluate import makespan_lower_bound, run_episodes
from src.experiment import (LABELS, MODES, P_VALUES, POLICIES, eval_specs, make_rule, pools,
                            run_job, select_yield_age, TRAIN_POOL_SIZE, TEST_POOL_SIZE)

ROOT = os.path.dirname(os.path.abspath(__file__))
NEAR = ["near_down_hold", "near_down_downhill", "near_down_other_move", "near_down_bump"]
METRICS = ["success_rate", "collision_rate", "timeout_rate", "steps_success", "team_return",
           "hold_frac_link_down", "hold_frac_link_up", "near_down_decisions",
           "near_down_steer_choices", "near_down_steer_away"] + NEAR
LEARNED = ["tabular_q", "dqn", "dqn_mem"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--tab-episodes", type=int, default=60_000)
    ap.add_argument("--dqn-steps", type=int, default=400_000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = ROOT
    if args.quick:
        args.seeds, args.tab_episodes, args.dqn_steps, args.n_eval = 2, 3000, 20_000, 100
        out = os.path.join(ROOT, "_quick")
    res_dir, fig_dir, model_dir = (os.path.join(out, d) for d in ("results", "figures", "models"))
    for d in (res_dir, fig_dir, model_dir):
        os.makedirs(d, exist_ok=True)

    t_start = time.time()
    device, bench = bench_devices()
    print(f"[bench] per-update / per-action ms: {bench} -> using {device}", flush=True)

    train_pool, test_pool = pools()
    yield_age, yield_table = select_yield_age()
    print(f"[val] memory-yield rule: success by yield_age {yield_table} -> {yield_age}", flush=True)

    seeds = list(range(args.seeds))
    base = {"tab_episodes": args.tab_episodes, "dqn_steps": args.dqn_steps, "n_eval": args.n_eval,
            "model_dir": model_dir, "device": device, "yield_age": yield_age}
    jobs = []
    for pol in ["dqn_mem", "dqn", "tabular_q"]:           # longest first for load balancing
        for p in P_VALUES[::-1]:
            for s in seeds:
                jobs.append({**base, "policy": pol, "train_p": p, "seed": s})
    for pol in ["rule_hold", "rule_nohold", "rule_memyield"]:
        for p in P_VALUES:
            for s in seeds:
                jobs.append({**base, "policy": pol, "train_p": p, "seed": s})

    t_pool = time.time()
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if r["policy"] in LEARNED:
                diag = [e for e in r["evals"] if np.isclose(e["eval_p"], r["train_p"])]
                s = " ".join(f"{e['mode']}={e['success_rate']:.3f}" for e in diag)
                print(f"[{k}/{len(jobs)}] {r['policy']:9s} p={r['train_p']:.2f} seed={r['seed']} "
                      f"train {r['train_seconds']:.0f}s  success {s}  ({(time.time() - t_start) / 60:.1f} min)",
                      flush=True)
    pool_minutes = (time.time() - t_pool) / 60

    # ------------------------------------------------------------ tables
    rows, curves = [], []
    for r in results:
        for e in r["evals"]:
            rows.append({"policy": r["policy"], "train_p": r["train_p"], "seed": r["seed"], **e})
        for c in r["train_info"].get("curve", []):
            curves.append({"policy": r["policy"], "train_p": r["train_p"], "seed": r["seed"], **c})
    allrows = pd.DataFrame(rows)
    main_rows = allrows[np.isclose(allrows.train_p, allrows.eval_p)].copy()
    main_rows["p"] = main_rows.eval_p
    main_rows = main_rows.sort_values(["policy", "mode", "p", "seed"])
    agg = main_rows.groupby(["policy", "mode", "p"])[METRICS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    summ = agg.reset_index()
    cross = allrows[allrows.policy.isin(LEARNED)].copy()
    curves = pd.DataFrame(curves)

    main_rows.to_csv(os.path.join(res_dir, "metrics_per_seed.csv"), index=False)
    summ.to_csv(os.path.join(res_dir, "summary.csv"), index=False)
    cross.groupby(["policy", "mode", "train_p", "eval_p"])[METRICS].mean().reset_index().to_csv(
        os.path.join(res_dir, "cross_p_generalization.csv"), index=False)
    curves.to_csv(os.path.join(res_dir, "training_curves.csv"), index=False)

    lower_bound = {m: float(np.mean([makespan_lower_bound(eval_specs(s, m, args.n_eval)) for s in seeds]))
                   for m in MODES}

    # ------------------------------------------------------------ figures
    plots.plot_sweep(summ, "success_rate", "Team success rate (%)",
                     "Team success vs. link degradation", os.path.join(fig_dir, "success_vs_p.png"), ylim=(0, 102),
                     n_seeds=len(seeds), n_eval=args.n_eval)
    plots.plot_sweep(summ, "collision_rate", "Episodes ending in a collision (%)",
                     "Collisions vs. link degradation", os.path.join(fig_dir, "collisions_vs_p.png"),
                     n_seeds=len(seeds), n_eval=args.n_eval)
    plots.plot_sweep(summ, "steps_success", "Steps until both robots reach the target",
                     "Time to target (successful episodes only) vs. link degradation",
                     os.path.join(fig_dir, "steps_vs_p.png"), pct=False, lower_bound=lower_bound,
                     n_seeds=len(seeds), n_eval=args.n_eval)
    for p in (0.75, 0.9):
        plots.plot_outcomes(summ, p, os.path.join(fig_dir, f"outcomes_p{int(p * 100):03d}.png"))
    plots.plot_training_curves(curves, os.path.join(fig_dir, "training_curves.png"))
    plots.plot_cross_p(cross, os.path.join(fig_dir, "cross_p_generalization.png"))
    example = make_example_figure(model_dir, fig_dir, yield_age, args.n_eval)

    # ------------------------------------------------------------ results.json
    def cell(pol, mode, p, metric):
        d = summ[(summ.policy == pol) & (summ["mode"] == mode) & np.isclose(summ.p, p)].iloc[0]
        return {"mean": round(float(d[f"{metric}_mean"]), 4), "std": round(float(d[f"{metric}_std"]), 4)}

    learned_train_s = sum(r["train_seconds"] for r in results if r["policy"] in LEARNED)
    headline = {}
    for mode in MODES:
        for p in (0.0, 0.75, 0.9):
            headline[f"{mode}_p{p:g}"] = {pol: {m: cell(pol, mode, p, m) for m in
                                                ("success_rate", "collision_rate", "timeout_rate", "steps_success")}
                                          for pol in POLICIES}
    res = {
        "project": "multi-agent-rl-degraded-comms",
        "question": "How much does losing the comm link hurt two cooperating robots, and do learned "
                    "policies cope better than the 'Hold when comms are degraded' rule?",
        "config": {
            "grid": "7x9", "max_steps": 40, "p_values": P_VALUES, "seeds": seeds,
            "train_layouts": TRAIN_POOL_SIZE, "held_out_test_layouts": TEST_POOL_SIZE,
            "eval_episodes_per_seed_per_scenario": args.n_eval,
            "tabular_train_episodes": args.tab_episodes, "dqn_train_env_steps": args.dqn_steps,
            "dqn_loss": "mse", "training_mix": "50% open, 50% close-quarters",
            "rewards": {"step": -0.1, "reach_target": 10.0, "collision": -10.0, "collision_ends_episode": True},
            "memory_yield_rule_yield_age": yield_age,
        },
        "memory_yield_selection_on_training_layouts": {str(k): round(v, 4) for k, v in yield_table.items()},
        "device_benchmark_ms": bench, "device_used": device,
        "runtime": {
            "total_wall_clock_minutes": round((time.time() - t_start) / 60, 2),
            "train_and_eval_pool_wall_clock_minutes": round(pool_minutes, 2),
            "learned_policy_training_cpu_minutes_summed": round(learned_train_s / 60, 2),
            "workers": args.workers, "machine": f"{platform.machine()} {platform.system()} "
                                                f"torch {torch.__version__}",
        },
        "shortest_path_makespan_lower_bound_steps": {k: round(v, 3) for k, v in lower_bound.items()},
        "headline": headline,
        "example_episode": example,
    }
    with open(os.path.join(res_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    write_tables(summ, os.path.join(res_dir, "tables.md"))
    print(f"done in {(time.time() - t_start) / 60:.1f} min "
          f"(pool {pool_minutes:.1f} min, summed training {learned_train_s / 60:.1f} CPU-min)")


def make_example_figure(model_dir, fig_dir, yield_age, n_eval, p=0.75):
    """Illustration only: first close-quarters test episode (seed 0) where the
    original rule, the keep-driving rule and DQN+memory do not all end the same way."""
    from src.dqn import QNet, DQNPolicy
    net = QNet()
    net.load_state_dict(torch.load(os.path.join(model_dir, f"dqn_mem_p{p:.2f}_s0.pt")))
    pols = {"rule_hold": make_rule("rule_hold", 0), "rule_nohold": make_rule("rule_nohold", 0),
            "dqn_mem": DQNPolicy(net, True)}
    specs = eval_specs(0, "close", n_eval)
    for idx, spec in enumerate(specs):
        runs = {k: run_episodes(pol, p, [spec], keep_traj=True) for k, pol in pols.items()}
        outcomes = {k: ("success" if v["success"][0] else "collision" if v["collision"][0] else "timeout")
                    for k, v in runs.items()}
        if len(set(outcomes.values())) == 3:
            note = (f"Illustration, not evidence: close-quarters test episode #{idx} (seed 0) - the first one in "
                    "which the three policies end three different ways.")
            plots.plot_example(spec, runs, p, os.path.join(fig_dir, "example_episode.png"), note)
            return {"episode_index": idx, "p": p, "outcomes": outcomes}
    return None


def write_tables(summ: pd.DataFrame, path: str):
    def fmt(pol, mode, p, metric, pct=True):
        d = summ[(summ.policy == pol) & (summ["mode"] == mode) & np.isclose(summ.p, p)].iloc[0]
        m, s = d[f"{metric}_mean"], d[f"{metric}_std"]
        if np.isnan(m):
            return "n/a"
        return f"{m * 100:.1f} ± {s * 100:.1f}" if pct else f"{m:.2f} ± {s:.2f}"

    lines = []
    for mode in MODES:
        lines.append(f"\n### Team success rate (%) — {mode}\n")
        lines.append("| Policy | " + " | ".join(f"p={p:g}" for p in P_VALUES) + " |")
        lines.append("|---|" + "---:|" * len(P_VALUES))
        for pol in POLICIES:
            lines.append(f"| {LABELS[pol]} | " + " | ".join(fmt(pol, mode, p, "success_rate") for p in P_VALUES) + " |")
    for p in (0.0, 0.9):
        for mode in MODES:
            lines.append(f"\n### Detail at p={p:g} — {mode}\n")
            lines.append("| Policy | Success % | Collision % | Timeout % | Steps (successful eps) | "
                         "Hold % when link down |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for pol in POLICIES:
                lines.append(f"| {LABELS[pol]} | {fmt(pol, mode, p, 'success_rate')} | "
                             f"{fmt(pol, mode, p, 'collision_rate')} | {fmt(pol, mode, p, 'timeout_rate')} | "
                             f"{fmt(pol, mode, p, 'steps_success', False)} | "
                             f"{fmt(pol, mode, p, 'hold_frac_link_down') if p > 0 else 'n/a'} |")
    for mode in MODES:
        lines.append(f"\n### What each policy does at a risky moment (p=0.9, {mode}): link down and the ally "
                     "was last seen within 2 cells\n")
        lines.append("| Policy | Hold % | Move downhill % | Other move (side-step/back) % | Bump into obstacle (stays put) % | "
                     "Risky decisions per seed | Of two downhill moves, picks the one AWAY from ally's last-known cell % |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for pol in POLICIES:
            d = summ[(summ.policy == pol) & (summ["mode"] == mode) & np.isclose(summ.p, 0.9)].iloc[0]
            away = (f"{d['near_down_steer_away_mean'] * 100:.1f} (n≈{d['near_down_steer_choices_mean']:.0f})"
                    if d["near_down_steer_choices_mean"] > 0 else "n/a")
            lines.append(f"| {LABELS[pol]} | " + " | ".join(f"{d[k + '_mean'] * 100:.1f}" for k in NEAR)
                         + f" | {d['near_down_decisions_mean']:.0f} | {away} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

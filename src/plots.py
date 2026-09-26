"""All figures. Static PNGs (matplotlib) for the README."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .experiment import LABELS, MODES, P_VALUES, POLICIES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
OBSTACLE = "#52514e"
# categorical slots in fixed order (validated palette), one per policy
COLORS = dict(zip(POLICIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARKERS = dict(zip(POLICIES, ["s", "D", "^", "o", "v", "P"]))
DASH = {p: ((0, (5, 2)) if p.startswith("rule") else "-") for p in POLICIES}
STATUS = {"success": "#0ca30c", "timeout": "#fab219", "collision": "#d03b3b"}
MODE_TITLE = {"open": "Open field (random starts)", "close": "Close quarters (robots start ≤ 2 cells apart)"}
MODE_SHORT = {"open": "Open field", "close": "Close quarters"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _sweep(ax, summ, mode, metric, pct=True):
    for pol in POLICIES:
        d = summ[(summ.policy == pol) & (summ["mode"] == mode)].sort_values("p")
        y, e = d[f"{metric}_mean"].to_numpy(), d[f"{metric}_std"].fillna(0).to_numpy()
        k = 100 if pct else 1
        ax.errorbar(d.p, y * k, yerr=e * k, color=COLORS[pol], lw=2, ls=DASH[pol], marker=MARKERS[pol],
                    ms=6.5, mec=SURFACE, mew=1.5, capsize=3, elinewidth=1.2, label=LABELS[pol], zorder=3)
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels([f"{p:g}" for p in P_VALUES])
    ax.set_xlabel("Link-drop probability per step, p")
    ax.set_title(MODE_TITLE[mode], loc="left")


def plot_sweep(summ: pd.DataFrame, metric: str, ylabel: str, title: str, path: str,
               pct=True, lower_bound=None, ylim=None, n_seeds=5, n_eval=500):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, mode in zip(axes, MODES):
        _sweep(ax, summ, mode, metric, pct)
        if lower_bound is not None:
            ax.axhline(lower_bound[mode], color=INK2, lw=1.2, ls=":", zorder=1,
                       label="Shortest-path lower bound (perfect link, no conflicts)")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.905, f"Mean ± std over {n_seeds} seeds; each seed = {n_eval} held-out episodes per "
             "scenario on obstacle layouts never seen in training.", color=INK2, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_outcomes(summ: pd.DataFrame, p: float, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, mode in zip(axes, MODES):
        d = summ[(summ["mode"] == mode) & (np.isclose(summ.p, p))].set_index("policy").loc[POLICIES[::-1]]
        left = np.zeros(len(d))
        for key in ("success", "timeout", "collision"):
            v = d[f"{key}_rate_mean"].to_numpy() * 100
            ax.barh(np.arange(len(d)), v, left=left, height=0.55, color=STATUS[key],
                    edgecolor=SURFACE, linewidth=2, label={"success": "Success", "timeout": "Timeout (ran out of steps)",
                                                           "collision": "Collision (crash)"}[key])
            for i, (x0, w) in enumerate(zip(left, v)):
                if w >= 9:
                    ax.text(x0 + w / 2, i, f"{w:.1f}%", ha="center", va="center", fontsize=8.5,
                            color="white" if key != "timeout" else INK)
            left += v
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels([LABELS[x] for x in d.index], fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of episodes")
        ax.grid(axis="y", visible=False)
        ax.set_title(MODE_SHORT[mode], loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"How episodes end when the link is down {int(p * 100)}% of the time (p = {p:g})",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(curves: pd.DataFrame, path: str):
    algos = ["tabular_q", "dqn", "dqn_mem"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    ramp = BLUES[2:]
    for ax, algo in zip(axes, algos):
        for k, p in enumerate(P_VALUES):
            d = curves[(curves.policy == algo) & np.isclose(curves.train_p, p)]
            g = d.groupby("episode").success_rate.agg(["mean", "std"]).reset_index()
            ax.plot(g.episode / 1000, g["mean"] * 100, color=ramp[k], lw=2, label=f"p = {p:g}")
        ax.set_title(LABELS[algo], loc="left")
        ax.set_xlabel("Training episodes (thousands)")
    axes[0].set_ylabel("Team success rate during training (%)")
    axes[-1].legend(loc="lower right", title="train link-drop p", fontsize=9)
    fig.suptitle("Learning curves (training episodes, 50/50 open + close-quarters, with ε-exploration)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cross_p(cross: pd.DataFrame, path: str):
    algos = ["tabular_q", "dqn", "dqn_mem"]
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for r, mode in enumerate(MODES):
        for c, algo in enumerate(algos):
            ax = axes[r, c]
            d = cross[(cross.policy == algo) & (cross["mode"] == mode)]
            m = d.pivot_table(index="train_p", columns="eval_p", values="success_rate", aggfunc="mean") * 100
            m = m.loc[P_VALUES, P_VALUES]
            im = ax.imshow(m.to_numpy(), cmap=cmap, vmin=50, vmax=100)
            for i in range(len(P_VALUES)):
                for j in range(len(P_VALUES)):
                    v = m.iat[i, j]
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                            color="white" if v > 80 else INK)
            ax.set_xticks(range(len(P_VALUES)))
            ax.set_xticklabels([f"{p:g}" for p in P_VALUES])
            ax.set_yticks(range(len(P_VALUES)))
            ax.set_yticklabels([f"{p:g}" for p in P_VALUES])
            ax.grid(False)
            ax.set_xlabel("Evaluated at link-drop p")
            if c == 0:
                ax.set_ylabel(f"{'Open field' if mode == 'open' else 'Close quarters'}\nTrained at link-drop p")
            ax.set_title(LABELS[algo], loc="left", fontsize=10.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("Success rate (%)")
    cb.outline.set_visible(False)
    fig.suptitle("Robustness: success when the real link is better or worse than the one trained on "
                 "(mean of 5 seeds; colour clipped at 50%)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_example(spec, runs: dict, p: float, path: str, note: str):
    """runs: {policy_name: output of run_episodes(..., keep_traj=True) on [spec]}"""
    lay = spec.layout
    names = list(runs)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.6))
    robot_col = ["#2a78d6", "#eb6834"]
    for ax, name in zip(axes, names):
        out = runs[name]
        traj, links = out["trajs"][0], out["links"][0]
        for (r, c) in lay.obstacles:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=OBSTACLE))
        tr, tc = spec.target
        ax.plot(tc, tr, marker="*", ms=20, color="#0ca30c", mec=SURFACE, mew=1.5, zorder=5)
        for i in (0, 1):
            pts = [pos[i] for pos in traj]
            ys = [q[0] + (i - 0.5) * 0.12 for q in pts]
            xs = [q[1] + (i - 0.5) * 0.12 for q in pts]
            ax.plot(xs, ys, color=robot_col[i], lw=2, alpha=0.9, zorder=3)
            ax.plot(xs[0], ys[0], "o", ms=10, color=robot_col[i], mec=SURFACE, mew=2, zorder=4)
            if pts[-1] != pts[0]:
                ax.plot(xs[-1], ys[-1], "X", ms=10, color=robot_col[i], mec=SURFACE, mew=1.5, zorder=4)
        ax.set_xlim(-0.5, lay.cols - 0.5)
        ax.set_ylim(lay.rows - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, lay.cols, 1))
        ax.set_yticks(np.arange(-0.5, lay.rows, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.set_aspect("equal")
        outcome = "success" if out["success"][0] else ("COLLISION" if out["collision"][0] else "timeout")
        steps = int(out["steps"][0])
        ups = sum(links[:steps])
        ax.set_title(f"{LABELS[name]}\n{outcome} after {steps} steps (link up {ups}/{steps})",
                     loc="left", fontsize=10)
    fig.suptitle(f"Same episode, same link outages (p = {p:g}). ● start, ✖ end, ★ target; "
                 "blue = robot A, orange = robot B", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.01, note, color=INK2, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)

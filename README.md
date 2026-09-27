# Multi-Agent RL Under Degraded Communications

Two simulated robots need to reach a shared target while their radio link keeps dropping. I compared the "hold when the link is down" rule from my Senior Design project with tabular Q-learning and DQN policies.

This grew out of my UMass Boston Senior Design project, where two Raspberry Pi robots move on a 7x9 grid and the controller makes them hold whenever the link degrades. I wrote the tabular Q-learning code for that project. Here I rebuilt the setup as a small simulator so I could test it on many maps and seeds. Everything below is simulation only; none of it ran on the robots.

![Success rate vs. link drop probability](figures/success_vs_p.png)

## Setup

- 7x9 grid with 6–10 obstacles, 2 robots and 1 shared target. Both robots move at the same time (hold, N, E, S, W).
- Each step the link is down with probability p (0, 0.25, 0.5, 0.75 or 0.9). A robot sees its partner's position only while the link is up.
- A collision ends the episode. Success means both robots reach the target within 40 steps.
- Two scenarios: open field (random starts) and close quarters (robots start within 2 cells of each other, so their paths overlap).
- 400 training layouts and 200 separate test layouts. Every policy is evaluated on the same 500 test episodes per scenario, with 5 seeds.

## Policies

| Policy | Idea |
|---|---|
| Hold rule (original) | Move greedily toward the target; hold whenever the link is down |
| Keep driving | Same rule, but never hold (ablation) |
| Memory rule | Hold only if the partner was last seen nearby, recently, and had right-of-way |
| Tabular Q-learning | One shared Q-table over local features |
| DQN | Double DQN with two hidden layers of 128 units |
| DQN + memory | Same DQN, plus the partner's last-known position and how old that sighting is |

## Results

Success rate (%) with the link down 90% of the time (p = 0.9), mean ± std over 5 seeds:

| Policy | Open field | Close quarters |
|---|---:|---:|
| Hold rule (original) | 17.6 ± 1.4 | 7.8 ± 1.4 |
| Keep driving | 94.7 ± 1.1 | 84.9 ± 1.7 |
| Memory rule | 97.6 ± 0.7 | 98.8 ± 0.4 |
| Tabular Q-learning | 93.4 ± 1.0 | 82.9 ± 3.3 |
| DQN | 94.7 ± 0.8 | 85.9 ± 2.1 |
| DQN + memory | 98.6 ± 0.8 | 96.0 ± 0.7 |

What I took from it:

- The hold rule never crashes, but it stalls. Up to p = 0.5 it still finishes every mission, just about twice as slowly (14.8 vs 7.4 steps). At p = 0.75 and 0.9 it mostly runs out of time.
- Most of the improvement comes from simply not holding. Plain DQN and tabular Q-learning were no better than the one-line "keep driving" rule.
- Remembering where the partner was last seen is what prevents crashes. In close quarters at p = 0.9, collisions dropped from 15.1% (keep driving) to 3.9% (DQN + memory).
- The DQN + memory policy almost never waits. When two moves are equally good, it picks the one that leads away from the partner's last-known cell 91.9% of the time, compared with about 50% for the other policies.
- The hand-written memory rule is still the best in close quarters with a very bad link, but it is slower (12.6 vs 8.2 steps at p = 0.9).
- A Q-table trained with a perfect link and then run at p = 0.9 behaves almost exactly like the hold rule (17.1% / 6.2%). It never saw a "link down" state, so those rows stayed at zero, and the argmax of all zeros is action 0, which is hold.

The full tables for every p, the cross-p robustness heatmap, learning curves and an example episode are in [docs/DETAILS.md](docs/DETAILS.md) and `results/`.

## What didn't work

- With Huber loss, both DQNs learned to never yield. Switching to MSE fixed it in a development check on training layouts.
- A Manhattan-distance greedy baseline kept getting stuck behind obstacles, so every policy uses shortest-path moves on the known map instead.
- My first Q-tables looped, because distance was bucketed too coarsely and the step size was constant. Exact distance (capped at 12) and a visit-count step size fixed most of it.

## Limitations

- Simulation only. Link drops are independent per step, while real links tend to fail in bursts.
- Perfect localisation, exactly two robots, and the robots know the static map.
- Differences of about 1 point (for example 98.6 vs 97.6) are within the noise across seeds.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiments.py --quick   # about 30 s smoke test
python run_experiments.py           # full run, about 6 min with 12 worker processes
```

Everything is seeded, so repeated runs give the same tables. The code is in `src/`: `env.py` (grid and link model), `policies.py` (rules), `features.py`, `tabular.py`, `dqn.py`, `evaluate.py` and `plots.py`.

---

Built with AI assistance (Claude). MIT License, Copyright (c) 2026 Murad Eyvazov.

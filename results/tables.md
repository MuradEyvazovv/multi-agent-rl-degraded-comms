
### Team success rate (%) — open

| Policy | p=0 | p=0.25 | p=0.5 | p=0.75 | p=0.9 |
|---|---:|---:|---:|---:|---:|
| Rule: hold when degraded (original) | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 79.2 ± 1.5 | 17.6 ± 1.4 |
| Rule: keep driving (ablation) | 100.0 ± 0.0 | 98.2 ± 0.5 | 96.8 ± 0.8 | 95.8 ± 0.6 | 94.7 ± 1.1 |
| Rule: last-known-ally yield | 100.0 ± 0.0 | 98.9 ± 0.4 | 97.8 ± 0.7 | 97.6 ± 0.5 | 97.6 ± 0.7 |
| Tabular Q-learning | 99.0 ± 0.4 | 98.5 ± 0.7 | 97.2 ± 1.0 | 95.6 ± 0.3 | 93.4 ± 1.0 |
| DQN | 99.6 ± 0.3 | 98.3 ± 0.5 | 97.2 ± 0.8 | 96.3 ± 0.8 | 94.7 ± 0.8 |
| DQN + last-known-ally memory | 99.6 ± 0.3 | 99.0 ± 0.4 | 98.9 ± 0.5 | 98.6 ± 0.6 | 98.6 ± 0.8 |

### Team success rate (%) — close

| Policy | p=0 | p=0.25 | p=0.5 | p=0.75 | p=0.9 |
|---|---:|---:|---:|---:|---:|
| Rule: hold when degraded (original) | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.8 ± 0.1 | 69.6 ± 3.1 | 7.8 ± 1.4 |
| Rule: keep driving (ablation) | 100.0 ± 0.0 | 95.5 ± 0.6 | 91.3 ± 1.7 | 86.6 ± 2.1 | 84.9 ± 1.7 |
| Rule: last-known-ally yield | 100.0 ± 0.0 | 97.8 ± 0.3 | 96.6 ± 0.9 | 98.0 ± 0.5 | 98.8 ± 0.4 |
| Tabular Q-learning | 96.9 ± 0.7 | 94.6 ± 1.0 | 92.0 ± 1.3 | 88.1 ± 1.3 | 82.9 ± 3.3 |
| DQN | 99.0 ± 0.7 | 95.6 ± 1.0 | 93.5 ± 1.3 | 88.4 ± 1.1 | 85.9 ± 2.1 |
| DQN + last-known-ally memory | 99.0 ± 0.7 | 98.6 ± 0.8 | 98.8 ± 0.5 | 97.0 ± 1.0 | 96.0 ± 0.7 |

### Detail at p=0 — open

| Policy | Success % | Collision % | Timeout % | Steps (successful eps) | Hold % when link down |
|---|---:|---:|---:|---:|---:|
| Rule: hold when degraded (original) | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 7.41 ± 0.09 | n/a |
| Rule: keep driving (ablation) | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 7.41 ± 0.09 | n/a |
| Rule: last-known-ally yield | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 7.41 ± 0.09 | n/a |
| Tabular Q-learning | 99.0 ± 0.4 | 0.1 ± 0.1 | 0.9 ± 0.5 | 7.49 ± 0.09 | n/a |
| DQN | 99.6 ± 0.3 | 0.0 ± 0.1 | 0.4 ± 0.4 | 7.42 ± 0.10 | n/a |
| DQN + last-known-ally memory | 99.6 ± 0.3 | 0.0 ± 0.1 | 0.4 ± 0.4 | 7.42 ± 0.10 | n/a |

### Detail at p=0 — close

| Policy | Success % | Collision % | Timeout % | Steps (successful eps) | Hold % when link down |
|---|---:|---:|---:|---:|---:|
| Rule: hold when degraded (original) | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8.67 ± 0.14 | n/a |
| Rule: keep driving (ablation) | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8.67 ± 0.14 | n/a |
| Rule: last-known-ally yield | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8.67 ± 0.14 | n/a |
| Tabular Q-learning | 96.9 ± 0.7 | 0.2 ± 0.3 | 2.9 ± 0.5 | 8.85 ± 0.14 | n/a |
| DQN | 99.0 ± 0.7 | 0.1 ± 0.1 | 0.8 ± 0.7 | 8.61 ± 0.17 | n/a |
| DQN + last-known-ally memory | 99.0 ± 0.7 | 0.1 ± 0.1 | 0.8 ± 0.7 | 8.61 ± 0.17 | n/a |

### Detail at p=0.9 — open

| Policy | Success % | Collision % | Timeout % | Steps (successful eps) | Hold % when link down |
|---|---:|---:|---:|---:|---:|
| Rule: hold when degraded (original) | 17.6 ± 1.4 | 0.0 ± 0.0 | 82.4 ± 1.4 | 28.06 ± 0.78 | 100.0 ± 0.0 |
| Rule: keep driving (ablation) | 94.7 ± 1.1 | 5.3 ± 1.1 | 0.0 ± 0.0 | 7.17 ± 0.08 | 0.0 ± 0.0 |
| Rule: last-known-ally yield | 97.6 ± 0.7 | 2.4 ± 0.7 | 0.0 ± 0.0 | 8.70 ± 0.08 | 13.5 ± 0.6 |
| Tabular Q-learning | 93.4 ± 1.0 | 5.2 ± 0.7 | 1.4 ± 0.5 | 7.72 ± 0.28 | 2.8 ± 0.8 |
| DQN | 94.7 ± 0.8 | 4.5 ± 0.7 | 0.8 ± 0.7 | 7.29 ± 0.20 | 1.4 ± 2.5 |
| DQN + last-known-ally memory | 98.6 ± 0.8 | 1.4 ± 0.7 | 0.1 ± 0.1 | 7.35 ± 0.16 | 0.5 ± 0.7 |

### Detail at p=0.9 — close

| Policy | Success % | Collision % | Timeout % | Steps (successful eps) | Hold % when link down |
|---|---:|---:|---:|---:|---:|
| Rule: hold when degraded (original) | 7.8 ± 1.4 | 0.0 ± 0.0 | 92.2 ± 1.4 | 31.93 ± 0.64 | 100.0 ± 0.0 |
| Rule: keep driving (ablation) | 84.9 ± 1.7 | 15.1 ± 1.7 | 0.0 ± 0.0 | 7.92 ± 0.08 | 0.0 ± 0.0 |
| Rule: last-known-ally yield | 98.8 ± 0.4 | 1.2 ± 0.4 | 0.0 ± 0.0 | 12.57 ± 0.13 | 29.8 ± 0.3 |
| Tabular Q-learning | 82.9 ± 3.3 | 15.2 ± 3.4 | 1.8 ± 0.9 | 8.60 ± 0.24 | 3.1 ± 1.6 |
| DQN | 85.9 ± 2.1 | 13.4 ± 2.4 | 0.7 ± 0.7 | 7.93 ± 0.17 | 1.4 ± 0.9 |
| DQN + last-known-ally memory | 96.0 ± 0.7 | 3.9 ± 0.6 | 0.1 ± 0.3 | 8.17 ± 0.24 | 0.6 ± 0.9 |

### What each policy does at a risky moment (p=0.9, open): link down and the ally was last seen within 2 cells

| Policy | Hold % | Move downhill % | Other move (side-step/back) % | Bump into obstacle (stays put) % | Risky decisions per seed | Of two downhill moves, picks the one AWAY from ally's last-known cell % |
|---|---:|---:|---:|---:|---:|---:|
| Rule: hold when degraded (original) | 100.0 | 0.0 | 0.0 | 0.0 | 5530 | n/a |
| Rule: keep driving (ablation) | 0.0 | 100.0 | 0.0 | 0.0 | 946 | 47.3 (n≈172) |
| Rule: last-known-ally yield | 50.2 | 49.8 | 0.0 | 0.0 | 1536 | 48.6 (n≈129) |
| Tabular Q-learning | 1.0 | 96.4 | 1.3 | 1.3 | 968 | 48.8 (n≈180) |
| DQN | 0.6 | 98.3 | 0.0 | 1.1 | 891 | 53.7 (n≈184) |
| DQN + last-known-ally memory | 0.9 | 94.9 | 1.9 | 2.3 | 795 | 87.6 (n≈141) |

### What each policy does at a risky moment (p=0.9, close): link down and the ally was last seen within 2 cells

| Policy | Hold % | Move downhill % | Other move (side-step/back) % | Bump into obstacle (stays put) % | Risky decisions per seed | Of two downhill moves, picks the one AWAY from ally's last-known cell % |
|---|---:|---:|---:|---:|---:|---:|
| Rule: hold when degraded (original) | 100.0 | 0.0 | 0.0 | 0.0 | 21735 | n/a |
| Rule: keep driving (ablation) | 0.0 | 100.0 | 0.0 | 0.0 | 2653 | 50.5 (n≈511) |
| Rule: last-known-ally yield | 54.8 | 45.2 | 0.0 | 0.0 | 4548 | 49.9 (n≈355) |
| Tabular Q-learning | 5.4 | 86.9 | 1.0 | 6.7 | 3087 | 53.3 (n≈539) |
| DQN | 2.0 | 94.4 | 0.1 | 3.5 | 2656 | 55.2 (n≈511) |
| DQN + last-known-ally memory | 1.4 | 91.8 | 3.1 | 3.7 | 2452 | 91.9 (n≈510) |

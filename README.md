# diayn-min

A minimal replication of **DIAYN** — *Diversity is All You Need: Learning Skills
without a Reward Function* (Eysenbach et al., 2018) — in plain PyTorch.

DIAYN learns a set of distinguishable skills with no task reward. A skill
`z - p(z)` is sampled at the start of each episode and held fixed. A
skill-conditioned SAC policy `pi(a | s, z)` is trained on the pseudo-reward

```
r(s, z) = log q(z | s') - log p(z)
```

where `q(z | s)` is a discriminator trained to recover the skill from visited
states. Maximizing this reward pushes the skills to visit distinguishable
regions of the state space.

## What's here

- `diayn/` — the core algorithm: networks, agent, replay buffer, the 2D
  navigation environment, evaluation and plotting helpers.
- `scripts/train.py` — train one run.
- `scripts/compare_ratios.py` — the **N:M asynchronous update rule** study on
  InvertedPendulum (N discriminator steps per M policy steps).
- `scripts/robustness.py` — multi-seed robustness study.
- `scripts/hierarchical.py` — the **downstream goal-reaching experiment** on
  pointnav (paper Fig. 6): are the discovered skills useful for a classical RL
  problem?
- `scripts/visualize.py` — skill trajectory plot + a GIF per skill.
- `scripts/plot_training.py` — training curves of a single run (Fig. 12).
- `config.json` — the single place to change every runnable setting.

`train.py` only writes `log.csv` + `checkpoint.pt`; pair it with `plot_training.py`
and `visualize.py` to get figures. `compare_ratios.py`, `robustness.py`,
`hierarchical.py` and `visualize.py` already produce their own figures.

## Setup

```
uv sync
```

## Usage

Everything reads `config.json`; edit it to change the experiment.

**2D navigation** (set `"env": "pointnav"`, `"n_skills": 6`, `"steps": 100000`):

```
uv run python scripts/train.py            # learn the skills
uv run python scripts/plot_training.py    # -> training_curves.png
uv run python scripts/visualize.py        # -> skills.png + a GIF per skill
```

**InvertedPendulum** (set `"env": "InvertedPendulum-v5"`, `"n_skills": 20`,
`"steps": 300000`, `"max_episode_steps": null`):

```
uv run python scripts/train.py
uv run python scripts/plot_training.py    # -> training_curves.png
uv run python scripts/visualize.py        # -> skills.png + a GIF per skill
```

**N:M update-ratio comparison** (uses the `ratios` list, always on
InvertedPendulum). Self-contained — trains every ratio and plots:

```
uv run python scripts/compare_ratios.py   # -> ratio_curves.png, ratio_returns.png
```

**Multi-seed robustness** (uses the `seeds` list, on `config["env"]`).
Self-contained — trains every seed and plots:

```
uv run python scripts/robustness.py       # -> seed_curves.png, seed_return_heatmap.png
```

**Downstream goal-reaching on pointnav** (paper Section 4.2.2 / Fig. 6 — needs a
trained pointnav model first):

```
uv run python scripts/train.py            # with env = pointnav
uv run python scripts/plot_training.py    # -> training_curves.png (optional)
uv run python scripts/hierarchical.py     # -> hierarchical_*.png
```

This freezes the discovered skills, adds the goal reward `r_g(s) = -||s - g||^2`,
and lets a meta-controller pick which skill to run. On pointnav one skill per goal
suffices (Appendix C.2), so the meta-controller greedily picks the skill that gets
closest to the goal. It writes `hierarchical_reward_vs_skills.png` (task reward
grows with the number of skills — and beats picking skills at random) and
`hierarchical_goals.png` (which skill is chosen for each of the 25 goals). A larger
`n_skills` (e.g. 20) makes the curve clearer.

Outputs (logs, checkpoints, figures, GIFs) are written under `runs/`.

## config.json

| key | meaning |
| --- | --- |
| `env` | `pointnav` or any gymnasium id (we use `InvertedPendulum-v5`) |
| `n_skills` | number of skills `|Z|` |
| `steps` | total environment steps |
| `seed` | random seed for the single-run scripts |
| `disc_updates` / `policy_updates` | the N:M update rule for `train.py` |
| `ratios` | `[N, M]` pairs swept by `compare_ratios.py` |
| `seeds` | seeds swept by `robustness.py` |
| `hierarchical` | `goal_grid` (NxN goals) and `subset_samples` for `hierarchical.py` |

The remaining keys (`gamma`, `tau`, `alpha`, `lr`, `batch_size`, `hidden`,
`replay_size`, `start_steps`, `update_after`, `max_episode_steps`, `log_every`,
`save_every`) are the standard SAC / training hyperparameters and follow the
paper's Appendix C.

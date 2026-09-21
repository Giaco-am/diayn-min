# diayn-min

Minimal PyTorch implementation of **DIAYN** (*Diversity Is All You Need*).
A skill-conditioned SAC policy learns distinguishable behaviors without task
rewards, using a discriminator reward: `log q(z | s′) - log p(z)`.
Includes 2D navigation (`pointnav`) and MuJoCo InvertedPendulum experiments.

## Quick start

Requires Python 3.10+ and `uv`. Run from the repository root:

```sh
uv sync
uv run python scripts/train.py
uv run python scripts/plot_training.py
uv run python scripts/visualize.py
```

Edit [config.json](config.json) to set the environment, skills, training steps,
seed, and discriminator:SAC update ratio. Results go to
`runs/{env}_{n_skills}skills_seed{seed}/`: logs, a checkpoint, training curves,
and skill plots/GIFs.

## Experiments

- `scripts/compare_ratios.py`: discriminator:SAC update-ratio comparison.
- `scripts/robustness.py`: multi-seed robustness study.
- `scripts/hierarchical.py`: downstream goal-reaching with learned skills.

Run these with `uv run python <script>`; settings are in `config.json`.

The **controlled multi-seed study** uses its own configuration and protocol:

```sh
uv run python scripts/controlled_experiment.py
uv run python scripts/plot_controlled.py --root runs/controlled_ratios_v1
```


# diayn-min

Minimal PyTorch implementation of **DIAYN** (*Diversity Is All You Need*).
A skill-conditioned SAC policy learns distinguishable behaviors without task
rewards, using a discriminator reward: `log q(z | s′) - log p(z)`.
Includes 2D navigation (`pointnav`) and MuJoCo InvertedPendulum experiments.

The project studies two extensions: how the **discriminator:SAC update schedule**
affects learning, and whether **continuous Dirichlet skills** produce gradual
changes in behavior while preserving diversity.

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

The **controlled multi-seed study** uses
[configs/controlled_ratios_v1.json](configs/controlled_ratios_v1.json) and its own
training and evaluation protocol:

```sh
uv run python scripts/controlled_experiment.py
uv run python scripts/plot_controlled.py --root runs/controlled_ratios_v1
```

## Continuous Dirichlet skills

The continuous variant replaces the categorical skill with a vector `z` whose
components are nonnegative and sum to one. With three components, the skill
space is a triangle (the simplex). Skills are sampled from a symmetric
Dirichlet prior and held fixed during each episode. The discriminator learns a
Dirichlet posterior; the intrinsic reward remains `log q(z | s′) - log p(z)`,
now using probability densities. The SAC policy and critics are shared with
the categorical implementation.

The question is whether nearby points in the simplex lead to nearby trajectories
and destinations in PointNav. A continuous skill input alone does not guarantee
smooth behavioral transitions.

### Training

The runner uses [config_dirichlet.json](config_dirichlet.json). Defaults are
PointNav, three skill components, 20,000 environment steps per run, a 100-step
episode horizon, and a 1:1 discriminator:SAC update schedule.

Run a two-condition pilot (uniform Dirichlet and curriculum, seed 10):

```sh
uv run python scripts/run_dirichlet_study.py
```

Available conditions:

| Condition | Skill prior and replay resets |
| --- | --- |
| `categorical` | Uniform categorical prior over one-hot skills. |
| `dirichlet_a0p05` | Fixed Dirichlet concentration 0.05, favoring the vertices. |
| `dirichlet_a1` | Fixed concentration 1, uniform over the simplex. |
| `dirichlet_a1_reset` | Concentration 1, with replay resets at 25%, 50% and 75% of training. |
| `curriculum_reset` | Concentration 0.05 → 0.1 → 0.3 → 1 across four equal training phases, with replay resets at phase changes. |

For example, compare categorical and uniform Dirichlet skills:

```sh
uv run python scripts/run_dirichlet_study.py \
  --conditions categorical dirichlet_a1 --seeds 10 --steps 20000
```

Run all five conditions with three training seeds:

```sh
uv run python scripts/run_dirichlet_study.py \
  --conditions categorical dirichlet_a0p05 dirichlet_a1 dirichlet_a1_reset curriculum_reset \
  --seeds 10 11 12 --steps 20000
```

### Evaluation and outputs

Each run is evaluated automatically after training. By default, evaluation uses
the same 256 uniformly sampled interior skills and 101 probes along each simplex
edge for every model. Each deterministic rollout starts at the PointNav center
and holds its skill fixed for the episode horizon.


Each invocation creates a new directory:

```text
runs_dirichlet_minimal/pilot_<timestamp>/
  summary.csv
  <condition>_seed<seed>/
    resolved_config.json
    log.csv
    checkpoint.pt
    evaluation/
      summary.json
      *_trajectories.npz
```

The Dirichlet study runner saves one final checkpoint per run. Evaluation
exports metrics and trajectories; historical experiment outputs are not
required.

"""Robustness across random seeds.

Training the same configuration with different seeds should give qualitatively
the same result: the discriminability curves should rise the same way and a
similar number of skills should incidentally solve the task. This trains one run
per seed in config["seeds"] and writes two figures:

  - seed_curves.png        training dynamics as a mean +/- std band across seeds
  - seed_return_heatmap.png each seed's per-skill task return, sorted best->worst

Run with:

    uv run python scripts/robustness.py
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diayn.agent import DIAYNAgent
from diayn.eval import skill_returns
from train import load_config, train_one

CURVES = [
    ("pseudo_reward", "pseudo-reward  log q(z|s) - log p(z)"),
    ("disc_acc", "discriminator accuracy"),
    ("policy_entropy", "policy entropy  H[a|s,z]"),
]


def read_log(run_dir):
    rows = list(csv.DictReader(open(run_dir / "log.csv")))
    steps = np.array([int(r["step"]) for r in rows])
    metrics = {k: np.array([float(r[k]) for r in rows]) for k, _ in CURVES}
    return steps, metrics


def main():
    config = load_config()
    env_name = config["env"]
    n_skills = config["n_skills"]
    out_dir = ROOT / f"runs/robustness_{env_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in config["seeds"]:
        run_dir = ROOT / f"runs/{env_name}_{n_skills}skills_seed{seed}"
        train_one(config, env_name, seed,
                  config["disc_updates"], config["policy_updates"], run_dir)
        runs.append((seed, run_dir))

    # training dynamics: mean +/- std across seeds, individual seeds underneath
    logs = [read_log(run_dir) for _, run_dir in runs]
    n = min(len(steps) for steps, _ in logs)
    steps = logs[0][0][:n]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, title) in zip(axes, CURVES):
        stacked = np.stack([m[key][:n] for _, m in logs])
        mean, std = stacked.mean(0), stacked.std(0)
        for s in stacked:
            ax.plot(steps, s, color="tab:blue", alpha=0.18, lw=0.8)
        ax.plot(steps, mean, color="tab:blue", lw=2, label="mean")
        ax.fill_between(steps, mean - std, mean + std, color="tab:blue",
                        alpha=0.25, label="+/- 1 std")
        ax.set(title=title, xlabel="environment steps")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"DIAYN training dynamics across {len(runs)} seeds ({env_name})")
    fig.tight_layout()
    fig.savefig(out_dir / "seed_curves.png", dpi=150)
    print(f"Saved {out_dir / 'seed_curves.png'}")

    # per-skill task return heatmap, sorted best->worst within each seed
    profiles, labels = [], []
    for seed, run_dir in runs:
        agent, payload = DIAYNAgent.load(str(run_dir / "checkpoint.pt"))
        returns = skill_returns(agent, payload["env"], payload["max_episode_steps"])
        profiles.append(np.sort(returns)[::-1])
        labels.append(f"seed{seed}")
        print(f"seed{seed}: best {returns.max():8.1f} | mean {returns.mean():8.1f} "
              f"| median {np.median(returns):8.1f}")

    mat = np.stack(profiles)
    fig, ax = plt.subplots(figsize=(0.45 * mat.shape[1] + 2, 0.5 * mat.shape[0] + 2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set(xlabel="skill rank within seed (best -> worst)", ylabel="seed",
           title=f"Per-skill task return across seeds ({env_name})")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(mat.shape[1]))
    fig.colorbar(im, ax=ax, label="task return (never seen in training)")
    fig.tight_layout()
    fig.savefig(out_dir / "seed_return_heatmap.png", dpi=150)
    print(f"Saved {out_dir / 'seed_return_heatmap.png'}")


if __name__ == "__main__":
    main()

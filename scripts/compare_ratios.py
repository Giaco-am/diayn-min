"""Compare discriminator:policy update ratios (the N:M asynchronous update rule).

Trains one run per ratio in config["ratios"] on InvertedPendulum (at config's
seed), then writes a comparison figure. 

Example usage:
    uv run python scripts/compare_ratios.py
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


from train import load_config, train_one

ENV = "InvertedPendulum-v5"
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
    seed = config["seed"]
    out_dir = ROOT / "runs/ratios"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for n, m in config["ratios"]:
        run_dir = out_dir / f"N{n}_M{m}_seed{seed}"
        train_one(config, ENV, seed, n, m, run_dir)
        runs.append(((n, m), run_dir))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(runs)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ((n, m), run_dir), color in zip(runs, colors):
        steps, metrics = read_log(run_dir)
        is_sync = (n, m) == (1, 1)
        for ax, (key, _title) in zip(axes, CURVES):
            ax.plot(steps, metrics[key], color="black" if is_sync else color,
                    lw=2.4 if is_sync else 1.8, ls="--" if is_sync else "-",
                    label=f"{n}:{m}" + ("  [sync]" if is_sync else ""))
    for ax, (_key, title) in zip(axes, CURVES):
        ax.set(title=title, xlabel="environment steps")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, title="disc:policy")
    fig.suptitle("DIAYN training dynamics vs discriminator:policy update ratio "
                 f"(InvertedPendulum, seed {seed})")
    fig.tight_layout()
    fig.savefig(out_dir / "ratio_curves.png", dpi=150)
    print(f"Saved {out_dir / 'ratio_curves.png'}")

    

if __name__ == "__main__":
    main()

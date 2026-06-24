"""Plot the training curves of a single run (paper Figure 12).

Reads the run described by config.json and saves training_curves.png: the
pseudo-reward log q(z|s') - log p(z), the discriminator accuracy and the policy
entropy H[a|s,z] against environment steps. The discriminability term keeps
rising while the entropy stays high, i.e. the skills become diverse without the
policy collapsing.

Run with (after training):

    uv run python scripts/plot_training.py
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import load_config

PANELS = [
    ("pseudo_reward", "pseudo-reward  log q(z|s) - log p(z)"),
    ("disc_acc", "discriminator accuracy"),
    ("policy_entropy", "policy entropy  H[a|s,z]"),
]


def main():
    config = load_config()
    run_dir = ROOT / f"runs/{config['env']}_{config['n_skills']}skills_seed{config['seed']}"

    rows = list(csv.DictReader(open(run_dir / "log.csv")))
    steps = [int(r["step"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, (key, title) in zip(axes, PANELS):
        ax.plot(steps, [float(r[key]) for r in rows])
        ax.set(title=title, xlabel="environment steps")
        ax.grid(alpha=0.3)
    fig.suptitle(f"DIAYN training: {run_dir.name}  [cf. Fig. 12]")
    fig.tight_layout()
    out = run_dir / "training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

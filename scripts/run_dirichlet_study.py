# === MODIFICA DIRICHLET: piccolo pilot sul trainer di Giaco, con risultati finali. ===
"""Default: uniform and curriculum, seed 10, same 20k budget as the completed study."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch

from train import ROOT, load_config, train_one
from evaluate_dirichlet import evaluate_checkpoint

CONDITIONS = ["categorical", "dirichlet_a0p05", "dirichlet_a1", "curriculum_reset", "dirichlet_a1_reset"]
METRICS = ["interior_coverage", "interior_separation", "interior_p95_endpoint_sensitivity",
           "interior_p95_trajectory_sensitivity", "max_adjacent_endpoint_distance", "mean_path_to_direct_ratio"]


# === MODIFICA DIRICHLET: CSV nuovi e confronto con le stesse run gia concluse. ===
def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_reference(output, rows, reference):
    with open(reference) as handle:
        previous = {r["run"]: r for r in csv.DictReader(handle)}
    comparisons = []
    for row in rows:
        old = previous.get(row["run"])
        if old is None or int(old["checkpoint_step"]) != row["checkpoint_step"]:
            print(f"{row['run']}: nessun confronto v2 allo stesso budget.")
            continue
        for metric in METRICS:
            current, historical = row[metric], float(old[metric])
            ratio = current / historical if current is not None and historical != 0 else None
            comparisons.append(dict(run=row["run"], metric=metric, historical=historical,
                                    current=current, ratio=ratio))
            print(f"{row['run']} | {metric}: nuovo={current}, v2={historical:.6g}")
    if comparisons:
        write_csv(output / "comparison_v2.csv", comparisons)


# === MODIFICA DIRICHLET: condizioni e seed selezionabili, senza resume o checkpoint intermedi. ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_dirichlet.json")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=["dirichlet_a1", "curriculum_reset"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[10])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs_dirichlet_minimal")
    parser.add_argument("--reference", type=Path, default=ROOT /
                        "runs_dirichlet_v2/study_pointnav_K3_20000steps/study_summary_v2.csv")
    args = parser.parse_args()
    base = load_config(args.config)
    if args.steps is not None:
        base["steps"] = args.steps
    torch.set_num_threads(base.get("torch_threads", 1))
    output = args.output_root / datetime.now().strftime("pilot_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    print(f"Risultati: {output}", flush=True)
    rows = []
    for seed in args.seeds:
        for condition in args.conditions:
            config = {**base, "seed": seed, "latent_type": "categorical" if condition == "categorical" else "dirichlet",
                      "prior_alpha": 0.05 if condition in ("dirichlet_a0p05", "curriculum_reset") else 1.0}
            starts = [1, base["steps"] // 4 + 1, base["steps"] // 2 + 1, 3 * base["steps"] // 4 + 1]
            config["prior_schedule"] = ([dict(start_step=s, alpha=a) for s, a in zip(starts, [.05, .1, .3, 1.0])]
                                        if condition == "curriculum_reset" else [])
            config["replay_reset_steps"] = starts[1:] if condition == "dirichlet_a1_reset" else []
            run = f"{condition}_seed{seed}"
            run_dir = train_one(config, config["env"], seed, 1, 1, output / run)
            print(f"Valutazione finale: {run}", flush=True)
            rows.append(dict(run=run, condition=condition, seed=seed,
                             **evaluate_checkpoint(run_dir / "checkpoint.pt")))
            write_csv(output / "summary.csv", rows)
    if args.reference.exists():
        compare_reference(output, rows, args.reference)
    print(f"Risultati salvati in {output}", flush=True)


if __name__ == "__main__":
    main()

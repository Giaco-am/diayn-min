"""Run or inspect the controlled multi-seed experiment; never resumes partial runs."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diayn.experiment import run_one, validate, verify_completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/controlled_ratios_v1.json')
    parser.add_argument('--smoke', action='store_true', help='120 steps, 20-step horizon, 2 eval episodes, 2 seeds; separate output')
    parser.add_argument('--output', type=Path, help='New output root; existing run directories are never overwritten')
    parser.add_argument('--status', action='store_true', help='Verify/report only, without training or analysis')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(experiment_id='controlled_ratios_smoke', steps=120, save_every=60,
            eval_every=60, max_episode_steps=20, evaluation_reset_seeds=[91000, 91001],
            seeds=[10, 11], batch_size=16, start_steps=16, update_after=16,
            replay_size=1000, log_every=30)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        config['output_dir'] = f'runs/controlled_ratios_smoke_{stamp}'
    if args.output:
        config['output_dir'] = str(args.output.resolve())
    root = Path(config['output_dir'])
    if not root.is_absolute():
        root = ROOT / root
    config['output_dir'] = str(root)
    validate(config)
    print(f'Experiment: {config["experiment_id"]}; output: {root}', flush=True)
    incomplete = []
    for n, m in config['ratios']:
        for seed in config['seeds']:
            run_dir = root / f'N{n}_M{m}' / f'seed{seed}'
            run_config = copy.deepcopy(config)
            run_config.update(seed=seed, ratio=[n, m], disc_updates=n, policy_updates=m)
            if run_dir.exists():
                valid, reason = verify_completed(run_dir, run_config)
                print(f'{run_dir}: {"SKIP" if valid else "INCOMPLETE — refusing overwrite"}: {reason}', flush=True)
                if not valid:
                    incomplete.append(str(run_dir))
                continue
            if args.status or args.analyze_only:
                print(f'{run_dir}: NOT STARTED', flush=True)
                continue
            print(f'START discriminator:SAC={n}:{m}, seed={seed}', flush=True)
            run_one(run_config, run_dir)
    if not args.status:
        from diayn.experiment_analysis import analyze
        analyze(root)
    if incomplete:
        print('Incomplete runs require inspection. Use a new output root for reruns; exact resume is unsupported.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

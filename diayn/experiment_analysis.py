"""Seed-level analysis; never pools deterministic/stochastic or historical runs."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'diayn-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .experiment import verify_completed, write_json

TRAIN = ['disc_acc', 'disc_loss', 'pseudo_reward', 'policy_entropy']
FRESH = ['mean_skill_return', 'full_horizon_skill_count', 'pooled_accuracy',
         'episode_skill_balanced_accuracy', 'pooled_nll', 'skill_mean_nll',
         'worst_skill_accuracy', 'worst_skill_nll']


def write_csv(path, rows):
    if rows:
        with path.open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def stats(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def load_runs(root, *, ratios=None, seeds=None):
    """Read verified controlled runs; return (runs, excluded) without plotting.

    Optional filters use ratio strings such as "1:5" and integer training seeds.
    No checkpoint is loaded and no environment is created.
    """
    root = Path(root)
    runs, excluded = [], []
    for path in sorted(root.glob('N*_M*/seed*')):
        if not path.is_dir():
            continue
        parts = path.parent.name[1:].split('_M')
        ratio = ':'.join(parts)
        if ratios is not None and ratio not in ratios:
            continue
        if seeds is not None and path.name not in {f'seed{s}' for s in seeds}:
            continue
        valid, reason = verify_completed(path)
        if not valid:
            excluded.append(dict(path=str(path), reason=reason))
            continue
        config = json.loads((path / 'resolved_config.json').read_text())
        with (path / 'log.csv').open() as f:
            log = [{k: float(v) if v else np.nan for k, v in row.items()}
                   for row in csv.DictReader(f)]
        evals = [json.loads(p.read_text()) for p in sorted(path.glob('evaluation_*.json'))]
        runs.append(dict(path=path, config=config, log=log, evals=evals,
                         ratio=':'.join(map(str, config['ratio'])), seed=config['seed']))
    if not runs:
        return runs, excluded
    
    protocols = [{k: v for k, v in r['config'].items()
                  if k not in ('ratio', 'seed', 'disc_updates', 'policy_updates', 'runtime')}
                 for r in runs]
    if any(p != protocols[0] for p in protocols[1:]):
        raise ValueError('Analysis root contains incompatible resolved configurations; keep protocols separate')
    implementations = []
    for run in runs:
        provenance = json.loads((run['path'] / 'provenance.json').read_text())
        implementations.append({k: v for k, v in provenance['source_sha256'].items()
                                if k in {f'diayn/{name}.py' for name in
                                    ('agent', 'networks', 'buffer', 'envs', 'controlled',
                                     'controlled_eval', 'experiment')}})
    if any(p != implementations[0] for p in implementations[1:]):
        raise ValueError('Analysis root contains different training/evaluation implementations; keep them separate')
    return runs, excluded


def analyze(root):
    root = Path(root)
    runs, excluded = load_runs(root)
    if not runs:
        print('No verified completed runs to analyze.')
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = root / 'analysis' / stamp
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'included_runs.json', dict(included=[str(r['path']) for r in runs], excluded=excluded))
    ratios = list(dict.fromkeys(r['ratio'] for r in runs))
    colors = dict(zip(ratios, plt.cm.tab10(np.arange(len(ratios)))))
    seeds = sorted({r['seed'] for r in runs})
    patterns = ['-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1))]
    styles = {seed: patterns[i % len(patterns)] for i, seed in enumerate(seeds)}
    seed_handles = [Line2D([0], [0], color='gray', lw=1, ls=styles[s], label=str(s)) for s in seeds]
    aggregate_rows = []
    for axis_key in ('step', 'training_seconds'):
        fig, axes = plt.subplots(1, 4, figsize=(17, 4))
        # Use a single common time interval across ALL runs and ratios
        valid_logs = [[p for p in r['log'] if np.isfinite(p['disc_acc'])] for r in runs]
        if not all(valid_logs):
            plt.close(fig)
            continue
        low = max(rows[0][axis_key] for rows in valid_logs)
        high = min(rows[-1][axis_key] for rows in valid_logs)
        if high < low:
            plt.close(fig)
            continue
        grid = (np.linspace(low, high, 150) if axis_key == 'training_seconds'
                else np.array(sorted({p['step'] for rows in valid_logs for p in rows if low <= p['step'] <= high})))
        for ratio in ratios:
            group = [r for r in runs if r['ratio'] == ratio]
            for ax, metric in zip(axes, TRAIN):
                curves = []
                for run in group:
                    rows = [r for r in run['log'] if np.isfinite(r[metric])]
                    x, y = [r[axis_key] for r in rows], [r[metric] for r in rows]
                    ax.plot(x, y, color=colors[ratio], alpha=.4, lw=.8, ls=styles[run['seed']])
                    curves.append(np.interp(grid, x, y))
                values = np.asarray(curves)
                mean = values.mean(0)
                sd = values.std(0, ddof=1) if len(group) > 1 else np.zeros(len(grid))
                ax.plot(grid, mean, color=colors[ratio], label=f'{ratio} (n={len(group)})', lw=2)
                if len(group) > 1:
                    ax.fill_between(grid, mean-sd, mean+sd, color=colors[ratio], alpha=.12)
                for x, mu, sig in zip(grid, mean, sd):
                    aggregate_rows.append(dict(axis=axis_key, budget=float(x), ratio=ratio,
                        metric=metric, mean=float(mu), seed_sd=float(sig), n_seeds=len(group)))
        for ax, metric in zip(axes, TRAIN):
            ax.set(xlabel=axis_key, title=metric)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        axes[1].legend(handles=seed_handles, title='Training seed', fontsize=7)
        fig.suptitle('Training: faint individual seeds; mean ±1 seed SD (not CI). Time curves interpolated on shared support.')
        finish(fig, out / f'training_{axis_key}.png')
    write_csv(out / 'training_aggregates.csv', aggregate_rows)
    eval_rows, skill_rows, episode_rows, seed_aggregates, deltas = [], [], [], [], []
    for mode in ('deterministic', 'stochastic'):
        fig, axes = plt.subplots(2, 4, figsize=(17, 8))
        for ratio in ratios:
            group = [r for r in runs if r['ratio'] == ratio]
            steps = sorted({e['step'] for r in group for e in r['evals']})
            for ax, metric in zip(axes.flat, FRESH):
                for r in group:
                    ax.plot([e['step'] for e in r['evals']],
                            [e['modes'][mode]['summary'][metric] for e in r['evals']],
                            color=colors[ratio], alpha=.4, lw=.8, marker='.', ls=styles[r['seed']])
                means, sds = [], []
                for step in steps:
                    vals = [e['modes'][mode]['summary'][metric] for r in group for e in r['evals'] if e['step'] == step]
                    mu, sd = stats(vals)
                    means.append(mu)
                    sds.append(sd)
                    seed_aggregates.append(dict(mode=mode, ratio=ratio, step=step, metric=metric,
                        mean=mu, seed_sd=sd, n_seeds=len(vals)))
                ax.plot(steps, means, color=colors[ratio], label=f'{ratio} (n={len(group)})', lw=2)
                if len(group) > 1:
                    ax.fill_between(steps, np.array(means)-sds, np.array(means)+sds,
                                    color=colors[ratio], alpha=.12)
        for ax, metric in zip(axes.flat, FRESH):
            ax.set(xlabel='environment steps', title=metric)
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        axes.flat[1].legend(handles=seed_handles, title='Training seed', fontsize=7)
        fig.suptitle(f'Fresh {mode}: individual seeds and mean ±1 seed SD (not CI)')
        finish(fig, out / f'fresh_{mode}.png')
        for r in runs:
            # Skill identities are local to each trained run; do not pool slot z across seeds.
            fig, axes = plt.subplots(1, 3, figsize=(13, 6))
            for ax, metric in zip(axes, ('mean_return', 'accuracy', 'nll')):
                matrix = np.array([[s[metric] for s in e['modes'][mode]['skills']] for e in r['evals']]).T
                im = ax.imshow(matrix, aspect='auto', origin='lower', interpolation='nearest',
                               vmin=0, vmax=(1 if metric == 'accuracy' else None))
                ax.set(title=metric, xlabel='checkpoint environment steps', ylabel='skill')
                ax.set_xticks(range(len(r['evals'])), [e['step'] for e in r['evals']], rotation=35)
                fig.colorbar(im, ax=ax)
            fig.suptitle(f'{mode}, disc:SAC {r["ratio"]}, training seed {r["seed"]}; un-clipped NLL')
            finish(fig, out / f'skills_{mode}_{r["ratio"].replace(":", "_")}_seed{r["seed"]}.png')
            for e in r['evals']:
                prefix = dict(mode=mode, ratio=r['ratio'], seed=r['seed'], step=e['step'])
                summary = dict(e['modes'][mode]['summary'])
                summary.pop('transition_skill_frequencies')  # Full vector retained in evaluation JSON.
                eval_rows.append(dict(**prefix, **summary))
                skill_rows.extend(dict(**prefix, **s) for s in e['modes'][mode]['skills'])
                episode_rows.extend(dict(**prefix, **s) for s in e['modes'][mode]['episodes'])
            by_step = {e['step']: e['modes'][mode]['summary'] for e in r['evals']}
            if 150000 in by_step and 200000 in by_step:
                for metric in FRESH:
                    deltas.append(dict(mode=mode, ratio=r['ratio'], seed=r['seed'], metric=metric,
                        change_150k_to_200k=by_step[200000][metric]-by_step[150000][metric]))
    write_csv(out / 'evaluation_summary.csv', eval_rows)
    write_csv(out / 'per_skill.csv', skill_rows)
    write_csv(out / 'per_episode.csv', episode_rows)
    write_csv(out / 'evaluation_seed_aggregates.csv', seed_aggregates)
    write_csv(out / 'changes_150k_to_200k.csv', deltas)
    delta_aggregates = []
    if deltas:
        for mode in ('deterministic', 'stochastic'):
            fig, axes = plt.subplots(2, 4, figsize=(17, 8))
            for ax, metric in zip(axes.flat, FRESH):
                for i, ratio in enumerate(ratios):
                    vals = [d['change_150k_to_200k'] for d in deltas if d['mode'] == mode and d['metric'] == metric and d['ratio'] == ratio]
                    if not vals:
                        continue
                    mu, sd = stats(vals)
                    ax.scatter(i + np.linspace(-.12, .12, len(vals)), vals, color=colors[ratio], alpha=.6)
                    ax.errorbar(i, mu, yerr=sd if len(vals) > 1 else None, color='black', fmt='o', capsize=5)
                    delta_aggregates.append(dict(mode=mode, ratio=ratio, metric=metric,
                        mean_change=mu, seed_sd=sd, n_seeds=len(vals)))
                ax.axhline(0, color='gray', lw=.8)
                ax.set(title=metric, ylabel='200k minus 150k')
                ax.set_xticks(range(len(ratios)), ratios)
            fig.suptitle(f'{mode}: paired changes within training seeds; mean ±1 seed SD (not CI)')
            finish(fig, out / f'changes_150k_to_200k_{mode}.png')
        write_csv(out / 'change_seed_aggregates.csv', delta_aggregates)
    expected = len(runs[0]['config']['ratios']) * len(runs[0]['config']['seeds'])
    lines = ['# Controlled discriminator:SAC comparison', '',
        f'{len(runs)} / {expected} planned runs verified complete; {len(excluded)} existing runs excluded.', '',
        'Faint curves/dots are individual training seeds. Bands/error bars are ±1 sample SD across training seeds, not confidence intervals. '
        'Evaluation episodes and training windows are not independent training replicates. '
        'Skill indices have no shared semantic identity across training seeds.', '',
        'Primary budget: environment steps. Training-time curves use linear interpolation only within the common time support of all included runs. '
        'Training time includes environment interaction and updates; initialization, checkpoint/log I/O and evaluation are excluded. '
        'Evaluation time includes saved-model loading and rollout/inference. Unequal update budgets are not equal computation.', '',
        '| Mode | Ratio | Last step | Seeds | Mean task return ± SD | Full-horizon skills ± SD | Worst-skill NLL ± SD |',
        '|---|---|---:|---:|---:|---:|---:|']
    for mode in ('deterministic', 'stochastic'):
        for ratio in ratios:
            rows = [s for s in seed_aggregates if s['mode'] == mode and s['ratio'] == ratio]
            step = max(s['step'] for s in rows)
            last = {s['metric']: s for s in rows if s['step'] == step}
            vals = [last[k] for k in ('mean_skill_return', 'full_horizon_skill_count', 'worst_skill_nll')]
            cells = ' | '.join(f'{s["mean"]:.3f} ± {s["seed_sd"]:.3f}' for s in vals)
            lines.append(f'| {mode} | {ratio} | {step} | {vals[0]["n_seeds"]} | {cells} |')
    lines.extend(['', '150k→200k paired change files/figures are available.' if deltas else
                  '150k→200k changes are unavailable until both checkpoints exist; smoke runs cannot assess convergence.', '',
                  'No best-skill-only criterion is used. Do not pool these controlled schedules with historical runs.'])
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(f'Analysis saved to {out}', flush=True)
    return out

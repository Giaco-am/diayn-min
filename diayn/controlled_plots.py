"""Reusable plots for controlled log.csv/evaluation JSON data, with no rollouts.

Each public plot function returns (matplotlib Figure, tables), where tables maps
names to lists of numeric records. Callers own the figure and may customize it.
Only training seeds are replication units. No function smooths or clips losses.
"""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

# Backend/cache configuration is shared with the existing noninteractive analyzer.
from .experiment_analysis import load_runs, plt, np, Line2D

MODES = ('deterministic', 'stochastic')
KINDS = ('training', 'fresh', 'skills', 'discriminator', 'outcomes', 'agreement', 'changes', 'budget')
TRAIN = ('disc_acc', 'disc_loss', 'pseudo_reward', 'policy_entropy')
FRESH = ('mean_skill_return', 'median_skill_return', 'full_horizon_episode_fraction',
         'full_horizon_skill_count', 'pooled_accuracy', 'episode_skill_balanced_accuracy',
         'pooled_nll', 'worst_skill_nll')
LABELS = dict(disc_acc='Replay discriminator accuracy', disc_loss='Replay discriminator NLL (nats)',
    pseudo_reward='Replay intrinsic reward (nats)', policy_entropy='Action entropy (nats)',
    mean_skill_return='Mean skill task return', median_skill_return='Median skill task return',
    full_horizon_episode_fraction='Episodes reaching the horizon', full_horizon_skill_count='Skills: all episodes reach horizon',
    pooled_accuracy='Fresh transition-pooled accuracy', episode_skill_balanced_accuracy='Fresh episode/skill-balanced accuracy',
    pooled_nll='Fresh transition-pooled NLL (nats)', worst_skill_nll='Worst skill mean NLL (nats)',
    worst_skill_accuracy='Worst skill accuracy', skill_mean_nll='Fresh equal-skill NLL (nats)',
    mean_return='Mean task return', accuracy='Discriminator accuracy', nll='NLL (nats)',
    intrinsic_reward='Intrinsic reward (nats)', min_true_log_probability='Minimum true-skill log probability',
    training_seconds='Training time (seconds)', evaluation_seconds='Evaluation time (seconds)',
    disc_updates='Discriminator updates', critic_updates='Joint critic updates',
    actor_updates='Actor updates', target_updates='Target updates')
COLORS = {'1:1': '#0072B2', '1:5': '#D55E00', '5:1': '#009E73'}
SPREAD = 'Faint lines = training seeds; thick lines = mean; bands = ±1 seed SD, not CI'


def _style(runs):
    ratios = sorted({r['ratio'] for r in runs}, key=lambda r: tuple(map(int, r.split(':'))))
    seeds = sorted({r['seed'] for r in runs})
    patterns = ['-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1))]
    colors = {r: COLORS.get(r, plt.cm.tab10(i % 10)) for i, r in enumerate(ratios)}
    styles = {s: patterns[i % len(patterns)] for i, s in enumerate(seeds)}
    return ratios, seeds, colors, styles


def _panels(count, title, cols=4):
    cols = min(cols, count)
    fig, axes = plt.subplots(math.ceil(count/cols), cols, figsize=(4.3*cols, 3.8*math.ceil(count/cols)), squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12)
    return fig, list(axes.flat[:count])


def _decorate(ax, metric, xlabel='Environment steps'):
    ax.set(title=LABELS.get(metric, metric), xlabel=xlabel)
    ax.grid(alpha=.18)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.spines[['top', 'right']].set_visible(False)


def _legends(axes, runs):
    ratios, seeds, colors, styles = _style(runs)
    axes[0].legend(handles=[Line2D([0], [0], color=colors[r], lw=2, label=r) for r in ratios],
                   title='Discriminator:SAC', fontsize=8)
    if len(axes) > 1:
        axes[1].legend(handles=[Line2D([0], [0], color='gray', ls=styles[s], label=str(s)) for s in seeds],
                       title='Training seed', fontsize=8)


def _band(ax, x, values, color):
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.full(mean.shape, np.nan)
    ax.plot(x, mean, color=color, lw=2)
    if len(values) > 1:
        ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.15)
    return mean, sd


def _evals(run, checkpoints=None):
    return [e for e in run['evals'] if checkpoints is None or e['step'] in checkpoints]


def _evaluation(run, step=None):
    candidates = _evals(run, None if step is None else [step])
    if not candidates:
        raise ValueError(f'No evaluation at {step} for {run["path"]}')
    return max(candidates, key=lambda e: e['step'])


def _identity(run, mode, step):
    return dict(ratio=run['ratio'], seed=run['seed'], mode=mode, step=step)


def plot_training_curves(runs, x='step', metrics=TRAIN, shared_time_points=150):
    """Exact common step grid, or interpolated COMMON time support; no smoothing.

    At step budgets, no interpolation is used. Time-grid endpoints are shared by
    every selected seed/ratio separately for each metric. Individual traces remain
    visible outside that support; shaded aggregate regions never extrapolate.
    """
    if x not in ('step', 'training_seconds'):
        raise ValueError('x must be step or training_seconds')
    fig, axes = _panels(len(metrics), f'Training — {SPREAD}')
    ratios, _, colors, styles = _style(runs)
    records = []
    for ax, metric in zip(axes, metrics):
        series = [[p for p in r['log'] if np.isfinite(p.get(metric, np.nan)) and np.isfinite(p.get(x, np.nan))] for r in runs]
        for run, rows in zip(runs, series):
            ax.plot([p[x] for p in rows], [p[metric] for p in rows],
                    color=colors[run['ratio']], ls=styles[run['seed']], lw=.8, alpha=.45)
        grid = []
        if series and all(series):
            if x == 'step':
                grid = sorted(set.intersection(*[{p[x] for p in rows} for rows in series]))
            else:
                lo, hi = max(p[0][x] for p in series), min(p[-1][x] for p in series)
                if hi >= lo:
                    grid = np.linspace(lo, hi, shared_time_points)
        if len(grid):
            for ratio in ratios:
                group = [(r, rows) for r, rows in zip(runs, series) if r['ratio'] == ratio]
                values = [np.interp(grid, [p[x] for p in rows], [p[metric] for p in rows]) for _, rows in group]
                mean, sd = _band(ax, grid, values, colors[ratio])
                records.extend(dict(axis=x, budget=float(t), ratio=ratio, metric=metric,
                    mean=float(mu), seed_sd=float(sig), n_seeds=len(group)) for t, mu, sig in zip(grid, mean, sd))
        else:
            ax.text(.5, .5, 'No common observed budget', transform=ax.transAxes, ha='center')
        _decorate(ax, metric, 'Environment steps' if x == 'step' else 'Elapsed training time (seconds)')
        if x == 'training_seconds':
            ax.set_ylabel('Aggregates only on shared time support')
    _legends(axes, runs)
    return fig, {'aggregates': records}


def plot_fresh_evaluation(runs, mode='deterministic', checkpoints=None, metrics=FRESH):
    """Equal weight per training seed, deterministic/stochastic kept separate."""
    fig, axes = _panels(len(metrics), f'Fresh {mode} evaluation — {SPREAD}')
    ratios, _, colors, styles = _style(runs)
    records = []
    for ax, metric in zip(axes, metrics):
        for ratio in ratios:
            group = [r for r in runs if r['ratio'] == ratio]
            for run in group:
                ev = _evals(run, checkpoints)
                ax.plot([e['step'] for e in ev], [e['modes'][mode]['summary'][metric] for e in ev],
                        color=colors[ratio], ls=styles[run['seed']], alpha=.45, lw=.8, marker='.')
            # Common checkpoints keep the seed cohort fixed along each aggregate curve.
            common = sorted(set.intersection(*[{e['step'] for e in _evals(r, checkpoints)} for r in group]))
            if common:
                values = [[next(e['modes'][mode]['summary'][metric] for e in r['evals'] if e['step'] == s) for s in common] for r in group]
                mean, sd = _band(ax, common, values, colors[ratio])
                records.extend(dict(mode=mode, ratio=ratio, step=s, metric=metric, mean=float(mu),
                                    seed_sd=float(sig), n_seeds=len(group)) for s, mu, sig in zip(common, mean, sd))
        _decorate(ax, metric)
    _legends(axes, runs)
    return fig, {'aggregates': records}


def plot_skill_heatmaps(run, mode='deterministic', checkpoints=None):
    """Skill rows stay in numerical order within a run; signed rewards preserved."""
    ev = _evals(run, checkpoints)
    if not ev:
        raise ValueError('No selected evaluations')
    metrics = ('mean_return', 'accuracy', 'nll', 'intrinsic_reward')
    fig, axes = _panels(4, f'{mode} • disc:SAC {run["ratio"]} • seed {run["seed"]} — skill identity is local to this run')
    rows = []
    skills = sorted(s['skill'] for s in ev[0]['modes'][mode]['skills'])
    for ax, metric in zip(axes, metrics):
        matrix = np.array([[next(s[metric] for s in e['modes'][mode]['skills'] if s['skill'] == z) for e in ev] for z in skills])
        kwargs = dict(vmin=0, vmax=1) if metric == 'accuracy' else {}
        # Never force a zero lower bound on signed intrinsic reward or task return.
        im = ax.imshow(matrix, aspect='auto', origin='lower', interpolation='nearest', **kwargs)
        ax.set(title=LABELS[metric], xlabel='Checkpoint environment steps', ylabel='Skill')
        ax.set_yticks(range(len(skills)), skills, fontsize=7)
        ax.set_xticks(range(len(ev)), [e['step'] for e in ev], rotation=35)
        fig.colorbar(im, ax=ax, shrink=.8)
    for e in ev:
        rows.extend(dict(**_identity(run, mode, e['step']), **s) for s in e['modes'][mode]['skills'])
    return fig, {'skills': rows}


def plot_discriminator_diagnostics(run, mode='deterministic', step=None):
    """Row-normalized confusion, transition frequencies and un-clipped tail NLL."""
    e = _evaluation(run, step)
    data = e['modes'][mode]
    identity = _identity(run, mode, e['step'])
    matrix = np.asarray(data['confusion_matrix'], dtype=float)
    totals = matrix.sum(axis=1)
    normalized = np.divide(matrix, totals[:, None], out=np.full_like(matrix, np.nan), where=totals[:, None] > 0)
    skills = sorted(data['skills'], key=lambda s: s['skill'])
    z = [s['skill'] for s in skills]
    fig, axes = _panels(4, f'{mode} • disc:SAC {run["ratio"]} • seed {run["seed"]} • step {e["step"]}', cols=2)
    im = axes[0].imshow(normalized, origin='lower', interpolation='nearest', vmin=0, vmax=1)
    axes[0].set(title='Confusion: normalized within true skill', xlabel='Predicted skill', ylabel='True skill')
    axes[0].set_xticks(z, z, fontsize=7, rotation=45)
    axes[0].set_yticks(z, z, fontsize=7)
    fig.colorbar(im, ax=axes[0], shrink=.8)
    freq = data['summary']['transition_skill_frequencies']
    axes[1].bar(z, freq, color='#0072B2')
    axes[1].axhline(1/len(z), color='black', ls=':', label='Uniform skill frequency')
    axes[1].set(title='Empirical transition-level skill frequencies', ylabel='Fraction of transitions', xlabel='True skill')
    axes[1].legend(fontsize=8)
    axes[2].bar(z, [s['nll'] for s in skills], color='#D55E00', label='Mean NLL within skill')
    axes[2].plot(z, [-s['min_true_log_probability'] for s in skills], 'k.', label='Maximum transition NLL')
    axes[2].set(title='Log loss: extreme mistakes remain visible', ylabel='NLL (nats), no clipping', xlabel='Skill')
    axes[2].legend(fontsize=8)
    axes[3].bar(z, [s['accuracy'] for s in skills], color='#009E73')
    axes[3].set(title='Per-skill transition accuracy', ylabel='Accuracy', xlabel='Skill', ylim=(0, 1.05))
    for ax in axes[1:]:
        ax.set_xticks(z, z, fontsize=7, rotation=45)
        ax.grid(axis='y', alpha=.15)
    confusion = [dict(**identity, true_skill=i, predicted_skill=j, count=int(matrix[i, j]),
        within_skill_fraction=float(normalized[i, j])) for i in range(len(matrix)) for j in range(len(matrix))]
    diagnostics = [dict(**identity, **s, transition_frequency=freq[s['skill']]) for s in skills]
    return fig, {'confusion': confusion, 'skills': diagnostics}


def plot_episode_outcomes(run, mode='deterministic', step=None):
    """Episode dots are diagnostics, never independent training-seed replicates.

    Outcome flags use mutually exclusive categories; simultaneous termination and
    truncation is shown explicitly instead of double-counting an episode.
    """
    e = _evaluation(run, step)
    data = e['modes'][mode]
    fig, axes = _panels(3, f'{mode} • {run["ratio"]} • seed {run["seed"]} • step {e["step"]} — episodes are descriptive, not training replicates', cols=3)
    outcomes = ('terminated only', 'truncated only', 'both', 'neither')
    color = ('#D55E00', '#0072B2', '#CC79A7', '#999999')
    ids = sorted(s['skill'] for s in data['skills'])
    counts = np.zeros((4, len(ids)))
    for j, z in enumerate(ids):
        episodes = [p for p in data['episodes'] if p['skill'] == z]
        offsets = np.linspace(-.2, .2, len(episodes))
        for ax, metric in zip(axes[:2], ('task_return', 'length')):
            values = [p[metric] for p in episodes]
            ax.scatter(z+offsets, values, s=12, alpha=.6, color='#0072B2')
            ax.plot(z, np.mean(values), 'k_', ms=10)
        for ep in episodes:
            key = 2 if ep['terminated'] and ep['truncated'] else 0 if ep['terminated'] else 1 if ep['truncated'] else 3
            counts[key, j] += 1/len(episodes)
    bottom = np.zeros(len(ids))
    for label, c, values in zip(outcomes, color, counts):
        axes[2].bar(ids, values, bottom=bottom, label=label, color=c)
        bottom += values
    axes[2].legend(fontsize=7)
    for ax, title in zip(axes, ('Task return: episodes and skill means', 'Episode length: episodes and skill means', 'Termination and truncation fractions')):
        ax.set(title=title, xlabel='Skill')
        ax.set_xticks(ids, ids, fontsize=7, rotation=45)
        ax.grid(axis='y', alpha=.15)
    axes[2].set_ylim(0, 1.05)
    if 'max_episode_steps' in run['config']:
        axes[1].axhline(run['config']['max_episode_steps'], color='gray', ls=':', label='Horizon')
        axes[1].legend(fontsize=8)
    return fig, {'episodes': [dict(**_identity(run, mode, e['step']), **p) for p in data['episodes']]}


def plot_replay_fresh_agreement(runs, mode='deterministic', checkpoints=None):
    """Pair fresh measurements with the EXACT checkpoint log window, no interpolation.

    Replay discriminator measurements are pre-optimization; fresh ones use the
    saved model. Differences are descriptive distribution/measurement differences.
    """
    pairs = [('disc_acc', 'pooled_accuracy'), ('disc_acc', 'episode_skill_balanced_accuracy'),
             ('disc_loss', 'pooled_nll'), ('pseudo_reward', None)]
    fig, axes = _panels(4, f'{mode}: replay log window versus fresh checkpoint — dots are seed/checkpoint pairs, not replicates')
    _, _, colors, _ = _style(runs)
    records = []
    markers = ['o', 's', '^', 'D', 'v']
    seeds = sorted({r['seed'] for r in runs})
    for ax, (training, fresh) in zip(axes, pairs):
        all_values = []
        for run in runs:
            logs = {int(p['step']): p for p in run['log']}
            for e in _evals(run, checkpoints):
                log = logs.get(e['step'])
                if log is None or not np.isfinite(log.get(training, np.nan)):
                    continue
                summary = e['modes'][mode]['summary']
                y = (math.log(len(e['modes'][mode]['skills'])) - summary['pooled_nll']) if fresh is None else summary[fresh]
                x = log[training]
                ax.scatter(x, y, color=colors[run['ratio']], marker=markers[seeds.index(run['seed']) % len(markers)], s=28, alpha=.7)
                all_values.extend([x, y])
                records.append(dict(**_identity(run, mode, e['step']), replay_metric=training,
                    fresh_metric=fresh or 'pooled_intrinsic_reward', replay=x, fresh=y, fresh_minus_replay=y-x))
        if all_values:
            low, high = min(all_values), max(all_values)
            pad = max((high-low)*.08, .01)
            ax.plot([low-pad, high+pad], [low-pad, high+pad], color='gray', ls=':', lw=1)
            ax.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad))
        ax.set(xlabel=LABELS[training], ylabel=LABELS[fresh] if fresh else 'Fresh pooled intrinsic reward (nats)')
        ax.grid(alpha=.18)
    ratios, _, colors, _ = _style(runs)
    axes[0].legend(handles=[Line2D([0], [0], color=colors[r], marker='o', ls='', label=r) for r in ratios], title='Discriminator:SAC', fontsize=8)
    axes[1].legend(handles=[Line2D([0], [0], color='gray', marker=markers[i % len(markers)], ls='', label=str(s)) for i,s in enumerate(seeds)], title='Training seed', fontsize=8)
    return fig, {'paired_measurements': records}


def plot_checkpoint_changes(runs, mode='deterministic', start=150000, end=200000, metrics=FRESH):
    """Paired within-training-seed changes. Missing pairs are reported, not inferred."""
    if start >= end:
        raise ValueError('start must be before end')
    fig, axes = _panels(len(metrics), f'{mode}: {end:,} minus {start:,} steps — paired seeds; mean ±1 seed SD, not CI')
    ratios, _, colors, _ = _style(runs)
    records, aggregates, missing = [], [], []
    for run in runs:
        by_step = {e['step']: e for e in run['evals']}
        if start not in by_step or end not in by_step:
            missing.append(dict(ratio=run['ratio'], seed=run['seed'], start=start, end=end, reason='missing checkpoint pair'))
            continue
        for metric in metrics:
            change = by_step[end]['modes'][mode]['summary'][metric] - by_step[start]['modes'][mode]['summary'][metric]
            records.append(dict(ratio=run['ratio'], seed=run['seed'], mode=mode, start=start, end=end, metric=metric, change=change))
    for ax, metric in zip(axes, metrics):
        for i, ratio in enumerate(ratios):
            vals = [r['change'] for r in records if r['ratio'] == ratio and r['metric'] == metric]
            if vals:
                mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals)>1 else np.nan
                ax.scatter(i+np.linspace(-.12, .12, len(vals)), vals, color=colors[ratio], alpha=.65)
                ax.errorbar(i, mu, yerr=sd if len(vals)>1 else None, fmt='ko', capsize=4)
                aggregates.append(dict(mode=mode, ratio=ratio, metric=metric, start=start, end=end, mean_change=mu, seed_sd=sd, n_seeds=len(vals)))
        if not records:
            ax.text(.5, .5, 'Checkpoint pair unavailable', transform=ax.transAxes, ha='center')
        ax.axhline(0, color='gray', lw=.8)
        ax.set(title=LABELS.get(metric, metric), ylabel='Change within training seed', xlabel='Discriminator:SAC')
        ax.set_xticks(range(len(ratios)), ratios)
        ax.grid(axis='y', alpha=.15)
    return fig, {'paired_changes': records, 'aggregates': aggregates, 'missing_pairs': missing}


def plot_update_budget(runs):
    """Actual counts and training/evaluation time versus environment steps."""
    metrics = ('disc_updates', 'critic_updates', 'actor_updates', 'target_updates', 'training_seconds', 'evaluation_seconds')
    fig, axes = _panels(6, f'Measured update and time budgets — {SPREAD}', cols=3)
    ratios, _, colors, styles = _style(runs)
    records = []
    for ax, metric in zip(axes, metrics):
        for ratio in ratios:
            group = [r for r in runs if r['ratio'] == ratio]
            for run in group:
                ax.plot([p['step'] for p in run['log']], [p.get(metric, np.nan) for p in run['log']],
                        color=colors[ratio], ls=styles[run['seed']], alpha=.45, lw=.8)
            common = sorted(set.intersection(*[{p['step'] for p in r['log'] if np.isfinite(p.get(metric,np.nan))} for r in group]))
            if common:
                values = [[next(p[metric] for p in r['log'] if p['step']==s) for s in common] for r in group]
                mu, sd = _band(ax, common, values, colors[ratio])
                records.extend(dict(ratio=ratio, step=s, metric=metric, mean=float(a), seed_sd=float(b), n_seeds=len(group)) for s,a,b in zip(common,mu,sd))
        _decorate(ax, metric)
    _legends(axes, runs)
    return fig, {'aggregates': records}


def save_plot(fig, path, formats=('png',), dpi=180):
    """Save without replacing files. Caller retains figure ownership."""
    path = Path(path)
    if any(fmt not in ('png', 'pdf', 'svg') for fmt in formats):
        raise ValueError('Supported formats: png, pdf, svg')
    paths = [path.parent / f'{path.name}.{fmt}' for fmt in formats]
    if any(p.exists() for p in paths):
        raise FileExistsError('Refusing to overwrite an existing plot')
    fig.tight_layout(rect=(0, 0, 1, .95))
    for dest, fmt in zip(paths, formats):
        with dest.open('xb') as f:
            fig.savefig(f, format=fmt, dpi=dpi, bbox_inches='tight')
    return paths


def generate_plots(root, *, output=None, ratios=None, seeds=None, modes=MODES,
                   kinds=KINDS, checkpoints=None, diagnostic_step=None,
                   change_steps=(150000,200000), formats=('png',), dpi=180):
    """Plot-only entry point; verified saved data only, never loads model tensors."""
    if not modes or any(m not in MODES for m in modes):
        raise ValueError('Select deterministic and/or stochastic mode')
    if not kinds or any(k not in KINDS for k in kinds):
        raise ValueError(f'Plot kinds must be selected from {KINDS}')
    if not formats or any(f not in ('png', 'pdf', 'svg') for f in formats) or dpi <= 0:
        raise ValueError('Choose png/pdf/svg and a positive dpi')
    if len(change_steps) != 2 or change_steps[0] >= change_steps[1]:
        raise ValueError('Change steps must be START END with START < END')
    runs, excluded = load_runs(root, ratios=ratios, seeds=seeds)
    if not runs:
        raise ValueError('No matching verified completed runs. Full training is not started by this command.')
    if checkpoints is not None:
        available = {e['step'] for r in runs for e in r['evals']}
        absent = set(checkpoints) - available
        if absent:
            raise ValueError(f'Requested checkpoints unavailable: {sorted(absent)}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(output) if output is not None else Path(root) / 'plots' / stamp
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(source_root=str(Path(root).resolve()), runs=[str(r['path']) for r in runs],
        excluded=excluded, modes=list(modes), kinds=list(kinds), checkpoints=checkpoints,
        diagnostic_step=diagnostic_step, change_steps=list(change_steps), formats=list(formats), dpi=dpi,
        uncertainty='±1 sample SD across training seeds; blank SD for one seed; not confidence intervals',
        skipped=[], files=[])
    def emit(name, result):
        fig, tables = result
        try:
            manifest['files'].extend(p.name for p in save_plot(fig, out/name, formats, dpi))
            for table, rows in tables.items():
                if not rows:
                    continue
                path = out / f'{name}_{table}.csv'
                with path.open('x', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows({k: '' if isinstance(v,float) and not np.isfinite(v) else v for k,v in row.items()} for row in rows)
                manifest['files'].append(path.name)
        finally:
            plt.close(fig)
    if 'training' in kinds:
        for x in ('step','training_seconds'):
            emit(f'training_{x}', plot_training_curves(runs, x=x))
    if 'budget' in kinds:
        emit('update_budget', plot_update_budget(runs))
    for mode in modes:
        if 'fresh' in kinds:
            emit(f'fresh_{mode}', plot_fresh_evaluation(runs, mode, checkpoints))
        if 'agreement' in kinds:
            emit(f'agreement_{mode}', plot_replay_fresh_agreement(runs, mode, checkpoints))
        if 'changes' in kinds:
            emit(f'changes_{mode}', plot_checkpoint_changes(runs, mode, *change_steps))
        for run in runs:
            tag = f'{mode}_N{run["ratio"].replace(":", "_M")}_seed{run["seed"]}'
            ev = _evals(run, checkpoints)
            if not ev:
                manifest['skipped'].append(f'{tag}: no selected evaluation')
                continue
            if 'skills' in kinds:
                emit(f'skills_{tag}', plot_skill_heatmaps(run, mode, checkpoints))
            step = diagnostic_step if diagnostic_step is not None else max(e['step'] for e in ev)
            for kind, func in (('discriminator', plot_discriminator_diagnostics), ('outcomes', plot_episode_outcomes)):
                if kind in kinds:
                    if step not in {e['step'] for e in ev}:
                        manifest['skipped'].append(f'{tag}/{kind}: checkpoint {step} unavailable or filtered out')
                    else:
                        emit(f'{kind}_{tag}_step{step}', func(run, mode, step))
    with (out/'plot_manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    print(f'Saved {len(manifest["files"])} plot/table files to {out}')
    for r in excluded:
        print(f'EXCLUDED {r["path"]}: {r["reason"]}')
    for reason in manifest['skipped']:
        print(f'SKIPPED {reason}')
    return out

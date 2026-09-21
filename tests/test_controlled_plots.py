"""Numerical/rendering checks for plot-only saved-data workflows."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from diayn.controlled_eval import aggregate
from diayn.controlled_plots import (plt, generate_plots, plot_training_curves,
    plot_skill_heatmaps, plot_discriminator_diagnostics, plot_episode_outcomes,
    plot_replay_fresh_agreement, plot_checkpoint_changes, save_plot)


def saved_run(seed=10, offset=0.):
    run = dict(path=Path(f'/fixture/N1_M1/seed{seed}'), seed=seed, ratio='1:1',
               config=dict(max_episode_steps=3), log=[], evals=[])
    for i, step in enumerate((60, 120)):
        run['log'].append(dict(step=step, training_seconds=1+i+offset,
            disc_acc=.5+i*.1, disc_loss=2.-i, pseudo_reward=-1.+i,
            policy_entropy=.3, disc_updates=step, critic_updates=step,
            actor_updates=step, target_updates=step, evaluation_seconds=.1*i))
        episodes = []
        for skill in range(2):
            for ep in range(2):
                length = 3 if skill == 0 else 1
                correct = length if skill == 0 else 0
                episodes.append(dict(skill=skill, length=length, correct=correct,
                    accuracy=correct/length, nll_sum=length*(1000 if skill else .1),
                    min_true_log_probability=-1000. if skill else -.1,
                    task_return=-2.+i*(seed-8), full_horizon=length==3,
                    terminated=skill==1 or ep==1, truncated=skill==0))
        data = aggregate(episodes, np.array([[6,0],[2,0]]), 2)
        run['evals'].append(dict(step=step, modes={m: copy.deepcopy(data) for m in ('deterministic','stochastic')}))
    return run


class PlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_common_time_budgets_and_single_seed_sd(self):
        a, b = saved_run(), saved_run(11, .2)
        _, tables = plot_training_curves([a,b], x='training_seconds')
        rows = tables['aggregates']
        self.assertEqual(min(r['budget'] for r in rows), 1.2)
        self.assertEqual(max(r['budget'] for r in rows), 2.)
        _, single = plot_training_curves([a])
        self.assertTrue(all(np.isnan(r['seed_sd']) for r in single['aggregates']))
        b['log'][0]['step'] = 70
        _, exact = plot_training_curves([a,b])
        self.assertEqual({r['budget'] for r in exact['aggregates']}, {120})
        b = saved_run(11, 10.)
        _, empty = plot_training_curves([a,b], x='training_seconds')
        self.assertFalse(empty['aggregates'])

    def test_signed_heatmaps_confusion_and_extreme_log_loss(self):
        run = saved_run()
        fig, _ = plot_skill_heatmaps(run)
        self.assertLess(fig.axes[0].images[0].get_clim()[0], 0)
        self.assertLess(fig.axes[3].images[0].get_clim()[0], -900)
        fig, tables = plot_discriminator_diagnostics(run)
        self.assertEqual([r['within_skill_fraction'] for r in tables['confusion']], [1.,0.,1.,0.])
        self.assertEqual([r['transition_frequency'] for r in tables['skills']], [.75,.25])
        self.assertEqual(tables['skills'][1]['nll'], 1000.)
        self.assertGreater(fig.axes[2].get_ylim()[1], 1000)
        fig, _ = plot_episode_outcomes(run)
        heights = np.array([[p.get_height() for p in bar] for bar in fig.axes[2].containers])
        np.testing.assert_allclose(heights.sum(0), 1)
        self.assertEqual(heights[2,0], .5)  # both terminated and truncated retained

    def test_exact_replay_pairing_and_missing_checkpoint_changes(self):
        run = saved_run()
        run['log'] = run['log'][1:]
        _, tables = plot_replay_fresh_agreement([run])
        self.assertEqual({r['step'] for r in tables['paired_measurements']}, {120})
        self.assertEqual(len(tables['paired_measurements']), 4)
        _, missing = plot_checkpoint_changes([run])
        self.assertEqual(len(missing['missing_pairs']), 1)
        self.assertFalse(missing['paired_changes'])
        _, changes = plot_checkpoint_changes([saved_run(),saved_run(12)], start=60,end=120)
        row = next(r for r in changes['aggregates'] if r['metric']=='mean_skill_return')
        self.assertEqual(row['mean_change'], 3.)
        self.assertAlmostEqual(row['seed_sd'], np.sqrt(2))

    def test_export_formats_filters_no_rollouts_and_no_overwrite(self):
        run = saved_run()
        original = copy.deepcopy(run)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'plots'
            with patch('diayn.controlled_plots.load_runs', return_value=([run],[])) as loader, \
                 patch('diayn.agent.DIAYNAgent.load', side_effect=AssertionError('must not load a checkpoint')), \
                 patch('diayn.envs.make_env', side_effect=AssertionError('must not create env')):
                generate_plots('/fixture', output=out, ratios=['1:1'], seeds=[10],
                    modes=['deterministic'], kinds=['discriminator'], checkpoints=[120], formats=['png','pdf','svg'])
                loader.assert_called_once_with('/fixture', ratios=['1:1'], seeds=[10])
                with self.assertRaises(FileExistsError):
                    generate_plots('/fixture', output=out, kinds=['training'])
            manifest = json.loads((out/'plot_manifest.json').read_text())
            self.assertEqual(len(manifest['files']), 5)
            for fmt in ('png','pdf','svg'):
                self.assertTrue(list(out.glob(f'*.{fmt}')))
            fig, _ = plot_skill_heatmaps(run)
            path = out/'standalone'
            save_plot(fig,path)
            original_bytes = (out/'standalone.png').read_bytes()
            with self.assertRaises(FileExistsError):
                save_plot(fig,path)
            self.assertEqual((out/'standalone.png').read_bytes(),original_bytes)
        self.assertEqual(run,original)


if __name__ == '__main__':
    unittest.main()

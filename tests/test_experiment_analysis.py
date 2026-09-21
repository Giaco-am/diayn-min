import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from diayn.controlled_eval import aggregate
from diayn.experiment_analysis import analyze, TRAIN


class AnalysisTests(unittest.TestCase):
    def test_paired_changes_seed_spread_and_shared_time_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, change, offset in ((10, 2., 0.), (11, 4., .2)):
                run = root / 'N1_M1' / f'seed{seed}'
                run.mkdir(parents=True)
                config = dict(seed=seed, ratio=[1, 1], ratios=[[1, 1]], seeds=[10, 11])
                (run / 'resolved_config.json').write_text(json.dumps(config))
                (run / 'provenance.json').write_text(json.dumps({'source_sha256': {'diayn/agent.py': 'fixture'}}))
                with (run / 'log.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['step', 'training_seconds', *TRAIN])
                    writer.writeheader()
                    for i, step in enumerate((150000, 200000)):
                        writer.writerow(dict(step=step, training_seconds=1+i+offset,
                                             **dict.fromkeys(TRAIN, .1+i)))
                for i, step in enumerate((150000, 200000)):
                    episodes = [dict(skill=z, length=1, correct=1, accuracy=1., task_return=10+i*change,
                                     full_horizon=True, nll_sum=1., min_true_log_probability=-1.) for z in range(2)]
                    metrics = aggregate(episodes, np.eye(2, dtype=int), 2)
                    evaluation = dict(step=step, modes={m: copy.deepcopy(metrics) for m in ('deterministic', 'stochastic')})
                    (run / f'evaluation_{step:09d}.json').write_text(json.dumps(evaluation))
            # Persistence has separate integration tests; these fixtures focus on analysis math.
            with patch('diayn.experiment_analysis.verify_completed', return_value=(True, 'fixture')):
                out = analyze(root)
            with (out / 'change_seed_aggregates.csv').open() as f:
                changes = list(csv.DictReader(f))
            values = [r for r in changes if r['metric'] == 'mean_skill_return']
            self.assertEqual(len(values), 2)
            for row in values:
                self.assertEqual(float(row['mean_change']), 3.)
                self.assertAlmostEqual(float(row['seed_sd']), np.sqrt(2))
                self.assertEqual(int(row['n_seeds']), 2)
            with (out / 'training_aggregates.csv').open() as f:
                time_rows = [r for r in csv.DictReader(f) if r['axis'] == 'training_seconds']
            self.assertEqual(min(float(r['budget']) for r in time_rows), 1.2)
            self.assertEqual(max(float(r['budget']) for r in time_rows), 2.)
            self.assertTrue((out / 'changes_150k_to_200k_stochastic.png').is_file())
            # Different implementation hashes must not be silently pooled.
            (root / 'N1_M1/seed11/provenance.json').write_text(json.dumps({'source_sha256': {'diayn/agent.py': 'changed'}}))
            with patch('diayn.experiment_analysis.verify_completed', return_value=(True, 'fixture')):
                with self.assertRaisesRegex(ValueError, 'different training/evaluation'):
                    analyze(root)


if __name__ == '__main__':
    unittest.main()

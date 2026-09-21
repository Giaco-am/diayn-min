"""Targeted scientific-protocol and historical-behavior regression checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from diayn.agent import DIAYNAgent, DIAYNConfig
from diayn.buffer import ReplayBuffer
from diayn.controlled import update_block
from diayn.controlled_eval import aggregate, evaluate, preserve_rng
from diayn.experiment import ROOT, run_one, verify_completed, sha256


def seed_all(seed=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def agent():
    return DIAYNAgent(DIAYNConfig(obs_dim=4, act_dim=1, act_limit=[3.], n_skills=2,
                                hidden=(8, 8), batch_size=4))


def replay():
    r = ReplayBuffer(4, 1, 10, 'cpu')
    for i in range(10):
        r.add(np.ones(4)*i/10, i % 2, np.zeros(1), np.ones(4)*(i+1)/10, False)
    return r


def tiny_config():
    c = json.loads((ROOT / 'configs/controlled_ratios_v1.json').read_text())
    c.update(steps=12, save_every=6, eval_every=6, max_episode_steps=3,
        n_skills=2, hidden=8, batch_size=4, replay_size=30, start_steps=4,
        update_after=4, evaluation_reset_seeds=[91000], log_every=6,
        seed=10, ratio=[1, 5], disc_updates=1, policy_updates=5)
    return c


class ControlledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schedule_order_fresh_batches_counts_and_metric_means(self):
        for n, m in ((1, 1), (1, 5), (5, 1)):
            events, batches = [], []
            class Buffer:
                def sample(self, size):
                    batch = object()
                    batches.append(batch)
                    events.append('sample')
                    return batch
            class Agent:
                def update_discriminator(self, batch):
                    events.append('disc')
                    return {'disc_loss': float(events.count('disc'))}
                def update_policy(self, batch, update_targets=True):
                    self_assert = not update_targets
                    if not self_assert:
                        raise AssertionError('SAC moved targets inside block')
                    events.append('sac')
                    return {'pseudo_reward': float(events.count('sac'))}
                def update_targets(self):
                    events.append('target')
            metrics, counts = update_block(Agent(), Buffer(), 4, n, m)
            self.assertEqual(events, ['sample', 'disc']*n + ['sample', 'sac']*m + ['target'])
            self.assertEqual(len({id(b) for b in batches}), n+m)
            self.assertEqual(counts, dict(disc_updates=n, critic_updates=m, actor_updates=m, target_updates=1))
            self.assertEqual(metrics['disc_loss'], (n+1)/2)
            self.assertEqual(metrics['pseudo_reward'], (m+1)/2)

    def test_real_optimizer_counts_and_polyak_timing(self):
        for n, m in ((1, 1), (1, 5), (5, 1)):
            a, b = agent(), replay()
            original = [p.clone() for net in (a.q1_target, a.q2_target) for p in net.parameters()]
            def check_before(*args, **kwargs):
                current = [p for net in (a.q1_target, a.q2_target) for p in net.parameters()]
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(original, current)))
            a.q_opt.register_step_pre_hook(check_before)
            with patch.object(a.disc_opt, 'step', wraps=a.disc_opt.step) as d, \
                 patch.object(a.q_opt, 'step', wraps=a.q_opt.step) as q, \
                 patch.object(a.actor_opt, 'step', wraps=a.actor_opt.step) as p, \
                 patch.object(a, 'update_targets', wraps=a.update_targets) as t:
                update_block(a, b, 4, n, m)
                self.assertEqual((d.call_count, q.call_count, p.call_count, t.call_count), (n, m, m, 1))
            source = [p for net in (a.q1, a.q2) for p in net.parameters()]
            targets = [p for net in (a.q1_target, a.q2_target) for p in net.parameters()]
            for old, new, actual in zip(original, source, targets):
                self.assertTrue(torch.equal(actual, old.mul(.995).add(.005*new)))

    def test_historical_default_matches_repository_head(self):
        # Independent reference: the unmodified agent from the repository revision.
        old_source = subprocess.check_output(['git', 'show', 'HEAD:diayn/agent.py'], cwd=ROOT, text=True)
        module = types.ModuleType('diayn._historical_test_agent')
        module.__package__ = 'diayn'
        import sys
        sys.modules[module.__name__] = module
        try:
            exec(compile(old_source, 'historical_agent.py', 'exec'), module.__dict__)
            for n, m in ((1, 1), (1, 5), (5, 1)):
                outputs = []
                for cls in (module.DIAYNAgent, DIAYNAgent):
                    seed_all()
                    a = cls(DIAYNConfig(obs_dim=4, act_dim=1, act_limit=[3.], n_skills=2, hidden=(8, 8)))
                    b = replay()
                    if (n, m) == (1, 1):
                        metric = a.update(b.sample(4))
                    else:
                        for _ in range(n):
                            d = a.update_discriminator(b.sample(4))
                        for _ in range(m):
                            p = a.update_policy(b.sample(4))
                        metric = {**d, **p}
                    states = {name: copy.deepcopy(getattr(a, name).state_dict()) for name in
                        ('actor', 'discriminator', 'q1', 'q2', 'q1_target', 'q2_target')}
                    outputs.append((metric, states, torch.get_rng_state()))
                self.assertEqual(outputs[0][0], outputs[1][0])
                for name in outputs[0][1]:
                    for key in outputs[0][1][name]:
                        self.assertTrue(torch.equal(outputs[0][1][name][key], outputs[1][1][name][key]))
                self.assertTrue(torch.equal(outputs[0][2], outputs[1][2]))
        finally:
            del sys.modules[module.__name__]

    def test_eval_rng_repeatability_and_extreme_log_loss(self):
        a, config = agent(), tiny_config()
        with torch.no_grad():
            for p in a.discriminator.parameters():
                p.zero_()
            a.discriminator.net[-1].bias[1] = -1000
        seed_all()
        py, ns, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        result = evaluate(a, config)
        self.assertEqual(random.getstate(), py)
        self.assertEqual(repr(np.random.get_state()), repr(ns))
        self.assertTrue(torch.equal(torch.get_rng_state(), ts))
        self.assertEqual(result, evaluate(a, config))
        for mode in result['modes'].values():
            self.assertEqual(mode['skills'][1]['nll'], 1000)
            self.assertEqual(mode['skills'][1]['min_true_log_probability'], -1000)
            self.assertEqual(mode['episodes'][0]['reset_seed'], mode['episodes'][1]['reset_seed'])
        with self.assertRaises(RuntimeError):
            with preserve_rng():
                random.random()
                np.random.rand()
                torch.rand(1)
                raise RuntimeError('intentional failure')
        self.assertEqual(random.getstate(), py)
        self.assertEqual(repr(np.random.get_state()), repr(ns))
        self.assertTrue(torch.equal(torch.get_rng_state(), ts))

    def test_aggregation_distinguishes_transition_and_episode_weighting(self):
        rows = []
        for z, length, correct, ret, full in ((0, 9, 9, 9, True), (0, 1, 0, 1, False),
                                            (1, 1, 0, 2, True), (1, 1, 0, 4, True)):
            rows.append(dict(skill=z, length=length, correct=correct, accuracy=correct/length,
                task_return=ret, full_horizon=full, nll_sum=length*(z+1),
                min_true_log_probability=-float(z+1)))
        result = aggregate(rows, np.array([[9, 1], [2, 0]]), 2)
        s = result['summary']
        self.assertEqual(s['pooled_accuracy'], .75)
        self.assertEqual(s['episode_skill_balanced_accuracy'], .25)
        self.assertEqual(s['mean_skill_return'], 4)
        self.assertEqual(s['median_skill_return'], 4)
        self.assertEqual(s['full_horizon_skill_count'], 1)
        self.assertEqual(s['full_horizon_episode_fraction'], .75)
        self.assertEqual(s['transition_skill_frequencies'], [10/12, 2/12])
        self.assertEqual(s['pooled_nll'], 14/12)
        self.assertEqual(s['skill_mean_nll'], 1.5)
        self.assertEqual(result['skills'][0]['accuracy'], .9)

    def test_evaluation_frequency_checkpoints_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            c1, c2 = tiny_config(), tiny_config()
            c2['eval_every'] = 12
            paths = [Path(tmp) / 'frequent', Path(tmp) / 'sparse']
            with contextlib.redirect_stdout(io.StringIO()):
                run_one(c1, paths[0])
                run_one(c2, paths[1])
            self.assertEqual((paths[0] / 'training_sequence_audit.json').read_text(),
                             (paths[1] / 'training_sequence_audit.json').read_text())
            for step in (6, 12):
                models = [torch.load(p / f'checkpoint_{step:09d}.pt', weights_only=False) for p in paths]
                for name in ('actor', 'discriminator', 'q1', 'q2', 'q1_target', 'q2_target'):
                    for key in models[0][name]:
                        self.assertTrue(torch.equal(models[0][name][key], models[1][name][key]))
                self.assertEqual(models[0]['counts']['target_updates'], step-3)
                self.assertEqual(models[0]['counts']['actor_updates'], (step-3)*5)
                self.assertEqual(models[0]['training_seed'], 10)
                self.assertEqual(models[0]['ratio'], [1, 5])
            for p, c in zip(paths, (c1, c2)):
                self.assertTrue(verify_completed(p, c)[0])
            self.assertFalse(verify_completed(paths[0], c2)[0])
            self.assertEqual(len(list(paths[0].glob('checkpoint_*.pt'))), 2)
            digest = sha256(paths[0] / 'checkpoint_000000012.pt')
            with self.assertRaises(FileExistsError):
                run_one(c1, paths[0])
            self.assertEqual(sha256(paths[0] / 'checkpoint_000000012.pt'), digest)
            with (paths[0] / 'log.csv').open('a') as f:
                f.write('corruption')
            self.assertFalse(verify_completed(paths[0], c1)[0])
            self.assertFalse(verify_completed(Path(tmp) / 'absent')[0])


if __name__ == '__main__':
    unittest.main()

"""Controlled experiment persistence and training. No exact-resume contract."""
import csv
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import random
import subprocess
import time

import gymnasium
import numpy as np
import torch

from .agent import DIAYNAgent, DIAYNConfig
from .buffer import ReplayBuffer
from .controlled import update_block
from .controlled_eval import evaluate, preserve_rng
from .envs import make_env

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['disc_loss', 'disc_acc', 'pseudo_reward', 'q_loss', 'actor_loss', 'policy_entropy']
COUNTS = ['disc_updates', 'critic_updates', 'actor_updates', 'target_updates']


def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_provenance():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    sources = sorted([*ROOT.glob('diayn/*.py'), *ROOT.glob('scripts/*.py'),
                      *ROOT.glob('configs/*.json'), ROOT / 'pyproject.toml', ROOT / 'uv.lock'])
    return dict(revision=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')),
                source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources})


def validate(config):
    if config.get('controlled_mode') is not True:
        raise ValueError('This dedicated runner requires controlled_mode=true; use train.py for historical behavior')
    if config['tau'] != 0.005:
        raise ValueError('Controlled schedules require tau=0.005')
    if config['device'] != 'cpu':
        raise ValueError('This reproducibility protocol currently supports CPU training only')
    for key in ('n_skills', 'steps', 'save_every', 'eval_every', 'max_episode_steps',
                'batch_size', 'replay_size', 'hidden', 'log_every', 'torch_threads'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['steps'] % config['save_every'] or config['steps'] % config['eval_every']:
        raise ValueError('Final step must align with checkpoint and evaluation intervals')
    if config['replay_size'] < config['batch_size']:
        raise ValueError('Replay must hold at least one batch')
    if not config['evaluation_reset_seeds'] or len(set(config['evaluation_reset_seeds'])) != len(config['evaluation_reset_seeds']):
        raise ValueError('Provide distinct evaluation reset seeds')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Provide distinct training seeds')
    if not config['ratios'] or len({tuple(r) for r in config['ratios']}) != len(config['ratios']):
        raise ValueError('Provide distinct ratios')
    for ratio in config['ratios']:
        if len(ratio) != 2 or any(not isinstance(v, int) or v <= 0 for v in ratio):
            raise ValueError('Ratios must be positive integer pairs')


def checkpoint_steps(config):
    # Any evaluation also has its own retained checkpoint.
    return sorted(set(range(config['save_every'], config['steps'] + 1, config['save_every'])) |
                  set(range(config['eval_every'], config['steps'] + 1, config['eval_every'])))


def expected_counts(config):
    eligible = max(0, config['steps'] - max(1, config['update_after'], config['batch_size']) + 1)
    n, m = config['ratio']
    return dict(disc_updates=eligible*n, critic_updates=eligible*m,
                actor_updates=eligible*m, target_updates=eligible)


def verify_completed(run_dir, requested=None):
    """Require config, expected artifacts, update budgets and all content hashes."""
    run_dir = Path(run_dir)
    try:
        complete = json.loads((run_dir / 'completed.json').read_text())
        config = json.loads((run_dir / 'resolved_config.json').read_text())
        if requested is not None and any(config.get(k) != v for k, v in requested.items()):
            return False, 'configuration mismatch'
        if complete['step'] != config['steps'] or complete['counts'] != expected_counts(config):
            return False, 'step/update budget mismatch'
        expected = {'resolved_config.json', 'provenance.json', 'log.csv'}
        expected |= {f'checkpoint_{s:09d}.pt' for s in checkpoint_steps(config)}
        expected |= {f'evaluation_{s:09d}.json' for s in range(config['eval_every'], config['steps']+1, config['eval_every'])}
        if not expected <= complete['artifacts'].keys():
            return False, 'missing expected artifacts in completion manifest'
        for name, digest in complete['artifacts'].items():
            if sha256(run_dir / name) != digest:
                return False, f'content mismatch: {name}'
        return True, 'verified completed run'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f'incomplete or invalid: {exc}'


def run_one(config, run_dir):
    """Create exclusively; never overwrite or resume a partial directory."""
    validate(config)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance()
    write_json(run_dir / 'provenance.json', provenance)
    # Capture the actual sources too: a dirty HEAD alone is insufficient provenance.
    for name in provenance['source_sha256']:
        dest = run_dir / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write((ROOT / name).read_bytes())
    torch.set_num_threads(config['torch_threads'])
    torch.use_deterministic_algorithms(True)
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(config['env'], config['max_episode_steps'])
    try:
        env.action_space.seed(seed)
        cfg = DIAYNConfig(n_skills=config['n_skills'], hidden=(config['hidden'],)*2,
            gamma=config['gamma'], tau=config['tau'], alpha=config['alpha'], lr=config['lr'],
            batch_size=config['batch_size'], device=config['device'],
            obs_dim=int(np.prod(env.observation_space.shape)), act_dim=int(np.prod(env.action_space.shape)),
            act_limit=np.asarray(env.action_space.high, dtype=np.float32).tolist())
        resolved = dict(config, agent_config=asdict(cfg), repository_revision=provenance['revision'],
            runtime=dict(python=platform.python_version(), torch=torch.__version__,
                         numpy=np.__version__, gymnasium=gymnasium.__version__, platform=platform.platform()),
            checkpoint_contract='evaluation only; no exact training resumption',
            critic_count_definition='joint optimizer steps, updating both critics',
            discriminator_measurement='pre-optimizer logits for each discriminator update')
        write_json(run_dir / 'resolved_config.json', resolved)
        agent = DIAYNAgent(cfg)
        replay = ReplayBuffer(cfg.obs_dim, cfg.act_dim, config['replay_size'], agent.device)
        obs, _ = env.reset(seed=seed)
        skill = agent.sample_skill()
        counts = dict.fromkeys(COUNTS, 0)
        train_seconds = eval_seconds = 0.0
        episodes = window_steps = window_updates = 0
        sums = dict.fromkeys(METRICS, 0.0)
        window_seconds = 0.0
        fields = ['step', 'environment_steps', 'episodes', *METRICS, 'sps', *COUNTS,
                  'training_seconds', 'evaluation_seconds', 'logged_update_blocks']
        save_steps = checkpoint_steps(config)
        with (run_dir / 'log.csv').open('x', newline='') as f:
            logger = csv.DictWriter(f, fieldnames=fields)
            logger.writeheader()
            for step in range(1, config['steps'] + 1):
                start = time.perf_counter()
                action = env.action_space.sample() if step <= config['start_steps'] else agent.act(obs, skill)
                next_obs, _, terminated, truncated, _ = env.step(action)
                replay.add(obs, skill, action, next_obs, terminated)
                obs = next_obs
                if terminated or truncated:
                    obs, _ = env.reset()
                    skill = agent.sample_skill()
                    episodes += 1
                if step >= config['update_after'] and replay.size >= config['batch_size']:
                    metrics, delta = update_block(agent, replay, config['batch_size'], *config['ratio'])
                    for key in counts:
                        counts[key] += delta[key]
                    for key in sums:
                        sums[key] += metrics[key]
                    window_updates += 1
                elapsed = time.perf_counter() - start
                train_seconds += elapsed
                window_seconds += elapsed
                window_steps += 1
                if step in save_steps:
                    # Exclusive file creation guards checkpoint names independently of directory creation.
                    with (run_dir / f'checkpoint_{step:09d}.pt').open('xb') as checkpoint:
                        agent.save(checkpoint, extra=dict(env=config['env'], step=step,
                            max_episode_steps=config['max_episode_steps'], training_seed=seed,
                            ratio=config['ratio'], resolved_config=resolved, counts=dict(counts),
                            repository_revision=provenance['revision'], training_seconds=train_seconds,
                            evaluation_seconds=eval_seconds, checkpoint_contract='evaluation only'))
                if step % config['eval_every'] == 0:
                    start = time.perf_counter()
                    # Load the saved model without consuming training RNG through constructor init.
                    with preserve_rng():
                        saved_agent, _ = DIAYNAgent.load(run_dir / f'checkpoint_{step:09d}.pt')
                        result = evaluate(saved_agent, config)
                    eval_seconds += time.perf_counter() - start
                    result.update(step=step, seed=seed, ratio=config['ratio'], counts=dict(counts),
                                  training_seconds=train_seconds, evaluation_seconds=eval_seconds)
                    write_json(run_dir / f'evaluation_{step:09d}.json', result)
                if step % config['log_every'] == 0 or step in save_steps or step == config['steps']:
                    row = {k: sums[k] / window_updates if window_updates else '' for k in METRICS}
                    row.update(step=step, environment_steps=step, episodes=episodes,
                        sps=window_steps / window_seconds, **counts, training_seconds=train_seconds,
                        evaluation_seconds=eval_seconds, logged_update_blocks=window_updates)
                    logger.writerow(row)
                    f.flush()
                    print(f'{run_dir.name} step={step} updates={counts} train={train_seconds:.2f}s eval={eval_seconds:.2f}s', flush=True)
                    window_steps = window_updates = 0
                    window_seconds = 0.0
                    sums = dict.fromkeys(METRICS, 0.0)
        # Diagnostic evidence for tests. This is NOT a replay/RNG/environment resume snapshot.
        h = hashlib.sha256()
        for arr in (replay.obs, replay.next_obs, replay.act, replay.skill, replay.done):
            h.update(arr[:replay.size].tobytes())
        audit = dict(replay_sha256=h.hexdigest(), replay_size=replay.size,
            python_state=repr(random.getstate()), numpy_state=repr(np.random.get_state()),
            torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            action_space_rng=repr(env.action_space.np_random.bit_generator.state),
            environment_rng=repr(env.unwrapped.np_random.bit_generator.state))
        write_json(run_dir / 'training_sequence_audit.json', audit)
        artifacts = {str(p.relative_to(run_dir)): sha256(p) for p in sorted(run_dir.rglob('*')) if p.is_file()}
        write_json(run_dir / 'completed.json', dict(step=config['steps'], counts=counts,
            training_seconds=train_seconds, evaluation_seconds=eval_seconds, artifacts=artifacts))
        return dict(training_seconds=train_seconds, evaluation_seconds=eval_seconds)
    finally:
        env.close()

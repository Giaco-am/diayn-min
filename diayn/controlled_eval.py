"""Fresh, RNG-isolated per-skill evaluation and explicit aggregation."""
from contextlib import contextmanager
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .envs import make_env


@contextmanager
def preserve_rng():
    """Restore Python, NumPy, Torch CPU and available accelerator RNG states."""
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if mps is not None:
            torch.mps.set_rng_state(mps)


def aggregate(episodes, confusion, n_skills):
    """Transition-weighted within skills; also report episode-balanced accuracy.

    Full-horizon counts mean length == configured horizon, including a failure
    exactly at the horizon. Termination and truncation are separately retained.
    """
    skills = []
    for z in range(n_skills):
        rows = [r for r in episodes if r['skill'] == z]
        count = sum(r['length'] for r in rows)
        skills.append(dict(
            skill=z, episodes=len(rows), transitions=count,
            mean_return=float(np.mean([r['task_return'] for r in rows])),
            mean_length=float(np.mean([r['length'] for r in rows])),
            full_horizon_fraction=float(np.mean([r['full_horizon'] for r in rows])),
            all_episodes_full_horizon=all(r['full_horizon'] for r in rows),
            accuracy=sum(r['correct'] for r in rows) / count,
            episode_mean_accuracy=float(np.mean([r['accuracy'] for r in rows])),
            nll=sum(r['nll_sum'] for r in rows) / count,
            intrinsic_reward=math.log(n_skills) - sum(r['nll_sum'] for r in rows) / count,
            min_true_log_probability=min(r['min_true_log_probability'] for r in rows),
        ))
    total = sum(r['length'] for r in episodes)
    returns = [r['mean_return'] for r in skills]
    worst_acc = min(skills, key=lambda s: s['accuracy'])
    worst_nll = max(skills, key=lambda s: s['nll'])
    summary = dict(
        mean_skill_return=float(np.mean(returns)), median_skill_return=float(np.median(returns)),
        full_horizon_episode_fraction=float(np.mean([r['full_horizon'] for r in episodes])),
        full_horizon_skill_count=sum(s['all_episodes_full_horizon'] for s in skills),
        pooled_accuracy=sum(r['correct'] for r in episodes) / total,
        episode_skill_balanced_accuracy=float(np.mean([s['episode_mean_accuracy'] for s in skills])),
        pooled_nll=sum(r['nll_sum'] for r in episodes) / total,
        skill_mean_nll=float(np.mean([s['nll'] for s in skills])),
        worst_skill_accuracy=worst_acc['accuracy'], worst_accuracy_skill=worst_acc['skill'],
        worst_skill_nll=worst_nll['nll'], worst_nll_skill=worst_nll['skill'],
        min_true_log_probability=min(r['min_true_log_probability'] for r in episodes),
        transitions=total,
        transition_skill_frequencies=[s['transitions'] / total for s in skills],
    )
    return dict(summary=summary, skills=skills, episodes=episodes, confusion_matrix=confusion.tolist())


@torch.no_grad()
def evaluate(agent, config):
    """New environment; no reference to the training env or replay buffer.

    Stochastic action seed = base + skill * episode_count + episode_index.
    Reset the action RNG at each episode, identically across checkpoints, ratios,
    and training seeds. Skills have separate streams; reset seeds are shared.
    """
    seeds = config['evaluation_reset_seeds']
    base = config['evaluation_action_seed_base']
    horizon = config['max_episode_steps']
    modes = {}
    with preserve_rng():
        env = make_env(config['env'], horizon)
        try:
            for mode in ('deterministic', 'stochastic'):
                rows = []
                confusion = np.zeros((agent.n_skills, agent.n_skills), dtype=np.int64)
                for z in range(agent.n_skills):
                    for ep, reset_seed in enumerate(seeds):
                        action_seed = base + z * len(seeds) + ep
                        random.seed(action_seed)
                        np.random.seed(action_seed)
                        torch.manual_seed(action_seed)
                        env.action_space.seed(action_seed)
                        obs, _ = env.reset(seed=reset_seed)
                        next_states, ret = [], 0.0
                        for _ in range(horizon):
                            action = agent.act(obs, z, deterministic=(mode == 'deterministic'))
                            obs, reward, terminated, truncated, _ = env.step(action)
                            next_states.append(np.asarray(obs).copy())
                            ret += float(reward)
                            if terminated or truncated:
                                break
                        # Batched discriminator inference on the exact next observations.
                        x = torch.as_tensor(np.asarray(next_states), dtype=torch.float32,
                                            device=agent.device)
                        logp = F.log_softmax(agent.discriminator(x), dim=-1)
                        predictions = logp.argmax(-1).cpu().numpy()
                        true_logp = logp[:, z].double().cpu().numpy()
                        length = len(next_states)
                        correct = int(np.sum(predictions == z))
                        nll_sum = float(-true_logp.sum())
                        np.add.at(confusion[z], predictions, 1)
                        rows.append(dict(skill=z, episode=ep, reset_seed=reset_seed,
                            action_seed=action_seed, task_return=ret, length=length,
                            terminated=bool(terminated), truncated=bool(truncated),
                            full_horizon=length == horizon, correct=correct,
                            accuracy=correct / length, nll_sum=nll_sum,
                            nll=nll_sum / length,
                            intrinsic_reward=math.log(agent.n_skills) - nll_sum / length,
                            min_true_log_probability=float(true_logp.min())))
                modes[mode] = aggregate(rows, confusion, agent.n_skills)
        finally:
            env.close()
    return dict(protocol=dict(reset_seeds=seeds, action_seed_base=base,
        stochastic_sampling='Torch tanh-Gaussian rsample; reseed per episode with '
            'base + skill * len(reset_seeds) + episode_index; shared across '
            'checkpoints, ratios and training seeds; training RNG restored',
        discriminator_observation='next observation', confusion_axes='true skill rows, predicted skill columns',
        horizon=horizon, episodes_per_skill_per_mode=len(seeds)), modes=modes)

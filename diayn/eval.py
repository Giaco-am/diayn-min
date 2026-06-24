"""Per-skill task return.

DIAYN never sees the task reward while training. To check whether any skill
happens to solve the task (the paper's Figure 15 analysis, and to rank skills
for plotting) we roll out each skill deterministically and sum the real reward.
A fixed per-(skill, episode) reset seed keeps the rankings reproducible."""

import numpy as np

from .envs import make_env


def skill_returns(agent, env_name, max_episode_steps=None,
                  episodes_per_skill=3, deterministic=True,
                  env=None, return_lengths=False):
    own_env = env is None
    if own_env:
        env = make_env(env_name, max_episode_steps)

    means = np.empty(agent.n_skills, dtype=np.float64)
    lengths = np.empty(agent.n_skills, dtype=np.float64)
    for z in range(agent.n_skills):
        rets, lens = [], []
        for ep in range(episodes_per_skill):
            obs, _ = env.reset(seed=20_000 + 100 * z + ep)
            done, ret, t = False, 0.0, 0
            while not done:
                action = agent.act(obs, z, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                ret += reward
                t += 1
                done = terminated or truncated
            rets.append(ret)
            lens.append(t)
        means[z] = float(np.mean(rets))
        lengths[z] = float(np.mean(lens))

    if own_env:
        env.close()
    return (means, lengths) if return_lengths else means

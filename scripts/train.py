"""Train DIAYN (Algorithm 1 of the paper).

The agent never sees the environment reward. A skill z - p(z) is sampled at the
start of every episode and held fixed; the policy is trained with SAC on the
discriminator pseudo-reward r = log q(z | s') - log p(z).

All settings come from config.json. Run it with:

    uv run python scripts/train.py
"""

import csv
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diayn.agent import DIAYNAgent, DIAYNConfig
from diayn.buffer import ReplayBuffer
from diayn.envs import make_env

LOG_FIELDS = ["step", "episodes", "disc_loss", "disc_acc", "pseudo_reward",
              "q_loss", "actor_loss", "policy_entropy", "sps"]


def load_config(path=None):
    return json.loads((Path(path) if path else ROOT / "config.json").read_text())


def train_one(config, env_name, seed, disc_updates, policy_updates, run_dir):
    """Train a single DIAYN run and return its directory. The N:M asynchronous
    update rule is 'disc_updates' discriminator steps for every 'policy_updates'
    policy steps."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(env_name, config["max_episode_steps"])
    env.action_space.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    cfg = DIAYNConfig(
        n_skills=config["n_skills"],
        hidden=(config["hidden"], config["hidden"]),
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config["alpha"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        device=config["device"],
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=np.asarray(env.action_space.high, dtype=np.float32).tolist(),
        **{key: config[key] for key in (
            "latent_type", "prior_alpha", "posterior_total_concentration_min",
            "posterior_total_concentration_max", "discriminator_grad_clip") if key in config},
    )
    agent = DIAYNAgent(cfg)
    continuous = cfg.latent_type == "dirichlet"
    buffer = ReplayBuffer(obs_dim, act_dim, config["replay_size"], agent.device,
                          skill_dim=cfg.n_skills if continuous else None)
    metric_fields = (['disc_loss', 'disc_nll', 'disc_cosine', 'disc_l1',
                      'posterior_concentration', 'disc_grad_norm', 'pseudo_reward',
                      'prior_log_prob', 'q_loss', 'actor_loss', 'policy_entropy']
                     if continuous else LOG_FIELDS[2:-1])
    log_fields = ['step', 'episodes', 'phase', 'prior_alpha', 'buffer_size',
                  'buffer_resets', 'updates', *metric_fields, 'sps']
    # Each training phase uses a fixed prior concentration.
    phases = config.get("prior_schedule") or [{"start_step": 1, "alpha": cfg.prior_alpha}]
    reset_steps = config.get("replay_reset_steps", [])
    phase = buffer_resets = updates = 0
    agent.cfg.prior_alpha = phases[0]["alpha"]

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps({**config, "env": env_name, "seed": seed,
                    "disc_updates": disc_updates, "policy_updates": policy_updates}, indent=2) + "\n")
    log_file = open(run_dir / "log.csv", "w", newline="")
    logger = csv.DictWriter(log_file, fieldnames=log_fields)
    logger.writeheader()

    steps = config["steps"]
    batch_size = config["batch_size"]
    log_every = config["log_every"]
    schedule = ("synchronous (1:1)" if disc_updates == 1 and policy_updates == 1
                else f"{disc_updates} disc : {policy_updates} policy")
    print(f"[{run_dir.name}] env={env_name} seed={seed} update schedule={schedule}")

    obs, _ = env.reset(seed=seed)
    skill = agent.sample_skill()
    episodes = 0
    recent = {k: deque(maxlen=log_every) for k in metric_fields}
    t0 = time.time()

    for step in range(1, steps + 1):
        phase_change = phase + 1 < len(phases) and step >= phases[phase + 1]["start_step"]
        if phase_change or step in reset_steps:
            if phase_change:
                phase += 1
                agent.cfg.prior_alpha = phases[phase]["alpha"]
            if (phase_change and config.get("reset_buffer_on_change", True)) or step in reset_steps:
                buffer.clear()
                buffer_resets += 1
            for values in recent.values():
                values.clear()
            obs, _ = env.reset()
            skill = agent.sample_skill()

        if step <= config["start_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, skill)

        next_obs, _task_reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, skill, action, next_obs, terminated)
        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
            skill = agent.sample_skill()  # z - p(z) each episode
            episodes += 1

        if step >= config["update_after"] and buffer.size >= batch_size:
            # N:M asynchronous update rule: N discriminator steps then M policy
            # steps, each on its own fresh batch. 1:1 is the synchronous baseline.
            if disc_updates == 1 and policy_updates == 1:
                metrics = agent.update(buffer.sample(batch_size))
            else:
                disc_m = policy_m = {}
                for _ in range(disc_updates):
                    disc_m = agent.update_discriminator(buffer.sample(batch_size))
                for _ in range(policy_updates):
                    policy_m = agent.update_policy(buffer.sample(batch_size))
                metrics = {**disc_m, **policy_m}
            updates += policy_updates
            for k in recent:
                recent[k].append(metrics[k])

        if step % log_every == 0 and recent["disc_loss"]:
            row = {k: float(np.mean(v)) for k, v in recent.items()}
            row.update(step=step, episodes=episodes,
                       sps=round(log_every / (time.time() - t0), 1))
            row.update(phase=phase, prior_alpha=agent.cfg.prior_alpha if continuous else "",
                       buffer_size=buffer.size, buffer_resets=buffer_resets, updates=updates)
            logger.writerow(row)
            log_file.flush()
            score = 'disc_cosine' if continuous else 'disc_acc'
            print(f"step {step:>8d} | ep {episodes:>5d} | "
                  f"{score} {row[score]:.3f} | "
                  f"pseudo_r {row['pseudo_reward']:+.3f} | "
                  f"H[a|s,z] {row['policy_entropy']:+.2f} | "
                  f"{row['sps']:.0f} steps/s")
            t0 = time.time()

    agent.save(run_dir / "checkpoint.pt",
               extra={"env": env_name, "step": steps, "seed": seed,
                      "max_episode_steps": config["max_episode_steps"],
                      "updates": updates, "buffer_resets": buffer_resets})
    log_file.close()
    env.close()
    print(f"Logs and checkpoint in {run_dir}/")
    return run_dir


if __name__ == "__main__":
    config = load_config()
    env_name = config["env"]
    seed = config["seed"]
    run_dir = ROOT / f"runs/{env_name}_{config['n_skills']}skills_seed{seed}"
    train_one(config, env_name, seed,
              config["disc_updates"], config["policy_updates"], run_dir)

"""Hierarchical RL on top of the learned pointnav skills (paper Section 4.2.2 /
Figure 6, Appendix C.2).

DIAYN learns the skills with no reward. Here we ask the downstream question: are
those skills useful for an ordinary goal-reaching RL problem? We freeze the
skills, introduce the goal reward

    r_g(s) = -||s - g||^2

and let a meta-controller pick which skill to run. As the paper notes, on
pointnav the skills cover the box well enough that the meta-controller only needs
to take a single action -- choose one skill per goal -- so it reduces to greedily
picking the skill that gets closest to the goal.

The key result (Figure 6): the achievable task reward grows with the number of
skills, because more skills cover more of the state space. We reproduce that
curve by subsampling the skills of a single trained model, and compare the greedy
meta-controller against a baseline that picks a skill at random (which shows that
it is the *choosing* of skills, not merely having them, that solves the task).

Run with (after training pointnav skills):

    uv run python scripts/hierarchical.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diayn.agent import DIAYNAgent
from diayn.envs import POINTNAV, make_env
from diayn.plotting import rollout
from train import load_config


def main():
    config = load_config()
    assert config["env"] == POINTNAV, "the goal-reaching task is defined on pointnav"
    seed = config["seed"]
    run_dir = ROOT / f"runs/{POINTNAV}_{config['n_skills']}skills_seed{seed}"

    agent, payload = DIAYNAgent.load(str(run_dir / "checkpoint.pt"))
    hp = config["hierarchical"]
    out_dir = run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    K = agent.n_skills

    # frozen skill trajectories (deterministic roll-outs)
    env = make_env(POINTNAV, payload["max_episode_steps"])
    trajs = [rollout(env, agent, z)[0] for z in range(K)]
    env.close()

    # 25 goals evenly spread across the box (paper: 25 goals on a grid)
    coords = np.linspace(0.1, 0.9, hp["goal_grid"])
    goals = np.array([[x, y] for x in coords for y in coords])

    # R[z, g] = -min_t ||s_t - g||^2 : how close skill z gets to goal g
    R = np.zeros((K, len(goals)))
    for z, tr in enumerate(trajs):
        d2 = ((tr[:, None, :] - goals[None, :, :]) ** 2).sum(-1)  # (T, goals)
        R[z] = -d2.min(0)

    # reward on the goal task vs. number of skills available to the meta-controller
    ks = list(range(1, K + 1))
    greedy_mean, greedy_std, random_mean = [], [], []
    for k in ks:
        greedy, rand = [], []
        n_subsets = 1 if k == K else hp["subset_samples"]
        for _ in range(n_subsets):
            subset = rng.choice(K, size=k, replace=False)
            greedy.append(R[subset].max(0).mean())   # pick the best skill per goal
            rand.append(R[subset].mean(0).mean())     # pick a skill at random
        greedy_mean.append(np.mean(greedy))
        greedy_std.append(np.std(greedy))
        random_mean.append(np.mean(rand))

    print(f"goal-reaching reward (averaged over {len(goals)} goals):")
    print(f"   1 skill : {greedy_mean[0]:+.3f}")
    print(f"  {K} skills: {greedy_mean[-1]:+.3f}   "
          f"(+{greedy_mean[-1] - greedy_mean[0]:.3f} from skill discovery)")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    greedy_mean, greedy_std = np.array(greedy_mean), np.array(greedy_std)
    ax.plot(ks, greedy_mean, "-o", color="tab:red", label="greedy meta-controller")
    ax.fill_between(ks, greedy_mean - greedy_std, greedy_mean + greedy_std,
                    color="tab:red", alpha=0.2)
    ax.plot(ks, random_mean, "--o", color="gray", label="random skill choice")
    ax.set(xlabel="number of skills", ylabel=r"task reward  $-\|s-g\|^2$",
           title=f"Hierarchical RL on pointnav ({len(goals)} goals)  [cf. Fig. 6]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "hierarchical_reward_vs_skills.png", dpi=150)
    print(f"Saved {out_dir / 'hierarchical_reward_vs_skills.png'}")

    # qualitative: which skill the meta-controller picks for each goal
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab20" if K > 10 else "tab10")
    for z, tr in enumerate(trajs):
        ax.plot(tr[:, 0], tr[:, 1], color=cmap(z % cmap.N), alpha=0.35, lw=1)
    chosen = R.argmax(0)  # best skill per goal, using all K skills
    for gi, g in enumerate(goals):
        z = chosen[gi]
        tr = trajs[z]
        nearest = tr[((tr - g) ** 2).sum(-1).argmin()]
        ax.plot([g[0], nearest[0]], [g[1], nearest[1]], color=cmap(z % cmap.N), lw=1.5)
        ax.scatter(*g, marker="*", color="black", s=90, zorder=4)
    ax.plot(0.5, 0.5, marker="o", color="black", markersize=6, zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y",
           title="Skill chosen per goal (star = goal, line to closest approach)")
    fig.tight_layout()
    fig.savefig(out_dir / "hierarchical_goals.png", dpi=150)
    print(f"Saved {out_dir / 'hierarchical_goals.png'}")


if __name__ == "__main__":
    main()

"""Visualize the learned skills of the run described by config.json.

Rolls out each skill with the deterministic policy and saves:
  - skills.png   2D trajectories (pointnav) or x-position vs time (pendulum)
  - skill_NN.gif a rendered GIF per skill (pointnav and InvertedPendulum)

Run with (after training):

    uv run python scripts/visualize.py
"""

import sys
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diayn.agent import DIAYNAgent
from diayn.envs import POINTNAV, make_env
from diayn.plotting import rollout, to_series
from train import load_config


def main():
    config = load_config()
    env_name = config["env"]
    seed = config["seed"]
    run_dir = ROOT / f"runs/{env_name}_{config['n_skills']}skills_seed{seed}"

    agent, payload = DIAYNAgent.load(str(run_dir / "checkpoint.pt"))
    out_dir = run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = agent.n_skills
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")

    env = make_env(env_name, payload["max_episode_steps"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for z in range(n):
        states, xs, ret, _ = rollout(env, agent, z)
        data = to_series(env_name, states, xs)
        color = cmap(z % cmap.N)
        if env_name == POINTNAV:
            ax.plot(data[:, 0], data[:, 1], color=color, alpha=0.8, label=f"skill {z}")
            ax.scatter(*data[-1], color=color, s=25, zorder=3)
        else:
            ax.plot(data, color=color, alpha=0.8, label=f"skill {z}")
        print(f"skill {z:>3d}: length {len(states) - 1:>4d}, task return {ret:8.1f}")
    if env_name == POINTNAV:
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y",
               title=f"DIAYN skills on 2D navigation ({n} skills)")
        ax.plot(0.5, 0.5, marker="*", color="black", markersize=15, zorder=4)
    else:
        ax.set(xlabel="time step", ylabel="x position",
               title=f"DIAYN skills on {env_name} ({n} skills)")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "skills.png", dpi=150)
    print(f"\nSaved {out_dir / 'skills.png'}")
    env.close()

    # one rendered GIF per skill
    render_env = make_env(env_name, payload["max_episode_steps"], render_mode="rgb_array")
    for z in range(n):
        _, _, ret, frames = rollout(render_env, agent, z, render=True)
        imageio.mimsave(out_dir / f"skill_{z:02d}_r{ret:.0f}.gif", frames[::2],
                        fps=20, loop=0)
    render_env.close()
    print(f"Saved {n} skill GIFs in {out_dir}/")


if __name__ == "__main__":
    main()

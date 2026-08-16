"""Rollout and drawing helpers shared by the visualization script."""

import numpy as np

from .envs import POINTNAV


def rollout(env, agent, skill, deterministic=True, render=False):
    """Roll out one episode of 'skill'; return (states, x_positions, return, frames)."""
    obs, _ = env.reset(seed=10_000 + skill)
    states, frames, xs, total_reward = [obs], [], [], 0.0
    done = False
    while not done:
        action = agent.act(obs, skill, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs)
        total_reward += reward
        if "x_position" in info:
            xs.append(info["x_position"])
        if render:
            frames.append(env.render())
        done = terminated or truncated
    return np.asarray(states), xs, total_reward, frames


def to_series(env_name, states, xs):
    """What we actually plot: the (T, 2) positions for pointnav, otherwise a
    (T,) x-position trace (obs[0] for classic control)."""
    if env_name == POINTNAV:
        return states
    return np.asarray(xs) if xs else states[:, 0]


def draw_skills(ax, env_name, series, cmap, t=None):
    """Draw one frame of skill trajectories. 'series' is a list of (skill, data)
    pairs; if 't' is given, only the first t points of each trajectory are drawn
    (used for the animation)."""
    for z, data in series:
        color = cmap(z % cmap.N)
        seg = data if t is None else data[:max(t, 2)]
        if env_name == POINTNAV:
            ax.plot(seg[:, 0], seg[:, 1], color=color, alpha=0.8)
            ax.scatter(*seg[-1], color=color, s=25, zorder=3)
        else:
            ax.plot(seg, color=color, alpha=0.8)
    if env_name == POINTNAV:
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y")
        ax.plot(0.5, 0.5, marker="*", color="black", markersize=15, zorder=4)
    else:
        ax.set(xlabel="time step", ylabel="x position")


def fig_to_rgb(fig):
    """Render a matplotlib figure to an RGB uint8 array (for GIF frames)."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

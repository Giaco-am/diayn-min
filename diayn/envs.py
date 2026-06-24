"""Environments.

The 2D navigation task from the paper (Appendix C.1): the agent starts in the
center of the unit box, observations s in [0, 1]^2 are its position, actions
a in [-0.1, 0.1]^2 directly change the position, and moves that would leave the
box are clipped back inside.

Everything else is a standard gymnasium id (we use InvertedPendulum-v5)."""

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

POINTNAV = "pointnav"


class PointNav2D(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-0.1, 0.1, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode
        self._pos = np.array([0.5, 0.5], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = np.array([0.5, 0.5], dtype=np.float32)
        return self._pos.copy(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._pos = np.clip(self._pos + action, 0.0, 1.0).astype(np.float32)
        return self._pos.copy(), 0.0, False, False, {}

    def render(self):
        # A small image of the box with the agent as a dot, so we can make GIFs.
        size = 256
        img = np.full((size, size, 3), 255, dtype=np.uint8)
        x = int(self._pos[0] * (size - 1))
        y = int((1.0 - self._pos[1]) * (size - 1))  # flip y so up is up
        r = 5
        img[max(y - r, 0):y + r, max(x - r, 0):x + r] = (200, 30, 30)
        return img


def make_env(name, max_episode_steps=None, render_mode=None):
    if name == POINTNAV:
        return TimeLimit(PointNav2D(render_mode=render_mode), max_episode_steps or 100)
    kwargs = {"render_mode": render_mode} if render_mode else {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    return gym.make(name, **kwargs)

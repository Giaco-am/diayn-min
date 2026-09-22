"""Replay buffer storing (s, z, a, s', done) transitions."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, device, skill_dim=None):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        # Keep continuous skill samples in float64 for density calculations.
        self.skill = (np.zeros(capacity, dtype=np.int64) if skill_dim is None
                      else np.zeros((capacity, skill_dim), dtype=np.float64))
        self.done = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device

    def clear(self):
        self.ptr = self.size = 0

    def add(self, obs, skill, act, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.skill[i] = skill
        self.act[i] = act
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda x: torch.as_tensor(x, device=self.device)
        return {
            "obs": to(self.obs[idx]),
            "skill": to(self.skill[idx]),
            "act": to(self.act[idx]),
            "next_obs": to(self.next_obs[idx]),
            "done": to(self.done[idx]),
        }

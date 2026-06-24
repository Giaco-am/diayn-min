"""Networks for DIAYN: the skill-conditioned policy, twin Q-functions and the
discriminator. Following Appendix C of the paper they are all MLPs with two
hidden layers of 300 units, and the skill z is fed in as a one-hot vector
concatenated to the observation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(in_dim, hidden, out_dim):
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TanhGaussianPolicy(nn.Module):
    """pi(a | s, z): a tanh-squashed Gaussian, as in SAC."""

    def __init__(self, obs_dim, n_skills, act_dim, act_limit, hidden=(300, 300)):
        super().__init__()
        self.net = mlp(obs_dim + n_skills, hidden, 2 * act_dim)
        self.register_buffer("act_limit", act_limit)

    def forward(self, obs_z, deterministic=False, with_logprob=True):
        mu, log_std = self.net(obs_z).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        dist = Normal(mu, log_std.exp())
        u = mu if deterministic else dist.rsample()
        action = torch.tanh(u) * self.act_limit

        logp = None
        if with_logprob:
            # log-prob with the tanh change-of-variables correction (the
            # numerically stable form from the SAC appendix)
            logp = dist.log_prob(u).sum(-1)
            logp = logp - (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return action, logp


class QNetwork(nn.Module):
    """Q(s, z, a)."""

    def __init__(self, obs_dim, n_skills, act_dim, hidden=(300, 300)):
        super().__init__()
        self.net = mlp(obs_dim + n_skills + act_dim, hidden, 1)

    def forward(self, obs_z, act):
        return self.net(torch.cat([obs_z, act], dim=-1)).squeeze(-1)


class Discriminator(nn.Module):
    """q(z | s): a categorical over the skills, conditioned on a single state."""

    def __init__(self, obs_dim, n_skills, hidden=(300, 300)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, n_skills)

    def forward(self, obs):
        return self.net(obs)  # unnormalized logits over skills

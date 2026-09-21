"""The DIAYN agent: a skill-conditioned SAC policy plus a skill discriminator
(Algorithm 1 of the paper).

At the start of each episode we sample a skill z -p(z) from a fixed uniform
prior. The policy gets the pseudo-reward

    r(s, z) = log q(z | s') - log p(z)        (Eq. 3)

and is trained with SAC (entropy term scaled by alpha = 0.1). The discriminator
is trained with cross-entropy to recover z from the visited states."""

from dataclasses import dataclass, field
import math

import numpy as np
import torch
import torch.nn.functional as F
# === MODIFICA DIRICHLET: distribuzione continua per prior e posterior. ===
from torch.distributions import Dirichlet

from .networks import Discriminator, QNetwork, TanhGaussianPolicy


@dataclass
class DIAYNConfig:
    n_skills: int = 20
    hidden: tuple = (300, 300)
    gamma: float = 0.99
    tau: float = 0.005          # polyak averaging for the target networks
    alpha: float = 0.1          # SAC entropy scale (paper, Appendix C)
    lr: float = 3e-4
    batch_size: int = 256
    device: str = "cpu"
    # filled in from the environment:
    obs_dim: int = 0
    act_dim: int = 0
    act_limit: list = field(default_factory=list)
    # === MODIFICA DIRICHLET: default categorico; parametri del protocollo v2. ===
    latent_type: str = "categorical"
    prior_alpha: float = 0.05
    posterior_total_concentration_min: float = 0.01
    posterior_total_concentration_max: float = 1000.0
    discriminator_grad_clip: float = 10.0


class DIAYNAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.n_skills = cfg.n_skills
        self.log_p_z = math.log(1.0 / cfg.n_skills)  # fixed uniform prior

        act_limit = torch.as_tensor(cfg.act_limit, dtype=torch.float32)
        self.actor = TanhGaussianPolicy(cfg.obs_dim, cfg.n_skills, cfg.act_dim,
                                        act_limit, cfg.hidden).to(self.device)
        self.q1 = QNetwork(cfg.obs_dim, cfg.n_skills, cfg.act_dim, cfg.hidden).to(self.device)
        self.q2 = QNetwork(cfg.obs_dim, cfg.n_skills, cfg.act_dim, cfg.hidden).to(self.device)
        self.q1_target = QNetwork(cfg.obs_dim, cfg.n_skills, cfg.act_dim, cfg.hidden).to(self.device)
        self.q2_target = QNetwork(cfg.obs_dim, cfg.n_skills, cfg.act_dim, cfg.hidden).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        # === MODIFICA DIRICHLET: cambia solo l'uscita del discriminatore. ===
        self.discriminator = Discriminator(
            cfg.obs_dim, cfg.n_skills, cfg.hidden,
            dirichlet=cfg.latent_type == "dirichlet").to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.lr)

    def one_hot(self, skill):
        return F.one_hot(skill, self.n_skills).float()

    def sample_skill(self):
        """z - p(z), the fixed uniform categorical prior."""
        # === MODIFICA DIRICHLET: conservare il campione float64, senza clamp. ===
        if self.cfg.latent_type == "dirichlet":
            return self.prior().sample().cpu().numpy()
        return int(np.random.randint(self.n_skills))

    # === MODIFICA DIRICHLET: densita in float64; solo actor/critic usano float32. ===
    def prior(self):
        concentration = torch.full((self.n_skills,), self.cfg.prior_alpha,
                                   dtype=torch.float64, device=self.device)
        return Dirichlet(concentration, validate_args=False)

    def posterior_log_prob(self, obs, skill):
        raw = self.discriminator(obs).double()
        mean = torch.softmax(raw[..., :self.n_skills], dim=-1)
        total = F.softplus(raw[..., -1]) + self.cfg.posterior_total_concentration_min
        total = total.clamp(max=self.cfg.posterior_total_concentration_max)
        concentration = (mean * total.unsqueeze(-1)).clamp_min(torch.finfo(torch.float64).tiny)
        total = concentration.sum(-1)
        mean = concentration / total.unsqueeze(-1)
        return Dirichlet(concentration, validate_args=False).log_prob(skill), mean, total

    @torch.no_grad()
    def act(self, obs, skill, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # === MODIFICA DIRICHLET: vettori continui; sul categorico sono probe OOD. ===
        skill_t = torch.as_tensor(skill, device=self.device).unsqueeze(0)
        z = self.one_hot(skill_t) if skill_t.ndim == 1 else skill_t.float()
        action, _ = self.actor(torch.cat([obs_t, z], dim=-1),
                               deterministic=deterministic, with_logprob=False)
        return action.squeeze(0).cpu().numpy()

    def update_discriminator(self, batch):
        """One discriminator step: maximize log q(z | s') (cross-entropy)."""
        skill, next_obs = batch["skill"], batch["next_obs"]
        # === MODIFICA DIRICHLET: NLL continua al posto della cross-entropy. ===
        if self.cfg.latent_type == "dirichlet":
            log_q, mean, total = self.posterior_log_prob(next_obs, skill)
            disc_loss = -log_q.mean()
            metrics = {"disc_cosine": F.cosine_similarity(mean, skill, dim=-1).mean().item(),
                       "disc_l1": (mean - skill).abs().mean().item(),
                       "posterior_concentration": total.mean().item()}
        else:
            logits = self.discriminator(next_obs)
            disc_loss = F.cross_entropy(logits, skill)
            metrics = {"disc_acc": (logits.argmax(dim=-1) == skill).float().mean().item()}
        self.disc_opt.zero_grad()
        disc_loss.backward()
        # === MODIFICA DIRICHLET: clipping usato nel training v2, parte dell'update. ===
        if self.cfg.latent_type == "dirichlet":
            norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                                  self.cfg.discriminator_grad_clip)
            metrics.update(disc_nll=disc_loss.item(), disc_grad_norm=float(norm))
        self.disc_opt.step()
        # === MODIFICA DIRICHLET: metriche specifiche, ottimizzatore condiviso. ===
        return {"disc_loss": disc_loss.item(), **metrics}

    def update_policy(self, batch, update_targets=True):
        """One SAC step (critic + actor + target polyak) on the discriminator
        pseudo-reward, with the discriminator held fixed."""
        obs, skill, act = batch["obs"], batch["skill"], batch["act"]
        next_obs, done = batch["next_obs"], batch["done"]
        # === MODIFICA DIRICHLET: il replay resta float64, l'input della rete no. ===
        z = skill.float() if self.cfg.latent_type == "dirichlet" else self.one_hot(skill)
        obs_z = torch.cat([obs, z], dim=-1)
        next_obs_z = torch.cat([next_obs, z], dim=-1)

        # pseudo-reward (Eq. 3), from the current discriminator
        with torch.no_grad():
            # === MODIFICA DIRICHLET: cambia il reward, SAC sottostante e comune. ===
            if self.cfg.latent_type == "dirichlet":
                log_q_z, _, _ = self.posterior_log_prob(next_obs, skill)
                log_p_z = self.prior().log_prob(skill)
                reward = log_q_z - log_p_z
            else:
                log_q_z = F.log_softmax(self.discriminator(next_obs), dim=-1)
                reward = log_q_z.gather(1, skill.unsqueeze(1)).squeeze(1) - self.log_p_z

        # critic update
        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs_z)
            q_next = torch.min(self.q1_target(next_obs_z, next_act),
                               self.q2_target(next_obs_z, next_act))
            target = reward + self.cfg.gamma * (1.0 - done) * (
                q_next - self.cfg.alpha * next_logp)
            # === MODIFICA DIRICHLET: cast dopo il calcolo del target in float64. ===
            target = target.to(obs.dtype)
        q1_loss = F.mse_loss(self.q1(obs_z, act), target)
        q2_loss = F.mse_loss(self.q2(obs_z, act), target)
        q_loss = q1_loss + q2_loss
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # actor update
        pi_act, pi_logp = self.actor(obs_z)
        q_pi = torch.min(self.q1(obs_z, pi_act), self.q2(obs_z, pi_act))
        actor_loss = (self.cfg.alpha * pi_logp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if update_targets:
            self.update_targets()

        return {
            # === MODIFICA DIRICHLET: log del prior per confrontare le run v2. ===
            **({"prior_log_prob": log_p_z.mean().item()}
               if self.cfg.latent_type == "dirichlet" else {}),
            "pseudo_reward": reward.mean().item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "policy_entropy": -pi_logp.mean().item(),
        }

    @torch.no_grad()
    def update_targets(self):
        """One Polyak update of both target critics (historically once per SAC)."""
        for net, target_net in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
            for p, p_t in zip(net.parameters(), target_net.parameters()):
                p_t.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p)

    def update(self, batch):
        """Synchronous update (the paper's baseline): one discriminator step and
        one policy step on the *same* batch. The discriminator goes first so the
        pseudo-reward reflects the freshly updated q(z | s')."""
        return {**self.update_discriminator(batch), **self.update_policy(batch)}

    def save(self, path, extra=None):
        payload = {
            # === MODIFICA DIRICHLET: il modello finale identifica il tipo di skill. ===
            "latent_type": self.cfg.latent_type,
            "config": vars(self.cfg),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = dict(payload["config"])
        cfg_dict["device"] = device
        cfg_dict["hidden"] = tuple(cfg_dict["hidden"])
        agent = cls(DIAYNConfig(**cfg_dict))
        agent.actor.load_state_dict(payload["actor"])
        agent.q1.load_state_dict(payload["q1"])
        agent.q2.load_state_dict(payload["q2"])
        agent.q1_target.load_state_dict(payload["q1_target"])
        agent.q2_target.load_state_dict(payload["q2_target"])
        agent.discriminator.load_state_dict(payload["discriminator"])
        return agent, payload

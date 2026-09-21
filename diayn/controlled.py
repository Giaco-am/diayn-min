"""Controlled update blocks. Historical entry points do not call this module."""
import numpy as np


def update_block(agent, replay, batch_size, n, m):
    """N fresh discriminator batches, M fresh complete SAC batches, one target.

    Discriminator metrics use logits BEFORE their respective optimizer step.
    SAC reward uses the discriminator AFTER all N discriminator steps. Loss and
    entropy measurements retain the original agent's measurement points.
    """
    disc = [agent.update_discriminator(replay.sample(batch_size)) for _ in range(n)]
    sac = [agent.update_policy(replay.sample(batch_size), update_targets=False)
           for _ in range(m)]
    agent.update_targets()
    metrics = {k: float(np.mean([r[k] for r in group]))
               for group in (disc, sac) for k in group[0]}
    # critic_updates counts joint optimizer steps; each updates BOTH critics.
    counts = dict(disc_updates=n, critic_updates=m, actor_updates=m, target_updates=1)
    return metrics, counts

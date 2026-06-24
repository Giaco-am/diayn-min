# diayn/ Module Architecture

How the files in `diayn/` interact with each other and with the rest of the project.

## File Overview

```
diayn/
├── __init__.py      # empty, marks the directory as a Python package
├── networks.py      # neural network definitions (policy, Q-functions, discriminator)
├── agent.py         # DIAYN agent: wires the networks together with optimizers and update logic
├── buffer.py        # replay buffer for storing (s, z, a, s', done) transitions
├── envs.py          # environment factory + custom PointNav2D environment
├── eval.py          # per-skill evaluation (task return measurement)
├── plotting.py      # rollout + drawing utilities for visualization
```

## Dependency Graph

```
networks.py
     │
     ▼
  agent.py ──────────────────────────────────┐
     │                                       │
     │                                       ▼
     │                                   buffer.py
     │                                  (used together
     │                                   by scripts/)
     ▼
  envs.py ◄──── eval.py
     ▲
     │
  plotting.py
```

## How They Interact

### networks.py → agent.py

`networks.py` defines three network classes — `TanhGaussianPolicy`, `QNetwork`, and `Discriminator` — as standalone `nn.Module` subclasses. It has no imports from the rest of the package.

`agent.py` imports all three and instantiates them inside `DIAYNAgent.__init__()`:
- **1 policy** (`TanhGaussianPolicy`): `pi(a | s, z)`, receives `[obs, one_hot(z)]` as input
- **2 Q-networks + 2 targets** (`QNetwork`): `Q(s, z, a)`, used as twin critics with Polyak-averaged targets
- **1 discriminator** (`Discriminator`): `q(z | s)`, maps raw observations to skill logits

The agent creates separate Adam optimizers for each group (actor, critics, discriminator).

### agent.py ↔ buffer.py

These two don't import each other, but they share a **batch dict interface**. `ReplayBuffer.sample()` returns a dict with keys `{"obs", "skill", "act", "next_obs", "done"}` as tensors, and `DIAYNAgent.update()` / `update_discriminator()` / `update_policy()` consume exactly that dict format. The training script (`scripts/train.py`) is the glue that calls `buffer.sample()` and passes the result to `agent.update()`.

### envs.py (standalone)

`envs.py` defines:
- `PointNav2D`: a custom Gymnasium environment (2D navigation in a unit box)
- `make_env(name, ...)`: factory that either creates a `PointNav2D` or delegates to `gym.make()` for standard envs (e.g. `InvertedPendulum-v5`)

It has no imports from the rest of the package.

### eval.py → envs.py, agent.py

`eval.py` imports `make_env` from `envs.py` to create evaluation environments. Its `skill_returns()` function takes an `agent` (a `DIAYNAgent` instance) and calls `agent.act()` and `agent.n_skills` to roll out each skill deterministically and measure the real task return (which DIAYN never uses during training).

### plotting.py → envs.py

`plotting.py` imports the `POINTNAV` constant from `envs.py` to branch between 2D trajectory plots (for pointnav) and 1D x-position traces (for classic control envs). Its `rollout()` function takes an `agent` and `env` as arguments (duck-typed, not imported) and calls `agent.act()` and `env.step()` / `env.reset()`.

## How Scripts Use the Package

The top-level `scripts/` directory contains the entry points that wire everything together:

### scripts/train.py
```
imports: agent.DIAYNAgent, agent.DIAYNConfig, buffer.ReplayBuffer, envs.make_env
```
Creates an environment, a config, an agent, and a replay buffer. Runs the training loop: collect transitions → store in buffer → sample batch → call `agent.update(batch)`. Periodically saves checkpoints via `agent.save()`.

### scripts/visualize.py
```
imports: agent.DIAYNAgent, envs.POINTNAV, envs.make_env, plotting.rollout, plotting.to_series
```
Loads a trained checkpoint with `DIAYNAgent.load()`, rolls out each skill using `plotting.rollout()`, and draws trajectory plots.

## Data Flow During Training

```
env.reset() / env.step()
        │
        ▼
   observations, actions, done
        │
        ▼
   buffer.add(obs, skill, act, next_obs, done)
        │
        ▼
   batch = buffer.sample(batch_size)
        │
        ├──► agent.update_discriminator(batch)
        │         └─ discriminator(next_obs) → cross-entropy loss with skill labels
        │
        └──► agent.update_policy(batch)
                  ├─ discriminator(next_obs) → pseudo-reward: log q(z|s') - log p(z)
                  ├─ Q-networks + targets → critic loss (MSE on Bellman target)
                  ├─ policy(obs, z) → actor loss (SAC objective with entropy)
                  └─ Polyak averaging → update target Q-networks
```

## Key Design Decisions

- **No circular imports**: the dependency graph is a DAG. `networks.py` and `envs.py` are leaf modules with no intra-package imports. `agent.py` only imports from `networks.py`. `eval.py` and `plotting.py` import from `envs.py` but not from each other.
- **Duck-typed agent interface**: `eval.py` and `plotting.py` call `agent.act()` and read `agent.n_skills` but don't import `DIAYNAgent` — any object with the same interface would work.
- **Batch dict as the contract**: the replay buffer and agent communicate through a plain dict of tensors, keeping them decoupled.

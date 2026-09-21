"""Save PointNav trajectories and metrics; no training or intermediate snapshots."""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diayn.agent import DIAYNAgent
from diayn.envs import make_env


def rollout(env, agent, skill, seed):
    if agent.cfg.latent_type == "categorical":
        skill = np.asarray(skill, dtype=np.float32)
        skill = skill / skill.sum()  # Gli input misti del categorico sono OOD.
    obs, _ = env.reset(seed=seed)
    states = [obs]
    while True:
        obs, _, terminated, truncated, _ = env.step(agent.act(obs, skill, deterministic=True))
        states.append(obs)
        if terminated or truncated:
            return np.asarray(states)


def behavior(trajectories, bins=20):
    endpoints = np.asarray([states[-1] for states in trajectories])
    separation = np.mean([float(np.linalg.norm(a - b)) for a, b in combinations(endpoints, 2)])
    cells = np.minimum((np.clip(np.concatenate(trajectories), 0, 1) * bins).astype(int), bins - 1)
    return float(separation), len(np.unique(cells, axis=0)) / bins**2


def trajectory_distance(a, b):
    length = max(len(a), len(b))
    a = np.pad(a, ((0, length - len(a)), (0, 0)), mode="edge")
    b = np.pad(b, ((0, length - len(b)), (0, 0)), mode="edge")
    return float(np.sqrt(np.mean(np.sum((a - b)**2, axis=-1))))


def save_trajectories(path, skills, trajectories):
    np.savez_compressed(path, skills=skills,
                        **{f"states_{i}": states for i, states in enumerate(trajectories)})


def evaluate_checkpoint(checkpoint, seed=12345):
    agent, payload = DIAYNAgent.load(checkpoint)
    if payload["env"] != "pointnav":
        raise ValueError("Questa valutazione geometrica e definita per PointNav.")
    out = Path(checkpoint).parent / "evaluation"
    out.mkdir(exist_ok=True)
    env = make_env(payload["env"], payload["max_episode_steps"])
    skills = np.random.default_rng(seed).dirichlet(np.ones(agent.n_skills), size=256)
    trajectories = [rollout(env, agent, z, seed) for z in skills]
    save_trajectories(out / "interior_trajectories.npz", skills, trajectories)
    separation, coverage = behavior(trajectories)
    distances = np.linalg.norm(skills[:, None] - skills[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    pairs = sorted({tuple(sorted((i, int(j)))) for i, j in enumerate(distances.argmin(1))})
    endpoint_sensitivity = [np.linalg.norm(trajectories[i][-1] - trajectories[j][-1])
                            / distances[i, j] for i, j in pairs]
    trajectory_sensitivity = [trajectory_distance(trajectories[i], trajectories[j])
                              / distances[i, j] for i, j in pairs]
    summary = dict(checkpoint_step=payload["step"], updates=payload["updates"],
                   latent_type=agent.cfg.latent_type, evaluation_seed=seed,
                   interior_samples=256, points_per_edge=101, coverage_bins=20,
                   interior_separation=separation, interior_coverage=coverage,
                   interior_p95_endpoint_sensitivity=float(np.quantile(endpoint_sensitivity, .95)),
                   interior_p95_trajectory_sensitivity=float(np.quantile(trajectory_sensitivity, .95)))
    edge_metrics = []
    eye = np.eye(agent.n_skills, dtype=np.float64)
    for left, right in combinations(range(agent.n_skills), 2):
        edge_skills = np.asarray([(1 - t) * eye[left] + t * eye[right]
                                 for t in np.linspace(0, 1, 101)])
        edge_skills = (1 - 1e-6 * agent.n_skills) * edge_skills + 1e-6
        edge_skills /= edge_skills.sum(axis=1, keepdims=True)
        paths = [rollout(env, agent, z, seed) for z in edge_skills]
        save_trajectories(out / f"edge_{left}_{right}_trajectories.npz", edge_skills, paths)
        endpoints = np.asarray([p[-1] for p in paths])
        adjacent = np.linalg.norm(np.diff(endpoints, axis=0), axis=-1)
        direct = float(np.linalg.norm(endpoints[-1] - endpoints[0]))
        edge_metrics.append(dict(edge=f"{left}-{right}", max_jump=float(adjacent.max()),
                                 path_to_direct_ratio=float(adjacent.sum()) / direct if direct > 1e-12 else None))
    ratios = [e["path_to_direct_ratio"] for e in edge_metrics if e["path_to_direct_ratio"] is not None]
    summary.update(max_adjacent_endpoint_distance=max(e["max_jump"] for e in edge_metrics),
                   mean_path_to_direct_ratio=float(np.mean(ratios)) if ratios else None,
                   degenerate_edges=len(edge_metrics) - len(ratios))
    if agent.cfg.latent_type == "categorical":
        native = [rollout(env, agent, z, seed) for z in eye]
        save_trajectories(out / "categorical_native_trajectories.npz", eye, native)
        separation, coverage = behavior(native)
        summary.update(native_categorical_separation=separation, native_categorical_coverage=coverage)
    env.close()
    (out / "summary.json").write_text(json.dumps({**summary, "edges": edge_metrics}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(evaluate_checkpoint(args.checkpoint), indent=2))

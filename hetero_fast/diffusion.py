from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .data import HeteroDataset, minmax_scale, node_types, relation_groups, row_normalize


TRUTH_DIR = Path(__file__).resolve().parent.parent / "hetero_fast_results" / "truth_cache"


DEFAULT_DIFFUSION = {
    "ACM": {"sir_beta": 0.045, "sir_gamma": 0.030, "si_beta": 0.040, "seed_k": 20, "mc_runs": 4, "t_steps": 20},
    "DBLP": {"sir_beta": 0.070, "sir_gamma": 0.028, "si_beta": 0.062, "seed_k": 50, "mc_runs": 3, "t_steps": 20},
    "Yelp": {"sir_beta": 0.030, "sir_gamma": 0.018, "si_beta": 0.024, "seed_k": 15, "mc_runs": 2, "t_steps": 20},
}


@dataclass
class HeteroDiffusionArtifacts:
    truth_sir: np.ndarray
    truth_si: np.ndarray
    truth_fused: np.ndarray
    config: Dict[str, float]
    sir_curves: List[List[float]]
    si_curves: List[List[float]]


@dataclass
class DiffusionState:
    rel_weight: np.ndarray
    neighbors: List[np.ndarray]
    rel_ids: List[np.ndarray]
    ntypes: np.ndarray


def relation_type_weights(data: HeteroDataset) -> np.ndarray:
    feats = row_normalize(data.features)
    weights = np.zeros(len(data.relation_names), dtype=np.float32)
    groups = relation_groups(data)
    for rel_id, (src, dst, _) in groups.items():
        if src.size == 0:
            weights[rel_id] = 0.5
            continue
        sample = min(src.size, 50000)
        if src.size > sample:
            choose = np.random.default_rng(42 + rel_id).choice(src.size, size=sample, replace=False)
            src_use = src[choose]
            dst_use = dst[choose]
        else:
            src_use = src
            dst_use = dst
        sims = np.einsum("ij,ij->i", feats[src_use], feats[dst_use])
        sims = np.clip(sims, 0.0, 1.0)
        weights[rel_id] = float(0.35 + 0.65 * np.mean(sims))
    return weights


def build_neighbor_lists(data: HeteroDataset) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    groups = relation_groups(data)
    neighbors: List[List[int]] = [[] for _ in range(data.num_nodes)]
    rel_ids: List[List[int]] = [[] for _ in range(data.num_nodes)]
    for rel_id, (src, dst, _) in groups.items():
        for s, d in zip(src.tolist(), dst.tolist()):
            neighbors[s].append(d)
            rel_ids[s].append(rel_id)
    neigh_arr = [np.asarray(x, dtype=np.int64) for x in neighbors]
    rel_arr = [np.asarray(x, dtype=np.int64) for x in rel_ids]
    return neigh_arr, rel_arr, node_types(data)


def build_diffusion_state(data: HeteroDataset) -> DiffusionState:
    rel_weight = relation_type_weights(data)
    neighbors, rel_ids, ntypes = build_neighbor_lists(data)
    return DiffusionState(rel_weight=rel_weight, neighbors=neighbors, rel_ids=rel_ids, ntypes=ntypes)


def edge_activation_weight(
    data: HeteroDataset,
    src: int,
    dst: int,
    rel_id: int,
    rel_weight: np.ndarray,
    ntypes: np.ndarray,
) -> float:
    fs = data.features[src]
    fd = data.features[dst]
    ss = data.semantic_features[src]
    sd = data.semantic_features[dst]
    feat_sim = float(np.dot(fs, fd) / (np.linalg.norm(fs) * np.linalg.norm(fd) + 1e-8))
    sem_sim = float(np.dot(ss, sd) / (np.linalg.norm(ss) * np.linalg.norm(sd) + 1e-8))
    feat_sim = max(0.0, min(1.0, 0.5 * (feat_sim + 1.0)))
    sem_sim = max(0.0, min(1.0, 0.5 * (sem_sim + 1.0)))
    type_bonus = 1.0 if ntypes[src] != ntypes[dst] else 0.85
    return float(rel_weight[rel_id] * type_bonus * (0.55 * feat_sim + 0.45 * sem_sim))


def simulate_sir_single(
    data: HeteroDataset,
    seed: int,
    t_steps: int,
    beta: float,
    gamma: float,
    neighbors: List[np.ndarray],
    rel_ids: List[np.ndarray],
    rel_weight: np.ndarray,
    ntypes: np.ndarray,
) -> List[float]:
    status = np.zeros(data.num_nodes, dtype=np.int8)
    infected = {int(seed)}
    recovered = set()
    status[seed] = 1
    target_covered = [1.0 / data.target_count if seed < data.target_count else 0.0]
    for _ in range(1, t_steps):
        new_infected = set()
        new_recovered = set()
        for node in infected:
            nbs = neighbors[node]
            rls = rel_ids[node]
            for idx, neigh in enumerate(nbs.tolist()):
                if status[neigh] != 0:
                    continue
                rel_id = int(rls[idx])
                act = edge_activation_weight(data, node, neigh, rel_id, rel_weight, ntypes)
                p = min(0.98, beta * act)
                if random.random() < p:
                    new_infected.add(neigh)
            if random.random() < gamma:
                new_recovered.add(node)
        for node in new_infected:
            status[node] = 1
        for node in new_recovered:
            status[node] = 2
        infected.update(new_infected)
        infected.difference_update(new_recovered)
        recovered.update(new_recovered)
        covered = np.count_nonzero(status[: data.target_count] != 0) / data.target_count
        target_covered.append(float(covered))
    return target_covered


def simulate_si_single(
    data: HeteroDataset,
    seed: int,
    t_steps: int,
    beta: float,
    neighbors: List[np.ndarray],
    rel_ids: List[np.ndarray],
    rel_weight: np.ndarray,
    ntypes: np.ndarray,
) -> List[float]:
    infected = {int(seed)}
    status = np.zeros(data.num_nodes, dtype=bool)
    status[seed] = True
    curve = [1.0 / data.target_count if seed < data.target_count else 0.0]
    for _ in range(1, t_steps):
        new_nodes = set()
        for node in infected:
            nbs = neighbors[node]
            rls = rel_ids[node]
            for idx, neigh in enumerate(nbs.tolist()):
                if status[neigh]:
                    continue
                rel_id = int(rls[idx])
                act = edge_activation_weight(data, node, neigh, rel_id, rel_weight, ntypes)
                if random.random() < min(0.98, beta * act):
                    new_nodes.add(neigh)
        for node in new_nodes:
            status[node] = True
        infected.update(new_nodes)
        curve.append(float(np.count_nonzero(status[: data.target_count]) / data.target_count))
    return curve


def monte_carlo_truth(
    data: HeteroDataset,
    force: bool = False,
    config_override: Dict[str, float] | None = None,
) -> HeteroDiffusionArtifacts:
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    cache_npz = TRUTH_DIR / f"{data.name.lower()}_truth.npz"
    cache_json = TRUTH_DIR / f"{data.name.lower()}_truth_meta.json"
    cfg = dict(DEFAULT_DIFFUSION[data.name])
    if config_override:
        cfg.update(config_override)
    if cache_npz.exists() and cache_json.exists() and not force:
        arr = np.load(cache_npz, allow_pickle=True)
        meta = json.loads(cache_json.read_text(encoding="utf-8"))
        return HeteroDiffusionArtifacts(
            truth_sir=arr["truth_sir"].astype(np.float32),
            truth_si=arr["truth_si"].astype(np.float32),
            truth_fused=arr["truth_fused"].astype(np.float32),
            config=meta["config"],
            sir_curves=arr["sir_curves"].tolist(),
            si_curves=arr["si_curves"].tolist(),
        )

    started = time.perf_counter()
    state = build_diffusion_state(data)
    sir_scores = np.zeros(data.target_count, dtype=np.float32)
    si_scores = np.zeros(data.target_count, dtype=np.float32)
    sir_curves: List[List[float]] = []
    si_curves: List[List[float]] = []

    for seed in range(data.target_count):
        sir_runs = []
        si_runs = []
        for run_idx in range(int(cfg["mc_runs"])):
            random.seed(1000 + seed * 997 + run_idx)
            sir_runs.append(
                simulate_sir_single(
                    data,
                    seed,
                    int(cfg["t_steps"]),
                    float(cfg["sir_beta"]),
                    float(cfg["sir_gamma"]),
                    state.neighbors,
                    state.rel_ids,
                    state.rel_weight,
                    state.ntypes,
                )
            )
            random.seed(2000 + seed * 997 + run_idx)
            si_runs.append(
                simulate_si_single(
                    data,
                    seed,
                    int(cfg["t_steps"]),
                    float(cfg["si_beta"]),
                    state.neighbors,
                    state.rel_ids,
                    state.rel_weight,
                    state.ntypes,
                )
            )
        sir_mean = np.mean(np.asarray(sir_runs, dtype=np.float32), axis=0)
        si_mean = np.mean(np.asarray(si_runs, dtype=np.float32), axis=0)
        sir_curves.append(sir_mean.tolist())
        si_curves.append(si_mean.tolist())
        sir_scores[seed] = float(sir_mean[-1])
        si_scores[seed] = float(si_mean[-1])

    truth_fused = minmax_scale(0.55 * minmax_scale(sir_scores) + 0.45 * minmax_scale(si_scores))
    np.savez_compressed(
        cache_npz,
        truth_sir=sir_scores.astype(np.float32),
        truth_si=si_scores.astype(np.float32),
        truth_fused=truth_fused.astype(np.float32),
        sir_curves=np.asarray(sir_curves, dtype=np.float32),
        si_curves=np.asarray(si_curves, dtype=np.float32),
    )
    cache_json.write_text(
        json.dumps(
            {
                "dataset": data.name,
                "config": cfg,
                "runtime_sec": float(time.perf_counter() - started),
                "mean_sir": float(np.mean(sir_scores)),
                "mean_si": float(np.mean(si_scores)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return HeteroDiffusionArtifacts(
        truth_sir=sir_scores,
        truth_si=si_scores,
        truth_fused=truth_fused,
        config=cfg,
        sir_curves=sir_curves,
        si_curves=si_curves,
    )


def simulate_seed_set(
    data: HeteroDataset,
    seeds: List[int],
    mode: str,
    t_steps: int,
    beta: float,
    gamma: float,
    runs: int,
    state: DiffusionState | None = None,
) -> List[float]:
    state = state or build_diffusion_state(data)
    curves = []
    for run_idx in range(runs):
        random.seed(50000 + run_idx)
        if mode == "SIR":
            status = np.zeros(data.num_nodes, dtype=np.int8)
            infected = set(int(x) for x in seeds)
            recovered = set()
            for s in infected:
                status[s] = 1
            curve = [float(np.count_nonzero(status[: data.target_count] != 0) / data.target_count)]
            for _ in range(1, t_steps):
                new_infected = set()
                new_recovered = set()
                for node in infected:
                    nbs = state.neighbors[node]
                    rls = state.rel_ids[node]
                    for idx, neigh in enumerate(nbs.tolist()):
                        if status[neigh] != 0:
                            continue
                        rel_id = int(rls[idx])
                        act = edge_activation_weight(data, node, neigh, rel_id, state.rel_weight, state.ntypes)
                        if random.random() < min(0.98, beta * act):
                            new_infected.add(neigh)
                    if random.random() < gamma:
                        new_recovered.add(node)
                for node in new_infected:
                    status[node] = 1
                for node in new_recovered:
                    status[node] = 2
                infected.update(new_infected)
                infected.difference_update(new_recovered)
                recovered.update(new_recovered)
                curve.append(float(np.count_nonzero(status[: data.target_count] != 0) / data.target_count))
            curves.append(curve)
        else:
            infected = set(int(x) for x in seeds)
            status = np.zeros(data.num_nodes, dtype=bool)
            if seeds:
                status[np.asarray(seeds, dtype=np.int64)] = True
            curve = [float(np.count_nonzero(status[: data.target_count]) / data.target_count)]
            for _ in range(1, t_steps):
                new_nodes = []
                for src in infected:
                    nbs = state.neighbors[src]
                    rls = state.rel_ids[src]
                    for idx, neigh in enumerate(nbs.tolist()):
                        if status[neigh]:
                            continue
                        rel_id = int(rls[idx])
                        act = edge_activation_weight(data, src, neigh, rel_id, state.rel_weight, state.ntypes)
                        if random.random() < min(0.98, beta * act):
                            new_nodes.append(neigh)
                if new_nodes:
                    status[np.asarray(new_nodes, dtype=np.int64)] = True
                    infected.update(int(x) for x in new_nodes)
                curve.append(float(np.count_nonzero(status[: data.target_count]) / data.target_count))
            curves.append(curve)
    return np.mean(np.asarray(curves, dtype=np.float32), axis=0).tolist()

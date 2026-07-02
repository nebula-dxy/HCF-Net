import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch

import credible_experiment_runner as credible
import experiment_runner as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_hcfnet_credible" / "true_sir"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_weighted_full_hetero_graph() -> Tuple[nx.Graph, List[List[Tuple[int, float]]], np.ndarray, Dict[str, float]]:
    payload = base.safe_torch_load(ROOT / "external_prepared" / "ACM" / "acm_nie.pt")
    num_nodes = int(payload["num_nodes"])
    target_count = int(payload["target_count"])
    features = np.asarray(payload["semantic_features"], dtype=np.float32)
    src = np.asarray(payload["edges"][0], dtype=np.int64)
    dst = np.asarray(payload["edges"][1], dtype=np.int64)

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats = features / norms

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for s, d in zip(src.tolist(), dst.tolist()):
        if s == d:
            continue
        sim = float(np.clip(np.dot(feats[s], feats[d]), 0.0, 1.0))
        weight = 0.15 + 0.85 * sim
        if graph.has_edge(s, d):
            if weight > graph[s][d].get("weight", 0.0):
                graph[s][d]["weight"] = weight
        else:
            graph.add_edge(s, d, weight=weight)

    weighted_adj = nx.to_scipy_sparse_array(graph, nodelist=range(num_nodes), weight="weight", dtype=np.float32)
    radius = credible.power_radius(weighted_adj)
    cfg = {
        "sir_beta": float(np.clip(1.15 / radius, 0.015, 0.25)),
        "sir_gamma": 0.025,
        "target_count": target_count,
        "num_nodes": num_nodes,
        "radius": float(radius),
    }
    target_mask = np.zeros(num_nodes, dtype=np.float32)
    target_mask[:target_count] = 1.0
    nbrs = [[] for _ in range(num_nodes)]
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        nbrs[u].append((int(v), w))
        nbrs[v].append((int(u), w))
    return graph, nbrs, target_mask, cfg


def simulate_target_coverage(neighbors: List[List[Tuple[int, float]]], target_mask: np.ndarray, beta: float, gamma: float, seed: int, t_steps: int) -> float:
    status = np.zeros(len(target_mask), dtype=np.int8)
    infected = {seed}
    recovered = set()
    status[seed] = 1
    for _ in range(t_steps):
        new_infected = set()
        new_recovered = set()
        for node in infected:
            for neigh, weight in neighbors[node]:
                if status[neigh] != 0:
                    continue
                if random.random() < min(0.98, beta * weight):
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
    covered = np.where((status != 0) & (target_mask > 0), 1.0, 0.0).astype(np.float32)
    return float(covered.sum() / max(float(target_mask.sum()), 1.0))


def main() -> None:
    dataset = "ACM"
    runs = 8
    t_steps = 20
    started = time.perf_counter()
    graph, neighbors, target_mask, cfg = build_weighted_full_hetero_graph()
    scores = np.zeros(int(cfg["target_count"]), dtype=np.float32)
    state = random.getstate()

    for node in range(int(cfg["target_count"])):
        vals = []
        for run_idx in range(runs):
            random.seed(credible.SEED + run_idx)
            vals.append(simulate_target_coverage(neighbors, target_mask, float(cfg["sir_beta"]), float(cfg["sir_gamma"]), node, t_steps))
        scores[node] = float(np.mean(vals))
        if node % 200 == 0 or node == int(cfg["target_count"]) - 1:
            print(f"node={node+1}/{int(cfg['target_count'])} score={scores[node]:.6f} elapsed={time.perf_counter()-started:.1f}s")

    random.setstate(state)
    npy_path = OUT_DIR / "ACM_true_sir_full_hetero_target_coverage_t20_runs8.npy"
    meta_path = OUT_DIR / "ACM_true_sir_full_hetero_target_coverage_t20_runs8_meta.json"
    np.save(npy_path, scores)
    meta = {
        "dataset": dataset,
        "definition": "full heterogeneous graph SIR, coverage counted on target-type nodes only",
        "runs": runs,
        "t_steps": t_steps,
        **cfg,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "top10_nodes": np.argsort(scores)[::-1][:10].astype(int).tolist(),
        "top10_scores": scores[np.argsort(scores)[::-1][:10]].astype(float).tolist(),
        "total_runtime_sec": float(time.perf_counter() - started),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved", npy_path)
    print("saved", meta_path)


if __name__ == "__main__":
    main()

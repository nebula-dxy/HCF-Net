import json
import argparse
import os
import random
import time
from multiprocessing import Pool
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


WORK_NEIGHBORS = None
WORK_TARGET_MASK = None
WORK_BETA = 0.0
WORK_GAMMA = 0.0
WORK_RUNS = 0
WORK_T_STEPS = 0


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


def init_worker(neighbors, target_mask, beta: float, gamma: float, runs: int, t_steps: int) -> None:
    global WORK_NEIGHBORS, WORK_TARGET_MASK, WORK_BETA, WORK_GAMMA, WORK_RUNS, WORK_T_STEPS
    WORK_NEIGHBORS = neighbors
    WORK_TARGET_MASK = target_mask
    WORK_BETA = float(beta)
    WORK_GAMMA = float(gamma)
    WORK_RUNS = int(runs)
    WORK_T_STEPS = int(t_steps)


def score_one_seed(seed: int) -> Tuple[int, float]:
    vals = []
    for run_idx in range(WORK_RUNS):
        random.seed(credible.SEED + 1000003 * int(seed) + run_idx)
        vals.append(
            simulate_target_coverage(
                WORK_NEIGHBORS,
                WORK_TARGET_MASK,
                WORK_BETA,
                WORK_GAMMA,
                int(seed),
                WORK_T_STEPS,
            )
        )
    return int(seed), float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ACM full heterogeneous SIR true scores")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--t-steps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Only compute the first N target nodes for quick validation; 0 means all")
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker processes; 0 means auto")
    args = parser.parse_args()

    dataset = "ACM"
    runs = args.runs
    t_steps = args.t_steps
    started = time.perf_counter()
    graph, neighbors, target_mask, cfg = build_weighted_full_hetero_graph()
    target_count = int(cfg["target_count"])
    limit = target_count if args.limit <= 0 else min(int(args.limit), target_count)
    scores = np.zeros(limit, dtype=np.float32)
    state = random.getstate()
    workers = int(args.workers) if int(args.workers) > 0 else max(1, min(8, (os.cpu_count() or 1) - 1))

    print(
        f"start ACM full hetero SIR: target_count={target_count} limit={limit} "
        f"runs={runs} t_steps={t_steps} workers={workers}"
    )
    if workers <= 1:
        init_worker(neighbors, target_mask, float(cfg["sir_beta"]), float(cfg["sir_gamma"]), runs, t_steps)
        for node in range(limit):
            _, score = score_one_seed(node)
            scores[node] = score
            if node % 25 == 0 or node == limit - 1:
                print(f"node={node+1}/{limit} score={scores[node]:.6f} elapsed={time.perf_counter()-started:.1f}s")
    else:
        with Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(neighbors, target_mask, float(cfg["sir_beta"]), float(cfg["sir_gamma"]), runs, t_steps),
        ) as pool:
            for finished, (node, score) in enumerate(pool.imap_unordered(score_one_seed, range(limit), chunksize=8), start=1):
                scores[node] = score
                if finished % 25 == 0 or finished == limit:
                    print(f"done={finished}/{limit} latest_node={node+1} score={score:.6f} elapsed={time.perf_counter()-started:.1f}s")

    random.setstate(state)
    suffix = f"t{t_steps}_runs{runs}" + ("" if limit == target_count else f"_limit{limit}")
    npy_path = OUT_DIR / f"ACM_true_sir_full_hetero_target_coverage_{suffix}.npy"
    meta_path = OUT_DIR / f"ACM_true_sir_full_hetero_target_coverage_{suffix}_meta.json"
    np.save(npy_path, scores)
    meta = {
        "dataset": dataset,
        "definition": "full heterogeneous graph SIR, coverage counted on target-type nodes only",
        "runs": runs,
        "t_steps": t_steps,
        "computed_target_count": limit,
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

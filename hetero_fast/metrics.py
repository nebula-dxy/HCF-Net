from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from .data import HeteroDataset, minmax_scale, target_adjacency
from .diffusion import DiffusionState, HeteroDiffusionArtifacts, simulate_seed_set


def ranking_metrics(target: np.ndarray, pred: np.ndarray, eval_idx: np.ndarray, k: int = 100) -> Dict[str, float]:
    truth = target[eval_idx].reshape(1, -1)
    score = pred[eval_idx].reshape(1, -1)
    ndcg = float(ndcg_score(truth, score, k=min(k, truth.shape[1])))
    spear = float(spearmanr(target[eval_idx], pred[eval_idx]).statistic)
    if not np.isfinite(spear):
        spear = 0.0
    return {"NDCG@100": ndcg, "Spearman": spear}


def topk_from_scores(scores: np.ndarray, k: int) -> List[int]:
    order = np.argsort(-scores)
    return [int(x) for x in order[:k]]


def discounted_topk(scores: np.ndarray, adj, k: int, penalty: float) -> List[int]:
    scores = np.asarray(scores, dtype=np.float32).copy()
    chosen: List[int] = []
    taken = np.zeros(scores.shape[0], dtype=bool)
    csr = adj.tocsr()
    for _ in range(k):
        masked = np.where(taken, -np.inf, scores)
        node = int(np.argmax(masked))
        if taken[node]:
            break
        chosen.append(node)
        taken[node] = True
        neigh = csr.indices[csr.indptr[node] : csr.indptr[node + 1]]
        scores[neigh] *= float(penalty)
    return chosen


def diversity_ls(data: HeteroDataset, seeds: List[int]) -> float:
    adj = target_adjacency(data)
    graph = nx.from_scipy_sparse_array(adj)
    if len(seeds) < 2:
        return 0.0
    dists = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            try:
                dists.append(nx.shortest_path_length(graph, seeds[i], seeds[j]))
            except nx.NetworkXNoPath:
                dists.append(float(data.target_count))
    return float(np.mean(dists)) if dists else 0.0


def evaluate_method(
    data: HeteroDataset,
    truth: HeteroDiffusionArtifacts,
    scores: np.ndarray,
    runtime_sec: float,
    seeds: List[int] | None = None,
    diffusion_state: DiffusionState | None = None,
) -> Dict[str, object]:
    scaled = minmax_scale(scores[: data.target_count])
    rows = ranking_metrics(truth.truth_fused, scaled, data.test_idx)
    seeds = seeds or topk_from_scores(scaled, int(truth.config["seed_k"]))
    sir_curve = simulate_seed_set(
        data,
        seeds,
        mode="SIR",
        t_steps=int(truth.config["t_steps"]),
        beta=float(truth.config["sir_beta"]),
        gamma=float(truth.config["sir_gamma"]),
        runs=max(12, int(truth.config["mc_runs"]) // 4),
        state=diffusion_state,
    )
    si_curve = simulate_seed_set(
        data,
        seeds,
        mode="SI",
        t_steps=int(truth.config["t_steps"]),
        beta=float(truth.config["si_beta"]),
        gamma=0.0,
        runs=max(12, int(truth.config["mc_runs"]) // 4),
        state=diffusion_state,
    )
    rows["F(20)-SIR"] = float(sir_curve[-1])
    rows["F(20)-SI"] = float(si_curve[-1])
    rows["Ls"] = diversity_ls(data, seeds)
    rows["RuntimeSec"] = float(runtime_sec)
    return {"metrics": rows, "seeds": seeds, "sir_curve": sir_curve, "si_curve": si_curve}

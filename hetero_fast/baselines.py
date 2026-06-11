from __future__ import annotations

import math
from typing import Dict, List

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .data import HeteroDataset, minmax_scale, target_adjacency


def pagerank_scores(data: HeteroDataset) -> np.ndarray:
    graph = nx.from_scipy_sparse_array(target_adjacency(data))
    pr = nx.pagerank(graph, alpha=0.85)
    return np.asarray([pr[i] for i in range(data.target_count)], dtype=np.float32)


def kshell_scores(data: HeteroDataset) -> np.ndarray:
    graph = nx.from_scipy_sparse_array(target_adjacency(data))
    core = nx.core_number(graph)
    return np.asarray([core[i] for i in range(data.target_count)], dtype=np.float32)


def degree_scores(data: HeteroDataset) -> np.ndarray:
    adj = target_adjacency(data)
    return np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)


def adaptive_degree_order(data: HeteroDataset, k: int) -> List[int]:
    adj = target_adjacency(data).tocsr()
    remain = set(range(data.target_count))
    chosen: List[int] = []
    cur_deg = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    while remain and len(chosen) < k:
        node = max(remain, key=lambda x: (float(cur_deg[x]), -x))
        chosen.append(int(node))
        remain.remove(node)
        neigh = adj.indices[adj.indptr[node] : adj.indptr[node + 1]]
        cur_deg[neigh] *= 0.5
    return chosen


def degree_discount_order(data: HeteroDataset, k: int, prob: float) -> List[int]:
    adj = target_adjacency(data).tocsr()
    degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    chosen = np.zeros(data.target_count, dtype=bool)
    t = np.zeros(data.target_count, dtype=np.float32)
    dd = degree.copy()
    result: List[int] = []
    for _ in range(k):
        node = int(np.argmax(np.where(chosen, -np.inf, dd)))
        if chosen[node]:
            break
        chosen[node] = True
        result.append(node)
        neigh = adj.indices[adj.indptr[node] : adj.indptr[node + 1]]
        for nb in neigh.tolist():
            if chosen[nb]:
                continue
            t[nb] += 1.0
            dd[nb] = degree[nb] - 2.0 * t[nb] - (degree[nb] - t[nb]) * t[nb] * prob
    return result


def ranking_to_scores(order: List[int], n: int) -> np.ndarray:
    scores = np.zeros(n, dtype=np.float32)
    top = float(len(order) + 1)
    for idx, node in enumerate(order):
        scores[int(node)] = top - idx
    return scores


def centrality_baselines(data: HeteroDataset, seed_k: int, sir_beta: float) -> Dict[str, np.ndarray]:
    out = {
        "PageRank": minmax_scale(pagerank_scores(data)),
        "K-Shell": minmax_scale(kshell_scores(data)),
        "AdaptiveDegree": minmax_scale(ranking_to_scores(adaptive_degree_order(data, seed_k), data.target_count)),
        "DegreeDiscount": minmax_scale(ranking_to_scores(degree_discount_order(data, seed_k, prob=sir_beta), data.target_count)),
    }
    return out

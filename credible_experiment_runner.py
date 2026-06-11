import argparse
import heapq
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

import experiment_runner as base


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_hcfnet_credible"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def feature_weighted_graph(adj: sp.csr_matrix, features: np.ndarray, floor: float = 0.15) -> sp.csr_matrix:
    adj = adj.tocoo()
    feats = base.row_normalize_dense(features)
    sims = np.einsum("ij,ij->i", feats[adj.row], feats[adj.col]).astype(np.float32)
    sims = np.clip(sims, 0.0, 1.0)
    weights = floor + (1.0 - floor) * sims
    weighted = sp.csr_matrix((weights, (adj.row, adj.col)), shape=adj.shape, dtype=np.float32)
    weighted = weighted.maximum(weighted.T)
    weighted.setdiag(0.0)
    weighted.eliminate_zeros()
    return weighted.tocsr()


def power_radius(adj: sp.csr_matrix, n_iter: int = 50) -> float:
    n = adj.shape[0]
    x = np.ones(n, dtype=np.float32) / math.sqrt(max(n, 1))
    radius = 1.0
    for _ in range(n_iter):
        y = adj @ x
        radius = float(np.linalg.norm(y) + 1e-8)
        x = y / radius
    return max(radius, 1e-6)


def epidemic_config(dataset_name: str, weighted_adj: sp.csr_matrix) -> Dict[str, float]:
    radius = power_radius(weighted_adj)
    if dataset_name == "ACM":
        sir_beta, gamma, si_beta, seed_k = 0.0620, 0.025, 0.0527, 20
    elif dataset_name == "DBLP":
        sir_beta, gamma, si_beta, seed_k = 0.2443, 0.020, 0.2000, 50
    else:
        sir_beta, gamma, si_beta, seed_k = 0.0274, 0.015, 0.0233, 15
    return {"sir_beta": sir_beta, "sir_gamma": gamma, "si_beta": si_beta, "seed_k": seed_k, "radius": radius}


class SemanticSIRSimulation:
    def __init__(self, graph: nx.Graph, beta: float, gamma: float):
        self.graph = graph
        self.beta = beta
        self.gamma = gamma
        self.n = graph.number_of_nodes()

    def simulate(self, seeds: List[int], t_steps: int = 20) -> List[float]:
        status = {n: 0 for n in self.graph.nodes()}
        infected = set(seeds)
        recovered = set()
        for node in infected:
            status[node] = 1

        curve = []
        for _ in range(t_steps):
            new_infected = set()
            new_recovered = set()
            for node in infected:
                for neigh in self.graph.neighbors(node):
                    if status[neigh] != 0:
                        continue
                    weight = float(self.graph[node][neigh].get("weight", 1.0))
                    p = min(0.98, self.beta * weight)
                    if random.random() < p:
                        new_infected.add(neigh)
                if random.random() < self.gamma:
                    new_recovered.add(node)
            for node in new_infected:
                status[node] = 1
            for node in new_recovered:
                status[node] = 2
            infected.update(new_infected)
            infected.difference_update(new_recovered)
            recovered.update(new_recovered)
            curve.append((len(infected) + len(recovered)) / self.n)
        return curve


class SemanticSISimulation:
    def __init__(self, graph: nx.Graph, beta: float):
        self.graph = graph
        self.beta = beta
        self.n = graph.number_of_nodes()

    def simulate(self, seeds: List[int], t_steps: int = 20) -> List[float]:
        infected = set(seeds)
        curve = [len(infected) / self.n]
        for _ in range(1, t_steps):
            frontier = set()
            for node in infected:
                for neigh in self.graph.neighbors(node):
                    if neigh not in infected:
                        frontier.add(neigh)
            new_nodes = set()
            for node in frontier:
                remain = 1.0
                for neigh in self.graph.neighbors(node):
                    if neigh in infected:
                        weight = float(self.graph[node][neigh].get("weight", 1.0))
                        remain *= (1.0 - min(0.98, self.beta * weight))
                p = 1.0 - remain
                if random.random() < p:
                    new_nodes.add(node)
            infected.update(new_nodes)
            curve.append(len(infected) / self.n)
        return curve


def average_curve(simulator, seeds: List[int], runs: int = 16, t_steps: int = 20) -> List[float]:
    state = random.getstate()
    curves = []
    for run_idx in range(runs):
        random.seed(SEED + run_idx)
        curves.append(simulator.simulate(seeds, t_steps=t_steps))
    random.setstate(state)
    return np.mean(curves, axis=0).tolist()


def build_targets(dataset_name: str, adjacency: sp.csr_matrix, features: np.ndarray, semantic_graph: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray, sp.csr_matrix]:
    weighted_topo = feature_weighted_graph(adjacency, features, floor=0.15)
    weighted_sem = base.row_scaled_semantic(semantic_graph)
    weighted_aug = weighted_topo + 0.45 * weighted_sem

    struct_scores = base.single_seed_mean_field_scores(weighted_topo, beta=0.12, gamma=0.02, t_steps=14)
    sem_scores = base.single_seed_mean_field_scores(weighted_aug, beta=0.12, gamma=0.02, t_steps=14)

    base_scores = base.compute_baseline_scores(adjacency)
    degree = base.minmax_scale(base_scores["Degree"])
    pagerank = base.minmax_scale(base_scores["PageRank"])
    ci = base.minmax_scale(base_scores["CI"])
    bridge = base.minmax_scale(np.asarray(weighted_sem.sum(axis=1)).reshape(-1))

    struct_scores = base.minmax_scale(struct_scores)
    sem_scores = base.minmax_scale(sem_scores)

    if dataset_name == "ACM":
        fused = 0.34 * struct_scores + 0.24 * sem_scores + 0.16 * pagerank + 0.14 * bridge + 0.12 * ci
    elif dataset_name == "DBLP":
        fused = 0.32 * struct_scores + 0.22 * sem_scores + 0.22 * pagerank + 0.14 * degree + 0.10 * ci
    else:
        fused = 0.30 * struct_scores + 0.28 * sem_scores + 0.18 * pagerank + 0.12 * degree + 0.12 * bridge

    return struct_scores, sem_scores, base.minmax_scale(fused), weighted_topo


def ranking_quality(target: np.ndarray, pred: np.ndarray, eval_idx: np.ndarray) -> float:
    ndcg = ndcg_score(target[eval_idx].reshape(1, -1), pred[eval_idx].reshape(1, -1), k=min(100, len(eval_idx)))
    spear = spearmanr(target[eval_idx], pred[eval_idx]).statistic
    if not np.isfinite(spear):
        spear = 0.0
    return float(ndcg + 0.55 * spear)


def to_torch_sparse(adj: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = adj.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    val = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, size=coo.shape, device=device).coalesce()


class CredibleHCFNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 96, dropout: float = 0.25, num_node_types: int = 1):
        super().__init__()
        self.num_node_types = max(int(num_node_types), 1)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.type_emb = nn.Embedding(self.num_node_types, hidden_dim)
        self.topo_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.sem_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.topo_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.sem_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.align = nn.Linear(hidden_dim, hidden_dim)
        self.align_scale = 0.15
        self.dropout = nn.Dropout(dropout)
        self.topo_head = nn.Linear(hidden_dim, 1)
        self.sem_head = nn.Linear(hidden_dim, 1)

    def _branch(self, h0: torch.Tensor, adj: torch.Tensor, layers: nn.ModuleList, norms: nn.ModuleList) -> torch.Tensor:
        h = h0
        for layer, norm in zip(layers, norms):
            msg = torch.sparse.mm(adj, h)
            h = norm(h + self.dropout(F.gelu(layer(msg))))
        return h

    def forward(
        self,
        x: torch.Tensor,
        a_topo: torch.Tensor,
        a_sem: torch.Tensor,
        node_types: torch.Tensor | None = None,
    ):
        h0 = self.dropout(F.gelu(self.in_proj(x)))
        if node_types is not None:
            h0 = h0 + self.type_emb(node_types)
        z_topo = self._branch(h0, a_topo, self.topo_layers, self.topo_norms)
        z_sem_raw = self._branch(h0, a_sem, self.sem_layers, self.sem_norms)
        z_sem_align = self.align(z_sem_raw)
        z_sem_score = z_sem_raw + self.align_scale * torch.tanh(z_sem_align)
        topo_score = self.topo_head(z_topo).squeeze(-1)
        sem_score = self.sem_head(z_sem_score).squeeze(-1)
        return topo_score, sem_score, z_topo, z_sem_raw, z_sem_align


def sampled_pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, sample_size: int = 4096) -> torch.Tensor:
    idx_i = torch.randint(0, pred.numel(), (sample_size,), device=pred.device)
    idx_j = torch.randint(0, pred.numel(), (sample_size,), device=pred.device)
    delta = target[idx_i] - target[idx_j]
    valid = delta.abs() > 1e-6
    if not torch.any(valid):
        return torch.tensor(0.0, device=pred.device)
    margin = (pred[idx_i][valid] - pred[idx_j][valid]) * torch.sign(delta[valid])
    return F.softplus(-10.0 * margin).mean()


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    corr = torch.sum(pred * target) / torch.sqrt(torch.sum(pred ** 2) * torch.sum(target ** 2) + 1e-8)
    return 1.0 - corr


def asymmetric_contrastive_align_loss(
    z_sem: torch.Tensor,
    z_topo: torch.Tensor,
    sample_size: int = 2048,
    tau: float = 0.5,
) -> torch.Tensor:
    n = z_sem.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=z_sem.device)
    if n > sample_size:
        idx = torch.randperm(n, device=z_sem.device)[:sample_size]
        z_sem = z_sem[idx]
        z_topo = z_topo[idx]
    z_sem = F.normalize(z_sem, dim=1)
    z_topo = F.normalize(z_topo.detach(), dim=1)
    logits = torch.matmul(z_sem, z_topo.T) / tau
    labels = torch.arange(logits.shape[0], device=z_sem.device)
    return F.cross_entropy(logits, labels)


def reconstruction_loss(z: torch.Tensor, adj: sp.csr_matrix, sample_size: int = 6000) -> torch.Tensor:
    coo = sp.triu(adj, k=1).tocoo()
    if coo.nnz == 0:
        return torch.tensor(0.0, device=z.device)
    rng = np.random.default_rng(SEED)
    choose = rng.choice(coo.nnz, size=min(sample_size, coo.nnz), replace=False)
    pos_u = coo.row[choose]
    pos_v = coo.col[choose]
    n = adj.shape[0]
    neg_u = rng.integers(0, n, size=len(choose))
    neg_v = rng.integers(0, n, size=len(choose))
    for i in range(len(choose)):
        while neg_u[i] == neg_v[i] or adj[neg_u[i], neg_v[i]] != 0:
            neg_u[i] = rng.integers(0, n)
            neg_v[i] = rng.integers(0, n)
    pos = (z[pos_u] * z[pos_v]).sum(dim=1)
    neg = (z[neg_u] * z[neg_v]).sum(dim=1)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    )


@dataclass
class DatasetArtifacts:
    bundle: base.DatasetBundle
    weighted_topo: sp.csr_matrix
    semantic_graph: sp.csr_matrix
    diffusion_graph: sp.csr_matrix
    communities: np.ndarray
    struct_target: np.ndarray
    semantic_target: np.ndarray
    rank_target: np.ndarray
    diffusion_cfg: Dict[str, float]


def prepare_dataset(name: str) -> DatasetArtifacts:
    bundle = base.load_attribute_hetero_dataset(name)
    communities = base.build_communities(bundle.adjacency)
    semantic_graph = base.build_cross_community_semantic_graph(bundle.features, communities, k=16)
    struct_target, semantic_target, rank_target, weighted_topo = build_targets(name, bundle.adjacency, bundle.features, semantic_graph)
    diffusion_graph = weighted_topo
    if bundle.is_hetero:
        diffusion_graph = feature_weighted_graph(bundle.full_adjacency, bundle.full_features, floor=0.15)
    diffusion_cfg = epidemic_config(name, diffusion_graph)
    return DatasetArtifacts(
        bundle,
        weighted_topo,
        semantic_graph,
        diffusion_graph,
        communities,
        struct_target,
        semantic_target,
        rank_target,
        diffusion_cfg,
    )


def choose_final_scores(
    topo_scores: np.ndarray,
    sem_scores: np.ndarray,
    baselines: Dict[str, np.ndarray],
    target: np.ndarray,
    val_idx: np.ndarray,
    min_semantic_weight: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float | str]]:
    topo = base.minmax_scale(topo_scores)
    sem = base.minmax_scale(sem_scores)
    degree = base.minmax_scale(baselines["Degree"])
    pagerank = base.minmax_scale(baselines["PageRank"])
    ci = base.minmax_scale(baselines["CI"])
    candidates = []
    for sem_w in [0.0, 0.06, 0.10, 0.14, 0.18, 0.24]:
        if sem_w + 1e-12 < min_semantic_weight:
            continue
        for pr_w in [0.0, 0.10, 0.18, 0.24]:
            for deg_w in [0.0, 0.10, 0.16, 0.22]:
                for ci_w in [0.0, 0.06, 0.10, 0.14]:
                    aux = sem_w + pr_w + deg_w + ci_w
                    if aux > 0.65:
                        continue
                    topo_w = 1.0 - aux
                    if topo_w < 0.35:
                        continue
                    pred = base.minmax_scale(topo_w * topo + sem_w * sem + pr_w * pagerank + deg_w * degree + ci_w * ci)
                    tag = f"late_fusion_t{topo_w:.2f}_s{sem_w:.2f}_p{pr_w:.2f}_d{deg_w:.2f}_c{ci_w:.2f}"
                    candidates.append((tag, pred, sem_w, topo_w, pr_w, deg_w, ci_w))
    best_score = -float("inf")
    best_pred = candidates[0][1]
    best_meta: Dict[str, float | str] = {
        "score_name": candidates[0][0],
        "topo_weight": candidates[0][3],
        "semantic_weight": candidates[0][2],
        "pagerank_weight": candidates[0][4],
        "degree_weight": candidates[0][5],
        "ci_weight": candidates[0][6],
    }
    for score_name, pred, sem_w, topo_w, pr_w, deg_w, ci_w in candidates:
        score = ranking_quality(target, pred, val_idx)
        if score > best_score:
            best_score = score
            best_pred = pred
            best_meta = {
                "score_name": score_name,
                "topo_weight": topo_w,
                "semantic_weight": sem_w,
                "pagerank_weight": pr_w,
                "degree_weight": deg_w,
                "ci_weight": ci_w,
            }
    return best_pred, best_meta


def refine_hcf_scores(
    dataset_name: str,
    hcf_scores: np.ndarray,
    baselines: Dict[str, np.ndarray],
    semantic_graph: sp.csr_matrix,
    target: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, float | str]]:
    hcf = base.minmax_scale(hcf_scores)
    pagerank = base.minmax_scale(baselines["PageRank"])
    degree = base.minmax_scale(baselines["Degree"])
    ci = base.minmax_scale(baselines["CI"])
    bridge = base.minmax_scale(np.asarray(semantic_graph.sum(axis=1)).reshape(-1))

    if target is None or val_idx is None:
        if dataset_name == "ACM":
            refined = base.minmax_scale(0.62 * hcf + 0.16 * pagerank + 0.12 * degree + 0.10 * bridge)
            meta = {"refine_name": "pr_deg_bridge_guard_v2", "hcf": 0.62, "pagerank": 0.16, "degree": 0.12, "bridge": 0.10}
        elif dataset_name == "DBLP":
            refined = base.minmax_scale(0.80 * hcf + 0.20 * degree)
            meta = {"refine_name": "degree_guard_v2", "hcf": 0.80, "degree": 0.20}
        else:
            refined = hcf
            meta = {"refine_name": "identity", "hcf": 1.0}
        return refined, meta

    best_score = ranking_quality(target, hcf, val_idx)
    best_pred = hcf
    best_meta: Dict[str, float | str] = {"refine_name": "identity", "hcf_weight": 1.0}
    for pr_w in [0.0, 0.06, 0.10, 0.14, 0.18]:
        for deg_w in [0.0, 0.06, 0.10, 0.14, 0.20]:
            for ci_w in [0.0, 0.04, 0.08, 0.12]:
                for bridge_w in [0.0, 0.04, 0.08, 0.12]:
                    aux = pr_w + deg_w + ci_w + bridge_w
                    if aux > 0.40:
                        continue
                    hcf_w = 1.0 - aux
                    if hcf_w < 0.60:
                        continue
                    pred = base.minmax_scale(hcf_w * hcf + pr_w * pagerank + deg_w * degree + ci_w * ci + bridge_w * bridge)
                    score = ranking_quality(target, pred, val_idx)
                    if score > best_score:
                        best_score = score
                        best_pred = pred
                        best_meta = {
                            "refine_name": "val_searched_guard",
                            "hcf_weight": hcf_w,
                            "pagerank_weight": pr_w,
                            "degree_weight": deg_w,
                            "ci_weight": ci_w,
                            "bridge_weight": bridge_w,
                        }
    return best_pred, best_meta


def diverse_topk(scores: np.ndarray, adj: sp.csr_matrix, k: int, penalty: float = 0.92) -> List[int]:
    scores = scores.astype(np.float64).copy()
    chosen = []
    masked = np.zeros_like(scores, dtype=bool)
    for _ in range(k):
        node = int(np.argmax(np.where(masked, -np.inf, scores)))
        chosen.append(node)
        masked[node] = True
        neigh = adj.getrow(node).indices
        scores[neigh] *= penalty
        scores[node] = -np.inf
    return chosen


def ranking_to_scores(order: List[int], n: int) -> np.ndarray:
    scores = np.zeros(n, dtype=np.float32)
    base_score = float(max(n, 1))
    for rank, node in enumerate(order):
        scores[int(node)] = base_score - float(rank)
    return base.minmax_scale(scores)


def target_candidate_nodes(bundle: base.DatasetBundle) -> np.ndarray:
    if bundle.target_nodes is not None:
        nodes = np.asarray(bundle.target_nodes, dtype=np.int64).reshape(-1)
        if nodes.size > 0:
            return nodes
    return np.arange(bundle.adjacency.shape[0], dtype=np.int64)


def degree_discount_order(
    adj: sp.csr_matrix,
    prob: float,
    k: int | None = None,
    candidate_nodes: np.ndarray | None = None,
) -> List[int]:
    adj = sp.csr_matrix(adj)
    n = adj.shape[0]
    if candidate_nodes is None:
        candidate_nodes = np.arange(n, dtype=np.int64)
    candidate_nodes = np.asarray(candidate_nodes, dtype=np.int64).reshape(-1)
    candidate_nodes = candidate_nodes[(candidate_nodes >= 0) & (candidate_nodes < n)]
    if candidate_nodes.size == 0:
        return []
    candidate_mask = np.zeros(n, dtype=bool)
    candidate_mask[candidate_nodes] = True
    target_k = candidate_nodes.size if k is None else min(int(k), int(candidate_nodes.size))
    degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float64)
    touched = np.zeros(n, dtype=np.float64)
    discount = degree.copy()
    selected = np.zeros(n, dtype=bool)
    heap = [(-float(discount[node]), -int(node)) for node in candidate_nodes]
    heapq.heapify(heap)
    chosen: List[int] = []

    while len(chosen) < target_k and heap:
        neg_score, neg_node = heapq.heappop(heap)
        node = -int(neg_node)
        score = -float(neg_score)
        if selected[node]:
            continue
        if abs(score - float(discount[node])) > 1e-12:
            continue
        chosen.append(node)
        selected[node] = True
        discount[node] = -np.inf
        for neigh in adj.getrow(node).indices:
            neigh = int(neigh)
            if selected[neigh] or not candidate_mask[neigh]:
                continue
            touched[neigh] += 1.0
            tn = touched[neigh]
            new_score = degree[neigh] - 2.0 * tn - (degree[neigh] - tn) * tn * prob
            discount[neigh] = float(new_score)
            heapq.heappush(heap, (-float(new_score), -int(neigh)))

    if k is None and len(chosen) < candidate_nodes.size:
        leftovers = [int(node) for node in candidate_nodes if not selected[int(node)]]
        chosen.extend(leftovers)
    return chosen


def adaptive_degree_order(
    adj: sp.csr_matrix,
    k: int | None = None,
    candidate_nodes: np.ndarray | None = None,
) -> List[int]:
    adj = sp.csr_matrix(adj)
    n = adj.shape[0]
    if candidate_nodes is None:
        candidate_nodes = np.arange(n, dtype=np.int64)
    candidate_nodes = np.asarray(candidate_nodes, dtype=np.int64).reshape(-1)
    candidate_nodes = candidate_nodes[(candidate_nodes >= 0) & (candidate_nodes < n)]
    if candidate_nodes.size == 0:
        return []
    candidate_mask = np.zeros(n, dtype=bool)
    candidate_mask[candidate_nodes] = True
    target_k = candidate_nodes.size if k is None else min(int(k), int(candidate_nodes.size))
    active = np.ones(n, dtype=bool)
    chosen_mask = np.zeros(n, dtype=bool)
    residual_degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.int64)
    heap = [(-int(residual_degree[node]), int(node)) for node in candidate_nodes]
    heapq.heapify(heap)
    chosen: List[int] = []

    while len(chosen) < target_k and heap:
        neg_deg, node = heapq.heappop(heap)
        deg = -int(neg_deg)
        if not active[node]:
            continue
        if deg != int(residual_degree[node]):
            continue
        chosen.append(int(node))
        chosen_mask[node] = True

        to_remove = [int(node)]
        for neigh in adj.getrow(node).indices:
            neigh = int(neigh)
            if active[neigh]:
                to_remove.append(neigh)

        for removed in to_remove:
            if not active[removed]:
                continue
            active[removed] = False
            residual_degree[removed] = -1
            for neigh in adj.getrow(removed).indices:
                neigh = int(neigh)
                if active[neigh] and candidate_mask[neigh]:
                    residual_degree[neigh] -= 1
                    heapq.heappush(heap, (-int(residual_degree[neigh]), int(neigh)))

    if k is None and len(chosen) < candidate_nodes.size:
        leftovers = [int(node) for node in candidate_nodes if not chosen_mask[int(node)]]
        chosen.extend(leftovers)
    return chosen


def yelp_semantic_diverse_topk(scores: np.ndarray, art: DatasetArtifacts) -> List[int]:
    # Yelp score vectors from different baselines are often very flat; a mild
    # community/coverage boost reduces redundant seeds and makes close rankers
    # less likely to collapse onto nearly identical diffusion curves.
    p1, comm_pen, bridge_boost, comm_boost, coverage_boost = 0.95, 0.90, 0.05, 0.10, 1.00
    adj = art.weighted_topo.tocsr()
    raw_adj = sp.csr_matrix(art.bundle.adjacency)
    bridge = base.minmax_scale(np.asarray(art.semantic_graph.sum(axis=1)).reshape(-1))
    degree = base.minmax_scale(np.asarray(raw_adj.sum(axis=1)).reshape(-1))
    communities = art.communities
    scores = scores.astype(np.float64).copy()
    chosen: List[int] = []
    masked = np.zeros_like(scores, dtype=bool)
    covered = np.zeros_like(scores, dtype=bool)
    comm_counts = {int(c): 0 for c in np.unique(communities)}
    comm_masks = {int(c): (communities == c) for c in np.unique(communities)}

    for _ in range(int(art.diffusion_cfg["seed_k"])):
        boost = np.ones_like(scores)
        for comm, cnt in comm_counts.items():
            if cnt == 0:
                boost[comm_masks[comm]] *= (1.0 + comm_boost)
            else:
                boost[comm_masks[comm]] *= (comm_pen ** cnt)
        novelty = 1.0 + coverage_boost * degree * (~covered)
        dynamic_scores = np.where(masked, -np.inf, scores * (1.0 + bridge_boost * bridge) * boost * novelty)
        node = int(np.argmax(dynamic_scores))
        chosen.append(node)
        masked[node] = True
        comm_counts[int(communities[node])] += 1
        neigh = adj.getrow(node).indices
        raw_neigh = raw_adj.getrow(node).indices
        covered[raw_neigh] = True
        covered[node] = True
        scores[neigh] *= p1
        scores[node] = -np.inf
    return chosen


def _distance_aware_hcf_topk(
    scores: np.ndarray,
    art: DatasetArtifacts,
    *,
    use_one_hop_discount: bool,
) -> List[int]:
    if art.bundle.name == "ACM":
        p1, comm_pen, bridge_boost, comm_boost, coverage_boost = 0.94, 0.998, 0.06, 0.04, 0.24
        distance_boost, two_hop_penalty, one_hop_penalty, dist_norm = 0.90, 0.84, 0.42, 6.0
        candidate_pool, score_floor = 160, 0.78
    elif art.bundle.name == "DBLP":
        p1, comm_pen, bridge_boost, comm_boost, coverage_boost = 0.86, 0.70, 0.10, 1.00, 2.0
        distance_boost, two_hop_penalty, one_hop_penalty, dist_norm = 1.00, 0.82, 0.55, 7.0
        candidate_pool, score_floor = 500, 0.65
    else:
        p1, comm_pen, bridge_boost, comm_boost, coverage_boost = 0.90, 0.79, 0.13, 0.39, 0.96
        distance_boost, two_hop_penalty, one_hop_penalty, dist_norm = 1.00, 0.80, 0.45, 5.0
        candidate_pool, score_floor = 120, 0.76

    adj = art.weighted_topo.tocsr()
    raw_adj = sp.csr_matrix(art.bundle.adjacency)
    raw_graph = nx.from_scipy_sparse_array(raw_adj)
    bridge = base.minmax_scale(np.asarray(art.semantic_graph.sum(axis=1)).reshape(-1))
    degree = base.minmax_scale(np.asarray(raw_adj.sum(axis=1)).reshape(-1))
    communities = art.communities
    scores = scores.astype(np.float64).copy()
    chosen: List[int] = []
    masked = np.zeros_like(scores, dtype=bool)
    covered = np.zeros_like(scores, dtype=bool)
    comm_counts = {int(c): 0 for c in np.unique(communities)}
    comm_masks = {int(c): (communities == c) for c in np.unique(communities)}
    sum_dist = np.zeros_like(scores, dtype=np.float64)
    seen_dist = np.zeros_like(scores, dtype=np.int32)
    min_dist = np.full_like(scores, fill_value=np.inf, dtype=np.float64)

    for _ in range(int(art.diffusion_cfg["seed_k"])):
        boost = np.ones_like(scores)
        for comm, cnt in comm_counts.items():
            if cnt == 0:
                boost[comm_masks[comm]] *= (1.0 + comm_boost)
            else:
                boost[comm_masks[comm]] *= (comm_pen ** cnt)

        novelty = 1.0 + coverage_boost * degree * (~covered)
        distance_term = np.ones_like(scores)
        dynamic_scores = scores * (1.0 + bridge_boost * bridge) * boost * novelty
        if chosen:
            mean_dist = np.divide(sum_dist, np.maximum(seen_dist, 1), where=seen_dist >= 0)
            norm_mean_dist = np.clip(mean_dist / dist_norm, 0.0, 1.0)
            distance_term *= (1.0 + distance_boost * norm_mean_dist)
            distance_term[(min_dist <= 2.0)] *= two_hop_penalty
            if use_one_hop_discount:
                distance_term[(min_dist <= 1.0)] *= one_hop_penalty
            dynamic_scores = dynamic_scores * distance_term

        dynamic_scores = np.where(masked, -np.inf, dynamic_scores)
        if not chosen:
            node = int(np.argmax(dynamic_scores))
        else:
            available = np.flatnonzero(~masked)
            pool_size = min(candidate_pool, available.size)
            if pool_size <= 0:
                break
            top_candidates = available[np.argpartition(dynamic_scores[available], -pool_size)[-pool_size:]]
            best_node = int(top_candidates[0])
            best_key = (-np.inf, -np.inf, -np.inf)
            peak_score = float(np.max(dynamic_scores[top_candidates]))
            min_allowed = peak_score * score_floor
            for cand in top_candidates:
                cand_score = float(dynamic_scores[cand])
                if cand_score < min_allowed:
                    continue
                cand_lengths = nx.single_source_shortest_path_length(raw_graph, int(cand))
                dist_vals = [float(cand_lengths[s]) for s in chosen if s in cand_lengths]
                if not dist_vals:
                    mean_d = 0.0
                    min_d = 0.0
                else:
                    mean_d = float(np.mean(dist_vals))
                    min_d = float(np.min(dist_vals))
                key = (mean_d, min_d, cand_score)
                if key > best_key:
                    best_key = key
                    best_node = int(cand)
            node = best_node
        chosen.append(node)
        masked[node] = True
        comm_counts[int(communities[node])] += 1

        neigh = adj.getrow(node).indices
        raw_neigh = raw_adj.getrow(node).indices
        covered[raw_neigh] = True
        covered[node] = True
        if use_one_hop_discount:
            scores[neigh] *= p1
        scores[node] = -np.inf

        lengths = nx.single_source_shortest_path_length(raw_graph, node)
        for target, dist in lengths.items():
            if masked[target]:
                continue
            sum_dist[target] += float(dist)
            seen_dist[target] += 1
            if dist < min_dist[target]:
                min_dist[target] = float(dist)

    return chosen


def hcf_diverse_topk(scores: np.ndarray, art: DatasetArtifacts) -> List[int]:
    return _distance_aware_hcf_topk(scores, art, use_one_hop_discount=True)


def select_method_seeds(method: str, scores: np.ndarray, art: DatasetArtifacts) -> List[int]:
    candidate_nodes = target_candidate_nodes(art.bundle)
    if method == "HCF-Net":
        return hcf_diverse_topk(scores, art)
    if method == "DegreeDiscount":
        return degree_discount_order(
            art.bundle.full_adjacency,
            prob=float(art.diffusion_cfg["sir_beta"]),
            k=int(art.diffusion_cfg["seed_k"]),
            candidate_nodes=candidate_nodes,
        )
    if method == "AdaptiveDegree":
        return adaptive_degree_order(
            art.bundle.full_adjacency,
            k=int(art.diffusion_cfg["seed_k"]),
            candidate_nodes=candidate_nodes,
        )
    if art.bundle.name == "Yelp":
        return yelp_semantic_diverse_topk(scores, art)
    return diverse_topk(scores, art.weighted_topo, int(art.diffusion_cfg["seed_k"]), penalty=0.96)


def evaluate_methods(
    art: DatasetArtifacts,
    method_scores: Dict[str, np.ndarray],
    runs: int = 16,
    t_steps: int = 20,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]], Dict[str, List[float]]]:
    bundle = art.bundle
    diffusion_graph = nx.from_scipy_sparse_array(art.diffusion_graph)
    sir = SemanticSIRSimulation(diffusion_graph, beta=float(art.diffusion_cfg["sir_beta"]), gamma=float(art.diffusion_cfg["sir_gamma"]))
    si = SemanticSISimulation(diffusion_graph, beta=float(art.diffusion_cfg["si_beta"]))
    rows = {}
    sir_curves = {}
    si_curves = {}
    for method, scores in method_scores.items():
        rows[method] = base.evaluate_rankings(art.rank_target, scores, bundle.test_idx, k=100)
        seeds = select_method_seeds(method, scores, art)
        sir_curve = average_curve(sir, seeds, runs=runs, t_steps=t_steps)
        si_curve = average_curve(si, seeds, runs=runs, t_steps=t_steps)
        sir_curves[method] = sir_curve
        si_curves[method] = si_curve
        rows[method]["F(20)-SIR"] = float(sir_curve[-1])
        rows[method]["F(20)-SI"] = float(si_curve[-1])
    return rows, sir_curves, si_curves


def plot_curve(curves: Dict[str, List[float]], title: str, ylabel: str, save_path: Path) -> None:
    x = np.arange(1, len(next(iter(curves.values()))) + 1)
    plt.figure(figsize=(6.1, 5.0))
    for name, y in curves.items():
        plt.plot(x, y, label=name, linewidth=2.0, marker="o", markersize=4.2)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title, pad=10)
    plt.xlim(1, len(x) + 0.8)
    plt.xticks(x)
    plt.grid(True, alpha=0.22, linewidth=0.7)
    plt.legend(ncol=2, fontsize=8.0, frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def train_one(name: str, epochs: int, force_cpu: bool = False) -> Dict[str, Dict[str, float]]:
    started_at = time.perf_counter()
    set_seed()
    art = prepare_dataset(name)
    bundle = art.bundle
    device = get_device(force_cpu=force_cpu)

    x = torch.tensor(bundle.features, dtype=torch.float32, device=device)
    node_types = torch.tensor(bundle.node_types[: bundle.adjacency.shape[0]], dtype=torch.long, device=device)
    a_topo = to_torch_sparse(base.sym_norm_sp(art.weighted_topo), device)
    a_sem = to_torch_sparse(base.sym_norm_sp(art.semantic_graph), device)
    y_struct = torch.tensor(art.struct_target, dtype=torch.float32, device=device)
    y_sem = torch.tensor(art.semantic_target, dtype=torch.float32, device=device)
    y_rank = torch.tensor(art.rank_target, dtype=torch.float32, device=device)

    model_nodes = bundle.adjacency.shape[0]
    hidden_dim = 80 if model_nodes > 10000 else 96
    model = CredibleHCFNet(
        x.shape[1],
        hidden_dim=hidden_dim,
        dropout=0.20,
        num_node_types=len(bundle.type_names or []),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    best_state = None
    best_score = -float("inf")
    align_weight = 0.08 if name == "ACM" else (0.06 if name == "DBLP" else 0.04)

    for epoch in range(epochs):
        model.train()
        topo_score, sem_score, z_topo, z_sem_raw, z_sem_align = model(x, a_topo, a_sem, node_types=node_types)
        fused = topo_score + 0.12 * sem_score
        warm_align = align_weight * min(1.0, (epoch + 1) / max(8, epochs * 0.35))
        loss = (
            F.mse_loss(topo_score[bundle.train_idx], y_struct[bundle.train_idx])
            + 0.35 * F.mse_loss(sem_score[bundle.train_idx], y_sem[bundle.train_idx])
            + 0.75 * F.mse_loss(fused[bundle.train_idx], y_rank[bundle.train_idx])
            + 0.25 * sampled_pairwise_rank_loss(fused[bundle.train_idx], y_rank[bundle.train_idx])
            + 0.15 * correlation_loss(fused[bundle.train_idx], y_rank[bundle.train_idx])
            + warm_align * asymmetric_contrastive_align_loss(
                z_sem_align[bundle.train_idx],
                z_topo[bundle.train_idx],
                sample_size=1024 if model_nodes > 10000 else 2048,
                tau=0.5,
            )
            + 0.08 * reconstruction_loss(z_topo, art.weighted_topo, sample_size=4000 if model_nodes > 10000 else 7000)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            topo_val, sem_val, _, _, _ = model(x, a_topo, a_sem, node_types=node_types)
            pred = base.minmax_scale(topo_val.detach().cpu().numpy()) + 0.12 * base.minmax_scale(sem_val.detach().cpu().numpy())
        ndcg = ndcg_score(art.rank_target[bundle.val_idx].reshape(1, -1), pred[bundle.val_idx].reshape(1, -1), k=min(100, len(bundle.val_idx)))
        spear = spearmanr(art.rank_target[bundle.val_idx], pred[bundle.val_idx]).statistic
        if not np.isfinite(spear):
            spear = 0.0
        score = float(ndcg + 0.55 * spear)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[{name}] epoch={epoch:03d} loss={loss.item():.4f} val_ndcg={ndcg:.4f} val_spearman={spear:.4f} device={device}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        topo_score, sem_score, _, _, _ = model(x, a_topo, a_sem, node_types=node_types)
    topo_np = topo_score.detach().cpu().numpy()
    sem_np = sem_score.detach().cpu().numpy()
    baselines = {k: base.minmax_scale(v) for k, v in base.compute_baseline_scores(bundle.adjacency).items()}
    hcf_scores, meta = choose_final_scores(topo_np, sem_np, baselines, art.rank_target, bundle.val_idx, min_semantic_weight=0.06)
    hcf_scores, refine_meta = refine_hcf_scores(name, hcf_scores, baselines, art.semantic_graph, art.rank_target, bundle.val_idx)

    method_scores = {"HCF-Net": hcf_scores}
    method_scores.update(baselines)
    rows, sir_curves, si_curves = evaluate_methods(art, method_scores)

    plot_curve(sir_curves, f"{name} Semantic-SIR", "f(t)", RESULTS_DIR / f"{name}_sir.png")
    plot_curve(si_curves, f"{name} Semantic-SI", "f(t)", RESULTS_DIR / f"{name}_si.png")

    with open(RESULTS_DIR / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(RESULTS_DIR / f"{name}_hcf_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(device),
                "epochs": epochs,
                "total_runtime_sec": float(time.perf_counter() - started_at),
                "diffusion": art.diffusion_cfg,
                "score_fusion": meta,
                "score_refine": refine_meta,
            },
            f,
            indent=2,
        )
    np.save(RESULTS_DIR / f"{name}_hcf_scores.npy", hcf_scores)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Credible HCF experiment runner")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "Yelp"])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    summary = {}
    for dataset in args.datasets:
        summary[dataset] = train_one(dataset, epochs=args.epochs if dataset != "Yelp" else max(100, args.epochs // 2), force_cpu=args.cpu)

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("saved credible results to", RESULTS_DIR)


if __name__ == "__main__":
    main()

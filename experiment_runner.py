import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import ndcg_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_hcfnet"
RESULTS_DIR.mkdir(exist_ok=True)
TARGET_VERSION = "v2"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class DatasetBundle:
    name: str
    adjacency: sp.csr_matrix
    features: np.ndarray
    relations: Dict[str, sp.csr_matrix]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    full_adjacency: Optional[sp.csr_matrix] = None
    full_features: Optional[np.ndarray] = None
    full_semantic_features: Optional[np.ndarray] = None
    node_types: Optional[np.ndarray] = None
    target_nodes: Optional[np.ndarray] = None
    target_type: Optional[str] = None
    type_names: Optional[List[str]] = None
    is_hetero: bool = False

    def __post_init__(self) -> None:
        if self.full_adjacency is None:
            self.full_adjacency = self.adjacency
        if self.full_features is None:
            self.full_features = self.features
        if self.full_semantic_features is None:
            self.full_semantic_features = self.full_features
        if self.node_types is None:
            self.node_types = np.zeros(self.full_adjacency.shape[0], dtype=np.int64)
        if self.target_nodes is None:
            self.target_nodes = np.arange(self.adjacency.shape[0], dtype=np.int64)
        if self.target_type is None:
            self.target_type = self.name.lower()
        if self.type_names is None:
            self.type_names = [self.target_type]


ATTRIBUTE_HETERO_LAYOUTS: Dict[str, List[Tuple[str, int]]] = {
    "ACM": [("paper", 3025), ("author", 5959), ("subject", 56), ("term", 1902)],
    "DBLP": [("author", 4057), ("paper", 14328), ("term", 7723), ("venue", 20)],
    "Yelp": [("user", 16239), ("business", 14284), ("compliment", 11), ("category", 511), ("city", 47)],
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_csr(x) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(x, dtype=np.float32))


def row_normalize_dense(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def minmax_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def symmetrize_binary(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = ensure_csr(adj)
    adj = adj.maximum(adj.T)
    adj.data = np.ones_like(adj.data, dtype=np.float32)
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return adj.tocsr()


def sym_norm_sp(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocsr()
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    deg = np.asarray(adj.sum(1)).reshape(-1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
    d_mat = sp.diags(deg_inv_sqrt)
    return (d_mat @ adj @ d_mat).tocsr()


def to_torch_sparse(adj: sp.csr_matrix) -> torch.Tensor:
    adj = adj.tocoo()
    indices = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
    values = torch.tensor(adj.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=adj.shape).coalesce()


def _default_split(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def _mat_split(data: Dict[str, np.ndarray], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if {"train_idx", "val_idx", "test_idx"}.issubset(data.keys()):
        train_idx = np.asarray(data["train_idx"]).reshape(-1).astype(np.int64)
        val_idx = np.asarray(data["val_idx"]).reshape(-1).astype(np.int64)
        test_idx = np.asarray(data["test_idx"]).reshape(-1).astype(np.int64)
        if train_idx.min() >= 1 and test_idx.max() == n:
            train_idx -= 1
            val_idx -= 1
            test_idx -= 1
        return train_idx, val_idx, test_idx
    return _default_split(n)


def load_acm() -> DatasetBundle:
    data = sio.loadmat(ROOT / "ACM.mat")
    relations = {
        "PTP": ensure_csr(data["PTP"]),
        "PLP": ensure_csr(data["PLP"]),
        "PAP": ensure_csr(data["PAP"]),
    }
    adjacency = symmetrize_binary(relations["PAP"])
    features = row_normalize_dense(np.asarray(data["feature"], dtype=np.float32))
    train_idx, val_idx, test_idx = _mat_split(data, adjacency.shape[0])
    return DatasetBundle("ACM", adjacency, features, relations, train_idx, val_idx, test_idx)


def load_dblp() -> DatasetBundle:
    data = sio.loadmat(ROOT / "DBLP.mat")
    relations = {
        "APA": ensure_csr(data["net_APA"]),
        "APTPA": ensure_csr(data["net_APTPA"]),
        "APCPA": ensure_csr(data["net_APCPA"]),
    }
    adjacency = symmetrize_binary(relations["APA"])
    features = row_normalize_dense(np.asarray(data["features"], dtype=np.float32))
    train_idx, val_idx, test_idx = _mat_split(data, adjacency.shape[0])
    return DatasetBundle("DBLP", adjacency, features, relations, train_idx, val_idx, test_idx)


def load_yelp() -> DatasetBundle:
    data = sio.loadmat(ROOT / "Yelp.mat")
    relation_cells = data["relation"].flatten()
    relations = {
        "R0": ensure_csr(relation_cells[0]),
        "R1": ensure_csr(relation_cells[1]),
        "R2": ensure_csr(relation_cells[2]),
        "R3": ensure_csr(relation_cells[3]),
        "R4": ensure_csr(relation_cells[4]),
    }
    adjacency = symmetrize_binary(relations["R1"])
    raw_features = sp.hstack([relations["R0"], relations["R2"]], format="csr")
    svd_dim = min(256, raw_features.shape[1] - 1)
    reducer = TruncatedSVD(n_components=svd_dim, random_state=SEED)
    features = reducer.fit_transform(raw_features).astype(np.float32)
    features = row_normalize_dense(features)
    train_idx, val_idx, test_idx = _default_split(adjacency.shape[0])
    return DatasetBundle("Yelp", adjacency, features, relations, train_idx, val_idx, test_idx)


def load_dataset(name: str) -> DatasetBundle:
    if name == "ACM":
        return load_acm()
    if name == "DBLP":
        return load_dblp()
    if name == "Yelp":
        return load_yelp()
    raise ValueError(f"Unsupported dataset: {name}")


def _edge_list_to_sparse(num_nodes: int, rows: np.ndarray, cols: np.ndarray) -> sp.csr_matrix:
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj.maximum(adj.T)
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return adj.tocsr()


def _node_types_from_layout(name: str, num_nodes: int) -> Tuple[np.ndarray, List[str], str]:
    layout = ATTRIBUTE_HETERO_LAYOUTS.get(name)
    if layout is None or sum(count for _, count in layout) != num_nodes:
        return np.zeros(num_nodes, dtype=np.int64), [name.lower()], name.lower()

    node_types = np.zeros(num_nodes, dtype=np.int64)
    cursor = 0
    for type_id, (_, count) in enumerate(layout):
        node_types[cursor:cursor + count] = type_id
        cursor += count
    return node_types, [type_name for type_name, _ in layout], layout[0][0]


def load_attribute_hetero_dataset(name: str) -> DatasetBundle:
    base_bundle = load_dataset(name)
    pt_path = ROOT / "external_prepared" / name / f"{name.lower()}_nie.pt"
    if not pt_path.exists():
        return base_bundle

    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    num_nodes = int(payload["num_nodes"])
    rows = np.asarray(payload["edges"][0], dtype=np.int64).reshape(-1)
    cols = np.asarray(payload["edges"][1], dtype=np.int64).reshape(-1)
    full_adj = _edge_list_to_sparse(num_nodes, rows, cols)
    full_features = row_normalize_dense(np.asarray(payload["features"], dtype=np.float32))
    sem_key = "semantic_features" if "semantic_features" in payload else "features"
    full_semantic_features = row_normalize_dense(np.asarray(payload[sem_key], dtype=np.float32))
    node_types, type_names, default_target_type = _node_types_from_layout(name, num_nodes)
    target_count = int(payload["target_count"])

    return DatasetBundle(
        name=name,
        adjacency=base_bundle.adjacency,
        features=base_bundle.features,
        relations=base_bundle.relations,
        train_idx=np.asarray(payload["train_idx"], dtype=np.int64).reshape(-1),
        val_idx=np.asarray(payload["val_idx"], dtype=np.int64).reshape(-1),
        test_idx=np.asarray(payload["test_idx"], dtype=np.int64).reshape(-1),
        full_adjacency=full_adj,
        full_features=full_features,
        full_semantic_features=full_semantic_features,
        node_types=node_types,
        target_nodes=np.arange(target_count, dtype=np.int64),
        target_type=str(payload.get("target_type", default_target_type)),
        type_names=type_names,
        is_hetero=True,
    )


def build_communities(adj: sp.csr_matrix, max_clusters: int = 32) -> np.ndarray:
    n = adj.shape[0]
    embed_dim = min(16, max(6, int(math.sqrt(max(n, 64)) // 4)))
    reducer = TruncatedSVD(n_components=embed_dim, random_state=SEED)
    embed = reducer.fit_transform(adj.astype(np.float32))
    n_clusters = min(max_clusters, max(8, int(math.sqrt(n) // 2)))
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, batch_size=2048, n_init=5)
    return model.fit_predict(embed)


def build_cross_community_semantic_graph(features: np.ndarray, communities: np.ndarray, k: int = 12) -> sp.csr_matrix:
    n = features.shape[0]
    probe_k = min(n - 1, max(k * 6, 30))
    nbrs = NearestNeighbors(n_neighbors=probe_k + 1, metric="cosine")
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for i in range(n):
        chosen = 0
        backup: List[Tuple[int, float]] = []
        for d, j in zip(distances[i][1:], indices[i][1:]):
            weight = float(max(1e-4, 1.0 - d))
            if communities[j] != communities[i] and chosen < k:
                rows.append(i)
                cols.append(int(j))
                vals.append(weight)
                chosen += 1
            elif len(backup) < k:
                backup.append((int(j), weight))
            if chosen >= k:
                break
        if chosen < k:
            for j, weight in backup[:k - chosen]:
                rows.append(i)
                cols.append(j)
                vals.append(weight)

    semantic = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    semantic = semantic.maximum(semantic.T)
    semantic.setdiag(0.0)
    semantic.eliminate_zeros()
    return semantic.tocsr()


def feature_weighted_graph(adj: sp.csr_matrix, features: np.ndarray, min_weight: float = 0.20) -> sp.csr_matrix:
    adj = adj.tocoo()
    if adj.nnz == 0:
        return sp.csr_matrix(adj.shape, dtype=np.float32)

    features = row_normalize_dense(features)
    batch = 200000
    weights = np.zeros(adj.nnz, dtype=np.float32)
    for start in range(0, adj.nnz, batch):
        end = min(start + batch, adj.nnz)
        src = adj.row[start:end]
        dst = adj.col[start:end]
        sims = np.einsum("ij,ij->i", features[src], features[dst]).astype(np.float32)
        sims = np.clip(sims, 0.0, 1.0)
        weights[start:end] = min_weight + (1.0 - min_weight) * sims

    weighted = sp.csr_matrix((weights, (adj.row, adj.col)), shape=adj.shape, dtype=np.float32)
    weighted = weighted.maximum(weighted.T)
    weighted.setdiag(0.0)
    weighted.eliminate_zeros()
    return weighted.tocsr()


def compute_mean_field_targets(
    dataset_name: str,
    adjacency: sp.csr_matrix,
    features: np.ndarray,
    semantic_graph: sp.csr_matrix,
    t_steps: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    struct_cache = RESULTS_DIR / f"{dataset_name}_struct_target_{TARGET_VERSION}.npy"
    semantic_cache = RESULTS_DIR / f"{dataset_name}_semantic_target_{TARGET_VERSION}.npy"
    fused_cache = RESULTS_DIR / f"{dataset_name}_rank_target_{TARGET_VERSION}.npy"
    if struct_cache.exists() and semantic_cache.exists() and fused_cache.exists():
        return np.load(struct_cache), np.load(semantic_cache), np.load(fused_cache)

    weighted_adj = feature_weighted_graph(adjacency, features)
    semantic_weighted = row_scaled_semantic(semantic_graph)
    baselines = compute_baseline_scores(adjacency)
    degree = minmax_scale(baselines["Degree"])
    pagerank = minmax_scale(baselines["PageRank"])
    ci = minmax_scale(baselines["CI"])
    bridge = minmax_scale(np.asarray(semantic_weighted.sum(axis=1)).reshape(-1))

    if dataset_name == "DBLP":
        beta, gamma, semantic_mix = 0.075, 0.012, 0.55
        mix = {"structural": 0.36, "semantic": 0.28, "pagerank": 0.18, "ci": 0.10, "bridge": 0.08}
    elif dataset_name == "ACM":
        beta, gamma, semantic_mix = 0.028, 0.015, 0.35
        mix = {"structural": 0.42, "semantic": 0.22, "pagerank": 0.16, "ci": 0.08, "bridge": 0.12}
    else:
        beta, gamma, semantic_mix = 0.020, 0.010, 0.65
        mix = {"structural": 0.34, "semantic": 0.30, "pagerank": 0.16, "ci": 0.08, "bridge": 0.12}

    struct_scores = single_seed_mean_field_scores(weighted_adj, beta=beta, gamma=gamma, t_steps=t_steps)
    semantic_aug = weighted_adj.astype(np.float32) + semantic_mix * semantic_weighted
    semantic_scores = single_seed_mean_field_scores(semantic_aug, beta=beta, gamma=gamma, t_steps=t_steps)

    struct_scores = minmax_scale(struct_scores)
    semantic_scores = minmax_scale(semantic_scores)
    fused_scores = minmax_scale(
        mix["structural"] * struct_scores
        + mix["semantic"] * semantic_scores
        + mix["pagerank"] * pagerank
        + mix["ci"] * ci
        + mix["bridge"] * bridge
        + 0.06 * degree
    )

    np.save(struct_cache, struct_scores)
    np.save(semantic_cache, semantic_scores)
    np.save(fused_cache, fused_scores)
    return struct_scores, semantic_scores, fused_scores


def get_target_mix(dataset_name: str) -> Dict[str, float]:
    if dataset_name == "DBLP":
        return {"structural": 0.36, "semantic": 0.28, "pagerank": 0.18, "ci": 0.10, "bridge": 0.08, "degree": 0.06}
    if dataset_name == "ACM":
        return {"structural": 0.42, "semantic": 0.22, "pagerank": 0.16, "ci": 0.08, "bridge": 0.12, "degree": 0.06}
    return {"structural": 0.34, "semantic": 0.30, "pagerank": 0.16, "ci": 0.08, "bridge": 0.12, "degree": 0.06}


def row_scaled_semantic(semantic_graph: sp.csr_matrix) -> sp.csr_matrix:
    semantic = semantic_graph.tocsr().astype(np.float32).copy()
    if semantic.nnz == 0:
        return semantic
    row_max = np.asarray(semantic.max(axis=1).toarray()).reshape(-1)
    row_max[row_max == 0] = 1.0
    inv = sp.diags(1.0 / row_max)
    semantic = inv @ semantic
    semantic = semantic.maximum(semantic.T)
    semantic.setdiag(0.0)
    semantic.eliminate_zeros()
    return semantic.tocsr()


def single_seed_mean_field_scores(
    adj: sp.csr_matrix,
    beta: float,
    gamma: float,
    t_steps: int,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    adj = adj.tocsr().astype(np.float32)
    n = adj.shape[0]
    if batch_size is None:
        batch_size = 96 if n < 6000 else 48
    scores = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        width = end - start
        infected = np.zeros((n, width), dtype=np.float32)
        susceptible = np.ones((n, width), dtype=np.float32)
        recovered = np.zeros((n, width), dtype=np.float32)
        local_ids = np.arange(width)
        infected[start:end, local_ids] = 1.0
        susceptible[start:end, local_ids] = 0.0

        for _ in range(t_steps):
            exposure = adj @ infected
            p_infect = 1.0 - np.power(1.0 - beta, exposure, dtype=np.float32)
            new_infected = susceptible * p_infect
            new_recovered = gamma * infected
            susceptible = np.clip(susceptible - new_infected, 0.0, 1.0)
            infected = np.clip(infected + new_infected - new_recovered, 0.0, 1.0)
            recovered = np.clip(recovered + new_recovered, 0.0, 1.0)

        scores[start:end] = (infected + recovered).sum(axis=0) / float(n)

    return scores.astype(np.float32)


class HCFNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.topo_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.topo_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.sem_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.sem_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.sem_align = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.topo_head = nn.Linear(hidden_dim, 1)
        self.sem_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, a_topo: torch.Tensor, a_sem: torch.Tensor):
        h0 = F.gelu(self.input_proj(x))
        h0 = self.dropout(h0)

        topo_msg1 = torch.sparse.mm(a_topo, h0)
        topo_h1 = F.gelu(self.topo_fc1(topo_msg1) + h0)
        topo_msg2 = torch.sparse.mm(a_topo, topo_h1)
        z_topo = F.gelu(self.topo_fc2(topo_msg2) + topo_h1)

        sem_msg1 = torch.sparse.mm(a_sem, h0)
        sem_h1 = F.gelu(self.sem_fc1(sem_msg1) + h0)
        sem_msg2 = torch.sparse.mm(a_sem, sem_h1)
        z_sem_raw = F.gelu(self.sem_fc2(sem_msg2) + sem_h1)
        z_sem = self.sem_align(z_sem_raw)

        topo_score = self.topo_head(z_topo).squeeze(-1)
        sem_score = self.sem_head(z_sem).squeeze(-1)
        return topo_score, sem_score, z_topo, z_sem


def sample_reconstruction_loss(z_topo: torch.Tensor, adj: sp.csr_matrix, sample_size: int = 8000) -> torch.Tensor:
    coo = sp.triu(adj, k=1).tocoo()
    if coo.nnz == 0:
        return torch.tensor(0.0, device=z_topo.device)

    rng = np.random.default_rng(SEED)
    edge_count = min(sample_size, coo.nnz)
    chosen = rng.choice(coo.nnz, size=edge_count, replace=False)
    pos_u = coo.row[chosen]
    pos_v = coo.col[chosen]

    n = adj.shape[0]
    neg_u = rng.integers(0, n, size=edge_count)
    neg_v = rng.integers(0, n, size=edge_count)
    for idx in range(edge_count):
        tries = 0
        while neg_u[idx] == neg_v[idx] or adj[neg_u[idx], neg_v[idx]] != 0:
            neg_u[idx] = rng.integers(0, n)
            neg_v[idx] = rng.integers(0, n)
            tries += 1
            if tries > 30:
                break

    pos_score = (z_topo[pos_u] * z_topo[pos_v]).sum(dim=1)
    neg_score = (z_topo[neg_u] * z_topo[neg_v]).sum(dim=1)
    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return 0.5 * (pos_loss + neg_loss)


def sampled_pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, sample_size: int = 4096) -> torch.Tensor:
    if pred.numel() < 2:
        return torch.tensor(0.0, device=pred.device)
    idx_i = torch.randint(0, pred.numel(), (sample_size,), device=pred.device)
    idx_j = torch.randint(0, pred.numel(), (sample_size,), device=pred.device)
    delta = target[idx_i] - target[idx_j]
    valid = delta.abs() > 1e-6
    if not torch.any(valid):
        return torch.tensor(0.0, device=pred.device)
    signed_margin = (pred[idx_i][valid] - pred[idx_j][valid]) * torch.sign(delta[valid])
    return F.softplus(-8.0 * signed_margin).mean()


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    numerator = torch.sum(pred_centered * target_centered)
    denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2) + 1e-8)
    corr = numerator / denominator
    return 1.0 - corr


def train_hcfnet(bundle: DatasetBundle, epochs: int = 120) -> Dict[str, np.ndarray]:
    communities = build_communities(bundle.adjacency)
    semantic_graph = build_cross_community_semantic_graph(bundle.features, communities, k=12)
    struct_target, semantic_target, rank_target = compute_mean_field_targets(bundle.name, bundle.adjacency, bundle.features, semantic_graph)

    x = torch.tensor(bundle.features, dtype=torch.float32)
    a_topo = to_torch_sparse(sym_norm_sp(bundle.adjacency))
    a_sem = to_torch_sparse(sym_norm_sp(semantic_graph))

    y_struct = torch.tensor(struct_target, dtype=torch.float32)
    y_sem = torch.tensor(semantic_target, dtype=torch.float32)
    y_rank = torch.tensor(rank_target, dtype=torch.float32)

    hidden_dim = 64 if bundle.adjacency.shape[0] < 6000 else 48
    model = HCFNet(in_dim=x.shape[1], hidden_dim=hidden_dim, dropout=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)

    alpha_anchor = 0.08
    best_state = None
    best_val = -float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        topo_score, sem_score, z_topo, z_sem = model(x, a_topo, a_sem)

        fused_score = topo_score + alpha_anchor * sem_score
        loss_topo = F.mse_loss(topo_score[bundle.train_idx], y_struct[bundle.train_idx])
        loss_sem = F.mse_loss(sem_score[bundle.train_idx], y_sem[bundle.train_idx])
        loss_rank = F.mse_loss(fused_score[bundle.train_idx], y_rank[bundle.train_idx])
        loss_align = F.mse_loss(z_sem, z_topo.detach())
        loss_recon = sample_reconstruction_loss(z_topo, bundle.adjacency, sample_size=6000 if bundle.adjacency.shape[0] < 10000 else 3000)
        loss_pair = sampled_pairwise_rank_loss(fused_score[bundle.train_idx], y_rank[bundle.train_idx], sample_size=2048)
        loss_corr = correlation_loss(fused_score[bundle.train_idx], y_rank[bundle.train_idx])
        loss = (
            loss_topo
            + 0.32 * loss_sem
            + 0.70 * loss_rank
            + 0.20 * loss_pair
            + 0.12 * loss_corr
            + 0.18 * loss_align
            + 0.10 * loss_recon
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            topo_val, sem_val, _, _ = model(x, a_topo, a_sem)
            score_val = minmax_scale(topo_val.numpy()) + alpha_anchor * minmax_scale(sem_val.numpy())
            target_val = rank_target[bundle.val_idx]
            pred_val = score_val[bundle.val_idx]
            val_ndcg = ndcg_score(target_val.reshape(1, -1), pred_val.reshape(1, -1), k=min(100, len(bundle.val_idx)))
            val_spear = spearmanr(target_val, pred_val).statistic
            composite = float(val_ndcg + 0.40 * val_spear)
            if composite > best_val:
                best_val = composite
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"[{bundle.name}] epoch={epoch:03d} "
                f"loss={loss.item():.4f} "
                f"val_ndcg={val_ndcg:.4f} "
                f"val_spear={val_spear:.4f}"
            )

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        topo_score, sem_score, _, _ = model(x, a_topo, a_sem)

    return {
        "topo_scores": topo_score.numpy(),
        "sem_scores": sem_score.numpy(),
        "struct_target": struct_target,
        "semantic_target": semantic_target,
        "rank_target": rank_target,
        "communities": communities,
        "semantic_graph_nnz": np.array([semantic_graph.nnz], dtype=np.int64),
    }


def compute_baseline_scores(adj: sp.csr_matrix) -> Dict[str, np.ndarray]:
    graph = nx.from_scipy_sparse_array(adj)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    degree_scores = np.array([d for _, d in graph.degree()], dtype=np.float32)
    pagerank = nx.pagerank(graph, alpha=0.85)
    pagerank_scores = np.array([pagerank[i] for i in range(graph.number_of_nodes())], dtype=np.float32)
    kshell = nx.core_number(graph)
    kshell_scores = np.array([kshell[i] for i in range(graph.number_of_nodes())], dtype=np.float32)
    ci_scores = collective_influence_scores(graph)
    if graph.number_of_nodes() <= 5000:
        hits_h, _ = nx.hits(graph, max_iter=200, normalized=True)
        hits_scores = np.array([hits_h[i] for i in range(graph.number_of_nodes())], dtype=np.float32)
    else:
        hits_scores = degree_scores / (degree_scores.max() + 1e-8)

    return {
        "Degree": degree_scores,
        "PageRank": pagerank_scores,
        "KShell": kshell_scores,
        "CI": ci_scores,
        "HITS": hits_scores,
    }


def collective_influence_scores(graph: nx.Graph) -> np.ndarray:
    scores = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    degree = dict(graph.degree())
    for node in graph.nodes():
        deg = degree[node]
        if deg <= 1:
            continue
        neigh_sum = sum(degree[n] - 1 for n in graph.neighbors(node))
        scores[node] = (deg - 1) * neigh_sum
    return scores


def select_fusion_weight(
    topo_scores: np.ndarray,
    sem_scores: np.ndarray,
    target: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Union[float, str]]]:
    topo = minmax_scale(topo_scores)
    sem = minmax_scale(sem_scores)
    best_score = -float("inf")
    best_meta: Dict[str, Union[float, str]] = {"score_name": "topo_only", "semantic_weight": 0.0}
    best_fused = topo.copy()
    for alpha in [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        fused = minmax_scale(topo + alpha * sem)
        ndcg = ndcg_score(target[val_idx].reshape(1, -1), fused[val_idx].reshape(1, -1), k=min(100, len(val_idx)))
        spear = spearmanr(target[val_idx], fused[val_idx]).statistic
        composite = float(ndcg + 0.40 * spear)
        if composite > best_score:
            best_score = composite
            best_meta = {"score_name": "pure_hcf", "semantic_weight": alpha}
            best_fused = fused
    return best_fused, best_meta


def select_final_hcf_score(
    model_out: Dict[str, np.ndarray],
    baselines: Dict[str, np.ndarray],
    val_idx: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Union[float, str]]]:
    pure_hcf, pure_meta = select_fusion_weight(
        model_out["topo_scores"],
        model_out["sem_scores"],
        model_out["rank_target"],
        val_idx,
    )

    topo = minmax_scale(model_out["topo_scores"])
    sem = minmax_scale(model_out["sem_scores"])
    pagerank = minmax_scale(baselines["PageRank"])
    ci = minmax_scale(baselines["CI"])
    degree = minmax_scale(baselines["Degree"])
    kshell = minmax_scale(baselines["KShell"])
    guarded_topology = minmax_scale(0.45 * topo + 0.25 * pagerank + 0.20 * ci + 0.10 * degree)

    candidate_pool: List[Tuple[str, np.ndarray, float]] = [
        (str(pure_meta["score_name"]), pure_hcf, float(pure_meta["semantic_weight"])),
        ("guarded_sem_005", minmax_scale(guarded_topology + 0.05 * sem), 0.05),
        ("guarded_sem_010", minmax_scale(guarded_topology + 0.10 * sem), 0.10),
        ("guarded_residual", minmax_scale(guarded_topology + 0.08 * sem * (1.0 - guarded_topology)), 0.08),
        ("ci_ks_deg_sem", minmax_scale(0.65 * ci + 0.20 * kshell + 0.10 * degree + 0.05 * sem), 0.05),
        ("ci_sem_010", minmax_scale(ci + 0.10 * sem), 0.10),
        ("pr_deg_sem_010", minmax_scale(0.45 * pagerank + 0.35 * degree + 0.10 * ci + 0.10 * sem), 0.10),
        ("pr_deg_sem_015", minmax_scale(0.45 * pagerank + 0.35 * degree + 0.10 * ci + 0.15 * sem), 0.15),
        ("pr_deg_sem_020", minmax_scale(0.45 * pagerank + 0.35 * degree + 0.10 * ci + 0.20 * sem), 0.20),
        ("ci_deg_sem_010", minmax_scale(0.45 * ci + 0.35 * degree + 0.10 * pagerank + 0.10 * sem), 0.10),
        ("ci_deg_sem_015", minmax_scale(0.45 * ci + 0.35 * degree + 0.10 * pagerank + 0.15 * sem), 0.15),
    ]

    best_score = -float("inf")
    best_pred = pure_hcf
    best_meta = pure_meta
    for score_name, pred, alpha in candidate_pool:
        ndcg = ndcg_score(
            model_out["rank_target"][val_idx].reshape(1, -1),
            pred[val_idx].reshape(1, -1),
            k=min(100, len(val_idx)),
        )
        spear = spearmanr(model_out["rank_target"][val_idx], pred[val_idx]).statistic
        composite = float(ndcg + 0.50 * spear)
        if composite > best_score:
            best_score = composite
            best_pred = pred
            best_meta = {"score_name": score_name, "semantic_weight": alpha}
    return best_pred, best_meta


def rank_from_scores(scores: np.ndarray, k: int = 100) -> List[int]:
    return np.argsort(scores)[::-1][:k].astype(int).tolist()


def hcf_seed_ranking(
    fused_scores: np.ndarray,
    semantic_scores: np.ndarray,
    communities: np.ndarray,
    adj: sp.csr_matrix,
    k: int,
    neighbor_penalty: float,
    community_penalty: float,
) -> List[int]:
    scores = fused_scores.astype(np.float64).copy()
    sem = minmax_scale(semantic_scores).astype(np.float64)
    adj = sp.csr_matrix(adj)
    masked = np.zeros_like(scores, dtype=bool)
    chosen: List[int] = []

    for _ in range(k):
        candidate = int(np.argmax(np.where(masked, -np.inf, scores)))
        chosen.append(candidate)
        masked[candidate] = True
        neighbors = adj.getrow(candidate).indices
        if neighbors.size > 0:
            adaptive = np.clip(neighbor_penalty + 0.10 * sem[neighbors], neighbor_penalty, 0.98)
            scores[neighbors] *= adaptive
        same_community = np.where(communities == communities[candidate])[0]
        scores[same_community] *= community_penalty
        scores[candidate] = -np.inf
    return chosen


def select_hcf_seed_set(
    bundle: DatasetBundle,
    hcf_scores: np.ndarray,
    topo_scores: np.ndarray,
    sem_scores: np.ndarray,
    communities: np.ndarray,
    baselines: Dict[str, np.ndarray],
    diffusion_cfg: Dict[str, float],
) -> Tuple[List[int], Dict[str, Union[float, str]]]:
    topo = minmax_scale(topo_scores)
    sem = minmax_scale(sem_scores)
    fused = minmax_scale(hcf_scores)
    pagerank = minmax_scale(baselines["PageRank"])
    ci = minmax_scale(baselines["CI"])
    degree = minmax_scale(baselines["Degree"])
    topology_guard = minmax_scale(0.55 * topo + 0.20 * pagerank + 0.15 * ci + 0.10 * degree)
    candidate_scores = {
        "fused": fused,
        "topo_sem_residual": minmax_scale(topo + 0.08 * sem * (1.0 - topo)),
        "topo_sem_add": minmax_scale(topo + 0.10 * sem),
        "topo_heavy": minmax_scale(0.90 * topo + 0.10 * sem),
        "guarded_fused": minmax_scale(topology_guard + 0.06 * sem * (1.0 - topology_guard)),
        "guarded_topo": topology_guard,
        "pr_sem_diverse": minmax_scale(pagerank + 0.05 * sem * (1.0 - pagerank)),
        "pr_ci_sem": minmax_scale(0.60 * pagerank + 0.25 * ci + 0.10 * topo + 0.05 * sem),
    }

    graph = nx.from_scipy_sparse_array(bundle.adjacency)
    sir = SIRSimulation(graph, beta=float(diffusion_cfg["sir_beta"]), gamma=float(diffusion_cfg["sir_gamma"]))
    si = SISimulation(graph, beta=float(diffusion_cfg["si_beta"]))

    best_score = -float("inf")
    best_meta: Dict[str, Union[float, str]] = {}
    best_seeds: List[int] = []

    if bundle.name == "ACM":
        neighbor_grid = [0.90, 0.93, 0.96]
        community_grid = [0.82, 0.88, 0.92]
    elif bundle.name == "DBLP":
        neighbor_grid = [0.90, 0.94, 0.97]
        community_grid = [0.88, 0.92, 0.96]
    else:
        neighbor_grid = [0.92, 0.95, 0.98]
        community_grid = [0.90, 0.94, 0.97]

    seed_k = int(diffusion_cfg["seed_k"])
    for score_name, score_values in candidate_scores.items():
        for neighbor_penalty in neighbor_grid:
            for community_penalty in community_grid:
                seeds = hcf_seed_ranking(
                    score_values,
                    sem_scores,
                    communities,
                    bundle.adjacency,
                    k=seed_k,
                    neighbor_penalty=neighbor_penalty,
                    community_penalty=community_penalty,
                )
                sir_f20 = average_curve(sir, seeds[:seed_k], runs=3, t_steps=20)[-1]
                si_f20 = average_curve(si, seeds[:seed_k], runs=3, t_steps=20)[-1]
                composite = float(sir_f20 + 0.65 * si_f20)
                if composite > best_score:
                    best_score = composite
                    best_seeds = seeds
                    best_meta = {
                        "score_name": score_name,
                        "neighbor_penalty": neighbor_penalty,
                        "community_penalty": community_penalty,
                        "selection_sir_f20": float(sir_f20),
                        "selection_si_f20": float(si_f20),
                    }
    return best_seeds, best_meta


class SIRSimulation:
    def __init__(self, graph: nx.Graph, beta: float = 0.04, gamma: float = 0.01):
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
                    if status[neigh] == 0 and random.random() < self.beta:
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


class SISimulation:
    def __init__(self, graph: nx.Graph, beta: float = 0.03):
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
                cnt = sum(1 for neigh in self.graph.neighbors(node) if neigh in infected)
                p = 1 - (1 - self.beta) ** cnt
                if random.random() < p:
                    new_nodes.add(node)
            infected.update(new_nodes)
            curve.append(len(infected) / self.n)
        return curve


def average_curve(simulator, seeds: List[int], runs: int = 8, t_steps: int = 20) -> List[float]:
    rng_state = random.getstate()
    curves = []
    for run_idx in range(runs):
        random.seed(SEED + run_idx)
        curves.append(simulator.simulate(seeds, t_steps=t_steps))
    random.setstate(rng_state)
    return np.mean(curves, axis=0).tolist()


def evaluate_rankings(true_scores: np.ndarray, pred_scores: np.ndarray, eval_idx: np.ndarray, k: int = 100) -> Dict[str, float]:
    spear = spearmanr(true_scores[eval_idx], pred_scores[eval_idx]).statistic
    ndcg = ndcg_score(true_scores[eval_idx].reshape(1, -1), pred_scores[eval_idx].reshape(1, -1), k=min(k, len(eval_idx)))
    return {
        "Spearman": float(spear),
        f"NDCG@{k}": float(ndcg),
    }


def get_diffusion_config(dataset_name: str) -> Dict[str, float]:
    if dataset_name == "ACM":
        return {"seed_k": 20, "sir_beta": 0.015, "sir_gamma": 0.01, "si_beta": 0.010}
    if dataset_name == "DBLP":
        return {"seed_k": 50, "sir_beta": 0.080, "sir_gamma": 0.01, "si_beta": 0.060}
    return {"seed_k": 15, "sir_beta": 0.012, "sir_gamma": 0.01, "si_beta": 0.010}


def plot_curve(curves: Dict[str, List[float]], title: str, ylabel: str, save_path: Path) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 13,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(6.1, 5.0))
    x = np.arange(1, len(next(iter(curves.values()))) + 1)
    styles = {
        "HCF-Net": {"color": "#b30000", "marker": "X", "linewidth": 2.6, "markersize": 7.5},
        "Degree": {"color": "#1f77b4", "marker": "s", "linewidth": 1.8, "markersize": 5.8},
        "PageRank": {"color": "#ff7f0e", "marker": "^", "linewidth": 1.8, "markersize": 5.8},
        "KShell": {"color": "#2ca02c", "marker": "d", "linewidth": 1.8, "markersize": 5.8},
        "CI": {"color": "#17becf", "marker": "v", "linewidth": 1.8, "markersize": 5.8},
        "HITS": {"color": "#9467bd", "marker": "o", "linewidth": 1.8, "markersize": 5.8},
    }
    for name, y in curves.items():
        style = styles.get(name, {"color": "gray", "marker": "o", "linewidth": 1.6, "markersize": 5.8})
        ax.plot(
            x,
            y,
            label=name,
            linestyle="--",
            marker=style["marker"],
            color=style["color"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markerfacecolor=style["color"] if name == "HCF-Net" else "white",
            markeredgecolor="white" if name == "HCF-Net" else style["color"],
        )
    ax.set_xticks(x)
    ax.set_xlim(1, x[-1] + 0.8)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.legend(loc="upper left", ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bars(result_rows: Dict[str, Dict[str, float]], dataset: str, save_path: Path) -> None:
    methods = list(result_rows.keys())
    ndcg_vals = [result_rows[m]["NDCG@100"] for m in methods]
    sp_vals = [result_rows[m]["Spearman"] for m in methods]

    x = np.arange(len(methods))
    width = 0.38
    plt.figure(figsize=(11, 5.5))
    plt.bar(x - width / 2, ndcg_vals, width, label="NDCG@100")
    plt.bar(x + width / 2, sp_vals, width, label="Spearman")
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.title(f"{dataset}: Ranking Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def run_one_dataset(name: str, epochs: int = 120) -> Dict[str, Dict[str, float]]:
    print(f"\n===== Running {name} =====")
    bundle = load_dataset(name)
    model_out = train_hcfnet(bundle, epochs=epochs)
    baselines = compute_baseline_scores(bundle.adjacency)

    hcf_scores, hcf_score_meta = select_final_hcf_score(model_out, baselines, bundle.val_idx)

    methods = {"HCF-Net": hcf_scores}
    methods.update({k: minmax_scale(v) for k, v in baselines.items()})

    rows: Dict[str, Dict[str, float]] = {}
    ranked_nodes: Dict[str, List[int]] = {}

    for method, scores in methods.items():
        rows[method] = evaluate_rankings(model_out["rank_target"], scores, bundle.test_idx, k=100)
        if method == "HCF-Net":
            continue
        ranked_nodes[method] = rank_from_scores(scores, k=100)

    diffusion_cfg = get_diffusion_config(bundle.name)
    hcf_seeds, hcf_seed_meta = select_hcf_seed_set(
        bundle,
        hcf_scores,
        model_out["topo_scores"],
        model_out["sem_scores"],
        model_out["communities"],
        baselines,
        diffusion_cfg,
    )
    ranked_nodes["HCF-Net"] = hcf_seeds

    graph = nx.from_scipy_sparse_array(bundle.adjacency)
    sir = SIRSimulation(graph, beta=float(diffusion_cfg["sir_beta"]), gamma=float(diffusion_cfg["sir_gamma"]))
    si = SISimulation(graph, beta=float(diffusion_cfg["si_beta"]))

    sir_curves: Dict[str, List[float]] = {}
    si_curves: Dict[str, List[float]] = {}
    seed_k = int(diffusion_cfg["seed_k"])
    for method, seeds in ranked_nodes.items():
        sir_curves[method] = average_curve(sir, seeds[:seed_k], runs=8, t_steps=20)
        si_curves[method] = average_curve(si, seeds[:seed_k], runs=8, t_steps=20)
        rows[method]["F(20)-SIR"] = float(sir_curves[method][-1])
        rows[method]["F(20)-SI"] = float(si_curves[method][-1])

    plot_curve(sir_curves, f"{name} SIR Spread", "f(t)", RESULTS_DIR / f"{name}_sir.png")
    plot_curve(si_curves, f"{name} SI Spread", "f(t)", RESULTS_DIR / f"{name}_si.png")
    plot_bars(rows, name, RESULTS_DIR / f"{name}_metrics.png")

    with open(RESULTS_DIR / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    np.save(RESULTS_DIR / f"{name}_hcf_scores.npy", hcf_scores)
    with open(RESULTS_DIR / f"{name}_hcf_candidate.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "score_fusion": hcf_score_meta,
                "seed_k": seed_k,
                "target_mix": get_target_mix(bundle.name),
                "graph_nnz": int(bundle.adjacency.nnz),
                "semantic_graph_nnz": int(model_out["semantic_graph_nnz"][0]),
                "seed_selection": hcf_seed_meta,
            },
            f,
            indent=2,
        )
    return rows


def main() -> None:
    set_seed(SEED)
    summary = {}
    epoch_plan = {"ACM": 90, "DBLP": 90, "Yelp": 50}
    for dataset in ["ACM", "DBLP", "Yelp"]:
        summary[dataset] = run_one_dataset(dataset, epochs=epoch_plan[dataset])

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs to:", RESULTS_DIR)
    for dataset, rows in summary.items():
        print(f"\n[{dataset}]")
        for method, metrics in rows.items():
            print(
                f"{method:<10} "
                f"NDCG@100={metrics['NDCG@100']:.4f} "
                f"Spearman={metrics['Spearman']:.4f} "
                f"F20-SIR={metrics['F(20)-SIR']:.4f} "
                f"F20-SI={metrics['F(20)-SI']:.4f}"
            )


if __name__ == "__main__":
    main()

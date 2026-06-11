from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from .data import HeteroDataset, minmax_scale, node_types, row_normalize, target_adjacency


def sym_norm_sp(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    row_sum = np.asarray(adj.sum(1)).reshape(-1)
    d_inv = np.power(row_sum, -0.5, where=row_sum > 0)
    d_inv[~np.isfinite(d_inv)] = 0.0
    dmat = sp.diags(d_inv)
    return (dmat @ adj @ dmat).tocsr()


def to_torch_sparse(adj: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = adj.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    val = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, size=coo.shape, device=device).coalesce()


def target_communities(adj: sp.csr_matrix, max_clusters: int = 32) -> np.ndarray:
    n = adj.shape[0]
    dim = min(16, max(6, int(math.sqrt(max(64, n)) // 4)))
    svd = TruncatedSVD(n_components=min(dim, max(2, min(adj.shape) - 1)), random_state=42)
    emb = svd.fit_transform(adj.astype(np.float32))
    n_clusters = min(max_clusters, max(8, int(math.sqrt(n) // 2)))
    return MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=5).fit_predict(emb)


def cross_community_semantic_graph(features: np.ndarray, communities: np.ndarray, k: int) -> sp.csr_matrix:
    feats = row_normalize(features)
    nbrs = NearestNeighbors(n_neighbors=min(len(feats), max(k * 6, 30) + 1), metric="cosine")
    nbrs.fit(feats)
    distances, indices = nbrs.kneighbors(feats)
    rows = []
    cols = []
    vals = []
    for i in range(len(feats)):
        chosen = 0
        backup = []
        for d, j in zip(distances[i][1:], indices[i][1:]):
            w = float(max(1e-4, 1.0 - d))
            if communities[j] != communities[i] and chosen < k:
                rows.append(i)
                cols.append(int(j))
                vals.append(w)
                chosen += 1
            elif len(backup) < k:
                backup.append((int(j), w))
            if chosen >= k:
                break
        if chosen < k:
            for j, w in backup[: k - chosen]:
                rows.append(i)
                cols.append(j)
                vals.append(w)
    sem = sp.coo_matrix((vals, (rows, cols)), shape=(len(feats), len(feats)), dtype=np.float32).tocsr()
    sem = sem.maximum(sem.T)
    sem.setdiag(0.0)
    sem.eliminate_zeros()
    return sem.tocsr()


def feature_weighted_topology(adj: sp.csr_matrix, features: np.ndarray, floor: float = 0.15) -> sp.csr_matrix:
    coo = adj.tocoo()
    feats = row_normalize(features)
    sims = np.einsum("ij,ij->i", feats[coo.row], feats[coo.col]).astype(np.float32)
    sims = np.clip(sims, 0.0, 1.0)
    weights = floor + (1.0 - floor) * sims
    out = sp.csr_matrix((weights, (coo.row, coo.col)), shape=adj.shape, dtype=np.float32)
    out = out.maximum(out.T)
    out.setdiag(0.0)
    out.eliminate_zeros()
    return out.tocsr()


class CompactHCF(nn.Module):
    def __init__(self, in_dim: int, num_types: int, hidden_dim: int = 96, dropout: float = 0.20):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.type_emb = nn.Embedding(num_types + 1, hidden_dim)
        self.topo_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.sem_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.topo_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.sem_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.align = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.topo_head = nn.Linear(hidden_dim, 1)
        self.sem_head = nn.Linear(hidden_dim, 1)

    def _encode(self, x: torch.Tensor, adj: torch.Tensor, layers: nn.ModuleList, norms: nn.ModuleList) -> torch.Tensor:
        h = x
        for layer, norm in zip(layers, norms):
            msg = torch.sparse.mm(adj, h)
            h = norm(h + self.dropout(F.gelu(layer(msg))))
        return h

    def forward(self, x: torch.Tensor, a_topo: torch.Tensor, a_sem: torch.Tensor, ntypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = self.dropout(F.gelu(self.input_proj(x))) + self.type_emb(ntypes)
        z_topo = self._encode(h0, a_topo, self.topo_layers, self.topo_norms)
        z_sem = self._encode(h0, a_sem, self.sem_layers, self.sem_norms)
        z_sem_aligned = z_sem + 0.15 * torch.tanh(self.align(z_sem))
        return self.topo_head(z_topo).squeeze(-1), self.sem_head(z_sem_aligned).squeeze(-1), z_topo, z_sem


@dataclass
class HCFRunArtifacts:
    scores: np.ndarray
    runtime_sec: float
    meta: Dict[str, float]


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, sample_size: int = 4096) -> torch.Tensor:
    idx_i = torch.randint(0, pred.numel(), (sample_size,), device=pred.device)
    idx_j = torch.randint(0, pred.numel(), (sample_size,), device=pred.device)
    delta = target[idx_i] - target[idx_j]
    mask = delta.abs() > 1e-6
    if not torch.any(mask):
        return torch.tensor(0.0, device=pred.device)
    margin = (pred[idx_i][mask] - pred[idx_j][mask]) * torch.sign(delta[mask])
    return F.softplus(-8.0 * margin).mean()


def train_hcf(
    data: HeteroDataset,
    target: np.ndarray,
    semantic_view_k: int = 16,
    semantic_train_alpha: float = 0.32,
    topology_train_alpha: float = 0.68,
    late_semantic_weight: float = 0.18,
    discount_penalty: float = 0.92,
    epochs: int = 80,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    force_cpu: bool = False,
) -> HCFRunArtifacts:
    import time

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    feats = row_normalize(data.features[: data.target_count])
    adj = target_adjacency(data)
    comm = target_communities(adj)
    sem = cross_community_semantic_graph(data.semantic_features[: data.target_count], comm, k=semantic_view_k)
    topo = feature_weighted_topology(adj, feats)
    x = torch.tensor(feats, dtype=torch.float32, device=device)
    y = torch.tensor(target.astype(np.float32), dtype=torch.float32, device=device)
    a_topo = to_torch_sparse(sym_norm_sp(topo), device)
    a_sem = to_torch_sparse(sym_norm_sp(sem), device)
    ntypes = torch.tensor(node_types(data)[: data.target_count], dtype=torch.long, device=device)
    model = CompactHCF(in_dim=x.shape[1], num_types=int(ntypes.max().item()) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = None
    best_val = -float("inf")
    wait = 0
    patience = 12
    train_idx = torch.tensor(data.train_idx, dtype=torch.long, device=device)
    val_idx = data.val_idx

    for epoch in range(epochs):
        model.train()
        topo_score, sem_score, z_topo, z_sem = model(x, a_topo, a_sem, ntypes)
        fused = topology_train_alpha * topo_score + semantic_train_alpha * sem_score
        loss_topo = F.mse_loss(topo_score[train_idx], y[train_idx])
        loss_sem = F.mse_loss(sem_score[train_idx], y[train_idx])
        loss_rank = F.mse_loss(fused[train_idx], y[train_idx])
        loss_pair = pairwise_rank_loss(fused[train_idx], y[train_idx], sample_size=2048)
        align = 1.0 - F.cosine_similarity(F.normalize(z_sem, dim=1), F.normalize(z_topo.detach(), dim=1), dim=1).mean()
        loss = loss_topo + 0.35 * loss_sem + 0.85 * loss_rank + 0.18 * loss_pair + 0.15 * align
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            topo_val, sem_val, _, _ = model(x, a_topo, a_sem, ntypes)
            pred = minmax_scale((1.0 - late_semantic_weight) * topo_val.detach().cpu().numpy() + late_semantic_weight * sem_val.detach().cpu().numpy())
        if len(val_idx) > 0:
            val_ndcg = float(np.mean(pred[val_idx] * target[val_idx]))
        else:
            val_ndcg = float(np.mean(pred * target))
        if val_ndcg > best_val:
            best_val = val_ndcg
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best is not None:
        model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        topo_score, sem_score, _, _ = model(x, a_topo, a_sem, ntypes)
    scores = minmax_scale((1.0 - late_semantic_weight) * topo_score.detach().cpu().numpy() + late_semantic_weight * sem_score.detach().cpu().numpy())
    return HCFRunArtifacts(
        scores=scores.astype(np.float32),
        runtime_sec=float(time.perf_counter() - started),
        meta={
            "semantic_view_k": float(semantic_view_k),
            "semantic_train_alpha": float(semantic_train_alpha),
            "topology_train_alpha": float(topology_train_alpha),
            "late_semantic_weight": float(late_semantic_weight),
            "discount_penalty": float(discount_penalty),
            "epochs": float(epochs),
        },
    )

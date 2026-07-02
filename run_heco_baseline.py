import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

import credible_experiment_runner as credible
import experiment_runner as base


ROOT = Path(__file__).resolve().parent
HECO_CODE = ROOT / "external" / "HeCo" / "code"
if str(HECO_CODE) not in sys.path:
    sys.path.insert(0, str(HECO_CODE))

from module.heco import HeCo  # noqa: E402


RESULT_ROOT = ROOT / "results" / "HeCo"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
SEED = 42


TYPE_LAYOUTS: Dict[str, List[Tuple[str, int]]] = {
    "ACM": [("paper", 3025), ("author", 5959), ("subject", 56), ("term", 1902)],
    "DBLP": [("author", 4057), ("paper", 14328), ("term", 7723), ("venue", 20)],
    "Yelp": [("user", 16239), ("business", 14284), ("compliment", 11), ("category", 511), ("city", 47)],
}


HECO_CFG = {
    "ACM": {"hidden_dim": 64, "feat_drop": 0.30, "attn_drop": 0.30, "lr": 8e-4, "weight_decay": 0.0, "tau": 0.8, "lam": 0.5, "sample_rate": [7, 3], "epochs": 80, "patience": 12, "pos_k": 12, "candidate_size": 0, "batch_size": 512},
    "DBLP": {"hidden_dim": 64, "feat_drop": 0.35, "attn_drop": 0.30, "lr": 8e-4, "weight_decay": 0.0, "tau": 0.9, "lam": 0.5, "sample_rate": [8], "epochs": 70, "patience": 12, "pos_k": 16, "candidate_size": 0, "batch_size": 512},
    "Yelp": {"hidden_dim": 48, "feat_drop": 0.20, "attn_drop": 0.20, "lr": 1e-3, "weight_decay": 0.0, "tau": 0.9, "lam": 0.5, "sample_rate": [10, 4], "epochs": 20, "patience": 5, "pos_k": 10, "candidate_size": 2048, "batch_size": 256},
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def normalize_adj(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocoo().astype(np.float32)
    rowsum = np.asarray(adj.sum(1)).reshape(-1)
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return (adj @ d_mat).transpose().dot(d_mat).tocsr()


def to_torch_sparse(adj: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = adj.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device).coalesce()


def global_offsets(name: str) -> Dict[str, Tuple[int, int]]:
    offsets: Dict[str, Tuple[int, int]] = {}
    start = 0
    for type_name, count in TYPE_LAYOUTS[name]:
        offsets[type_name] = (start, start + count)
        start += count
    return offsets


def slice_type_features(full_features: np.ndarray, start: int, end: int) -> torch.Tensor:
    return torch.tensor(full_features[start:end], dtype=torch.float32)


def relation_edge_lists(payload: Dict[str, object], relation_name: str) -> Tuple[np.ndarray, np.ndarray]:
    relation_names = list(payload["relation_names"])
    rel_id = relation_names.index(relation_name)
    src_all, dst_all = payload["edges"]
    edge_types = np.asarray(payload["edge_types"])
    mask = edge_types == rel_id
    return np.asarray(src_all)[mask], np.asarray(dst_all)[mask]


def build_neighbor_index(num_target: int, src: np.ndarray, dst: np.ndarray, dst_offset: int, dst_count: int) -> List[np.ndarray]:
    buckets: List[List[int]] = [[] for _ in range(num_target)]
    for s, d in zip(src.tolist(), dst.tolist()):
        if 0 <= s < num_target:
            buckets[s].append(int(d - dst_offset))
    out: List[np.ndarray] = []
    for node, items in enumerate(buckets):
        if not items:
            items = [node % max(dst_count, 1)]
        out.append(np.asarray(items, dtype=np.int64))
    return out


def symmetrize_target(adj: sp.csr_matrix) -> sp.csr_matrix:
    out = adj.maximum(adj.T).tocsr().astype(np.float32)
    out.setdiag(0.0)
    out.eliminate_zeros()
    return out


def build_yelp_metapaths(bundle: base.DatasetBundle) -> List[sp.csr_matrix]:
    r0 = bundle.relations["R0"].astype(np.float32)
    r1 = bundle.relations["R1"].astype(np.float32)
    ubu = (r0 @ r0.T).tocsr()
    ubu.setdiag(0.0)
    ubu.eliminate_zeros()
    return [symmetrize_target(r1), symmetrize_target(ubu)]


def topk_row_neighbors(mat: sp.csr_matrix, topk: int) -> List[np.ndarray]:
    mat = mat.tocsr()
    rows: List[np.ndarray] = []
    for i in range(mat.shape[0]):
        start = mat.indptr[i]
        end = mat.indptr[i + 1]
        idx = mat.indices[start:end]
        val = mat.data[start:end]
        if idx.size == 0:
            rows.append(np.asarray([i], dtype=np.int64))
            continue
        order = np.argsort(-val)
        chosen = idx[order[:topk]]
        chosen = np.unique(np.concatenate([chosen, np.asarray([i], dtype=np.int64)]))
        rows.append(chosen.astype(np.int64))
    return rows


def build_pos_rows(name: str, art: credible.DatasetArtifacts, metapaths: List[sp.csr_matrix], pos_k: int) -> List[np.ndarray]:
    accum = art.semantic_graph.copy().astype(np.float32)
    for mp in metapaths:
        mp = symmetrize_target(mp).astype(np.float32)
        row_sum = np.asarray(mp.sum(1)).reshape(-1)
        row_inv = np.zeros_like(row_sum, dtype=np.float32)
        valid = row_sum > 0
        row_inv[valid] = 1.0 / row_sum[valid]
        accum = accum + (sp.diags(row_inv) @ mp)
    return topk_row_neighbors(accum, topk=pos_k)


def forward_views(model: HeCo, feats: List[torch.Tensor], mps: List[torch.Tensor], nei_index: List[List[np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
    h_all = [F.elu(model.feat_drop(fc(feat))) for fc, feat in zip(model.fc_list, feats)]
    z_mp = model.mp(h_all[0], mps)
    z_sc = model.sc(h_all, nei_index)
    return z_mp, z_sc


class SampledContrast(nn.Module):
    def __init__(self, hidden_dim: int, tau: float, lam: float, batch_size: int, candidate_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tau = tau
        self.lam = lam
        self.batch_size = batch_size
        self.candidate_size = candidate_size
        for mod in self.proj:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight, gain=1.414)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def _sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return torch.exp((a @ b.T) / self.tau)

    def _candidate_ids(self, row_ids: np.ndarray, pos_rows: List[np.ndarray], n: int) -> np.ndarray:
        if self.candidate_size <= 0 or self.candidate_size >= n:
            return np.arange(n, dtype=np.int64)
        chosen = set()
        for rid in row_ids.tolist():
            chosen.update(pos_rows[rid].tolist())
            chosen.add(int(rid))
        target_size = max(self.candidate_size, len(chosen))
        if len(chosen) < target_size:
            need = target_size - len(chosen)
            mask = np.ones(n, dtype=bool)
            if chosen:
                mask[np.fromiter(chosen, dtype=np.int64)] = False
            pool = np.nonzero(mask)[0]
            if need >= len(pool):
                chosen.update(pool.tolist())
            else:
                sampled = np.random.choice(pool, size=need, replace=False)
                chosen.update(sampled.tolist())
        return np.fromiter(chosen, dtype=np.int64)

    def _loss_dir(self, z1: torch.Tensor, z2: torch.Tensor, pos_rows: List[np.ndarray]) -> torch.Tensor:
        n = z1.shape[0]
        losses: List[torch.Tensor] = []
        for start in range(0, n, self.batch_size):
            row_ids = np.arange(start, min(start + self.batch_size, n), dtype=np.int64)
            cand_ids = self._candidate_ids(row_ids, pos_rows, n)
            sim = self._sim(z1[row_ids], z2[cand_ids])
            denom = sim.sum(dim=1) + 1e-8
            local_map = {int(node): idx for idx, node in enumerate(cand_ids.tolist())}
            for local_row, row_id in enumerate(row_ids.tolist()):
                pos_local = [local_map[p] for p in pos_rows[row_id].tolist() if p in local_map]
                if not pos_local:
                    pos_local = [local_map[row_id]]
                numer = sim[local_row, pos_local].sum()
                losses.append(-torch.log(numer / denom[local_row] + 1e-8))
        return torch.stack(losses).mean()

    def forward(self, z_mp: torch.Tensor, z_sc: torch.Tensor, pos_rows: List[np.ndarray]) -> torch.Tensor:
        z_mp = self.proj(z_mp)
        z_sc = self.proj(z_sc)
        loss_mp = self._loss_dir(z_mp, z_sc, pos_rows)
        loss_sc = self._loss_dir(z_sc, z_mp, pos_rows)
        return self.lam * loss_mp + (1.0 - self.lam) * loss_sc


def build_inputs(name: str) -> Tuple[credible.DatasetArtifacts, List[torch.Tensor], List[sp.csr_matrix], List[List[np.ndarray]], List[np.ndarray]]:
    art = credible.prepare_dataset(name)
    bundle = art.bundle
    payload = base.safe_torch_load(ROOT / "external_prepared" / name / f"{name.lower()}_nie.pt")
    offsets = global_offsets(name)
    full_features = row_normalize(np.asarray(payload["features"], dtype=np.float32))

    if name == "ACM":
        feats = [
            slice_type_features(full_features, *offsets["paper"]),
            slice_type_features(full_features, *offsets["author"]),
            slice_type_features(full_features, *offsets["subject"]),
        ]
        metapaths = [bundle.relations["PAP"], bundle.relations["PLP"], bundle.relations["PTP"]]
        p2a_src, p2a_dst = relation_edge_lists(payload, "paper:to:author")
        p2s_src, p2s_dst = relation_edge_lists(payload, "paper:to:subject")
        nei_index = [
            build_neighbor_index(bundle.adjacency.shape[0], p2a_src, p2a_dst, offsets["author"][0], offsets["author"][1] - offsets["author"][0]),
            build_neighbor_index(bundle.adjacency.shape[0], p2s_src, p2s_dst, offsets["subject"][0], offsets["subject"][1] - offsets["subject"][0]),
        ]
    elif name == "DBLP":
        feats = [
            slice_type_features(full_features, *offsets["author"]),
            slice_type_features(full_features, *offsets["paper"]),
        ]
        metapaths = [bundle.relations["APA"], bundle.relations["APCPA"], bundle.relations["APTPA"]]
        a2p_src, a2p_dst = relation_edge_lists(payload, "author:to:paper")
        nei_index = [
            build_neighbor_index(bundle.adjacency.shape[0], a2p_src, a2p_dst, offsets["paper"][0], offsets["paper"][1] - offsets["paper"][0]),
        ]
    else:
        feats = [
            slice_type_features(full_features, *offsets["user"]),
            slice_type_features(full_features, *offsets["business"]),
            slice_type_features(full_features, *offsets["compliment"]),
        ]
        metapaths = build_yelp_metapaths(bundle)
        u2b_src, u2b_dst = relation_edge_lists(payload, "user:user_to_business:business")
        u2c_src, u2c_dst = relation_edge_lists(payload, "user:user_to_compliment:compliment")
        nei_index = [
            build_neighbor_index(bundle.adjacency.shape[0], u2b_src, u2b_dst, offsets["business"][0], offsets["business"][1] - offsets["business"][0]),
            build_neighbor_index(bundle.adjacency.shape[0], u2c_src, u2c_dst, offsets["compliment"][0], offsets["compliment"][1] - offsets["compliment"][0]),
        ]

    pos_rows = build_pos_rows(name, art, metapaths, pos_k=int(HECO_CFG[name]["pos_k"]))
    return art, feats, metapaths, nei_index, pos_rows


def score_embeddings(art: credible.DatasetArtifacts, embeds: np.ndarray) -> np.ndarray:
    bundle = art.bundle
    z = row_normalize(embeds)
    communities = base.build_communities(bundle.adjacency)
    k = 12 if z.shape[0] < 10000 else 8
    sem_graph = base.build_cross_community_semantic_graph(z, communities, k=k)
    bridge = base.minmax_scale(np.asarray(sem_graph.sum(axis=1)).reshape(-1))

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, z.shape[0]), metric="cosine")
    nbrs.fit(z)
    distances, _ = nbrs.kneighbors(z)
    density = base.minmax_scale(np.asarray(1.0 - distances[:, 1:]).mean(axis=1))

    # Keep HeCo scoring primarily tied to the learned representation instead of
    # boosting it with strong hand-crafted centrality baselines.
    return base.minmax_scale(0.60 * bridge + 0.40 * density)


def run_dataset(name: str, force_cpu: bool = False) -> Dict[str, float]:
    started_at = time.perf_counter()
    set_seed()
    cfg = HECO_CFG[name]
    art, feats, metapaths, nei_index, pos_rows = build_inputs(name)
    out_dir = RESULT_ROOT / f"CUSTOM_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() and not force_cpu:
        try:
            _ = torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except Exception as exc:
            print(f"[HeCo/{name}] CUDA unavailable at runtime, falling back to CPU: {exc}")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"[HeCo/{name}] device={device}")
    feats = [feat.to(device) for feat in feats]
    mps = [to_torch_sparse(normalize_adj(symmetrize_target(mp)), device) for mp in metapaths]

    model = HeCo(
        hidden_dim=int(cfg["hidden_dim"]),
        feats_dim_list=[int(feat.shape[1]) for feat in feats],
        feat_drop=float(cfg["feat_drop"]),
        attn_drop=float(cfg["attn_drop"]),
        P=len(mps),
        sample_rate=list(cfg["sample_rate"]),
        nei_num=len(nei_index),
        tau=float(cfg["tau"]),
        lam=float(cfg["lam"]),
    ).to(device)
    contrast = SampledContrast(
        hidden_dim=int(cfg["hidden_dim"]),
        tau=float(cfg["tau"]),
        lam=float(cfg["lam"]),
        batch_size=int(cfg["batch_size"]),
        candidate_size=int(cfg["candidate_size"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(contrast.parameters()),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(int(cfg["epochs"])):
        model.train()
        contrast.train()
        z_mp, z_sc = forward_views(model, feats, mps, nei_index)
        loss = contrast(z_mp, z_sc, pos_rows)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(contrast.parameters()), max_norm=5.0)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss - 1e-5:
            best_loss = loss_value
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "contrast": {k: v.detach().cpu().clone() for k, v in contrast.state_dict().items()},
                "epoch": epoch,
            }
            wait = 0
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == int(cfg["epochs"]) - 1:
            print(f"[HeCo/{name}] epoch={epoch:03d} loss={loss_value:.6f}")
        if wait >= int(cfg["patience"]):
            print(f"[HeCo/{name}] early stop at epoch={epoch:03d}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        contrast.load_state_dict(best_state["contrast"])

    model.eval()
    with torch.no_grad():
        embeds = model.get_embeds(feats, mps).detach().cpu().numpy()
    scores = score_embeddings(art, embeds)
    rows, sir_curves, si_curves = credible.evaluate_methods(art, {"HeCo": scores})
    row = rows["HeCo"]

    np.save(out_dir / f"{name.lower()}_heco_pred.npy", scores)
    np.save(out_dir / f"{name.lower()}_heco_embed.npy", embeds)
    with (out_dir / f"{name.lower()}_heco_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    with (out_dir / f"{name.lower()}_sir_curve.json").open("w", encoding="utf-8") as f:
        json.dump(sir_curves["HeCo"], f, indent=2)
    with (out_dir / f"{name.lower()}_si_curve.json").open("w", encoding="utf-8") as f:
        json.dump(si_curves["HeCo"], f, indent=2)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": name,
                "config": cfg,
                "best_loss": best_loss,
                "best_epoch": None if best_state is None else int(best_state["epoch"]),
                "total_runtime_sec": float(time.perf_counter() - started_at),
                "metrics": row,
            },
            f,
            indent=2,
        )
    print(f"[HeCo/{name}] saved to {out_dir}")
    return {k: float(v) for k, v in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HeCo baseline on prepared ACM/DBLP/Yelp datasets")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "Yelp"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    summary = {}
    for name in args.datasets:
        summary[name] = run_dataset(name, force_cpu=args.cpu)

    out_path = RESULT_ROOT / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()

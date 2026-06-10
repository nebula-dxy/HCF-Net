import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from torch_geometric.datasets import HGBDataset

import credible_experiment_runner as credible
import experiment_runner as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "external_prepared"
OUT_DIR.mkdir(exist_ok=True)
SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def svd_embed(mat: sp.csr_matrix, dim: int) -> np.ndarray:
    max_dim = min(mat.shape[0] - 1, mat.shape[1] - 1, dim)
    if max_dim < 2:
        return np.zeros((mat.shape[0], max(dim, 2)), dtype=np.float32)
    reducer = TruncatedSVD(n_components=max_dim, random_state=SEED)
    emb = reducer.fit_transform(mat).astype(np.float32)
    return row_normalize(emb)


def norm_adj(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    deg = np.asarray(adj.sum(1)).reshape(-1)
    deg_inv = np.zeros_like(deg, dtype=np.float32)
    valid = deg > 0
    deg_inv[valid] = 1.0 / deg[valid]
    return sp.diags(deg_inv) @ adj


def to_global_offsets(type_sizes: List[Tuple[str, int]]) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    cursor = 0
    for name, size in type_sizes:
        offsets[name] = cursor
        cursor += size
    return offsets


def hetero_edges_to_sparse(
    num_nodes: int,
    rows: List[np.ndarray],
    cols: List[np.ndarray],
) -> sp.csr_matrix:
    all_row = np.concatenate(rows)
    all_col = np.concatenate(cols)
    data = np.ones_like(all_row, dtype=np.float32)
    adj = sp.coo_matrix((data, (all_row, all_col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj.maximum(adj.T)
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return adj.tocsr()


def propagate_target_features(full_adj: sp.csr_matrix, target_features: np.ndarray, target_count: int) -> np.ndarray:
    init = np.zeros((full_adj.shape[0], target_features.shape[1]), dtype=np.float32)
    init[:target_count] = target_features.astype(np.float32)
    trans = norm_adj(full_adj)
    hop1 = trans @ init
    hop2 = trans @ hop1
    mixed = 0.60 * init + 0.28 * hop1 + 0.12 * hop2
    return mixed.astype(np.float32)


def export_mahe_triplets(path: Path, triplets: List[Tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for src, dst in triplets:
            f.write(f"{src}\t{dst}\n")


def prepare_hgb_dataset(name: str) -> Dict[str, object]:
    art = credible.prepare_dataset(name)
    if name == "ACM":
        raw = HGBDataset(root=str(ROOT / "external_datasets" / "pyg_hgb"), name="ACM")[0]
        ordered_types = [("paper", raw["paper"].num_nodes), ("author", raw["author"].num_nodes), ("subject", raw["subject"].num_nodes), ("term", raw["term"].num_nodes)]
        mahe_relations = [("paper", "author"), ("paper", "subject")]
    elif name == "DBLP":
        raw = HGBDataset(root=str(ROOT / "external_datasets" / "pyg_hgb"), name="DBLP")[0]
        ordered_types = [("author", raw["author"].num_nodes), ("paper", raw["paper"].num_nodes), ("term", raw["term"].num_nodes), ("venue", raw["venue"].num_nodes)]
        mahe_relations = [("author", "paper"), ("paper", "venue")]
    else:
        raise ValueError(name)

    offsets = to_global_offsets(ordered_types)
    target_type = ordered_types[0][0]
    target_count = ordered_types[0][1]
    total_nodes = sum(size for _, size in ordered_types)

    edge_rows: List[np.ndarray] = []
    edge_cols: List[np.ndarray] = []
    edge_types: List[np.ndarray] = []
    relation_names: List[str] = []
    mahe_triplets: List[Tuple[str, str]] = []

    for rel_id, edge_type in enumerate(raw.edge_types):
        src_type, rel_name, dst_type = edge_type
        edge_index = raw[edge_type].edge_index.cpu().numpy()
        src = edge_index[0] + offsets[src_type]
        dst = edge_index[1] + offsets[dst_type]
        edge_rows.append(src.astype(np.int64))
        edge_cols.append(dst.astype(np.int64))
        edge_types.append(np.full(edge_index.shape[1], rel_id, dtype=np.int64))
        relation_names.append(f"{src_type}:{rel_name}:{dst_type}")

        if (src_type, dst_type) in mahe_relations:
            for s, d in zip(edge_index[0], edge_index[1]):
                mahe_triplets.append((f"a{s}" if src_type == target_type else f"p{s}", f"p{d}" if dst_type == "paper" else f"c{d}"))

    full_adj = hetero_edges_to_sparse(total_nodes, edge_rows, edge_cols)
    struct_features = svd_embed(full_adj, dim=128)
    sem_dense = propagate_target_features(full_adj, art.bundle.features, target_count)
    semantic_features = svd_embed(sp.csr_matrix(sem_dense), dim=128)

    labels = np.zeros(total_nodes, dtype=np.float32)
    labels[:target_count] = art.rank_target.astype(np.float32)
    invalid_masks = np.ones(total_nodes, dtype=np.int64)
    invalid_masks[:target_count] = 0

    payload = {
        "dataset_name": name,
        "target_type": target_type,
        "target_count": int(target_count),
        "num_nodes": int(total_nodes),
        "edges": (np.concatenate(edge_rows), np.concatenate(edge_cols)),
        "edge_types": np.concatenate(edge_types),
        "relation_names": relation_names,
        "features": struct_features.astype(np.float32),
        "semantic_features": semantic_features.astype(np.float32),
        "labels": labels,
        "invalid_masks": invalid_masks,
        "train_idx": art.bundle.train_idx.astype(np.int64),
        "val_idx": art.bundle.val_idx.astype(np.int64),
        "test_idx": art.bundle.test_idx.astype(np.int64),
    }

    out_dir = OUT_DIR / name
    out_dir.mkdir(exist_ok=True)
    torch.save(payload, out_dir / f"{name.lower()}_nie.pt")
    export_mahe_triplets(out_dir / f"{name.lower()}_mahe_edges.txt", mahe_triplets)
    return {
        "dataset": name,
        "target_count": int(target_count),
        "num_nodes": int(total_nodes),
        "relation_count": len(relation_names),
        "pt_path": str(out_dir / f"{name.lower()}_nie.pt"),
        "mahe_edges": str(out_dir / f"{name.lower()}_mahe_edges.txt"),
    }


def prepare_yelp_dataset() -> Dict[str, object]:
    art = credible.prepare_dataset("Yelp")
    bundle = art.bundle
    relations = bundle.relations
    type_sizes = [("user", relations["R1"].shape[0]), ("business", relations["R0"].shape[1]), ("compliment", relations["R2"].shape[1]), ("category", relations["R3"].shape[1]), ("city", relations["R4"].shape[1])]
    offsets = to_global_offsets(type_sizes)
    total_nodes = sum(size for _, size in type_sizes)

    spec = [
        ("user", "user_to_business", "business", relations["R0"], True),
        ("user", "user_to_user", "user", relations["R1"], False),
        ("user", "user_to_compliment", "compliment", relations["R2"], True),
        ("business", "business_to_category", "category", relations["R3"], True),
        ("business", "business_to_city", "city", relations["R4"], True),
    ]

    edge_rows: List[np.ndarray] = []
    edge_cols: List[np.ndarray] = []
    edge_types: List[np.ndarray] = []
    relation_names: List[str] = []
    mahe_triplets: List[Tuple[str, str]] = []
    rel_id = 0

    for src_type, rel_name, dst_type, mat, add_reverse in spec:
        coo = mat.tocoo()
        src = coo.row.astype(np.int64) + offsets[src_type]
        dst = coo.col.astype(np.int64) + offsets[dst_type]
        edge_rows.append(src)
        edge_cols.append(dst)
        edge_types.append(np.full(coo.nnz, rel_id, dtype=np.int64))
        relation_names.append(f"{src_type}:{rel_name}:{dst_type}")
        if src_type == "user" and dst_type == "business":
            for s, d in zip(coo.row, coo.col):
                mahe_triplets.append((f"a{s}", f"p{d}"))
        if src_type == "business" and dst_type == "category":
            for s, d in zip(coo.row, coo.col):
                mahe_triplets.append((f"p{s}", f"c{d}"))
        rel_id += 1

        if add_reverse:
            edge_rows.append(dst)
            edge_cols.append(src)
            edge_types.append(np.full(coo.nnz, rel_id, dtype=np.int64))
            relation_names.append(f"{dst_type}:rev_{rel_name}:{src_type}")
            rel_id += 1

    full_adj = hetero_edges_to_sparse(total_nodes, edge_rows, edge_cols)
    struct_features = svd_embed(full_adj, dim=128)
    sem_dense = propagate_target_features(full_adj, bundle.features, type_sizes[0][1])
    semantic_features = svd_embed(sp.csr_matrix(sem_dense), dim=128)

    target_count = type_sizes[0][1]
    labels = np.zeros(total_nodes, dtype=np.float32)
    labels[:target_count] = art.rank_target.astype(np.float32)
    invalid_masks = np.ones(total_nodes, dtype=np.int64)
    invalid_masks[:target_count] = 0

    out_dir = OUT_DIR / "Yelp"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "dataset_name": "Yelp",
        "target_type": "user",
        "target_count": int(target_count),
        "num_nodes": int(total_nodes),
        "edges": (np.concatenate(edge_rows), np.concatenate(edge_cols)),
        "edge_types": np.concatenate(edge_types),
        "relation_names": relation_names,
        "features": struct_features.astype(np.float32),
        "semantic_features": semantic_features.astype(np.float32),
        "labels": labels,
        "invalid_masks": invalid_masks,
        "train_idx": bundle.train_idx.astype(np.int64),
        "val_idx": bundle.val_idx.astype(np.int64),
        "test_idx": bundle.test_idx.astype(np.int64),
    }
    torch.save(payload, out_dir / "yelp_nie.pt")
    export_mahe_triplets(out_dir / "yelp_mahe_edges.txt", mahe_triplets)
    return {
        "dataset": "Yelp",
        "target_count": int(target_count),
        "num_nodes": int(total_nodes),
        "relation_count": len(relation_names),
        "pt_path": str(out_dir / "yelp_nie.pt"),
        "mahe_edges": str(out_dir / "yelp_mahe_edges.txt"),
    }


def main() -> None:
    set_seed()
    summaries = [
        prepare_hgb_dataset("ACM"),
        prepare_hgb_dataset("DBLP"),
        prepare_yelp_dataset(),
    ]
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print("Prepared external NIE datasets in", OUT_DIR)


if __name__ == "__main__":
    main()

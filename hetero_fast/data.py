from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch


ROOT = Path(__file__).resolve().parent.parent
PREPARED_DIR = ROOT / "external_prepared"


@dataclass
class HeteroDataset:
    name: str
    target_type: str
    target_count: int
    num_nodes: int
    edges_src: np.ndarray
    edges_dst: np.ndarray
    edge_types: np.ndarray
    relation_names: List[str]
    features: np.ndarray
    semantic_features: np.ndarray
    labels: np.ndarray
    invalid_masks: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def load_dataset(name: str) -> HeteroDataset:
    path = PREPARED_DIR / name / f"{name.lower()}_nie.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    src, dst = payload["edges"]
    return HeteroDataset(
        name=str(payload["dataset_name"]),
        target_type=str(payload["target_type"]),
        target_count=int(payload["target_count"]),
        num_nodes=int(payload["num_nodes"]),
        edges_src=np.asarray(src, dtype=np.int64).reshape(-1),
        edges_dst=np.asarray(dst, dtype=np.int64).reshape(-1),
        edge_types=np.asarray(payload["edge_types"], dtype=np.int64).reshape(-1),
        relation_names=[str(x) for x in payload["relation_names"]],
        features=np.asarray(payload["features"], dtype=np.float32),
        semantic_features=np.asarray(payload["semantic_features"], dtype=np.float32),
        labels=np.asarray(payload["labels"], dtype=np.float32).reshape(-1),
        invalid_masks=np.asarray(payload["invalid_masks"], dtype=np.int64).reshape(-1),
        train_idx=np.asarray(payload["train_idx"], dtype=np.int64).reshape(-1),
        val_idx=np.asarray(payload["val_idx"], dtype=np.int64).reshape(-1),
        test_idx=np.asarray(payload["test_idx"], dtype=np.int64).reshape(-1),
    )


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def minmax_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def target_adjacency(data: HeteroDataset) -> sp.csr_matrix:
    mask = (data.edges_src < data.target_count) & (data.edges_dst < data.target_count)
    src = data.edges_src[mask]
    dst = data.edges_dst[mask]
    vals = np.ones_like(src, dtype=np.float32)
    adj = sp.coo_matrix((vals, (src, dst)), shape=(data.target_count, data.target_count), dtype=np.float32)
    adj = adj.maximum(adj.T)
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return adj.tocsr()


def type_ranges(data: HeteroDataset) -> Dict[str, Tuple[int, int]]:
    ranges: Dict[str, List[int]] = {}
    for rel_name in data.relation_names:
        src_type, _, dst_type = rel_name.split(":", 2)
        if src_type not in ranges:
            mask = np.zeros(0, dtype=np.int64)
        if dst_type not in ranges:
            mask = np.zeros(0, dtype=np.int64)
    rel_to_nodes: Dict[str, List[np.ndarray]] = {}
    for rel_id, rel_name in enumerate(data.relation_names):
        src_type, _, dst_type = rel_name.split(":", 2)
        mask = data.edge_types == rel_id
        rel_to_nodes.setdefault(src_type, []).append(data.edges_src[mask])
        rel_to_nodes.setdefault(dst_type, []).append(data.edges_dst[mask])

    out: Dict[str, Tuple[int, int]] = {}
    for type_name, chunks in rel_to_nodes.items():
        if not chunks:
            continue
        vals = np.concatenate(chunks)
        out[type_name] = (int(vals.min()), int(vals.max()) + 1)
    return out


def node_types(data: HeteroDataset) -> np.ndarray:
    ranges = type_ranges(data)
    names = sorted(ranges.keys(), key=lambda k: ranges[k][0])
    out = np.full(data.num_nodes, -1, dtype=np.int64)
    for idx, name in enumerate(names):
        start, end = ranges[name]
        out[start:end] = idx
    out[out < 0] = len(names)
    return out


def relation_groups(data: HeteroDataset) -> Dict[int, Tuple[np.ndarray, np.ndarray, str]]:
    groups: Dict[int, Tuple[np.ndarray, np.ndarray, str]] = {}
    for rel_id, rel_name in enumerate(data.relation_names):
        mask = data.edge_types == rel_id
        groups[rel_id] = (
            data.edges_src[mask],
            data.edges_dst[mask],
            rel_name,
        )
    return groups

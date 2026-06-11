import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

import build_full_comparison_artifacts as compare
import credible_experiment_runner as credible
import experiment_runner as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_external_credible" / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_RESULTS_DIR = ROOT / "results_external_credible"


DEFAULT_ALPHA_CENTER = 0.12
DEFAULT_ALPHA_STEP = 0.02
DEFAULT_ONE_HOP_STEP = 0.10
DEFAULT_TWO_HOP_STEP = 0.04
DEFAULT_LATE_SEM_STEP = 0.04
DEFAULT_GRID_POINTS = 5

RANKING_PLOT_METRICS = [
    ("NDCG@100", ("metrics", "NDCG@100")),
    ("Ls", ("diversity", "Ls")),
    ("F(20)-SIR", ("metrics", "F(20)-SIR")),
    ("F(20)-SI", ("metrics", "F(20)-SI")),
]

PENALTY_PLOT_METRICS = [
    ("F(20)-SIR", ("metrics", "F(20)-SIR")),
    ("F(20)-SI", ("metrics", "F(20)-SI")),
    ("Ls", ("diversity", "Ls")),
    ("SeedDiversityScore", ("diversity", "SeedDiversityScore")),
]

DATASET_STYLE = {
    "ACM": {"color": "#b22222", "marker": "o"},
    "DBLP": {"color": "#1f77b4", "marker": "s"},
    "Yelp": {"color": "#2a9d8f", "marker": "^"},
}


def clean_float(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return value


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_centered_grid(center: float, step: float, points: int, lower: float, upper: float) -> List[float]:
    half = points // 2
    values = [center + (idx - half) * step for idx in range(points)]
    clipped = [min(upper, max(lower, round(v, 6))) for v in values]
    ordered: List[float] = []
    for value in clipped:
        if value not in ordered:
            ordered.append(value)
    return ordered


def load_paper_defaults(dataset: str) -> Dict[str, float | int | str]:
    ablation_path = PAPER_RESULTS_DIR / f"{dataset}_ablation_fast.json"
    if not ablation_path.exists():
        late_sem = 0.06 if dataset == "ACM" else 0.18
        epochs = 18 if dataset == "Yelp" else 20
        return {
            "source": "fallback_from_code",
            "epochs": epochs,
            "sim_runs": 4,
            "t_steps": 20,
            "semantic_train_alpha": 0.12,
            "late_semantic_weight": late_sem,
            "score_fusion": {
                "topo_weight": 1.0 - late_sem,
                "semantic_weight": late_sem,
                "pagerank_weight": 0.0,
                "degree_weight": 0.0,
                "ci_weight": 0.0,
            },
            "score_refine": {"refine_name": "identity", "hcf_weight": 1.0},
        }

    payload = load_json(ablation_path)
    full_meta = payload["variants"]["HCF-Net (Full)"]["meta"]
    return {
        "source": str(ablation_path.name),
        "epochs": int(payload["epochs"]),
        "sim_runs": int(payload["sim_runs"]),
        "t_steps": int(payload["t_steps"]),
        "semantic_train_alpha": 0.12,
        "late_semantic_weight": float(full_meta["score_fusion"]["semantic_weight"]),
        "score_fusion": dict(full_meta["score_fusion"]),
        "score_refine": dict(full_meta["score_refine"]),
    }


def default_scan_plan(dataset: str) -> Dict[str, object]:
    paper_defaults = load_paper_defaults(dataset)
    discount_defaults = default_discount_config(dataset)
    return {
        "source": paper_defaults["source"],
        "epochs": int(paper_defaults["epochs"]),
        "sim_runs": int(paper_defaults["sim_runs"]),
        "t_steps": int(paper_defaults["t_steps"]),
        "semantic_train_alpha": float(paper_defaults["semantic_train_alpha"]),
        "late_semantic_weight": float(paper_defaults["late_semantic_weight"]),
        "one_hop_penalty": float(discount_defaults["one_hop_penalty"]),
        "two_hop_penalty": float(discount_defaults["two_hop_penalty"]),
        "score_fusion": dict(paper_defaults["score_fusion"]),
        "score_refine": dict(paper_defaults["score_refine"]),
        "alpha_grid": build_centered_grid(
            float(paper_defaults["semantic_train_alpha"]),
            DEFAULT_ALPHA_STEP,
            DEFAULT_GRID_POINTS,
            0.0,
            0.24,
        ),
        "one_hop_grid": build_centered_grid(
            float(discount_defaults["one_hop_penalty"]),
            DEFAULT_ONE_HOP_STEP,
            DEFAULT_GRID_POINTS,
            0.05,
            0.98,
        ),
        "two_hop_grid": build_centered_grid(
            float(discount_defaults["two_hop_penalty"]),
            DEFAULT_TWO_HOP_STEP,
            DEFAULT_GRID_POINTS,
            0.50,
            0.98,
        ),
        "late_sem_grid": build_centered_grid(
            float(paper_defaults["late_semantic_weight"]),
            DEFAULT_LATE_SEM_STEP,
            DEFAULT_GRID_POINTS,
            0.0,
            0.24,
        ),
    }


def ranking_score(target: np.ndarray, pred: np.ndarray, eval_idx: np.ndarray) -> float:
    ndcg = ndcg_score(
        target[eval_idx].reshape(1, -1),
        pred[eval_idx].reshape(1, -1),
        k=min(100, len(eval_idx)),
    )
    spear = spearmanr(target[eval_idx], pred[eval_idx]).statistic
    if not np.isfinite(spear):
        spear = 0.0
    return float(ndcg + 0.55 * spear)


def default_discount_config(dataset: str) -> Dict[str, float]:
    if dataset == "ACM":
        return {
            "p1": 0.94,
            "comm_pen": 0.998,
            "bridge_boost": 0.06,
            "comm_boost": 0.04,
            "coverage_boost": 0.24,
            "distance_boost": 0.90,
            "two_hop_penalty": 0.84,
            "one_hop_penalty": 0.42,
            "dist_norm": 6.0,
            "candidate_pool": 160,
            "score_floor": 0.78,
        }
    if dataset == "DBLP":
        return {
            "p1": 0.86,
            "comm_pen": 0.70,
            "bridge_boost": 0.10,
            "comm_boost": 1.00,
            "coverage_boost": 2.0,
            "distance_boost": 1.00,
            "two_hop_penalty": 0.82,
            "one_hop_penalty": 0.55,
            "dist_norm": 7.0,
            "candidate_pool": 500,
            "score_floor": 0.65,
        }
    return {
        "p1": 0.90,
        "comm_pen": 0.79,
        "bridge_boost": 0.13,
        "comm_boost": 0.39,
        "coverage_boost": 0.96,
        "distance_boost": 1.00,
        "two_hop_penalty": 0.80,
        "one_hop_penalty": 0.45,
        "dist_norm": 5.0,
        "candidate_pool": 120,
        "score_floor": 0.76,
    }


def paper_penalty_rerank_topk(
    scores: np.ndarray,
    art: credible.DatasetArtifacts,
    *,
    lambda_1: float,
    lambda_2: float,
) -> List[int]:
    raw_adj = sp.csr_matrix(art.bundle.adjacency)
    raw_graph = nx.from_scipy_sparse_array(raw_adj)
    communities = art.communities
    bridge = base.minmax_scale(np.asarray(art.semantic_graph.sum(axis=1)).reshape(-1))
    degree = base.minmax_scale(np.asarray(raw_adj.sum(axis=1)).reshape(-1))
    cfg = default_discount_config(art.bundle.name)

    chosen: List[int] = []
    work_scores = scores.astype(np.float64).copy()
    masked = np.zeros_like(work_scores, dtype=bool)
    covered = np.zeros_like(work_scores, dtype=bool)
    comm_counts = {int(c): 0 for c in np.unique(communities)}
    comm_masks = {int(c): (communities == c) for c in np.unique(communities)}
    min_dist = np.full_like(work_scores, fill_value=np.inf, dtype=np.float64)

    for _ in range(int(art.diffusion_cfg["seed_k"])):
        dynamic_scores = work_scores * (1.0 + cfg["bridge_boost"] * bridge)

        boost = np.ones_like(work_scores)
        for comm, cnt in comm_counts.items():
            if cnt == 0:
                boost[comm_masks[comm]] *= (1.0 + cfg["comm_boost"])
            else:
                boost[comm_masks[comm]] *= (cfg["comm_pen"] ** cnt)
        dynamic_scores *= boost

        novelty = 1.0 + cfg["coverage_boost"] * degree * (~covered)
        dynamic_scores *= novelty

        if chosen:
            penalty = np.ones_like(work_scores)
            penalty[(min_dist > 1.0) & (min_dist <= 2.0)] = float(lambda_2)
            penalty[min_dist <= 1.0] = float(lambda_1)
            dynamic_scores *= penalty

        dynamic_scores = np.where(masked, -np.inf, dynamic_scores)
        node = int(np.argmax(dynamic_scores))
        if not np.isfinite(dynamic_scores[node]):
            break

        chosen.append(node)
        masked[node] = True
        covered[node] = True
        covered[raw_adj.getrow(node).indices] = True
        comm_counts[int(communities[node])] += 1
        work_scores[node] = -np.inf

        lengths = nx.single_source_shortest_path_length(raw_graph, node, cutoff=2)
        for target, dist in lengths.items():
            if masked[target]:
                continue
            if dist < min_dist[target]:
                min_dist[target] = float(dist)

    return chosen


def distance_aware_topk_with_overrides(
    scores: np.ndarray,
    art: credible.DatasetArtifacts,
    *,
    use_one_hop_discount: bool = True,
    overrides: Dict[str, float] | None = None,
) -> List[int]:
    cfg = default_discount_config(art.bundle.name)
    if overrides:
        cfg.update(overrides)

    adj = art.weighted_topo.tocsr()
    raw_adj = sp.csr_matrix(art.bundle.adjacency)
    raw_graph = nx.from_scipy_sparse_array(raw_adj)
    bridge = base.minmax_scale(np.asarray(art.semantic_graph.sum(axis=1)).reshape(-1))
    degree = base.minmax_scale(np.asarray(raw_adj.sum(axis=1)).reshape(-1))
    communities = art.communities
    work_scores = scores.astype(np.float64).copy()
    chosen: List[int] = []
    masked = np.zeros_like(work_scores, dtype=bool)
    covered = np.zeros_like(work_scores, dtype=bool)
    comm_counts = {int(c): 0 for c in np.unique(communities)}
    comm_masks = {int(c): (communities == c) for c in np.unique(communities)}
    sum_dist = np.zeros_like(work_scores, dtype=np.float64)
    seen_dist = np.zeros_like(work_scores, dtype=np.int32)
    min_dist = np.full_like(work_scores, fill_value=np.inf, dtype=np.float64)

    for _ in range(int(art.diffusion_cfg["seed_k"])):
        boost = np.ones_like(work_scores)
        for comm, cnt in comm_counts.items():
            if cnt == 0:
                boost[comm_masks[comm]] *= (1.0 + cfg["comm_boost"])
            else:
                boost[comm_masks[comm]] *= (cfg["comm_pen"] ** cnt)

        novelty = 1.0 + cfg["coverage_boost"] * degree * (~covered)
        dynamic_scores = work_scores * (1.0 + cfg["bridge_boost"] * bridge) * boost * novelty
        if chosen:
            mean_dist = np.divide(sum_dist, np.maximum(seen_dist, 1), where=seen_dist >= 0)
            norm_mean_dist = np.clip(mean_dist / cfg["dist_norm"], 0.0, 1.0)
            distance_term = 1.0 + cfg["distance_boost"] * norm_mean_dist
            distance_term[(min_dist <= 2.0)] *= cfg["two_hop_penalty"]
            if use_one_hop_discount:
                distance_term[(min_dist <= 1.0)] *= cfg["one_hop_penalty"]
            dynamic_scores *= distance_term

        dynamic_scores = np.where(masked, -np.inf, dynamic_scores)
        if not chosen:
            node = int(np.argmax(dynamic_scores))
        else:
            available = np.flatnonzero(~masked)
            pool_size = min(int(cfg["candidate_pool"]), available.size)
            if pool_size <= 0:
                break
            top_candidates = available[np.argpartition(dynamic_scores[available], -pool_size)[-pool_size:]]
            best_node = int(top_candidates[0])
            best_key = (-np.inf, -np.inf, -np.inf)
            peak_score = float(np.max(dynamic_scores[top_candidates]))
            min_allowed = peak_score * cfg["score_floor"]
            for cand in top_candidates:
                cand_score = float(dynamic_scores[cand])
                if cand_score < min_allowed:
                    continue
                cand_lengths = nx.single_source_shortest_path_length(raw_graph, int(cand))
                dist_vals = [float(cand_lengths[s]) for s in chosen if s in cand_lengths]
                if dist_vals:
                    mean_d = float(np.mean(dist_vals))
                    min_d = float(np.min(dist_vals))
                else:
                    mean_d = 0.0
                    min_d = 0.0
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
            work_scores[neigh] *= cfg["p1"]
        work_scores[node] = -np.inf

        lengths = nx.single_source_shortest_path_length(raw_graph, node)
        for target, dist in lengths.items():
            if masked[target]:
                continue
            sum_dist[target] += float(dist)
            seen_dist[target] += 1
            if dist < min_dist[target]:
                min_dist[target] = float(dist)
    return chosen


def evaluate_setting(
    art: credible.DatasetArtifacts,
    scores: np.ndarray,
    seeds: List[int],
    runs: int,
    t_steps: int,
) -> Tuple[Dict[str, float], Dict[str, float], List[float], List[float]]:
    diffusion_graph = nx.from_scipy_sparse_array(art.diffusion_graph)
    raw_graph = nx.from_scipy_sparse_array(art.bundle.adjacency)
    sir = credible.SemanticSIRSimulation(
        diffusion_graph,
        beta=float(art.diffusion_cfg["sir_beta"]),
        gamma=float(art.diffusion_cfg["sir_gamma"]),
    )
    si = credible.SemanticSISimulation(diffusion_graph, beta=float(art.diffusion_cfg["si_beta"]))
    metrics = base.evaluate_rankings(art.rank_target, scores, art.bundle.test_idx, k=100)
    sir_curve = credible.average_curve(sir, seeds, runs=runs, t_steps=t_steps)
    si_curve = credible.average_curve(si, seeds, runs=runs, t_steps=t_steps)
    metrics["F(20)-SIR"] = float(sir_curve[-1])
    metrics["F(20)-SI"] = float(si_curve[-1])
    diversity = compare.diversity_metrics(art.bundle.adjacency, art.communities, raw_graph, seeds)
    return metrics, diversity, sir_curve, si_curve


def train_model_for_alpha(
    art: credible.DatasetArtifacts,
    dataset: str,
    epochs: int,
    force_cpu: bool,
    semantic_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    credible.set_seed()
    bundle = art.bundle
    device = credible.get_device(force_cpu=force_cpu)

    x = torch.tensor(bundle.features, dtype=torch.float32, device=device)
    node_types = torch.tensor(bundle.node_types[: bundle.adjacency.shape[0]], dtype=torch.long, device=device)
    a_topo = credible.to_torch_sparse(base.sym_norm_sp(art.weighted_topo), device)
    a_sem = credible.to_torch_sparse(base.sym_norm_sp(art.semantic_graph), device)
    y_struct = torch.tensor(art.struct_target, dtype=torch.float32, device=device)
    y_sem = torch.tensor(art.semantic_target, dtype=torch.float32, device=device)
    y_rank = torch.tensor(art.rank_target, dtype=torch.float32, device=device)

    model_nodes = bundle.adjacency.shape[0]
    hidden_dim = 80 if model_nodes > 10000 else 96
    model = credible.CredibleHCFNet(
        x.shape[1],
        hidden_dim=hidden_dim,
        dropout=0.20,
        num_node_types=len(bundle.type_names or []),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    best_state = None
    best_score = -float("inf")
    align_weight = 0.08 if dataset == "ACM" else (0.06 if dataset == "DBLP" else 0.04)

    for epoch in range(epochs):
        model.train()
        topo_score, sem_score, z_topo, _, z_sem_align = model(x, a_topo, a_sem, node_types=node_types)
        fused = topo_score + semantic_alpha * sem_score
        warm_align = align_weight * min(1.0, (epoch + 1) / max(8, epochs * 0.35))
        loss = (
            F.mse_loss(topo_score[bundle.train_idx], y_struct[bundle.train_idx])
            + 0.35 * F.mse_loss(sem_score[bundle.train_idx], y_sem[bundle.train_idx])
            + 0.75 * F.mse_loss(fused[bundle.train_idx], y_rank[bundle.train_idx])
            + 0.25 * credible.sampled_pairwise_rank_loss(fused[bundle.train_idx], y_rank[bundle.train_idx])
            + 0.15 * credible.correlation_loss(fused[bundle.train_idx], y_rank[bundle.train_idx])
            + warm_align * credible.asymmetric_contrastive_align_loss(
                z_sem_align[bundle.train_idx],
                z_topo[bundle.train_idx],
                sample_size=1024 if model_nodes > 10000 else 2048,
                tau=0.5,
            )
            + 0.08 * credible.reconstruction_loss(
                z_topo,
                art.weighted_topo,
                sample_size=4000 if model_nodes > 10000 else 7000,
            )
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            topo_val, sem_val, _, _, _ = model(x, a_topo, a_sem, node_types=node_types)
            pred = base.minmax_scale(topo_val.detach().cpu().numpy()) + semantic_alpha * base.minmax_scale(sem_val.detach().cpu().numpy())
        score = ranking_score(art.rank_target, pred, bundle.val_idx)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        topo_score, sem_score, _, _, _ = model(x, a_topo, a_sem, node_types=node_types)

    return (
        topo_score.detach().cpu().numpy(),
        sem_score.detach().cpu().numpy(),
        {
            "device": str(device),
            "epochs": int(epochs),
            "semantic_train_alpha": float(semantic_alpha),
            "best_val_score": float(best_score),
        },
    )


def build_late_fusion_scores(
    topo_scores: np.ndarray,
    sem_scores: np.ndarray,
    baselines: Dict[str, np.ndarray],
    base_meta: Dict[str, object],
    semantic_weight: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    topo = base.minmax_scale(topo_scores)
    sem = base.minmax_scale(sem_scores)
    pagerank = base.minmax_scale(baselines["PageRank"])
    degree = base.minmax_scale(baselines["Degree"])
    ci = base.minmax_scale(baselines["CI"])

    pr_w = float(base_meta.get("pagerank_weight", 0.0))
    deg_w = float(base_meta.get("degree_weight", 0.0))
    ci_w = float(base_meta.get("ci_weight", 0.0))
    aux = semantic_weight + pr_w + deg_w + ci_w
    topo_w = 1.0 - aux
    if topo_w < 0.35:
        raise ValueError(f"semantic_weight={semantic_weight:.2f} makes topo weight too small: {topo_w:.2f}")

    pred = base.minmax_scale(topo_w * topo + semantic_weight * sem + pr_w * pagerank + deg_w * degree + ci_w * ci)
    return pred, {
        "topo_weight": clean_float(topo_w),
        "semantic_weight": clean_float(semantic_weight),
        "pagerank_weight": clean_float(pr_w),
        "degree_weight": clean_float(deg_w),
        "ci_weight": clean_float(ci_w),
    }


def build_fixed_fusion_scores(
    topo_scores: np.ndarray,
    sem_scores: np.ndarray,
    baselines: Dict[str, np.ndarray],
    fusion_meta: Dict[str, object],
) -> Tuple[np.ndarray, Dict[str, float]]:
    topo = base.minmax_scale(topo_scores)
    sem = base.minmax_scale(sem_scores)
    pagerank = base.minmax_scale(baselines["PageRank"])
    degree = base.minmax_scale(baselines["Degree"])
    ci = base.minmax_scale(baselines["CI"])

    topo_w = float(fusion_meta.get("topo_weight", 1.0))
    sem_w = float(fusion_meta.get("semantic_weight", 0.0))
    pr_w = float(fusion_meta.get("pagerank_weight", 0.0))
    deg_w = float(fusion_meta.get("degree_weight", 0.0))
    ci_w = float(fusion_meta.get("ci_weight", 0.0))
    pred = base.minmax_scale(topo_w * topo + sem_w * sem + pr_w * pagerank + deg_w * degree + ci_w * ci)
    return pred, {
        "topo_weight": clean_float(topo_w),
        "semantic_weight": clean_float(sem_w),
        "pagerank_weight": clean_float(pr_w),
        "degree_weight": clean_float(deg_w),
        "ci_weight": clean_float(ci_w),
    }


def apply_fixed_refine_scores(
    fused_scores: np.ndarray,
    baselines: Dict[str, np.ndarray],
    semantic_graph: sp.csr_matrix,
    refine_meta: Dict[str, object],
) -> Tuple[np.ndarray, Dict[str, float | str]]:
    hcf = base.minmax_scale(fused_scores)
    pagerank = base.minmax_scale(baselines["PageRank"])
    degree = base.minmax_scale(baselines["Degree"])
    ci = base.minmax_scale(baselines["CI"])
    bridge = base.minmax_scale(np.asarray(semantic_graph.sum(axis=1)).reshape(-1))

    hcf_w = float(refine_meta.get("hcf_weight", 1.0))
    pr_w = float(refine_meta.get("pagerank_weight", 0.0))
    deg_w = float(refine_meta.get("degree_weight", 0.0))
    ci_w = float(refine_meta.get("ci_weight", 0.0))
    bridge_w = float(refine_meta.get("bridge_weight", 0.0))
    pred = base.minmax_scale(hcf_w * hcf + pr_w * pagerank + deg_w * degree + ci_w * ci + bridge_w * bridge)
    return pred, {
        "refine_name": str(refine_meta.get("refine_name", "fixed_default")),
        "hcf_weight": clean_float(hcf_w),
        "pagerank_weight": clean_float(pr_w),
        "degree_weight": clean_float(deg_w),
        "ci_weight": clean_float(ci_w),
        "bridge_weight": clean_float(bridge_w),
    }


def render_study_csv(path: Path, results: List[Dict[str, object]], param_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                param_name,
                "NDCG@100",
                "Spearman",
                "F(20)-SIR",
                "F(20)-SI",
                "Ls",
                "SeedDiversityScore",
                "Effective1HopCoverage",
                "Unique1HopCoverage",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item[param_name],
                    f"{float(item['metrics']['NDCG@100']):.6f}",
                    f"{float(item['metrics']['Spearman']):.6f}",
                    f"{float(item['metrics']['F(20)-SIR']):.6f}",
                    f"{float(item['metrics']['F(20)-SI']):.6f}",
                    f"{float(item['diversity']['Ls']):.6f}",
                    f"{float(item['diversity']['SeedDiversityScore']):.6f}",
                    f"{float(item['diversity']['Effective1HopCoverage']):.6f}",
                    f"{float(item['diversity']['Unique1HopCoverage']):.6f}",
                ]
            )


def render_lambda_heatmap_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lambda_1",
                "lambda_2",
                "NDCG@100",
                "Spearman",
                "F(20)-SIR",
                "F(20)-SI",
                "Ls",
                "SeedDiversityScore",
                "Effective1HopCoverage",
                "Unique1HopCoverage",
            ]
        )
        for item in rows:
            writer.writerow(
                [
                    f"{float(item['lambda_1']):.6f}",
                    f"{float(item['lambda_2']):.6f}",
                    f"{float(item['metrics']['NDCG@100']):.6f}",
                    f"{float(item['metrics']['Spearman']):.6f}",
                    f"{float(item['metrics']['F(20)-SIR']):.6f}",
                    f"{float(item['metrics']['F(20)-SI']):.6f}",
                    f"{float(item['diversity']['Ls']):.6f}",
                    f"{float(item['diversity']['SeedDiversityScore']):.6f}",
                    f"{float(item['diversity']['Effective1HopCoverage']):.6f}",
                    f"{float(item['diversity']['Unique1HopCoverage']):.6f}",
                ]
            )


def render_study_plot(
    dataset: str,
    study_name: str,
    param_name: str,
    results: List[Dict[str, object]],
    metric_specs: List[Tuple[str, Tuple[str, str]]] | None = None,
) -> None:
    plot_metrics = metric_specs if metric_specs is not None else RANKING_PLOT_METRICS
    x = [float(item[param_name]) for item in results]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    for ax, (metric_name, path) in zip(axes, plot_metrics):
        y = [float(item[path[0]][path[1]]) for item in results]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#9b2226")
        ax.set_title(metric_name)
        ax.set_xlabel(param_name)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{dataset} Sensitivity: {study_name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{dataset}_{study_name}.png", dpi=250)
    plt.close(fig)


def render_cross_dataset_ndcg_plot(
    summary: Dict[str, object],
    study_name: str,
    param_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for dataset in ["ACM", "DBLP", "Yelp"]:
        if dataset not in summary:
            continue
        results = summary[dataset]["studies"][study_name]["results"]
        x = [float(item[param_name]) for item in results]
        y = [float(item["metrics"]["NDCG@100"]) for item in results]
        style = DATASET_STYLE.get(dataset, {"color": "#444444", "marker": "o"})
        ax.plot(
            x,
            y,
            label=dataset,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6,
        )
    ax.set_xlabel(param_name)
    ax.set_ylabel("NDCG@100")
    ax.set_title(f"NDCG Sensitivity to {param_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"combined_ndcg_{study_name}.png", dpi=300)
    plt.close(fig)


def render_cross_dataset_metric_plot(
    summary: Dict[str, object],
    study_name: str,
    param_name: str,
    metric_name: str,
    metric_group: str,
    filename_tag: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for dataset in ["ACM", "DBLP", "Yelp"]:
        if dataset not in summary:
            continue
        results = summary[dataset]["studies"][study_name]["results"]
        x = [float(item[param_name]) for item in results]
        y = [float(item[metric_group][metric_name]) for item in results]
        style = DATASET_STYLE.get(dataset, {"color": "#444444", "marker": "o"})
        ax.plot(
            x,
            y,
            label=dataset,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6,
        )
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Sensitivity to {param_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"combined_{filename_tag}_{study_name}.png", dpi=300)
    plt.close(fig)


def render_penalty_heatmap(
    dataset: str,
    study_name: str,
    rows: List[Dict[str, object]],
    metric_name: str,
    metric_group: str,
    filename_tag: str,
) -> None:
    lambda_1_values = sorted({float(item["lambda_1"]) for item in rows})
    lambda_2_values = sorted({float(item["lambda_2"]) for item in rows})
    matrix = np.full((len(lambda_2_values), len(lambda_1_values)), np.nan, dtype=np.float64)

    l1_index = {value: idx for idx, value in enumerate(lambda_1_values)}
    l2_index = {value: idx for idx, value in enumerate(lambda_2_values)}
    for item in rows:
        matrix[l2_index[float(item["lambda_2"])]][l1_index[float(item["lambda_1"])]] = float(
            item[metric_group][metric_name]
        )

    fig, ax = plt.subplots(figsize=(6.6, 5.5))
    image = ax.imshow(matrix, cmap="YlOrRd", origin="lower", aspect="auto")
    ax.set_xticks(range(len(lambda_1_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in lambda_1_values])
    ax.set_yticks(range(len(lambda_2_values)))
    ax.set_yticklabels([f"{value:.2f}" for value in lambda_2_values])
    ax.set_xlabel("lambda_1")
    ax.set_ylabel("lambda_2")
    ax.set_title(f"{dataset} {metric_name} over (lambda_1, lambda_2)")
    for row_idx in range(len(lambda_2_values)):
        for col_idx in range(len(lambda_1_values)):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(image, ax=ax, shrink=0.9, label=metric_name)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{dataset}_{study_name}_{filename_tag}.png", dpi=300)
    plt.close(fig)


def summarize_best(results: List[Dict[str, object]], param_name: str, key: Tuple[str, str]) -> Dict[str, float]:
    best = max(results, key=lambda item: float(item[key[0]][key[1]]))
    return {
        param_name: clean_float(best[param_name]),
        key[1]: clean_float(best[key[0]][key[1]]),
    }


def summarize_best_pair(
    results: List[Dict[str, object]],
    key: Tuple[str, str],
) -> Dict[str, float]:
    best = max(results, key=lambda item: float(item[key[0]][key[1]]))
    return {
        "lambda_1": clean_float(best["lambda_1"]),
        "lambda_2": clean_float(best["lambda_2"]),
        key[1]: clean_float(best[key[0]][key[1]]),
    }


def run_alpha_study(
    dataset: str,
    art: credible.DatasetArtifacts,
    baselines: Dict[str, np.ndarray],
    epochs: int,
    force_cpu: bool,
    runs: int,
    t_steps: int,
    alpha_grid: List[float],
    default_fusion_meta: Dict[str, object],
    default_refine_meta: Dict[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for alpha in alpha_grid:
        topo_scores, sem_scores, train_meta = train_model_for_alpha(
            art=art,
            dataset=dataset,
            epochs=epochs,
            force_cpu=force_cpu,
            semantic_alpha=float(alpha),
        )
        fused_scores, choose_meta = build_fixed_fusion_scores(
            topo_scores,
            sem_scores,
            baselines,
            default_fusion_meta,
        )
        final_scores, refine_meta = apply_fixed_refine_scores(
            fused_scores,
            baselines,
            art.semantic_graph,
            default_refine_meta,
        )
        seeds = credible.hcf_diverse_topk(final_scores, art)
        metrics, diversity, _, _ = evaluate_setting(art, final_scores, seeds, runs=runs, t_steps=t_steps)
        results.append(
            {
                "semantic_train_alpha": float(alpha),
                "metrics": metrics,
                "diversity": diversity,
                "meta": {
                    "train": train_meta,
                    "score_fusion": choose_meta,
                    "score_refine": refine_meta,
                },
            }
        )
        print(
            f"[sensitivity:{dataset}:alpha] alpha={alpha:.2f} "
            f"ndcg={metrics['NDCG@100']:.4f} sir={metrics['F(20)-SIR']:.4f} ls={diversity['Ls']:.4f}"
        )
    results.sort(key=lambda item: float(item["semantic_train_alpha"]))
    summary = {
        "best_ndcg": summarize_best(results, "semantic_train_alpha", ("metrics", "NDCG@100")),
        "best_sir": summarize_best(results, "semantic_train_alpha", ("metrics", "F(20)-SIR")),
        "best_ls": summarize_best(results, "semantic_train_alpha", ("diversity", "Ls")),
    }
    return results, summary


def run_base_training(
    dataset: str,
    art: credible.DatasetArtifacts,
    baselines: Dict[str, np.ndarray],
    epochs: int,
    force_cpu: bool,
    default_fusion_meta: Dict[str, object],
    default_refine_meta: Dict[str, object],
) -> Dict[str, object]:
    topo_scores, sem_scores, train_meta = train_model_for_alpha(
        art=art,
        dataset=dataset,
        epochs=epochs,
        force_cpu=force_cpu,
        semantic_alpha=0.12,
    )
    fused_scores, choose_meta = build_fixed_fusion_scores(
        topo_scores,
        sem_scores,
        baselines,
        default_fusion_meta,
    )
    final_scores, refine_meta = apply_fixed_refine_scores(
        fused_scores,
        baselines,
        art.semantic_graph,
        default_refine_meta,
    )
    return {
        "topo_scores": topo_scores,
        "sem_scores": sem_scores,
        "fused_scores": fused_scores,
        "final_scores": final_scores,
        "train_meta": train_meta,
        "choose_meta": choose_meta,
        "refine_meta": refine_meta,
    }


def run_lambda_1_study(
    dataset: str,
    art: credible.DatasetArtifacts,
    base_scores: np.ndarray,
    runs: int,
    t_steps: int,
    lambda_1_grid: List[float],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    results: List[Dict[str, object]] = []
    default_cfg = default_discount_config(dataset)
    for value in lambda_1_grid:
        seeds = paper_penalty_rerank_topk(
            base_scores,
            art,
            lambda_1=float(value),
            lambda_2=float(default_cfg["two_hop_penalty"]),
        )
        metrics, diversity, _, _ = evaluate_setting(art, base_scores, seeds, runs=runs, t_steps=t_steps)
        results.append(
            {
                "lambda_1": float(value),
                "metrics": metrics,
                "diversity": diversity,
                "seeds": seeds,
                "meta": {
                    "paper_penalty_defaults": {
                        "lambda_1": float(default_cfg["one_hop_penalty"]),
                        "lambda_2": float(default_cfg["two_hop_penalty"]),
                    },
                },
            }
        )
        print(
            f"[sensitivity:{dataset}:lambda1] lambda_1={value:.2f} "
            f"sir={metrics['F(20)-SIR']:.4f} si={metrics['F(20)-SI']:.4f} ls={diversity['Ls']:.4f}"
        )
    results.sort(key=lambda item: float(item["lambda_1"]))
    summary = {
        "best_sir": summarize_best(results, "lambda_1", ("metrics", "F(20)-SIR")),
        "best_si": summarize_best(results, "lambda_1", ("metrics", "F(20)-SI")),
        "best_ls": summarize_best(results, "lambda_1", ("diversity", "Ls")),
        "best_diversity": summarize_best(results, "lambda_1", ("diversity", "SeedDiversityScore")),
        "default_lambda_1": clean_float(default_cfg["one_hop_penalty"]),
        "fixed_lambda_2": clean_float(default_cfg["two_hop_penalty"]),
    }
    return results, summary


def run_lambda_2_study(
    dataset: str,
    art: credible.DatasetArtifacts,
    base_scores: np.ndarray,
    runs: int,
    t_steps: int,
    lambda_2_grid: List[float],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    results: List[Dict[str, object]] = []
    default_cfg = default_discount_config(dataset)
    for value in lambda_2_grid:
        seeds = paper_penalty_rerank_topk(
            base_scores,
            art,
            lambda_1=float(default_cfg["one_hop_penalty"]),
            lambda_2=float(value),
        )
        metrics, diversity, _, _ = evaluate_setting(art, base_scores, seeds, runs=runs, t_steps=t_steps)
        results.append(
            {
                "lambda_2": float(value),
                "metrics": metrics,
                "diversity": diversity,
                "seeds": seeds,
                "meta": {
                    "paper_penalty_defaults": {
                        "lambda_1": float(default_cfg["one_hop_penalty"]),
                        "lambda_2": float(default_cfg["two_hop_penalty"]),
                    },
                },
            }
        )
        print(
            f"[sensitivity:{dataset}:lambda2] lambda_2={value:.2f} "
            f"sir={metrics['F(20)-SIR']:.4f} si={metrics['F(20)-SI']:.4f} ls={diversity['Ls']:.4f}"
        )
    results.sort(key=lambda item: float(item["lambda_2"]))
    summary = {
        "best_sir": summarize_best(results, "lambda_2", ("metrics", "F(20)-SIR")),
        "best_si": summarize_best(results, "lambda_2", ("metrics", "F(20)-SI")),
        "best_ls": summarize_best(results, "lambda_2", ("diversity", "Ls")),
        "best_diversity": summarize_best(results, "lambda_2", ("diversity", "SeedDiversityScore")),
        "fixed_lambda_1": clean_float(default_cfg["one_hop_penalty"]),
        "default_lambda_2": clean_float(default_cfg["two_hop_penalty"]),
    }
    return results, summary


def run_lambda_heatmap_study(
    dataset: str,
    art: credible.DatasetArtifacts,
    base_scores: np.ndarray,
    runs: int,
    t_steps: int,
    lambda_1_grid: List[float],
    lambda_2_grid: List[float],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for lambda_2 in lambda_2_grid:
        for lambda_1 in lambda_1_grid:
            seeds = paper_penalty_rerank_topk(
                base_scores,
                art,
                lambda_1=float(lambda_1),
                lambda_2=float(lambda_2),
            )
            metrics, diversity, _, _ = evaluate_setting(art, base_scores, seeds, runs=runs, t_steps=t_steps)
            rows.append(
                {
                    "lambda_1": float(lambda_1),
                    "lambda_2": float(lambda_2),
                    "metrics": metrics,
                    "diversity": diversity,
                    "seeds": seeds,
                }
            )
            print(
                f"[sensitivity:{dataset}:lambda-grid] lambda_1={lambda_1:.2f} lambda_2={lambda_2:.2f} "
                f"sir={metrics['F(20)-SIR']:.4f} si={metrics['F(20)-SI']:.4f} ls={diversity['Ls']:.4f}"
            )
    summary = {
        "best_sir": summarize_best_pair(rows, ("metrics", "F(20)-SIR")),
        "best_si": summarize_best_pair(rows, ("metrics", "F(20)-SI")),
        "best_ls": summarize_best_pair(rows, ("diversity", "Ls")),
        "best_diversity": summarize_best_pair(rows, ("diversity", "SeedDiversityScore")),
        "grid_shape": {"lambda_1_points": len(lambda_1_grid), "lambda_2_points": len(lambda_2_grid)},
    }
    return rows, summary


def run_late_fusion_study(
    dataset: str,
    art: credible.DatasetArtifacts,
    baselines: Dict[str, np.ndarray],
    topo_scores: np.ndarray,
    sem_scores: np.ndarray,
    choose_meta: Dict[str, object],
    default_refine_meta: Dict[str, object],
    runs: int,
    t_steps: int,
    late_sem_grid: List[float],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for sem_w in late_sem_grid:
        try:
            fused_scores, fusion_meta = build_late_fusion_scores(
                topo_scores=topo_scores,
                sem_scores=sem_scores,
                baselines=baselines,
                base_meta=choose_meta,
                semantic_weight=float(sem_w),
            )
        except ValueError:
            continue
        final_scores, refine_meta = apply_fixed_refine_scores(
            fused_scores,
            baselines,
            art.semantic_graph,
            default_refine_meta,
        )
        seeds = credible.hcf_diverse_topk(final_scores, art)
        metrics, diversity, _, _ = evaluate_setting(art, final_scores, seeds, runs=runs, t_steps=t_steps)
        results.append(
            {
                "late_semantic_weight": float(sem_w),
                "metrics": metrics,
                "diversity": diversity,
                "meta": {
                    "score_fusion": fusion_meta,
                    "score_refine": refine_meta,
                    "base_choose_meta": choose_meta,
                },
            }
        )
        print(
            f"[sensitivity:{dataset}:late] sem_w={sem_w:.2f} "
            f"ndcg={metrics['NDCG@100']:.4f} sir={metrics['F(20)-SIR']:.4f} ls={diversity['Ls']:.4f}"
        )
    results.sort(key=lambda item: float(item["late_semantic_weight"]))
    summary = {
        "best_ndcg": summarize_best(results, "late_semantic_weight", ("metrics", "NDCG@100")),
        "best_sir": summarize_best(results, "late_semantic_weight", ("metrics", "F(20)-SIR")),
        "best_ls": summarize_best(results, "late_semantic_weight", ("diversity", "Ls")),
        "default_semantic_weight": clean_float(choose_meta.get("semantic_weight", 0.0)),
    }
    return results, summary


def write_dataset_markdown(dataset: str, payload: Dict[str, object]) -> None:
    studies = payload["studies"]
    lines = [
        f"# {dataset} Sensitivity Analysis",
        "",
        "Parameter-to-code mapping:",
        "",
        "- `semantic_train_alpha`: training/inference semantic fusion coefficient in `fused = topo_score + alpha * sem_score`.",
        "- `lambda_1`: one-hop neighborhood penalty in the paper-style seed re-ranking stage.",
        "- `lambda_2`: two-hop neighborhood penalty in the paper-style seed re-ranking stage.",
        "- `late_semantic_weight`: semantic weight in the validation-guided late score fusion stage.",
        "",
        "Important note:",
        "",
        "- `semantic_train_alpha` and `late_semantic_weight` directly affect the ranking scores, so `NDCG@100` is meaningful for them.",
        "- `lambda_1` and `lambda_2` only affect the seed re-ranking / selection stage after node scores are fixed, so they are expected to change diffusion and diversity metrics much more than `NDCG@100`.",
        "",
        "Default parameter values used by the current main pipeline:",
        "",
        f"- `semantic_train_alpha = {float(payload['defaults']['semantic_train_alpha']):.4f}`",
        f"- `lambda_1 = {float(payload['defaults']['lambda_1']):.4f}`",
        f"- `lambda_2 = {float(payload['defaults']['lambda_2']):.4f}`",
        f"- `late_semantic_weight = {float(payload['defaults']['late_semantic_weight']):.4f}`",
        f"- `source = {payload['defaults']['source']}`",
        f"- `epochs = {int(payload['epochs'])}, sim_runs = {int(payload['sim_runs'])}, t_steps = {int(payload['t_steps'])}`",
        "",
    ]
    for study_key in ["semantic_train_alpha", "lambda_1", "lambda_2", "late_semantic_weight"]:
        block = studies[study_key]
        lines.extend(
            [
                f"## {study_key}",
                "",
                "| Value | NDCG@100 | Spearman | F(20)-SIR | F(20)-SI | Ls |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in block["results"]:
            value = item[study_key]
            lines.append(
                f"| {float(value):.4f} | {float(item['metrics']['NDCG@100']):.4f} | "
                f"{float(item['metrics']['Spearman']):.4f} | {float(item['metrics']['F(20)-SIR']):.4f} | "
                f"{float(item['metrics']['F(20)-SI']):.4f} | {float(item['diversity']['Ls']):.4f} |"
            )
        lines.append("")
        lines.append("Best summary:")
        lines.append("")
        for key, val in block["summary"].items():
            lines.append(f"- `{key}`: `{json.dumps(val, ensure_ascii=False)}`")
        lines.append("")
    heatmap = studies["lambda_grid"]
    lines.extend(
        [
            "## lambda_grid",
            "",
            "| lambda_1 | lambda_2 | F(20)-SIR | F(20)-SI | Ls | SeedDiversityScore |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in heatmap["results"]:
        lines.append(
            f"| {float(item['lambda_1']):.4f} | {float(item['lambda_2']):.4f} | "
            f"{float(item['metrics']['F(20)-SIR']):.4f} | {float(item['metrics']['F(20)-SI']):.4f} | "
            f"{float(item['diversity']['Ls']):.4f} | {float(item['diversity']['SeedDiversityScore']):.4f} |"
        )
    lines.append("")
    lines.append("Best summary:")
    lines.append("")
    for key, val in heatmap["summary"].items():
        lines.append(f"- `{key}`: `{json.dumps(val, ensure_ascii=False)}`")
    lines.append("")
    (OUT_DIR / f"{dataset}_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")


def run_dataset(
    dataset: str,
    epochs: int,
    force_cpu: bool,
    runs: int,
    t_steps: int,
    alpha_grid: List[float],
    lambda_1_grid: List[float],
    two_hop_grid: List[float],
    late_sem_grid: List[float],
    defaults_meta: Dict[str, object],
) -> Dict[str, object]:
    art = credible.prepare_dataset(dataset)
    baselines = {k: base.minmax_scale(v) for k, v in base.compute_baseline_scores(art.bundle.adjacency).items()}
    default_fusion_meta = dict(defaults_meta["score_fusion"])
    default_refine_meta = dict(defaults_meta["score_refine"])

    alpha_results, alpha_summary = run_alpha_study(
        dataset=dataset,
        art=art,
        baselines=baselines,
        epochs=epochs,
        force_cpu=force_cpu,
        runs=runs,
        t_steps=t_steps,
        alpha_grid=alpha_grid,
        default_fusion_meta=default_fusion_meta,
        default_refine_meta=default_refine_meta,
    )

    base_run = run_base_training(
        dataset=dataset,
        art=art,
        baselines=baselines,
        epochs=epochs,
        force_cpu=force_cpu,
        default_fusion_meta=default_fusion_meta,
        default_refine_meta=default_refine_meta,
    )

    lambda_1_results, lambda_1_summary = run_lambda_1_study(
        dataset=dataset,
        art=art,
        base_scores=base_run["final_scores"],
        runs=runs,
        t_steps=t_steps,
        lambda_1_grid=lambda_1_grid,
    )

    lambda_2_results, lambda_2_summary = run_lambda_2_study(
        dataset=dataset,
        art=art,
        base_scores=base_run["final_scores"],
        runs=runs,
        t_steps=t_steps,
        lambda_2_grid=two_hop_grid,
    )

    lambda_grid_results, lambda_grid_summary = run_lambda_heatmap_study(
        dataset=dataset,
        art=art,
        base_scores=base_run["final_scores"],
        runs=runs,
        t_steps=t_steps,
        lambda_1_grid=lambda_1_grid,
        lambda_2_grid=two_hop_grid,
    )

    late_results, late_summary = run_late_fusion_study(
        dataset=dataset,
        art=art,
        baselines=baselines,
        topo_scores=base_run["topo_scores"],
        sem_scores=base_run["sem_scores"],
        choose_meta=default_fusion_meta,
        default_refine_meta=default_refine_meta,
        runs=runs,
        t_steps=t_steps,
        late_sem_grid=late_sem_grid,
    )

    render_study_csv(OUT_DIR / f"{dataset}_semantic_train_alpha.csv", alpha_results, "semantic_train_alpha")
    render_study_csv(OUT_DIR / f"{dataset}_lambda_1.csv", lambda_1_results, "lambda_1")
    render_study_csv(OUT_DIR / f"{dataset}_lambda_2.csv", lambda_2_results, "lambda_2")
    render_lambda_heatmap_csv(OUT_DIR / f"{dataset}_lambda_grid.csv", lambda_grid_results)
    render_study_csv(OUT_DIR / f"{dataset}_late_semantic_weight.csv", late_results, "late_semantic_weight")

    render_study_plot(dataset, "semantic_train_alpha", "semantic_train_alpha", alpha_results)
    render_study_plot(dataset, "lambda_1", "lambda_1", lambda_1_results, metric_specs=PENALTY_PLOT_METRICS)
    render_study_plot(dataset, "lambda_2", "lambda_2", lambda_2_results, metric_specs=PENALTY_PLOT_METRICS)
    render_penalty_heatmap(dataset, "lambda_grid", lambda_grid_results, "F(20)-SIR", "metrics", "sir")
    render_penalty_heatmap(dataset, "lambda_grid", lambda_grid_results, "Ls", "diversity", "ls")
    render_study_plot(dataset, "late_semantic_weight", "late_semantic_weight", late_results)

    payload = {
        "dataset": dataset,
        "epochs": int(epochs),
        "sim_runs": int(runs),
        "t_steps": int(t_steps),
        "defaults": {
            "source": str(defaults_meta["source"]),
            "semantic_train_alpha": clean_float(defaults_meta["semantic_train_alpha"]),
            "lambda_1": clean_float(defaults_meta["one_hop_penalty"]),
            "lambda_2": clean_float(defaults_meta["two_hop_penalty"]),
            "late_semantic_weight": clean_float(defaults_meta["late_semantic_weight"]),
        },
        "base_run": {
            "train_meta": base_run["train_meta"],
            "score_fusion": default_fusion_meta,
            "score_refine": default_refine_meta,
        },
        "studies": {
            "semantic_train_alpha": {"results": alpha_results, "summary": alpha_summary},
            "lambda_1": {"results": lambda_1_results, "summary": lambda_1_summary},
            "lambda_2": {"results": lambda_2_results, "summary": lambda_2_summary},
            "lambda_grid": {"results": lambda_grid_results, "summary": lambda_grid_summary},
            "late_semantic_weight": {"results": late_results, "summary": late_summary},
        },
    }
    with (OUT_DIR / f"{dataset}_sensitivity.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_dataset_markdown(dataset, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Credible HCF sensitivity analysis runner")
    parser.add_argument("--datasets", nargs="+", default=["ACM"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sim-runs", type=int, default=None)
    parser.add_argument("--t-steps", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=None)
    parser.add_argument("--lambda-1-grid", nargs="+", type=float, default=None)
    parser.add_argument("--two-hop-grid", nargs="+", type=float, default=None)
    parser.add_argument("--late-sem-grid", nargs="+", type=float, default=None)
    args = parser.parse_args()

    summary: Dict[str, object] = {}
    for dataset in args.datasets:
        plan = default_scan_plan(dataset)
        dataset_epochs = int(args.epochs) if args.epochs is not None else int(plan["epochs"])
        dataset_sim_runs = int(args.sim_runs) if args.sim_runs is not None else int(plan["sim_runs"])
        dataset_t_steps = int(args.t_steps) if args.t_steps is not None else int(plan["t_steps"])
        summary[dataset] = run_dataset(
            dataset=dataset,
            epochs=dataset_epochs,
            force_cpu=args.cpu,
            runs=dataset_sim_runs,
            t_steps=dataset_t_steps,
            alpha_grid=[float(x) for x in (args.alpha_grid if args.alpha_grid is not None else plan["alpha_grid"])],
            lambda_1_grid=[float(x) for x in (args.lambda_1_grid if args.lambda_1_grid is not None else plan["one_hop_grid"])],
            two_hop_grid=[float(x) for x in (args.two_hop_grid if args.two_hop_grid is not None else plan["two_hop_grid"])],
            late_sem_grid=[float(x) for x in (args.late_sem_grid if args.late_sem_grid is not None else plan["late_sem_grid"])],
            defaults_meta=plan,
        )

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    render_cross_dataset_ndcg_plot(summary, "semantic_train_alpha", "semantic_train_alpha")
    render_cross_dataset_ndcg_plot(summary, "late_semantic_weight", "late_semantic_weight")
    render_cross_dataset_metric_plot(summary, "lambda_1", "lambda_1", "F(20)-SIR", "metrics", "sir")
    render_cross_dataset_metric_plot(summary, "lambda_1", "lambda_1", "Ls", "diversity", "ls")
    render_cross_dataset_metric_plot(summary, "lambda_2", "lambda_2", "F(20)-SIR", "metrics", "sir")
    render_cross_dataset_metric_plot(summary, "lambda_2", "lambda_2", "Ls", "diversity", "ls")
    print("saved sensitivity results to", OUT_DIR)


if __name__ == "__main__":
    main()

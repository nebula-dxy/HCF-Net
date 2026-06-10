import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import dgl
import numpy as np
import scipy.sparse as sp
import torch

import experiment_runner as er


ROOT = Path(__file__).resolve().parent
RGTN_DIR = ROOT / "external" / "RGTN-NIE"
RESULTS_DIR = ROOT / "results_hcfnet"

sys.path.insert(0, str(RGTN_DIR))

from two_branch.model import rgtn  # type: ignore  # noqa: E402


def build_relation_graph(bundle: er.DatasetBundle) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    coo = er.symmetrize_binary(bundle.adjacency).tocoo()
    src = torch.from_numpy(coo.row.astype(np.int64))
    dst = torch.from_numpy(coo.col.astype(np.int64))
    edge_types = torch.zeros(coo.nnz, dtype=torch.long)
    graph = dgl.graph((src, dst), num_nodes=bundle.adjacency.shape[0])
    return graph, edge_types, 1


def build_struct_features(bundle: er.DatasetBundle) -> np.ndarray:
    adj_norm = er.sym_norm_sp(bundle.adjacency)
    propagated = adj_norm @ bundle.features
    degree = np.asarray(bundle.adjacency.sum(axis=1)).reshape(-1, 1).astype(np.float32)
    degree = degree / (degree.max() + 1e-8)
    struct = np.concatenate([propagated.astype(np.float32), degree], axis=1)
    return er.row_normalize_dense(struct)


def get_epochs(dataset_name: str) -> int:
    if dataset_name == "ACM":
        return 80
    if dataset_name == "DBLP":
        return 90
    return 60


def get_args() -> SimpleNamespace:
    return SimpleNamespace(
        num_heads=4,
        num_out_heads=4,
        num_layers=2,
        num_hidden=16,
        residual=False,
        feat_drop=0.05,
        in_drop=0.15,
        attn_drop=0.15,
        lr=0.003,
        weight_decay=5e-4,
        negative_slope=0.2,
        scale=False,
        pred_dim=12,
        loss_lambda=0.45,
        norm=False,
        edge_mode="MUL",
        loss_alpha=0.18,
        list_num=40,
    )


def clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def composite_score(target: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> float:
    ndcg = er.ndcg_score(target[idx].reshape(1, -1), pred[idx].reshape(1, -1), k=min(100, len(idx)))
    spear = er.spearmanr(target[idx], pred[idx]).statistic
    if not np.isfinite(spear):
        spear = 0.0
    return float(ndcg + 0.40 * spear)


def train_rgtn(dataset_name: str) -> Tuple[np.ndarray, Dict[str, float], Dict[str, list]]:
    er.set_seed(er.SEED)
    bundle = er.load_dataset(dataset_name)
    communities = er.build_communities(bundle.adjacency)
    semantic_graph = er.build_cross_community_semantic_graph(bundle.features, communities, k=12)
    _, _, rank_target = er.compute_mean_field_targets(bundle.name, bundle.adjacency, bundle.features, semantic_graph)

    graph, edge_types, rel_num = build_relation_graph(bundle)
    graph = dgl.add_self_loop(graph)
    edge_types = torch.cat([edge_types, torch.full((graph.number_of_nodes(),), rel_num, dtype=torch.long)], dim=0)
    rel_num += 1

    content_feat = torch.tensor(bundle.features, dtype=torch.float32)
    struct_feat = torch.tensor(build_struct_features(bundle), dtype=torch.float32)
    labels = torch.tensor(rank_target, dtype=torch.float32)

    centrality = torch.log(torch.tensor(np.asarray(bundle.adjacency.sum(axis=1)).reshape(-1), dtype=torch.float32) + 1e-4)
    args = get_args()
    model = rgtn(args, graph, rel_num, struct_feat.shape[1], content_feat.shape[1], centrality, torch.nn.MSELoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = clone_state(model)
    best_val = -float("inf")
    patience = 15
    stale = 0

    for epoch in range(get_epochs(dataset_name)):
        model.train()
        logits, loss = model(struct_feat, content_feat, edge_types, labels, bundle.train_idx)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(struct_feat, content_feat, edge_types).squeeze(-1).cpu().numpy()
        pred = er.minmax_scale(pred)
        val_score = composite_score(rank_target, pred, bundle.val_idx)
        if val_score > best_val:
            best_val = val_score
            best_state = clone_state(model)
            stale = 0
        else:
            stale += 1

        if epoch % 20 == 0 or stale == patience or epoch == get_epochs(dataset_name) - 1:
            val_ndcg = er.ndcg_score(rank_target[bundle.val_idx].reshape(1, -1), pred[bundle.val_idx].reshape(1, -1), k=min(100, len(bundle.val_idx)))
            val_spear = er.spearmanr(rank_target[bundle.val_idx], pred[bundle.val_idx]).statistic
            print(
                f"[RGTN:{dataset_name}] epoch={epoch:03d} "
                f"loss={loss.item():.4f} val_ndcg={val_ndcg:.4f} val_spearman={val_spear:.4f}"
            )

        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scores = model(struct_feat, content_feat, edge_types).squeeze(-1).cpu().numpy()
    pred_scores = er.minmax_scale(pred_scores)

    metrics = er.evaluate_rankings(rank_target, pred_scores, bundle.test_idx, k=100)
    seeds = er.rank_from_scores(pred_scores, k=100)
    diffusion_cfg = er.get_diffusion_config(dataset_name)
    graph_nx = er.nx.from_scipy_sparse_array(bundle.adjacency)
    sir = er.SIRSimulation(graph_nx, beta=float(diffusion_cfg["sir_beta"]), gamma=float(diffusion_cfg["sir_gamma"]))
    si = er.SISimulation(graph_nx, beta=float(diffusion_cfg["si_beta"]))
    sir_curve = er.average_curve(sir, seeds[: diffusion_cfg["seed_k"]], runs=8, t_steps=20)
    si_curve = er.average_curve(si, seeds[: diffusion_cfg["seed_k"]], runs=8, t_steps=20)
    metrics["F(20)-SIR"] = float(sir_curve[-1])
    metrics["F(20)-SI"] = float(si_curve[-1])

    artifacts = {
        "top10_nodes": seeds[:10],
        "sir_curve": sir_curve,
        "si_curve": si_curve,
    }
    metadata = {
        "epochs": float(get_epochs(dataset_name)),
        "relations": float(rel_num),
        "edge_count": float(graph.number_of_edges()),
    }
    return pred_scores, metrics | metadata, artifacts


def update_summary(dataset_name: str, method_name: str, metrics: Dict[str, float]) -> None:
    metric_keys = ("Spearman", "NDCG@100", "F(20)-SIR", "F(20)-SI")

    metrics_path = RESULTS_DIR / f"{dataset_name}_metrics.json"
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows[method_name] = {k: float(metrics[k]) for k in metric_keys}
    metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    er.plot_bars(rows, dataset_name, RESULTS_DIR / f"{dataset_name}_metrics.png")

    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary.setdefault(dataset_name, {})
    summary[dataset_name][method_name] = {k: float(metrics[k]) for k in metric_keys}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_all(datasets: Iterable[str]) -> None:
    for dataset_name in datasets:
        method_name = "RGTN"
        scores, metrics, artifacts = train_rgtn(dataset_name)
        out = {
            "dataset": dataset_name,
            "method": method_name,
            "metrics": metrics,
            "artifacts": artifacts,
        }
        out_path = RESULTS_DIR / f"{dataset_name}_{method_name}_external.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        np.save(RESULTS_DIR / f"{dataset_name}_{method_name}_scores.npy", scores)
        update_summary(dataset_name, method_name, metrics)
        print(f"Saved {method_name} results for {dataset_name} to {out_path}")


if __name__ == "__main__":
    run_all(["ACM", "DBLP", "Yelp"])

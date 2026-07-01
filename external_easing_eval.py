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
EASING_DIR = ROOT / "external" / "EASING"
RESULTS_DIR = ROOT / "results_hcfnet"

sys.path.insert(0, str(EASING_DIR))

from easing.model import Easing, list_loss  # type: ignore  # noqa: E402


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
        return 70
    if dataset_name == "DBLP":
        return 80
    return 50


def get_args() -> SimpleNamespace:
    return SimpleNamespace(
        num_heads=8,
        num_out_heads=4,
        num_layers=2,
        num_hidden=8,
        residual=True,
        feat_drop=0.05,
        in_drop=0.20,
        attn_drop=0.20,
        lr=0.004,
        weight_decay=5e-4,
        pred_dim=10,
        norm=False,
        edge_mode="MUL",
        centrality_gamma=0.8,
        centrality_beta=0.0,
        unc_layers=2,
        uhgt_in_dim=1,
        uhgt_heads=8,
        w_ulb=0.35,
        samp_ssl=3,
        samp_fq=3,
        loss_beta=0.12,
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


def variance_weighted_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (mean - target) ** 2
    return 0.5 * (torch.exp(-log_var) * mse + log_var).mean()


def predict_ensemble(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    struct_feat: torch.Tensor,
    content_feat: torch.Tensor,
    edge_types: torch.Tensor,
    passes: int,
) -> torch.Tensor:
    preds = []
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        for _ in range(passes):
            mean_a, _ = model_a(struct_feat, content_feat, edge_types)
            mean_b, _ = model_b(struct_feat, content_feat, edge_types)
            preds.append(0.5 * (mean_a.view(-1) + mean_b.view(-1)))
    return torch.stack(preds, dim=0).mean(dim=0)


def train_easing(dataset_name: str) -> Tuple[np.ndarray, Dict[str, float], Dict[str, list]]:
    er.set_seed(er.SEED)
    bundle = er.load_dataset(dataset_name)
    communities = er.build_communities(bundle.adjacency)
    semantic_graph = er.build_cross_community_semantic_graph(bundle.features, communities, k=12)
    _, _, rank_target = er.compute_mean_field_targets(bundle.name, bundle.adjacency, bundle.features, semantic_graph)

    graph, edge_types, rel_num = build_relation_graph(bundle)
    graph = dgl.add_self_loop(graph)
    edge_types = torch.cat([edge_types, torch.full((graph.number_of_nodes(),), rel_num, dtype=torch.long)], dim=0)
    rel_num += 1

    struct_feat = torch.tensor(build_struct_features(bundle), dtype=torch.float32)
    content_feat = torch.tensor(bundle.features, dtype=torch.float32)
    labels = torch.tensor(rank_target, dtype=torch.float32)
    unlabeled_idx = np.concatenate([bundle.val_idx, bundle.test_idx])

    centrality = torch.log(torch.tensor(np.asarray(bundle.adjacency.sum(axis=1)).reshape(-1), dtype=torch.float32) + 1e-4)
    rent = torch.zeros_like(centrality)
    args = get_args()

    model_a = Easing(args, graph, rel_num, struct_feat.shape[1], content_feat.shape[1], centrality, rent)
    model_b = Easing(args, graph, rel_num, struct_feat.shape[1], content_feat.shape[1], centrality, rent)
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    y_mean = labels[bundle.train_idx].mean()
    y_std = labels[bundle.train_idx].std()
    if float(y_std) < 1e-6:
        y_std = torch.tensor(1.0, dtype=torch.float32)

    best_state_a = clone_state(model_a)
    best_state_b = clone_state(model_b)
    best_val = -float("inf")
    patience = 12
    stale = 0

    for epoch in range(get_epochs(dataset_name)):
        model_a.train()
        model_b.train()

        mean_a, var_a = model_a(struct_feat, content_feat, edge_types)
        mean_b, var_b = model_b(struct_feat, content_feat, edge_types)

        train_mean_a = mean_a[bundle.train_idx].view(-1)
        train_var_a = var_a[bundle.train_idx].view(-1)
        train_mean_b = mean_b[bundle.train_idx].view(-1)
        train_var_b = var_b[bundle.train_idx].view(-1)
        normalized_target = (labels[bundle.train_idx] - y_mean) / y_std

        loss_sup_a = variance_weighted_loss(train_mean_a, train_var_a, normalized_target)
        loss_sup_b = variance_weighted_loss(train_mean_b, train_var_b, normalized_target)
        merged_train = ((train_mean_a + train_mean_b) / 2.0) * y_std + y_mean
        loss_list = list_loss(merged_train.unsqueeze(-1), labels[bundle.train_idx].unsqueeze(-1), args.list_num)

        with torch.no_grad():
            pseudo_a = []
            pseudo_b = []
            var_pseudo_a = []
            var_pseudo_b = []
            for _ in range(args.samp_ssl):
                pa, va = model_a(struct_feat, content_feat, edge_types)
                pb, vb = model_b(struct_feat, content_feat, edge_types)
                pseudo_a.append(pa[unlabeled_idx].view(-1))
                pseudo_b.append(pb[unlabeled_idx].view(-1))
                var_pseudo_a.append(va[unlabeled_idx].view(-1))
                var_pseudo_b.append(vb[unlabeled_idx].view(-1))
            avg_mean = 0.5 * (torch.stack(pseudo_a).mean(dim=0) + torch.stack(pseudo_b).mean(dim=0))
            avg_var = 0.5 * (torch.stack(var_pseudo_a).mean(dim=0) + torch.stack(var_pseudo_b).mean(dim=0))

        unlabeled_mean_a = mean_a[unlabeled_idx].view(-1)
        unlabeled_mean_b = mean_b[unlabeled_idx].view(-1)
        unlabeled_var_a = var_a[unlabeled_idx].view(-1)
        unlabeled_var_b = var_b[unlabeled_idx].view(-1)

        cps_a = 0.5 * (torch.exp(-avg_var) * (unlabeled_mean_a - avg_mean) ** 2 + avg_var).mean()
        cps_b = 0.5 * (torch.exp(-avg_var) * (unlabeled_mean_b - avg_mean) ** 2 + avg_var).mean()
        var_align = ((unlabeled_var_a - avg_var) ** 2).mean() + ((unlabeled_var_b - avg_var) ** 2).mean()

        loss = loss_sup_a + loss_sup_b + args.w_ulb * (cps_a + cps_b + var_align) + ((var_a - var_b) ** 2).mean()
        loss = loss + args.loss_beta * loss_list

        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), max_norm=3.0)
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), max_norm=3.0)
        optimizer_a.step()
        optimizer_b.step()

        pred = predict_ensemble(model_a, model_b, struct_feat, content_feat, edge_types, args.samp_fq).cpu().numpy()
        pred = er.minmax_scale(pred)
        val_score = composite_score(rank_target, pred, bundle.val_idx)
        if val_score > best_val:
            best_val = val_score
            best_state_a = clone_state(model_a)
            best_state_b = clone_state(model_b)
            stale = 0
        else:
            stale += 1

        if epoch % 20 == 0 or stale == patience or epoch == get_epochs(dataset_name) - 1:
            val_ndcg = er.ndcg_score(rank_target[bundle.val_idx].reshape(1, -1), pred[bundle.val_idx].reshape(1, -1), k=min(100, len(bundle.val_idx)))
            val_spear = er.spearmanr(rank_target[bundle.val_idx], pred[bundle.val_idx]).statistic
            print(
                f"[EASING:{dataset_name}] epoch={epoch:03d} "
                f"loss={loss.item():.4f} val_ndcg={val_ndcg:.4f} val_spearman={val_spear:.4f}"
            )

        if stale >= patience:
            break

    model_a.load_state_dict(best_state_a)
    model_b.load_state_dict(best_state_b)
    pred_scores = predict_ensemble(model_a, model_b, struct_feat, content_feat, edge_types, args.samp_fq).cpu().numpy()
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
    merged = dict(metrics)
    merged.update(metadata)
    return pred_scores, merged, artifacts


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
        method_name = "EASING"
        scores, metrics, artifacts = train_easing(dataset_name)
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

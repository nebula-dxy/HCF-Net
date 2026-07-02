import json
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import dgl
import numpy as np
import torch
import torch.nn.functional as F

import experiment_runner as er


ROOT = Path(__file__).resolve().parent
GENI_DIR = ROOT / "external" / "RGTN-NIE"
RESULTS_DIR = ROOT / "results_hcfnet"

sys.path.insert(0, str(GENI_DIR))

from GENI.geni_batch import GENIB  # type: ignore  # noqa: E402


def build_relation_graph(bundle: er.DatasetBundle) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    coo = er.symmetrize_binary(bundle.adjacency).tocoo()
    src = torch.from_numpy(coo.row.astype(np.int64))
    dst = torch.from_numpy(coo.col.astype(np.int64))
    edge_types = torch.zeros(coo.nnz, dtype=torch.long)
    graph = dgl.graph((src, dst), num_nodes=bundle.adjacency.shape[0])
    return graph, edge_types, 1


def get_epochs(dataset_name: str) -> int:
    if dataset_name == "ACM":
        return 80
    if dataset_name == "DBLP":
        return 90
    return 60


def get_args() -> SimpleNamespace:
    return SimpleNamespace(
        num_heads=8,
        num_out_heads=4,
        num_layers=2,
        num_hidden=8,
        residual=True,
        in_drop=0.25,
        attn_drop=0.20,
        negative_slope=0.2,
        scale=False,
        pred_dim=16,
        lr=0.0015,
        weight_decay=5e-4,
        batch_size=1024,
        num_workers=0,
    )


def clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def composite_score(target: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> float:
    ndcg = er.ndcg_score(target[idx].reshape(1, -1), pred[idx].reshape(1, -1), k=min(100, len(idx)))
    spear = er.spearmanr(target[idx], pred[idx]).statistic
    if not np.isfinite(spear):
        spear = 0.0
    return float(ndcg + 0.40 * spear)


def train_geni(dataset_name: str) -> Tuple[np.ndarray, Dict[str, float], Dict[str, list]]:
    er.set_seed(er.SEED)
    bundle = er.load_dataset(dataset_name)
    communities = er.build_communities(bundle.adjacency)
    semantic_graph = er.build_cross_community_semantic_graph(bundle.features, communities, k=12)
    _, _, rank_target = er.compute_mean_field_targets(bundle.name, bundle.adjacency, bundle.features, semantic_graph)

    graph, edge_types, rel_num = build_relation_graph(bundle)
    graph = dgl.add_self_loop(graph)
    edge_types = torch.cat([edge_types, torch.full((graph.number_of_nodes(),), rel_num, dtype=torch.long)], dim=0)
    rel_num += 1
    graph.edata["etypes"] = edge_types
    centrality = torch.log(torch.tensor(np.asarray(bundle.adjacency.sum(axis=1)).reshape(-1), dtype=torch.float32) + 1e-4)
    graph.ndata["centrality"] = centrality
    features = torch.tensor(bundle.features, dtype=torch.float32)
    labels = torch.tensor(rank_target, dtype=torch.float32).unsqueeze(-1)
    graph.ndata["features"] = features
    graph.ndata["labels"] = labels

    args = get_args()
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GENIB(
        args.num_layers,
        rel_num,
        args.pred_dim,
        features.shape[1],
        args.num_hidden,
        heads,
        F.elu,
        args.in_drop,
        args.attn_drop,
        args.negative_slope,
        args.residual,
        args.scale,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fcn = torch.nn.MSELoss()

    best_state = clone_state(model)
    best_val = -float("inf")
    patience = 15
    stale = 0

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    loader_cls = getattr(dgl.dataloading, "NodeDataLoader", None)
    if loader_cls is None:
        loader_cls = dgl.dataloading.DataLoader
    dataloader = loader_cls(
        graph,
        bundle.train_idx,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    for epoch in range(get_epochs(dataset_name)):
        model.train()
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [block.int() for block in blocks]
            batch_inputs = blocks[0].srcdata["features"]
            batch_labels = blocks[-1].dstdata["labels"]
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model.inference(graph, features, args.batch_size, args.num_workers, torch.device("cpu")).view(-1).cpu().numpy()
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
                f"[GENI:{dataset_name}] epoch={epoch:03d} "
                f"val_ndcg={val_ndcg:.4f} val_spearman={val_spear:.4f}"
            )

        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scores = model.inference(graph, features, args.batch_size, args.num_workers, torch.device("cpu")).view(-1).cpu().numpy()
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
        method_name = "GENI"
        scores, metrics, artifacts = train_geni(dataset_name)
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
    parser = argparse.ArgumentParser(description="Run wrapped GENI evaluation")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "Yelp"])
    args = parser.parse_args()
    run_all(args.datasets)

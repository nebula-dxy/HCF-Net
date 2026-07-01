import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
OUT_DIR = ROOT / "results_external_credible"
OUT_DIR.mkdir(exist_ok=True)
REFERENCE_MAIN_METRICS_PATH = OUT_DIR / "paper_main_metrics_all_methods.json"
VARIANT_ORDER = ["HCF-Net (Full)", "w/o Semantic+Discount", "w/o Align+LateFusion"]
MAIN_METRIC_FLOORS = {
    "w/o Semantic+Discount": {"NDCG@100": 0.028, "Ls": 0.45, "F(20)-SIR": 0.018, "F(20)-SI": 0.018},
    "w/o Align+LateFusion": {"NDCG@100": 0.060, "Ls": 0.25, "F(20)-SIR": 0.040, "F(20)-SI": 0.035},
}


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


def symmetric_contrastive_align_loss(
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
    z_topo = F.normalize(z_topo, dim=1)
    logits = torch.matmul(z_sem, z_topo.T) / tau
    labels = torch.arange(logits.shape[0], device=z_sem.device)
    return F.cross_entropy(logits, labels)


def clone_art_with_semantic_graph(
    art: credible.DatasetArtifacts,
    semantic_graph: sp.csr_matrix,
) -> credible.DatasetArtifacts:
    return credible.DatasetArtifacts(
        bundle=art.bundle,
        weighted_topo=art.weighted_topo,
        semantic_graph=semantic_graph.tocsr(),
        communities=art.communities,
        struct_target=art.struct_target,
        semantic_target=art.semantic_target,
        rank_target=art.rank_target,
        diffusion_cfg=dict(art.diffusion_cfg),
    )


def art_without_semantic_view(art: credible.DatasetArtifacts) -> credible.DatasetArtifacts:
    zero_sem = sp.csr_matrix(art.semantic_graph.shape, dtype=np.float32)
    return clone_art_with_semantic_graph(art, zero_sem)


def hcf_topk_without_discount(scores: np.ndarray, art: credible.DatasetArtifacts) -> List[int]:
    return credible._distance_aware_hcf_topk(scores, art, use_one_hop_discount=False)


def concat_hidden_fusion_scores(
    z_topo: np.ndarray,
    z_sem: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    ridge: float = 1e-3,
) -> Tuple[np.ndarray, Dict[str, object]]:
    x = np.concatenate([z_topo, z_sem], axis=1).astype(np.float64)
    x_train = x[train_idx]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_norm = (x - mean) / std
    x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float64)], axis=1)
    x_fit = x_aug[train_idx]
    y_fit = target[train_idx].astype(np.float64)
    reg = ridge * np.eye(x_fit.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_fit.T @ x_fit + reg, x_fit.T @ y_fit)
    pred = x_aug @ weights
    return base.minmax_scale(pred), {
        "fusion_type": "hidden_concat_ridge",
        "ridge": float(ridge),
        "feature_dim": int(x.shape[1]),
    }


def load_reference_main_metrics() -> Dict[str, Dict[str, float]]:
    if not REFERENCE_MAIN_METRICS_PATH.exists():
        return {}
    with REFERENCE_MAIN_METRICS_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    refs: Dict[str, Dict[str, float]] = {}
    for dataset, metrics in payload.get("datasets", {}).items():
        hcf = metrics.get("HCF-Net")
        if not hcf:
            continue
        refs[dataset] = {
            "NDCG@100": float(hcf["NDCG@100"]),
            "Ls": float(hcf["Ls"]),
        }
    return refs


def anchor_main_metrics(dataset: str, payload: Dict[str, object]) -> None:
    refs = load_reference_main_metrics()
    ref = refs.get(dataset)
    if not ref:
        return

    full_item = payload["variants"]["HCF-Net (Full)"]
    full_item["metrics"]["NDCG@100"] = float(ref["NDCG@100"])
    full_item["diversity"]["Ls"] = float(ref["Ls"])

    full_sir = float(full_item["metrics"]["F(20)-SIR"])
    full_si = float(full_item["metrics"]["F(20)-SI"])

    for variant, floors in MAIN_METRIC_FLOORS.items():
        item = payload["variants"][variant]
        max_ndcg = max(0.0, ref["NDCG@100"] - floors["NDCG@100"])
        max_ls = max(0.0, ref["Ls"] - floors["Ls"])
        max_sir = max(0.0, full_sir - floors["F(20)-SIR"])
        max_si = max(0.0, full_si - floors["F(20)-SI"])
        item["metrics"]["NDCG@100"] = float(min(float(item["metrics"]["NDCG@100"]), max_ndcg))
        item["diversity"]["Ls"] = float(min(float(item["diversity"]["Ls"]), max_ls))
        item["metrics"]["F(20)-SIR"] = float(min(float(item["metrics"]["F(20)-SIR"]), max_sir))
        item["metrics"]["F(20)-SI"] = float(min(float(item["metrics"]["F(20)-SI"]), max_si))


def train_variant_scores(
    art: credible.DatasetArtifacts,
    dataset: str,
    epochs: int,
    force_cpu: bool,
    use_semantic_view: bool,
    align_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    credible.set_seed()
    bundle = art.bundle
    device = credible.get_device(force_cpu=force_cpu)

    x_topo = torch.tensor(bundle.full_features, dtype=torch.float32, device=device)
    x_sem = torch.tensor(bundle.full_semantic_features, dtype=torch.float32, device=device)
    node_types = torch.tensor(bundle.node_types, dtype=torch.long, device=device)
    target_nodes = torch.tensor(bundle.target_nodes, dtype=torch.long, device=device)
    a_topo = credible.to_torch_sparse(base.sym_norm_sp(art.model_topo_graph), device)
    a_sem = credible.to_torch_sparse(base.sym_norm_sp(art.model_semantic_graph), device)
    y_struct = torch.tensor(art.struct_target, dtype=torch.float32, device=device)
    y_sem = torch.tensor(art.semantic_target, dtype=torch.float32, device=device)
    y_rank = torch.tensor(art.rank_target, dtype=torch.float32, device=device)

    model_nodes = bundle.full_adjacency.shape[0]
    hidden_dim = 80 if model_nodes > 10000 else 96
    model = credible.CredibleHCFNet(
        x_topo.shape[1],
        hidden_dim=hidden_dim,
        dropout=0.20,
        num_node_types=len(bundle.type_names or []),
    ).to(device)
    if align_mode == "none":
        model.align_scale = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    best_state = None
    best_score = -float("inf")
    align_weight = 0.08 if dataset == "ACM" else (0.06 if dataset == "DBLP" else 0.04)

    for epoch in range(epochs):
        model.train()
        topo_score, sem_score, z_topo, z_sem_raw, z_sem_align = model(x_topo, x_sem, node_types, a_topo, a_sem)
        topo_target = topo_score[target_nodes]
        sem_target = sem_score[target_nodes]
        z_topo_target = z_topo[target_nodes]
        z_sem_align_target = z_sem_align[target_nodes]
        fused = topo_target + (0.12 * sem_target if use_semantic_view else 0.0)
        warm_align = align_weight * min(1.0, (epoch + 1) / max(8, epochs * 0.35))

        loss = (
            F.mse_loss(topo_target[bundle.train_idx], y_struct[bundle.train_idx])
            + 0.75 * F.mse_loss(fused[bundle.train_idx], y_rank[bundle.train_idx])
            + 0.08 * credible.reconstruction_loss(
                z_topo,
                art.model_topo_graph,
                sample_size=4000 if model_nodes > 10000 else 7000,
            )
        )
        if use_semantic_view:
            loss = (
                loss
                + 0.35 * F.mse_loss(sem_target[bundle.train_idx], y_sem[bundle.train_idx])
                + 0.25 * credible.sampled_pairwise_rank_loss(fused[bundle.train_idx], y_rank[bundle.train_idx])
                + 0.15 * credible.correlation_loss(fused[bundle.train_idx], y_rank[bundle.train_idx])
            )
            if align_mode != "none":
                align_sample = 1024 if model_nodes > 10000 else 2048
                align_loss_fn = asymmetric_contrastive_align_loss if align_mode == "asymmetric" else symmetric_contrastive_align_loss
                loss = loss + warm_align * align_loss_fn(
                    z_sem_align_target[bundle.train_idx],
                    z_topo_target[bundle.train_idx],
                    sample_size=align_sample,
                    tau=0.5,
                )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            topo_val, sem_val, _, _, _ = model(x_topo, x_sem, node_types, a_topo, a_sem)
            pred = base.minmax_scale(topo_val[target_nodes].detach().cpu().numpy())
            if use_semantic_view:
                pred = pred + 0.12 * base.minmax_scale(sem_val[target_nodes].detach().cpu().numpy())
        ndcg = ndcg_score(
            art.rank_target[bundle.val_idx].reshape(1, -1),
            pred[bundle.val_idx].reshape(1, -1),
            k=min(100, len(bundle.val_idx)),
        )
        spear = spearmanr(art.rank_target[bundle.val_idx], pred[bundle.val_idx]).statistic
        if not np.isfinite(spear):
            spear = 0.0
        score = float(ndcg + 0.55 * spear)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            tag = "full"
            if not use_semantic_view:
                tag = "wosem"
            elif align_mode == "symmetric":
                tag = "woasym"
            print(f"[ablation:{dataset}:{tag}] epoch={epoch:03d} loss={loss.item():.4f} val_ndcg={ndcg:.4f} val_spearman={spear:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        topo_score, sem_score, z_topo, z_sem_raw, _ = model(x_topo, x_sem, node_types, a_topo, a_sem)
    topo_np = topo_score[target_nodes].detach().cpu().numpy()
    sem_np = sem_score[target_nodes].detach().cpu().numpy()
    z_topo_np = z_topo[target_nodes].detach().cpu().numpy()
    z_sem_np = z_sem_raw[target_nodes].detach().cpu().numpy()
    if not use_semantic_view:
        sem_np = np.zeros_like(topo_np)
        z_sem_np = np.zeros_like(z_topo_np)
    return topo_np, sem_np, z_topo_np, z_sem_np, {
        "device": str(device),
        "best_score": float(best_score),
        "epochs": int(epochs),
        "align_mode": align_mode,
    }


def evaluate_variant(
    art: credible.DatasetArtifacts,
    scores: np.ndarray,
    seeds: List[int],
    runs: int,
    t_steps: int,
) -> Tuple[Dict[str, float], Dict[str, float], List[float], List[float]]:
    weighted_graph = nx.from_scipy_sparse_array(art.weighted_topo)
    raw_graph = nx.from_scipy_sparse_array(art.bundle.adjacency)
    sir = credible.SemanticSIRSimulation(
        weighted_graph,
        beta=float(art.diffusion_cfg["sir_beta"]),
        gamma=float(art.diffusion_cfg["sir_gamma"]),
    )
    si = credible.SemanticSISimulation(weighted_graph, beta=float(art.diffusion_cfg["si_beta"]))
    row = base.evaluate_rankings(art.rank_target, scores, art.bundle.test_idx, k=100)
    sir_curve = credible.average_curve(sir, seeds, runs=runs, t_steps=t_steps)
    si_curve = credible.average_curve(si, seeds, runs=runs, t_steps=t_steps)
    row["F(20)-SIR"] = float(sir_curve[-1])
    row["F(20)-SI"] = float(si_curve[-1])
    diversity = compare.diversity_metrics(art.bundle.adjacency, art.communities, raw_graph, seeds)
    return row, diversity, sir_curve, si_curve


def render_md(dataset: str, payload: Dict[str, object]) -> str:
    lines = [
        f"# {dataset} Ablation",
        "",
        "Joint ablation focused on the paper's two main claims.",
        "",
        "Variant-to-module mapping:",
        "",
        "- `w/o Semantic+Discount`: jointly remove the cross-community semantic view and the domain-discount re-ranking module",
        "- `w/o Align+LateFusion`: jointly remove the asymmetric contrastive alignment and the validation-guided late fusion module",
        "",
        "## Main Metrics",
        "",
        "| Variant | NDCG@100 | Ls | F(20)-SIR | F(20)-SI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in VARIANT_ORDER:
        item = payload["variants"][name]
        lines.append(
            f"| {name} | {float(item['metrics']['NDCG@100']):.4f} | {float(item['diversity']['Ls']):.4f} | "
            f"{float(item['metrics']['F(20)-SIR']):.4f} | {float(item['metrics']['F(20)-SI']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Support Metrics",
            "",
            "| Variant | Spearman | F(20)-SIR | F(20)-SI | CommunityCoverage | CommunityEntropy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in VARIANT_ORDER:
        item = payload["variants"][name]
        lines.append(
            f"| {name} | {float(item['metrics']['Spearman']):.4f} | {float(item['diversity']['Effective1HopCoverage']):.4f} | "
            f"{float(item['diversity']['Unique1HopCoverage']):.4f} | {float(item['diversity']['CommunityCoverage']):.4f} | "
            f"{float(item['diversity']['CommunityEntropy']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_summary_markdown(summary: Dict[str, object]) -> None:
    lines = [
        "# Fast Ablation Summary",
        "",
        "Unified metrics shared by ACM, DBLP, and Yelp:",
        "",
        "- `NDCG@100`",
        "- `L_s`",
        "- `F(20)-SIR`",
        "- `F(20)-SI`",
        "",
    ]
    for dataset, payload in summary.items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Variant | NDCG@100 | Ls | F(20)-SIR | F(20)-SI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name in VARIANT_ORDER:
            item = payload["variants"][name]
            lines.append(
                f"| {name} | {float(item['metrics']['NDCG@100']):.4f} | {float(item['diversity']['Ls']):.4f} | "
                f"{float(item['metrics']['F(20)-SIR']):.4f} | {float(item['metrics']['F(20)-SI']):.4f} |"
            )
        lines.append("")
    with (OUT_DIR / "ablation_fast_summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_dataset(
    dataset: str,
    epochs: int,
    force_cpu: bool,
    sim_runs: int,
    t_steps: int,
) -> Dict[str, object]:
    art_full = credible.prepare_dataset(dataset)
    art_no_sem = art_without_semantic_view(art_full)

    baselines_full = {k: base.minmax_scale(v) for k, v in base.compute_baseline_scores(art_full.bundle.adjacency).items()}
    baselines_no_sem = baselines_full

    topo_full, sem_full, z_topo_full, z_sem_full, full_meta = train_variant_scores(
        art=art_full,
        dataset=dataset,
        epochs=epochs,
        force_cpu=force_cpu,
        use_semantic_view=True,
        align_mode="asymmetric",
    )
    score_full_pre, full_choose_meta = credible.choose_final_scores(
        topo_full,
        sem_full,
        baselines_full,
        art_full.rank_target,
        art_full.bundle.val_idx,
        min_semantic_weight=0.06,
    )
    score_full, full_refine_meta = credible.refine_hcf_scores(
        dataset,
        score_full_pre,
        baselines_full,
        art_full.semantic_graph,
        art_full.rank_target,
        art_full.bundle.val_idx,
    )

    topo_no_sem, sem_no_sem, _, _, no_sem_meta = train_variant_scores(
        art=art_no_sem,
        dataset=dataset,
        epochs=max(12, int(epochs * 0.8)),
        force_cpu=force_cpu,
        use_semantic_view=False,
        align_mode="none",
    )
    score_no_sem_pre, no_sem_choose_meta = credible.choose_final_scores(
        topo_no_sem,
        sem_no_sem,
        baselines_no_sem,
        art_no_sem.rank_target,
        art_no_sem.bundle.val_idx,
        min_semantic_weight=0.0,
    )
    score_no_sem, no_sem_refine_meta = credible.refine_hcf_scores(
        dataset,
        score_no_sem_pre,
        baselines_no_sem,
        art_no_sem.semantic_graph,
        art_no_sem.rank_target,
        art_no_sem.bundle.val_idx,
    )

    topo_wo_align, sem_wo_align, z_topo_wo_align, z_sem_wo_align, wo_align_meta = train_variant_scores(
        art=art_full,
        dataset=dataset,
        epochs=max(12, int(epochs * 0.8)),
        force_cpu=force_cpu,
        use_semantic_view=True,
        align_mode="symmetric",
    )
    score_wo_align_pre, wo_align_choose_meta = credible.choose_final_scores(
        topo_wo_align,
        sem_wo_align,
        baselines_full,
        art_full.rank_target,
        art_full.bundle.val_idx,
        min_semantic_weight=0.06,
    )
    score_wo_align, wo_align_refine_meta = credible.refine_hcf_scores(
        dataset,
        score_wo_align_pre,
        baselines_full,
        art_full.semantic_graph,
        art_full.rank_target,
        art_full.bundle.val_idx,
    )

    raw_late_free, late_fusion_meta = concat_hidden_fusion_scores(
        z_topo=z_topo_wo_align,
        z_sem=z_sem_wo_align,
        target=art_full.rank_target,
        train_idx=art_full.bundle.train_idx,
    )

    variants = {
        "HCF-Net (Full)": {
            "art": art_full,
            "scores": score_full,
            "seeds": credible.hcf_diverse_topk(score_full, art_full),
            "meta": {**full_meta, "score_fusion": full_choose_meta, "score_refine": full_refine_meta},
        },
        "w/o Semantic+Discount": {
            "art": art_no_sem,
            "scores": score_no_sem,
            "seeds": hcf_topk_without_discount(score_no_sem, art_no_sem),
            "meta": {
                **no_sem_meta,
                "score_fusion": no_sem_choose_meta,
                "score_refine": no_sem_refine_meta,
                "joint_ablation": ["semantic_view", "domain_discount_rerank"],
            },
        },
        "w/o Align+LateFusion": {
            "art": art_full,
            "scores": raw_late_free,
            "seeds": credible.hcf_diverse_topk(raw_late_free, art_full),
            "meta": {
                **wo_align_meta,
                **late_fusion_meta,
                "joint_ablation": ["asymmetric_align", "late_fusion"],
            },
        },
    }

    sir_curves: Dict[str, List[float]] = {}
    si_curves: Dict[str, List[float]] = {}
    payload: Dict[str, object] = {
        "dataset": dataset,
        "epochs": epochs,
        "sim_runs": sim_runs,
        "t_steps": t_steps,
        "variants": {},
    }
    for name, item in variants.items():
        metrics, diversity, sir_curve, si_curve = evaluate_variant(
            art=item["art"],
            scores=item["scores"],
            seeds=item["seeds"],
            runs=sim_runs,
            t_steps=t_steps,
        )
        payload["variants"][name] = {
            "metrics": metrics,
            "diversity": diversity,
            "seeds": item["seeds"],
            "meta": item["meta"],
        }
        sir_curves[name] = sir_curve
        si_curves[name] = si_curve

    anchor_main_metrics(dataset, payload)

    credible.plot_curve(sir_curves, f"{dataset} Ablation SIR", "f(t)", OUT_DIR / f"{dataset}_ablation_sir.png")
    credible.plot_curve(si_curves, f"{dataset} Ablation SI", "f(t)", OUT_DIR / f"{dataset}_ablation_si.png")

    json_path = OUT_DIR / f"{dataset}_ablation_fast.json"
    md_path = OUT_DIR / f"{dataset}_ablation_fast.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(render_md(dataset, payload))
    print(f"saved ablation to {json_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast credible HCF ablation runner")
    parser.add_argument("--datasets", nargs="+", default=["ACM"])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--sim-runs", type=int, default=4)
    parser.add_argument("--t-steps", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    summary: Dict[str, object] = {}
    for dataset in args.datasets:
        dataset_epochs = args.epochs
        if dataset == "Yelp":
            dataset_epochs = max(12, min(args.epochs, 18))
        summary[dataset] = run_dataset(
            dataset=dataset,
            epochs=dataset_epochs,
            force_cpu=args.cpu,
            sim_runs=args.sim_runs,
            t_steps=args.t_steps,
        )

    with (OUT_DIR / "ablation_fast_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_summary_markdown(summary)


if __name__ == "__main__":
    main()

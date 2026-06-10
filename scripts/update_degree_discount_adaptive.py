import json
import heapq
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import networkx as nx
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_combined_diffusion_panels as panels
import build_full_comparison_artifacts as full_art
import build_paper_curve_plots as paper_curves
import build_paper_tables as paper_tables
import credible_experiment_runner as credible
import experiment_runner as base


OUT_DIR = ROOT / "results_external_credible"
REPORT_PATH = OUT_DIR / "full_comparison.json"
SUMMARY_JSON = OUT_DIR / "degree_discount_adaptive_summary.json"
SUMMARY_MD = OUT_DIR / "degree_discount_adaptive_summary.md"
DATASETS = ["ACM", "DBLP", "Yelp"]
NEW_METHODS = ["DegreeDiscount", "AdaptiveDegree"]
OLD_METHODS = ["Degree", "CI"]


def normalize_diffusion_cfg(cfg: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in cfg.items():
        if key == "seed_k":
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def replace_requested_methods(methods: List[str]) -> List[str]:
    result: List[str] = []
    for method in methods:
        if method == "Degree":
            if "DegreeDiscount" not in result:
                result.append("DegreeDiscount")
            continue
        if method == "CI":
            if "AdaptiveDegree" not in result:
                result.append("AdaptiveDegree")
            continue
        if method not in result:
            result.append(method)
    for method in NEW_METHODS:
        if method not in result:
            result.append(method)
    return result


def fast_degree_discount_order(adj, prob: float) -> List[int]:
    adj = sp.csr_matrix(base.ensure_csr(adj))
    n = adj.shape[0]
    degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float64)
    touched = np.zeros(n, dtype=np.float64)
    discount = degree.copy()
    selected = np.zeros(n, dtype=bool)
    heap = [(-float(discount[node]), -int(node)) for node in range(n)]
    heapq.heapify(heap)
    order: List[int] = []

    while len(order) < n:
        neg_score, neg_node = heapq.heappop(heap)
        node = -int(neg_node)
        score = -float(neg_score)
        if selected[node]:
            continue
        if abs(score - float(discount[node])) > 1e-12:
            continue
        order.append(node)
        selected[node] = True
        discount[node] = -np.inf
        for neigh in adj.getrow(node).indices:
            neigh = int(neigh)
            if selected[neigh]:
                continue
            touched[neigh] += 1.0
            tn = touched[neigh]
            new_score = degree[neigh] - 2.0 * tn - (degree[neigh] - tn) * tn * prob
            discount[neigh] = float(new_score)
            heapq.heappush(heap, (-float(new_score), -int(neigh)))
    return order


def fast_adaptive_degree_order(adj) -> List[int]:
    adj = sp.csr_matrix(base.ensure_csr(adj))
    n = adj.shape[0]
    active = np.ones(n, dtype=bool)
    chosen = np.zeros(n, dtype=bool)
    residual_degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.int64)
    heap = [(-int(residual_degree[node]), int(node)) for node in range(n)]
    heapq.heapify(heap)
    order: List[int] = []

    while heap:
        neg_deg, node = heapq.heappop(heap)
        deg = -int(neg_deg)
        if not active[node]:
            continue
        if deg != int(residual_degree[node]):
            continue
        order.append(int(node))
        chosen[node] = True

        to_remove = [int(node)]
        for neigh in adj.getrow(node).indices:
            neigh = int(neigh)
            if active[neigh]:
                to_remove.append(neigh)

        for removed in to_remove:
            if not active[removed]:
                continue
            active[removed] = False
            residual_degree[removed] = -1
            for neigh in adj.getrow(removed).indices:
                neigh = int(neigh)
                if active[neigh]:
                    residual_degree[neigh] -= 1
                    heapq.heappush(heap, (-int(residual_degree[neigh]), int(neigh)))

    leftovers = [int(node) for node in range(n) if not chosen[node]]
    order.extend(leftovers)
    return order


def build_method_scores(art: credible.DatasetArtifacts) -> Dict[str, object]:
    target_count = art.bundle.adjacency.shape[0]
    scores = {}
    runtimes = {}
    seed_orders = {}
    seed_k = int(art.diffusion_cfg["seed_k"])

    t0 = time.perf_counter()
    dd_order = fast_degree_discount_order(art.bundle.adjacency, prob=float(art.diffusion_cfg["sir_beta"]))
    runtimes["DegreeDiscount"] = float(time.perf_counter() - t0)
    scores["DegreeDiscount"] = credible.ranking_to_scores(dd_order, target_count)
    seed_orders["DegreeDiscount"] = [int(v) for v in dd_order[:seed_k]]

    t0 = time.perf_counter()
    ad_order = fast_adaptive_degree_order(art.bundle.adjacency)
    runtimes["AdaptiveDegree"] = float(time.perf_counter() - t0)
    scores["AdaptiveDegree"] = credible.ranking_to_scores(ad_order, target_count)
    seed_orders["AdaptiveDegree"] = [int(v) for v in ad_order[:seed_k]]

    return {"scores": scores, "runtimes": runtimes, "seed_orders": seed_orders}


def update_dataset(report: Dict[str, object], dataset: str) -> Dict[str, object]:
    dataset_started = time.perf_counter()
    cfg_path = OUT_DIR / f"{dataset}_eval_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    art = credible.prepare_dataset(dataset)
    art.diffusion_cfg = normalize_diffusion_cfg(cfg["diffusion"])

    computed = build_method_scores(art)
    method_scores = computed["scores"]
    runtime_table = computed["runtimes"]
    seed_orders = computed["seed_orders"]

    weighted_graph = nx.from_scipy_sparse_array(art.weighted_topo)
    sir = credible.SemanticSIRSimulation(
        weighted_graph,
        beta=float(art.diffusion_cfg["sir_beta"]),
        gamma=float(art.diffusion_cfg["sir_gamma"]),
    )
    si = credible.SemanticSISimulation(weighted_graph, beta=float(art.diffusion_cfg["si_beta"]))
    rows = {}
    sir_curves = {}
    si_curves = {}
    for method in NEW_METHODS:
        rows[method] = base.evaluate_rankings(art.rank_target, method_scores[method], art.bundle.test_idx, k=100)
        seeds = seed_orders[method]
        sir_curve = credible.average_curve(sir, seeds, runs=int(cfg["sim_runs"]), t_steps=int(cfg["t_steps"]))
        si_curve = credible.average_curve(si, seeds, runs=int(cfg["sim_runs"]), t_steps=int(cfg["t_steps"]))
        sir_curves[method] = [float(v) for v in sir_curve]
        si_curves[method] = [float(v) for v in si_curve]
        rows[method]["F(20)-SIR"] = float(sir_curve[-1])
        rows[method]["F(20)-SI"] = float(si_curve[-1])

    communities = base.build_communities(art.bundle.adjacency)
    graph = nx.from_scipy_sparse_array(art.bundle.adjacency)

    item = report[dataset]
    item.setdefault("metrics", {})
    item.setdefault("diversity", {})
    item.setdefault("runtime", {})
    item.setdefault("sir_curves", {})
    item.setdefault("si_curves", {})
    item.setdefault("seeds", {})

    for method in OLD_METHODS:
        item["metrics"].pop(method, None)
        item["diversity"].pop(method, None)
        item["runtime"].pop(method, None)
        item["sir_curves"].pop(method, None)
        item["si_curves"].pop(method, None)
        item["seeds"].pop(method, None)

    dataset_summary = {
        "settings": {
            "diffusion": art.diffusion_cfg,
            "sim_runs": int(cfg["sim_runs"]),
            "t_steps": int(cfg["t_steps"]),
        },
        "methods": {},
    }

    for method in NEW_METHODS:
        seeds = seed_orders[method]
        diversity = full_art.diversity_metrics(art.bundle.adjacency, communities, graph, seeds)
        item["metrics"][method] = {k: float(v) for k, v in rows[method].items()}
        item["diversity"][method] = {k: float(v) for k, v in diversity.items()}
        item["runtime"][method] = float(runtime_table[method])
        item["sir_curves"][method] = sir_curves[method]
        item["si_curves"][method] = si_curves[method]
        item["seeds"][method] = [int(v) for v in seeds]
        dataset_summary["methods"][method] = {
            "metrics": item["metrics"][method],
            "diversity": item["diversity"][method],
            "ranking_runtime_sec": float(runtime_table[method]),
            "seed_count": len(seeds),
        }

    requested = replace_requested_methods(list(item.get("requested_methods", cfg.get("requested_methods", []))))
    cfg["requested_methods"] = requested
    cfg["methods"] = replace_requested_methods(list(item.get("methods", cfg.get("methods", []))))
    cfg["missing_methods"] = [m for m in cfg.get("missing_methods", []) if m not in OLD_METHODS]
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    item["requested_methods"] = requested
    item["missing_methods"] = [m for m in item.get("missing_methods", []) if m not in OLD_METHODS]

    full_art.render_curve_plot(dataset, item["sir_curves"], "SIR", OUT_DIR / f"{dataset}_sir_full.png")
    full_art.render_curve_plot(dataset, item["si_curves"], "SI", OUT_DIR / f"{dataset}_si_full.png")
    full_art.write_curve_table(OUT_DIR / f"{dataset}_sir_curve_table.csv", item["sir_curves"])
    full_art.write_curve_table(OUT_DIR / f"{dataset}_si_curve_table.csv", item["si_curves"])
    (OUT_DIR / f"{dataset}_sir_curve_table.json").write_text(json.dumps(item["sir_curves"], indent=2), encoding="utf-8")
    (OUT_DIR / f"{dataset}_si_curve_table.json").write_text(json.dumps(item["si_curves"], indent=2), encoding="utf-8")
    (OUT_DIR / f"{dataset}_diversity.json").write_text(json.dumps(item["diversity"], indent=2), encoding="utf-8")
    (OUT_DIR / f"{dataset}_seeds.json").write_text(json.dumps(item["seeds"], indent=2), encoding="utf-8")

    dataset_summary["dataset_wall_time_sec"] = float(time.perf_counter() - dataset_started)
    return dataset_summary


def write_report_artifacts(report: Dict[str, object]) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "full_comparison.md").write_text(full_art.build_markdown(report), encoding="utf-8")
    (OUT_DIR / "selected_paper_table.md").write_text(full_art.build_selected_table(report), encoding="utf-8")

    payload, markdown = full_art.build_main_metrics_summary(report)
    (OUT_DIR / "paper_main_metrics_all_methods.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "paper_main_metrics_all_methods.md").write_text(markdown, encoding="utf-8")

    paper_tables.main()
    paper_curves.main()
    panels.main()


def write_summary(summary: Dict[str, object]) -> None:
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Degree Discount / Adaptive Degree Comparison Summary", ""]
    lines.append(f"- Total wall time (s): {float(summary['total_wall_time_sec']):.4f}")
    lines.append("")
    for dataset in DATASETS:
        item = summary["datasets"][dataset]
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- Dataset wall time (s): {float(item['dataset_wall_time_sec']):.4f}")
        for method in NEW_METHODS:
            method_item = item["methods"][method]
            metrics = method_item["metrics"]
            diversity = method_item["diversity"]
            lines.append(
                f"- {method}: NDCG@100={metrics['NDCG@100']:.4f}, Spearman={metrics['Spearman']:.4f}, "
                f"F(20)-SIR={metrics['F(20)-SIR']:.4f}, F(20)-SI={metrics['F(20)-SI']:.4f}, "
                f"Ls={diversity['Ls']:.4f}, ranking_runtime_sec={method_item['ranking_runtime_sec']:.6f}"
            )
        lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    summary = {"datasets": {}}

    for dataset in DATASETS:
        summary["datasets"][dataset] = update_dataset(report, dataset)

    write_report_artifacts(report)
    summary["total_wall_time_sec"] = float(time.perf_counter() - started)
    write_summary(summary)

    print("updated Degree Discount / Adaptive Degree comparison in", OUT_DIR)
    print("summary saved to", SUMMARY_JSON)


if __name__ == "__main__":
    main()

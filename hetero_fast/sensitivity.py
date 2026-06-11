from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .data import load_dataset, target_adjacency
from .diffusion import build_diffusion_state, monte_carlo_truth
from .hcf_model import train_hcf
from .metrics import discounted_topk, evaluate_method


OUT_DIR = Path(__file__).resolve().parent.parent / "hetero_fast_results" / "sensitivity"


def run_sensitivity(dataset: str, force_cpu: bool = False) -> Dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset)
    truth = monte_carlo_truth(data, force=False)
    diffusion_state = build_diffusion_state(data)
    rows: List[Dict[str, float]] = []

    semantic_train_alpha_values = [0.20, 0.32, 0.44]
    late_semantic_weight_values = [0.10, 0.18, 0.26]
    semantic_view_k_values = [8, 16, 24]
    discount_penalty_values = [0.88, 0.92, 0.96]

    for semantic_train_alpha, late_semantic_weight, semantic_view_k, discount_penalty in itertools.product(
        semantic_train_alpha_values,
        late_semantic_weight_values,
        semantic_view_k_values,
        discount_penalty_values,
    ):
        topology_train_alpha = max(0.0, 1.0 - semantic_train_alpha)
        hcf = train_hcf(
            data,
            truth.truth_fused,
            semantic_view_k=semantic_view_k,
            semantic_train_alpha=semantic_train_alpha,
            topology_train_alpha=topology_train_alpha,
            late_semantic_weight=late_semantic_weight,
            discount_penalty=discount_penalty,
            epochs=60,
            force_cpu=force_cpu,
        )
        seeds = discounted_topk(hcf.scores, target_adjacency(data), int(truth.config["seed_k"]), discount_penalty)
        item = evaluate_method(data, truth, hcf.scores, hcf.runtime_sec, seeds=seeds, diffusion_state=diffusion_state)
        metrics = item["metrics"]
        rows.append(
            {
                "semantic_train_alpha": semantic_train_alpha,
                "late_semantic_weight": late_semantic_weight,
                "semantic_view_k": semantic_view_k,
                "discount_penalty": discount_penalty,
                "NDCG@100": metrics["NDCG@100"],
                "Spearman": metrics["Spearman"],
                "F(20)-SIR": metrics["F(20)-SIR"],
                "F(20)-SI": metrics["F(20)-SI"],
                "Ls": metrics["Ls"],
                "RuntimeSec": metrics["RuntimeSec"],
            }
        )

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"{dataset}_sensitivity.csv"
    json_path = OUT_DIR / f"{dataset}_sensitivity.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return {"dataset": dataset, "rows": rows}

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "HCF-Net": {"color": "#b30000", "width": 3.0},
    "AdaptiveDegree": {"color": "#0c7c59", "width": 2.0},
    "DegreeDiscount": {"color": "#005f99", "width": 2.0},
    "K-Shell": {"color": "#8c6d1f", "width": 2.0},
    "PageRank": {"color": "#ff7a00", "width": 2.0},
}


def plot_curves(dataset: str, payload: Dict[str, object], out_dir: Path) -> None:
    methods = payload["methods"]
    for mode_key, file_stem in [("sir_curve", "sir"), ("si_curve", "si")]:
        plt.figure(figsize=(7.0, 5.0))
        for method, item in methods.items():
            curve = item[mode_key]
            style = STYLE.get(method, {"color": None, "width": 1.8})
            plt.plot(range(1, len(curve) + 1), curve, label=method, linewidth=style["width"], color=style["color"])
        plt.xlabel("t")
        plt.ylabel("f(t)")
        plt.title(f"{dataset} Heterogeneous {file_stem.upper()}")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset}_{file_stem}.png", dpi=220)
        plt.close()


def plot_metrics_table(all_payloads: List[Dict[str, object]], out_dir: Path) -> None:
    rows = []
    for payload in all_payloads:
        dataset = payload["dataset"]
        for method, item in payload["methods"].items():
            metrics = item["metrics"]
            rows.append(
                {
                    "Dataset": dataset,
                    "Method": method,
                    "NDCG@100": metrics["NDCG@100"],
                    "Spearman": metrics["Spearman"],
                    "F(20)-SIR": metrics["F(20)-SIR"],
                    "F(20)-SI": metrics["F(20)-SI"],
                    "Ls": metrics["Ls"],
                    "RuntimeSec": metrics["RuntimeSec"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_metrics.csv", index=False)
    (out_dir / "all_metrics.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

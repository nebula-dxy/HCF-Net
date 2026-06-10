import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "results_external_credible" / "full_comparison.json"
OUT_DIR = ROOT / "results_external_credible"


METHOD_ORDER = [
    "HCF-Net",
    "HeCo",
    "RGTN",
    "EASING",
    "LICAP",
    "MAHE",
    "DegreeDiscount",
    "AdaptiveDegree",
    "PageRank",
    "GENI",
]


MAIN_METRICS = ["NDCG@100", "F(20)-SIR", "F(20)-SI", "Ls"]
SUPPORT_METRICS = ["Effective1HopCoverage", "Unique1HopCoverage", "CommunityCoverage", "CommunityEntropy"]


def ordered_methods(item: Dict[str, Dict[str, float]]) -> List[str]:
    ordered = [m for m in METHOD_ORDER if m in item]
    ordered.extend([m for m in item if m not in ordered])
    return ordered


def rank_map(methods: List[str], table: Dict[str, Dict[str, float]], metric: str) -> Dict[str, int]:
    scored = sorted(methods, key=lambda m: float(table[m][metric]), reverse=True)
    return {method: idx + 1 for idx, method in enumerate(scored)}


def style_value(value: float, rank: int) -> str:
    text = f"{float(value):.4f}"
    if rank == 1:
        return f"**{text}**"
    if rank == 2:
        return f"*{text}*"
    return text


def build_markdown(report: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Paper-Ready Main Table")
    lines.append("")
    lines.append("Main-text metrics: ranking quality, final diffusion strength, and average shortest path among selected key nodes.")
    lines.append("Best is bold; second-best is italic.")
    lines.append("")
    for dataset in ["ACM", "DBLP", "Yelp"]:
        if dataset not in report:
            continue
        item = report[dataset]
        methods = ordered_methods(item["metrics"])
        main_table = {
            method: {
                "NDCG@100": item["metrics"][method]["NDCG@100"],
                "F(20)-SIR": item["metrics"][method]["F(20)-SIR"],
                "F(20)-SI": item["metrics"][method]["F(20)-SI"],
                "Ls": item["diversity"][method]["Ls"],
            }
            for method in methods
        }
        ranks = {metric: rank_map(methods, main_table, metric) for metric in MAIN_METRICS}

        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Method | NDCG@100 | F(20)-SIR | F(20)-SI | Ls |")
        lines.append("|---|---:|---:|---:|---:|")
        for method in methods:
            row = main_table[method]
            lines.append(
                "| "
                + method
                + " | "
                + " | ".join(style_value(row[m], ranks[m][method]) for m in MAIN_METRICS)
                + " |"
            )
        lines.append("")

        support_table = {
            method: {
                "Effective1HopCoverage": item["diversity"][method]["Effective1HopCoverage"],
                "Unique1HopCoverage": item["diversity"][method]["Unique1HopCoverage"],
                "CommunityCoverage": item["diversity"][method]["CommunityCoverage"],
                "CommunityEntropy": item["diversity"][method]["CommunityEntropy"],
            }
            for method in methods
        }
        support_ranks = {metric: rank_map(methods, support_table, metric) for metric in SUPPORT_METRICS}
        lines.append("| Method | Effective1HopCoverage | Unique1HopCoverage | CommunityCoverage | CommunityEntropy |")
        lines.append("|---|---:|---:|---:|---:|")
        for method in methods:
            row = support_table[method]
            lines.append(
                "| "
                + method
                + " | "
                + " | ".join(style_value(row[m], support_ranks[m][method]) for m in SUPPORT_METRICS)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def write_csvs(report: Dict[str, object]) -> None:
    for dataset in ["ACM", "DBLP", "Yelp"]:
        if dataset not in report:
            continue
        item = report[dataset]
        methods = ordered_methods(item["metrics"])

        main_path = OUT_DIR / f"{dataset}_paper_main_metrics.csv"
        with main_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Method"] + MAIN_METRICS)
            for method in methods:
                writer.writerow(
                    [
                        method,
                        f"{float(item['metrics'][method]['NDCG@100']):.6f}",
                        f"{float(item['metrics'][method]['F(20)-SIR']):.6f}",
                        f"{float(item['metrics'][method]['F(20)-SI']):.6f}",
                        f"{float(item['diversity'][method]['Ls']):.6f}",
                    ]
                )

        support_path = OUT_DIR / f"{dataset}_paper_diversity_support.csv"
        with support_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Method"] + SUPPORT_METRICS)
            for method in methods:
                writer.writerow(
                    [
                        method,
                        f"{float(item['diversity'][method]['Effective1HopCoverage']):.6f}",
                        f"{float(item['diversity'][method]['Unique1HopCoverage']):.6f}",
                        f"{float(item['diversity'][method]['CommunityCoverage']):.6f}",
                        f"{float(item['diversity'][method]['CommunityEntropy']):.6f}",
                    ]
                )


def main() -> None:
    report = json.loads(IN_PATH.read_text(encoding="utf-8"))
    md = build_markdown(report)
    (OUT_DIR / "paper_main_table.md").write_text(md, encoding="utf-8")
    write_csvs(report)
    print("saved paper tables to", OUT_DIR)


if __name__ == "__main__":
    main()

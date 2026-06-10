import json
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results_external_credible"
JSON_OUT = OUT_DIR / "all_method_runtimes.json"
MD_OUT = OUT_DIR / "all_method_runtimes.md"
DATASETS = ["ACM", "DBLP", "Yelp"]


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_hcf_runtime(dataset: str) -> Optional[float]:
    payload = load_json(ROOT / "results_hcfnet_credible" / f"{dataset}_hcf_meta.json")
    if not payload:
        return None
    return payload.get("total_runtime_sec")


def get_heco_runtime(dataset: str) -> Optional[float]:
    payload = load_json(ROOT / "results" / "HeCo" / f"CUSTOM_{dataset}" / "summary.json")
    if not payload:
        return None
    return payload.get("total_runtime_sec")


def get_external_runtime(dataset: str, rel_path: str) -> Optional[float]:
    payload = load_json(ROOT / rel_path)
    if not payload:
        return None
    return payload.get("total_runtime_sec")


def get_mahe_runtime(dataset: str) -> Optional[float]:
    payload = load_json(ROOT / "results" / "MAHE" / f"CUSTOM_{dataset}" / "summary.json")
    if not payload:
        return None
    return payload.get("total_runtime_sec")


def get_degree_summary() -> dict:
    return load_json(OUT_DIR / "degree_discount_adaptive_summary.json") or {}


def main() -> None:
    degree_summary = get_degree_summary()
    summary: Dict[str, Dict[str, float]] = {}

    for dataset in DATASETS:
        summary[dataset] = {}
        values = {
            "HCF-Net": get_hcf_runtime(dataset),
            "HeCo": get_heco_runtime(dataset),
            "GENI": get_external_runtime(dataset, f"results/CUSTOM_{dataset}_rel_GENI/runtime_summary.json"),
            "RGTN": get_external_runtime(dataset, f"results/CUSTOM_{dataset}_two_RGTN/runtime_summary.json"),
            "EASING": get_external_runtime(dataset, f"results/CUSTOM_{dataset}_EASING/runtime_summary.json"),
            "LICAP": get_external_runtime(dataset, f"external/LICAP/pretrain/results/CUSTOM_{dataset}_two_pregat_struct_pretrain_rgtn/runtime_summary.json"),
            "MAHE": get_mahe_runtime(dataset),
            "DegreeDiscount": ((degree_summary.get("datasets") or {}).get(dataset) or {}).get("methods", {}).get("DegreeDiscount", {}).get("ranking_runtime_sec"),
            "AdaptiveDegree": ((degree_summary.get("datasets") or {}).get(dataset) or {}).get("methods", {}).get("AdaptiveDegree", {}).get("ranking_runtime_sec"),
        }
        for method, value in values.items():
            if value is not None:
                summary[dataset][method] = float(value)

    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = ["# All Method Runtimes", ""]
    for dataset in DATASETS:
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Method | Total Runtime (s) |")
        lines.append("|---|---:|")
        for method, value in summary[dataset].items():
            lines.append(f"| {method} | {value:.4f} |")
        lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print("saved", JSON_OUT)
    print("saved", MD_OUT)


if __name__ == "__main__":
    main()

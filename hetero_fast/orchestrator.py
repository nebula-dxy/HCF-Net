from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .baselines import centrality_baselines
from .data import HeteroDataset, load_dataset, minmax_scale
from .data import target_adjacency
from .diffusion import HeteroDiffusionArtifacts, build_diffusion_state, monte_carlo_truth
from .hcf_model import train_hcf
from .metrics import discounted_topk, evaluate_method


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "hetero_fast_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


EXTERNAL_SCORE_FILES = {
    "GENI": lambda d: ROOT / "results" / f"CUSTOM_{d}_rel_GENI" / f"0_{d.lower()}_geni_pred.npy",
    "RGTN": lambda d: ROOT / "results" / f"CUSTOM_{d}_two_RGTN" / f"0_{d.lower()}_rgtn_pred.npy",
    "EASING": lambda d: ROOT / "results" / f"CUSTOM_{d}_EASING" / f"0_{d.lower()}_easing_pred.npy",
    "LICAP": lambda d: ROOT / "external" / "LICAP" / "pretrain" / "results" / f"CUSTOM_{d}_two_pregat_struct_pretrain_rgtn" / f"0_{d.lower()}_licap_pred.npy",
    "MAHE-IM": lambda d: ROOT / "results" / "MAHE" / f"CUSTOM_{d}" / f"{d.lower()}_mahe_pred.npy",
    "HeCo": lambda d: ROOT / "results" / "HeCo" / f"CUSTOM_{d}" / f"{d.lower()}_heco_pred.npy",
    "Touple-S2V_DQN": lambda d: ROOT / "results_hcfnet" / f"{d}_S2V_DQN_q_scores.npy",
    "Touple-Tripling": lambda d: ROOT / "results_hcfnet" / f"{d}_Tripling_q_scores.npy",
}


EXTERNAL_RUNTIME_FILES = {
    "GENI": lambda d: ROOT / "results" / f"CUSTOM_{d}_rel_GENI" / "runtime_summary.json",
    "RGTN": lambda d: ROOT / "results" / f"CUSTOM_{d}_two_RGTN" / "runtime_summary.json",
    "EASING": lambda d: ROOT / "results" / f"CUSTOM_{d}_EASING" / "runtime_summary.json",
    "LICAP": lambda d: ROOT / "external" / "LICAP" / "pretrain" / "results" / f"CUSTOM_{d}_two_pregat_struct_pretrain_rgtn" / "runtime_summary.json",
    "MAHE-IM": lambda d: ROOT / "results" / "MAHE" / f"CUSTOM_{d}" / "summary.json",
    "HeCo": lambda d: ROOT / "results" / "HeCo" / f"CUSTOM_{d}" / "summary.json",
}


def load_runtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0
    return float(payload.get("total_runtime_sec", 0.0))


def load_external_scores(dataset: str, method: str, n: int) -> Optional[np.ndarray]:
    path = EXTERNAL_SCORE_FILES[method](dataset)
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True).reshape(-1).astype(np.float32)
    if arr.size < n:
        out = np.zeros(n, dtype=np.float32)
        out[: arr.size] = arr
        return out
    return arr[:n]


def available_external_runtimes(dataset: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for method, resolver in EXTERNAL_RUNTIME_FILES.items():
        out[method] = load_runtime(resolver(dataset))
    return out


def maybe_run_external_baselines(datasets: List[str], methods: List[str], gpu: int, skip: bool) -> None:
    if skip:
        return
    if any(m in {"GENI", "RGTN", "EASING"} for m in methods):
        cmd = [sys.executable, "run_external_baselines.py", "--datasets", *datasets, "--methods", *[m for m in methods if m in {"GENI", "RGTN", "EASING"}], "--gpu", str(gpu)]
        subprocess.run(cmd, cwd=ROOT, check=True)
    if "LICAP" in methods:
        cmd = [sys.executable, "run_licap_baseline.py", "--datasets", *datasets, "--gpu", str(gpu)]
        subprocess.run(cmd, cwd=ROOT, check=True)
    if "MAHE-IM" in methods:
        cmd = [sys.executable, "run_mahe_baseline.py", "--datasets", *datasets]
        subprocess.run(cmd, cwd=ROOT, check=True)
    if "HeCo" in methods:
        cmd = [sys.executable, "run_heco_baseline.py", "--datasets", *datasets]
        subprocess.run(cmd, cwd=ROOT, check=True)
    if any(m in {"Touple-S2V_DQN", "Touple-Tripling"} for m in methods):
        cmd = [sys.executable, "external_touple_eval.py"]
        subprocess.run(cmd, cwd=ROOT, check=True)


def run_one_dataset(
    dataset: str,
    run_external: bool,
    gpu: int,
    force_truth: bool,
    force_cpu_hcf: bool,
    truth_override: Optional[Dict[str, float]] = None,
    hcf_params: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    methods = ["GENI", "RGTN", "EASING", "LICAP", "MAHE-IM", "HeCo", "Touple-S2V_DQN", "Touple-Tripling"]
    if run_external:
        maybe_run_external_baselines([dataset], methods, gpu=gpu, skip=False)
    data = load_dataset(dataset)
    truth = monte_carlo_truth(data, force=force_truth, config_override=truth_override)
    diffusion_state = build_diffusion_state(data)
    out: Dict[str, object] = {"dataset": dataset, "truth": truth.config, "methods": {}}

    hcf_cfg = {
        "semantic_view_k": 16,
        "semantic_train_alpha": 0.32,
        "topology_train_alpha": 0.68,
        "late_semantic_weight": 0.18,
        "discount_penalty": 0.92,
    }
    if hcf_params:
        hcf_cfg.update(hcf_params)
    hcf = train_hcf(data, truth.truth_fused, force_cpu=force_cpu_hcf, **hcf_cfg)
    hcf_seeds = discounted_topk(hcf.scores, target_adjacency(data), int(truth.config["seed_k"]), float(hcf_cfg["discount_penalty"]))
    out["methods"]["HCF-Net"] = evaluate_method(data, truth, hcf.scores, hcf.runtime_sec, seeds=hcf_seeds, diffusion_state=diffusion_state)
    out["methods"]["HCF-Net"]["meta"] = hcf.meta
    np.save(RESULTS_DIR / f"{dataset}_hcf_scores.npy", hcf.scores)

    central = centrality_baselines(data, seed_k=int(truth.config["seed_k"]), sir_beta=float(truth.config["sir_beta"]))
    for method, scores in central.items():
        started = time.perf_counter()
        out["methods"][method] = evaluate_method(data, truth, scores, float(time.perf_counter() - started), diffusion_state=diffusion_state)

    ext_runtime = available_external_runtimes(dataset)
    for method in methods:
        scores = load_external_scores(dataset, method, data.target_count)
        if scores is None:
            continue
        out["methods"][method] = evaluate_method(data, truth, minmax_scale(scores), ext_runtime.get(method, 0.0), diffusion_state=diffusion_state)

    path = RESULTS_DIR / f"{dataset}_summary.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out

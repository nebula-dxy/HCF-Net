from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestrator import RESULTS_DIR, run_one_dataset
from .plots import plot_curves, plot_metrics_table
from .sensitivity import run_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact heterogeneous influence experiment runner")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "Yelp"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--force-truth", action="store_true")
    parser.add_argument("--force-cpu-hcf", action="store_true")
    parser.add_argument("--with-sensitivity", action="store_true")
    parser.add_argument("--truth-mc-runs", type=int, default=0)
    parser.add_argument("--truth-steps", type=int, default=0)
    args = parser.parse_args()

    payloads = []
    truth_override = {}
    if args.truth_mc_runs > 0:
        truth_override["mc_runs"] = int(args.truth_mc_runs)
    if args.truth_steps > 0:
        truth_override["t_steps"] = int(args.truth_steps)
    for dataset in args.datasets:
        payload = run_one_dataset(
            dataset,
            run_external=not args.skip_external,
            gpu=args.gpu,
            force_truth=args.force_truth,
            force_cpu_hcf=args.force_cpu_hcf,
            truth_override=truth_override or None,
        )
        payloads.append(payload)
        plot_curves(dataset, payload, RESULTS_DIR)
        if args.with_sensitivity:
            run_sensitivity(dataset, force_cpu=args.force_cpu_hcf)

    plot_metrics_table(payloads, RESULTS_DIR)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(payloads, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

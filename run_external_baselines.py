import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
PYTHON_EXE = Path(sys.executable)


def script_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def prepared_path(dataset: str) -> str:
    return str(ROOT / "external_prepared" / dataset / f"{dataset.lower()}_nie.pt")


def run(cmd: Sequence[str], dry_run: bool = False, cwd: Path = ROOT) -> None:
    print("RUN", " ".join(str(c) for c in cmd))
    if dry_run:
        return
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )


def runtime_output_path(method: str, dataset: str) -> Path:
    if method == "GENI":
        return ROOT / "results" / f"CUSTOM_{dataset}_rel_GENI" / "runtime_summary.json"
    if method == "RGTN":
        return ROOT / "results" / f"CUSTOM_{dataset}_two_RGTN" / "runtime_summary.json"
    if method == "EASING":
        return ROOT / "results" / f"CUSTOM_{dataset}_EASING" / "runtime_summary.json"
    raise ValueError(method)


def run_geni(dataset: str, epochs: int, gpu: int, dry_run: bool = False) -> None:
    tag = f"{dataset.lower()}_geni"
    cmd = [
        str(PYTHON_EXE),
        str(script_path("external", "RGTN-NIE", "GENI", "geni_batch_train.py")),
        "--dataset", f"CUSTOM_{dataset}_rel",
        "--data_path", prepared_path(dataset),
        "--cross-num", "1",
        "--gpu", str(gpu),
        "--epochs", str(epochs),
        "--batch-size", "1024",
        "--num-workers", "0",
        "--num-hidden", "16" if dataset == "Yelp" else "8",
        "--num-heads", "8",
        "--num-out-heads", "4",
        "--pred-dim", "32" if dataset == "Yelp" else "16",
        "--save-path", f"{tag}_checkpoint.pt",
    ]
    if dataset != "Yelp":
        cmd.append("--spm")
    else:
        cmd.extend(["--no-scale", "--patience", "20"])
    run(cmd, dry_run=dry_run, cwd=script_path("external", "RGTN-NIE"))


def run_rgtn(dataset: str, epochs: int, gpu: int, dry_run: bool = False) -> None:
    tag = f"{dataset.lower()}_rgtn"
    cmd = [
        str(PYTHON_EXE),
        str(script_path("external", "RGTN-NIE", "two_branch", "two_branch_batch_train.py")),
        "--dataset", f"CUSTOM_{dataset}_two",
        "--data_path", prepared_path(dataset),
        "--cross-num", "1",
        "--gpu", str(gpu),
        "--epochs", str(epochs),
        "--batch-size", "1024",
        "--num-workers", "0",
        "--loss-lambda", "0.7",
        "--loss-alpha", "0.6",
        "--list-num", "100",
        "--residual",
        "--norm",
        "--num-hidden", "8",
        "--num-heads", "8",
        "--num-out-heads", "8",
        "--save-path", f"{tag}_checkpoint.pt",
        "--spm",
        "--pred-dim", "16",
    ]
    run(cmd, dry_run=dry_run, cwd=script_path("external", "RGTN-NIE"))


def run_easing(dataset: str, epochs: int, gpu: int, dry_run: bool = False) -> None:
    tag = f"{dataset.lower()}_easing"
    cmd = [
        str(PYTHON_EXE),
        str(script_path("external", "EASING", "easing", "main_easing.py")),
        "--dataset", f"CUSTOM_{dataset}",
        "--data_path", str(ROOT),
        "--graph_data", str(script_path("external_prepared", dataset, f"{dataset.lower()}_nie.pt")),
        "--semantic_data", "placeholder.pk",
        "--structure_data", "placeholder.pk",
        "--cross-num", "1",
        "--gpu", str(gpu),
        "--epochs", str(epochs),
        "--train_num", "1.0",
        "--samp_ssl", "5",
        "--unc_layers", "1",
        "--save-path", f"{tag}_checkpoint.pt",
        "--early-stop",
        "--patience", "60",
        "--min-epoch", "10",
    ]
    run(cmd, dry_run=dry_run, cwd=script_path("external", "EASING"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external baselines on prepared custom hetero datasets")
    parser.add_argument("--datasets", nargs="+", default=["ACM"])
    parser.add_argument("--methods", nargs="+", default=["GENI", "RGTN", "EASING"])
    parser.add_argument("--geni-epochs", type=int, default=40)
    parser.add_argument("--rgtn-epochs", type=int, default=40)
    parser.add_argument("--easing-epochs", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing them")
    args = parser.parse_args()

    for dataset in args.datasets:
        for method in args.methods:
            started_at = time.perf_counter()
            if method == "GENI":
                run_geni(dataset, args.geni_epochs, args.gpu, dry_run=args.dry_run)
            elif method == "RGTN":
                run_rgtn(dataset, args.rgtn_epochs, args.gpu, dry_run=args.dry_run)
            elif method == "EASING":
                run_easing(dataset, args.easing_epochs, args.gpu, dry_run=args.dry_run)
            else:
                raise ValueError(f"Unsupported method: {method}")
            out_path = runtime_output_path(method, dataset)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "method": method,
                        "total_runtime_sec": float(time.perf_counter() - started_at),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()

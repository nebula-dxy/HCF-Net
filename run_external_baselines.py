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
    cmd = [
        str(PYTHON_EXE),
        str(script_path("external_geni_eval.py")),
        "--datasets", dataset,
    ]
    run(cmd, dry_run=dry_run, cwd=ROOT)


def run_rgtn(dataset: str, epochs: int, gpu: int, dry_run: bool = False) -> None:
    cmd = [
        str(PYTHON_EXE),
        str(script_path("external_rgtn_eval.py")),
        "--datasets", dataset,
    ]
    run(cmd, dry_run=dry_run, cwd=ROOT)


def run_easing(dataset: str, epochs: int, gpu: int, dry_run: bool = False) -> None:
    cmd = [
        str(PYTHON_EXE),
        str(script_path("external_easing_eval.py")),
        "--datasets", dataset,
    ]
    run(cmd, dry_run=dry_run, cwd=ROOT)


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

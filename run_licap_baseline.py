import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LICAP_ROOT = ROOT / "external" / "LICAP"
PYTHON_EXE = Path(sys.executable)


def prepared_path(dataset: str) -> Path:
    return ROOT / "external_prepared" / dataset / f"{dataset.lower()}_nie.pt"


def run(cmd):
    print("RUN", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=LICAP_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LICAP on prepared custom heterogeneous datasets")
    parser.add_argument("--datasets", nargs="+", default=["ACM"])
    parser.add_argument("--pretrain-model", default="pregat_struct", choices=["pregat_struct", "pregat_semantic"])
    parser.add_argument("--cross-num", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--downstream-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--important-ratio", type=float, default=0.1)
    args = parser.parse_args()

    pretrain_script = "pregat_pretrain_struct.py" if args.pretrain_model == "pregat_struct" else "pregat_pretrain_semantic.py"
    for dataset in args.datasets:
        started_at = time.perf_counter()
        tag = dataset.lower()
        common = [
            "--dataset", f"CUSTOM_{dataset}_two",
            "--data_path", str(prepared_path(dataset)),
            "--cross-num", str(args.cross_num),
            "--gpu", str(args.gpu),
            "--important-ratio", str(args.important_ratio),
            "--pretrain-patience", str(args.patience),
        ]
        run([
            str(PYTHON_EXE),
            f"pretrain/{pretrain_script}",
            *common,
            "--epochs", str(args.pretrain_epochs),
            "--save-path", f"{tag}_licap_{args.pretrain_model}_checkpoint.pt",
        ])
        run([
            str(PYTHON_EXE),
            "downstream/rgtn_downstream.py",
            *common,
            "--epochs", str(args.downstream_epochs),
            "--patience", str(args.patience),
            "--pretrain-model", args.pretrain_model,
            "--save-path", f"{tag}_licap_rgtn_checkpoint.pt",
            "--list-num", "100",
            "--spm",
        ])
        out_dir = LICAP_ROOT / "pretrain" / "results" / f"CUSTOM_{dataset}_two_pregat_struct_pretrain_rgtn"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "runtime_summary.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "method": "LICAP",
                    "total_runtime_sec": float(time.perf_counter() - started_at),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

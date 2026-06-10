import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ROOT = ROOT / "dlc_bundle" / "gemini"


FILES = [
    "credible_experiment_runner.py",
    "credible_ablation_runner.py",
    "credible_sensitivity_runner.py",
    "experiment_runner.py",
    "build_full_comparison_artifacts.py",
    "build_paper_curve_plots.py",
    "build_paper_tables.py",
    "build_combined_diffusion_panels.py",
    "prepare_external_nie_data.py",
    "run_external_baselines.py",
    "run_heco_baseline.py",
    "run_licap_baseline.py",
    "run_mahe_baseline.py",
    "external_baselines.py",
    "external_easing_eval.py",
    "external_rgtn_eval.py",
    "external_touple_eval.py",
    "CLOUD_RUN.md",
    "PAI_DLC_RUN.md",
]


DIRS = [
    "scripts",
    "results",
    "results_hcfnet",
    "results_external_credible",
    "external",
    "external_datasets/pyg_hgb",
    "external_prepared",
    "external_inputs",
]


DATA_FILES = [
    "ACM.mat",
    "DBLP.mat",
    "Yelp.mat",
]


def copy_file(rel_path: str) -> None:
    src = ROOT / rel_path
    dst = BUNDLE_ROOT / rel_path
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(rel_path: str) -> None:
    src = ROOT / rel_path
    dst = BUNDLE_ROOT / rel_path
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def main() -> None:
    if BUNDLE_ROOT.exists():
        shutil.rmtree(BUNDLE_ROOT)
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)

    for rel_path in FILES:
        copy_file(rel_path)
    for rel_path in DIRS:
        copy_dir(rel_path)

    data_dir = BUNDLE_ROOT / "data_manifest"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "README.txt").write_text(
        "Place ACM.mat, DBLP.mat, Yelp.mat under /mnt/data when running on DLC.\n"
        "The DLC entry script will symlink them into the workspace root.\n",
        encoding="utf-8",
    )
    (data_dir / "expected_files.txt").write_text("\n".join(DATA_FILES) + "\n", encoding="utf-8")

    print("prepared DLC bundle at", BUNDLE_ROOT)


if __name__ == "__main__":
    main()

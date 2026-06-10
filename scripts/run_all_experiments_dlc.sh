#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/root/workspace}"
WORKDIR="${WORKDIR:-/root/workspace/gemini}"
DATA_DIR="${DATA_DIR:-/mnt/data}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/root/workspace/gemini_bundle.zip}"

if [ ! -d "${WORKDIR}" ]; then
  if [ -f "${ARCHIVE_PATH}" ]; then
    mkdir -p "${WORKDIR}"
    python - <<PY
import zipfile
zip_path = r"${ARCHIVE_PATH}"
out_dir = r"${WORKDIR}"
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
print("extracted", zip_path, "to", out_dir)
PY
  else
    echo "[DLC] WORKDIR not found and ARCHIVE_PATH not found"
    exit 1
  fi
fi

LOG_DIR="${WORKDIR}/dlc_logs"

mkdir -p "${LOG_DIR}"
cd "${WORKDIR}"

echo "[DLC] workdir=${WORKDIR}"
echo "[DLC] data_dir=${DATA_DIR}"

for f in ACM.mat DBLP.mat Yelp.mat; do
  if [ -f "${DATA_DIR}/${f}" ] && [ ! -e "${WORKDIR}/${f}" ]; then
    ln -s "${DATA_DIR}/${f}" "${WORKDIR}/${f}"
    echo "[DLC] linked ${f} from ${DATA_DIR}"
  fi
done

python -V | tee "${LOG_DIR}/python_version.log"
python - <<'PY' | tee "${LOG_DIR}/torch_cuda.log"
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn matplotlib networkx pillow gensim tensorflow-cpu

python prepare_external_nie_data.py | tee "${LOG_DIR}/prepare_external_nie_data.log"

python run_heco_baseline.py --datasets ACM DBLP Yelp | tee "${LOG_DIR}/run_heco_baseline.log"
python run_external_baselines.py --datasets ACM DBLP Yelp --methods GENI RGTN EASING | tee "${LOG_DIR}/run_external_baselines.log"
python run_licap_baseline.py --datasets ACM DBLP Yelp --gpu 0 --pretrain-epochs 3 --downstream-epochs 3 | tee "${LOG_DIR}/run_licap_baseline.log"
python run_mahe_baseline.py --datasets ACM DBLP Yelp | tee "${LOG_DIR}/run_mahe_baseline.log"

python credible_experiment_runner.py --datasets ACM DBLP Yelp --epochs 180 | tee "${LOG_DIR}/credible_experiment_runner.log"
python build_full_comparison_artifacts.py | tee "${LOG_DIR}/build_full_comparison_artifacts.log"
python build_paper_curve_plots.py | tee "${LOG_DIR}/build_paper_curve_plots.log"
python build_paper_tables.py | tee "${LOG_DIR}/build_paper_tables.log"
python build_combined_diffusion_panels.py | tee "${LOG_DIR}/build_combined_diffusion_panels.log"
python scripts/update_degree_discount_adaptive.py | tee "${LOG_DIR}/update_degree_discount_adaptive.log"

echo "[DLC] all tasks completed"

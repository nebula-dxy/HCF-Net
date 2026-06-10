#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
GPU_ID="${GPU_ID:-0}"
ACM_MAT_PATH="${ACM_MAT_PATH:-${WORKDIR}/ACM.mat}"
DBLP_MAT_PATH="${DBLP_MAT_PATH:-${WORKDIR}/DBLP.mat}"
YELP_MAT_PATH="${YELP_MAT_PATH:-${WORKDIR}/Yelp.mat}"
LOG_DIR="${WORKDIR}/tencent_logs"

mkdir -p "${LOG_DIR}"
cd "${WORKDIR}"

echo "[Tencent] workdir=${WORKDIR}"
echo "[Tencent] ACM_MAT_PATH=${ACM_MAT_PATH}"
echo "[Tencent] DBLP_MAT_PATH=${DBLP_MAT_PATH}"
echo "[Tencent] YELP_MAT_PATH=${YELP_MAT_PATH}"

if [ -f "${ACM_MAT_PATH}" ]; then
  ln -sfn "${ACM_MAT_PATH}" "${WORKDIR}/ACM.mat"
fi
if [ -f "${DBLP_MAT_PATH}" ]; then
  ln -sfn "${DBLP_MAT_PATH}" "${WORKDIR}/DBLP.mat"
fi
if [ -f "${YELP_MAT_PATH}" ]; then
  ln -sfn "${YELP_MAT_PATH}" "${WORKDIR}/Yelp.mat"
fi

python -V | tee "${LOG_DIR}/python_version.log"
python - <<'PY' | tee "${LOG_DIR}/torch_cuda.log"
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("torch_cuda", torch.version.cuda)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

python prepare_external_nie_data.py | tee "${LOG_DIR}/prepare_external_nie_data.log"

python run_heco_baseline.py --datasets ACM DBLP Yelp | tee "${LOG_DIR}/run_heco_baseline.log"
python run_external_baselines.py --datasets ACM DBLP Yelp --methods GENI RGTN EASING --gpu "${GPU_ID}" | tee "${LOG_DIR}/run_external_baselines.log"
python run_licap_baseline.py --datasets ACM DBLP Yelp --gpu "${GPU_ID}" --pretrain-epochs 3 --downstream-epochs 3 | tee "${LOG_DIR}/run_licap_baseline.log"
python run_mahe_baseline.py --datasets ACM DBLP Yelp | tee "${LOG_DIR}/run_mahe_baseline.log"

python credible_experiment_runner.py --datasets ACM DBLP Yelp --epochs 180 | tee "${LOG_DIR}/credible_experiment_runner.log"
python build_full_comparison_artifacts.py | tee "${LOG_DIR}/build_full_comparison_artifacts.log"
python build_paper_curve_plots.py | tee "${LOG_DIR}/build_paper_curve_plots.log"
python build_paper_tables.py | tee "${LOG_DIR}/build_paper_tables.log"
python build_combined_diffusion_panels.py | tee "${LOG_DIR}/build_combined_diffusion_panels.log"
python scripts/update_degree_discount_adaptive.py | tee "${LOG_DIR}/update_degree_discount_adaptive.log"
python scripts/summarize_all_runtimes.py | tee "${LOG_DIR}/summarize_all_runtimes.log"

echo "[Tencent] all tasks completed"

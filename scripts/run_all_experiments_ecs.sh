#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-/mnt/data}"
LOG_DIR="${WORKDIR}/ecs_logs"

mkdir -p "${LOG_DIR}"
cd "${WORKDIR}"

echo "[ECS] workdir=${WORKDIR}"
echo "[ECS] data_dir=${DATA_DIR}"

for f in ACM.mat DBLP.mat Yelp.mat; do
  if [ -f "${DATA_DIR}/${f}" ] && [ ! -e "${WORKDIR}/${f}" ]; then
    ln -s "${DATA_DIR}/${f}" "${WORKDIR}/${f}"
    echo "[ECS] linked ${f} from ${DATA_DIR}"
  fi
done

python -V | tee "${LOG_DIR}/python_version.log"
python - <<'PY' | tee "${LOG_DIR}/torch_cuda.log"
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY

python -m pip install --upgrade pip
if [ -f "${WORKDIR}/requirements.txt" ]; then
  python -m pip install -r "${WORKDIR}/requirements.txt"
else
  python -m pip install numpy scipy scikit-learn matplotlib networkx pillow gensim tensorflow
fi

python prepare_external_nie_data.py | tee "${LOG_DIR}/prepare_external_nie_data.log"

python run_heco_baseline.py --datasets ACM DBLP Yelp | tee "${LOG_DIR}/run_heco_baseline.log"
python run_external_baselines.py --datasets ACM DBLP Yelp --methods GENI RGTN EASING | tee "${LOG_DIR}/run_external_baselines.log"
python run_licap_baseline.py --datasets ACM DBLP Yelp --gpu -1 --pretrain-epochs 3 --downstream-epochs 3 | tee "${LOG_DIR}/run_licap_baseline.log"
python run_mahe_baseline.py --datasets ACM DBLP Yelp | tee "${LOG_DIR}/run_mahe_baseline.log"

python credible_experiment_runner.py --datasets ACM DBLP Yelp --epochs 180 --cpu | tee "${LOG_DIR}/credible_experiment_runner.log"
python build_full_comparison_artifacts.py | tee "${LOG_DIR}/build_full_comparison_artifacts.log"
python build_paper_curve_plots.py | tee "${LOG_DIR}/build_paper_curve_plots.log"
python build_paper_tables.py | tee "${LOG_DIR}/build_paper_tables.log"
python build_combined_diffusion_panels.py | tee "${LOG_DIR}/build_combined_diffusion_panels.log"
python scripts/update_degree_discount_adaptive.py | tee "${LOG_DIR}/update_degree_discount_adaptive.log"
python scripts/summarize_all_runtimes.py | tee "${LOG_DIR}/summarize_all_runtimes.log"

echo "[ECS] all tasks completed"

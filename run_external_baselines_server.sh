#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
DATASETS=${DATASETS:-"ACM DBLP Yelp"}
METHODS=${METHODS:-"GENI RGTN EASING"}
GENI_EPOCHS=${GENI_EPOCHS:-40}
RGTN_EPOCHS=${RGTN_EPOCHS:-40}
EASING_EPOCHS=${EASING_EPOCHS:-20}
GPU_ID=${GPU_ID:--1}

export PYTHONUNBUFFERED=1

$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install -r requirements.txt
$PYTHON_BIN -m pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

$PYTHON_BIN prepare_external_nie_data.py

$PYTHON_BIN run_external_baselines.py \
  --datasets $DATASETS \
  --methods $METHODS \
  --geni-epochs $GENI_EPOCHS \
  --rgtn-epochs $RGTN_EPOCHS \
  --easing-epochs $EASING_EPOCHS \
  --gpu $GPU_ID

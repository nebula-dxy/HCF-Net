#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-$(pwd)}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${BASE_DIR}/gemini_bundle.zip}"
WORKDIR="${WORKDIR:-${BASE_DIR}/gemini}"
DATA_DIR="${DATA_DIR:-/mnt/data}"

echo "[BOOT] base_dir=${BASE_DIR}"
echo "[BOOT] archive_path=${ARCHIVE_PATH}"
echo "[BOOT] workdir=${WORKDIR}"
echo "[BOOT] data_dir=${DATA_DIR}"

if [ ! -f "${ARCHIVE_PATH}" ]; then
  echo "[BOOT] archive not found: ${ARCHIVE_PATH}"
  exit 1
fi

mkdir -p "${WORKDIR}"

python - <<PY
import zipfile
zip_path = r"${ARCHIVE_PATH}"
out_dir = r"${WORKDIR}"
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
print("extracted", zip_path, "to", out_dir)
PY

cd "${WORKDIR}"
bash scripts/run_all_experiments_dlc.sh

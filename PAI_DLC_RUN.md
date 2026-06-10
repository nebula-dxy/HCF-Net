# PAI-DLC Full Run Guide

This repo can be uploaded to DLC as one preserved folder tree instead of many loose files.

## Goal

Run the full experiment suite and keep per-method total runtime records, including:

- `HCF-Net`
- `HeCo`
- `GENI`
- `RGTN`
- `EASING`
- `LICAP`
- `MAHE`
- `Degree Discount`
- `Adaptive Degree`
- figure/table rebuild steps

## Your DLC image

Use:

- `ppu-training:2.1.0-pytorch2.6.0-ppu-py312-cu128-ubuntu24.04`

This is suitable for the current codebase and is much better than the local laptop environment.

## How DLC runs

In DLC, you normally provide:

- an image
- a code directory
- optional mounted datasets
- one startup command

DLC starts a container, mounts your code and datasets, then executes the startup command inside that container.

## Important path rule for your case

You said the dataset mount path is:

- `/mnt/data`

The current code expects:

- `ACM.mat`
- `DBLP.mat`
- `Yelp.mat`

to exist in the repo root.

To avoid modifying many scripts, the DLC startup script will automatically create symlinks:

- `/root/workspace/gemini/ACM.mat -> /mnt/data/ACM.mat`
- `/root/workspace/gemini/DBLP.mat -> /mnt/data/DBLP.mat`
- `/root/workspace/gemini/Yelp.mat -> /mnt/data/Yelp.mat`

## What to upload

Do not upload files one by one.

Instead, prepare a preserved folder bundle:

```powershell
python scripts/prepare_dlc_bundle.py
```

Then build a single zip archive:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_dlc_archive.ps1
```

This creates:

- `dlc_bundle/gemini/`
- `dlc_bundle/gemini_bundle.zip`

Upload **`gemini_bundle.zip` as one single code artifact**.

This avoids the DLC uploader flattening your directory tree.

## What the bundle contains

It includes:

- all core experiment scripts
- `scripts/`
- `external/`
- `external_datasets/pyg_hgb/`
- `external_prepared/`
- `external_inputs/`
- existing `results/`
- existing `results_hcfnet/`
- existing `results_external_credible/`

It does **not** copy the three `.mat` datasets into the bundle, because you already mount them separately at `/mnt/data`.

## Suggested code placement in DLC

If the platform lets you choose a code destination, place the uploaded zip at:

- `/root/workspace/gemini_bundle.zip`

The startup script will automatically extract it into:

- `/root/workspace/gemini`

## Suggested resource config

- `1 x NVIDIA A10 24GB`
- `8 vCPU`
- `30 GiB RAM`
- system disk at least `100 GiB`

## Startup command

Use this as the DLC startup command:

```bash
bash /root/workspace/gemini/scripts/run_all_experiments_dlc.sh
```

If you uploaded only the zip and not an extracted folder, use:

```bash
ARCHIVE_PATH=/root/workspace/gemini_bundle.zip WORKDIR=/root/workspace/gemini bash /root/workspace/gemini/scripts/run_all_experiments_dlc.sh
```

## What the startup script does

It will:

1. link `ACM.mat / DBLP.mat / Yelp.mat` from `/mnt/data`
2. print Python and CUDA status
3. install a few missing Python packages
4. regenerate prepared external inputs if needed
5. run baseline methods
6. run `credible_experiment_runner.py`
7. build comparison tables and figures
8. run `scripts/update_degree_discount_adaptive.py`
9. write logs under `dlc_logs/`

## Runtime outputs

Method runtime summaries are written to these locations:

- `results_hcfnet_credible/*_hcf_meta.json`
- `results/HeCo/CUSTOM_*/summary.json`
- `results/CUSTOM_*_rel_GENI/runtime_summary.json`
- `results/CUSTOM_*_two_RGTN/runtime_summary.json`
- `results/CUSTOM_*_EASING/runtime_summary.json`
- `external/LICAP/pretrain/results/CUSTOM_*_two_pregat_struct_pretrain_rgtn/runtime_summary.json`
- `results/MAHE/CUSTOM_*/summary.json`
- `results_external_credible/degree_discount_adaptive_summary.json`
- `results_external_credible/full_comparison.json`
- `results_external_credible/paper_main_metrics_all_methods.json`

## If you only want to test the job first

Use:

```bash
cd /root/workspace/gemini && python credible_experiment_runner.py --datasets ACM --epochs 20
```

after the symlink step, or just temporarily edit the startup command to:

```bash
cd /root/workspace/gemini && python -m pip install --upgrade pip && python -m pip install numpy scipy scikit-learn matplotlib networkx pillow gensim tensorflow-cpu && ln -sf /mnt/data/ACM.mat ACM.mat && python credible_experiment_runner.py --datasets ACM --epochs 20
```

## Recommended workflow

1. Run `python scripts/prepare_dlc_bundle.py` locally
2. Upload `dlc_bundle/gemini/` as the code source
3. Mount your dataset directory to `/mnt/data`
4. Use the startup command above
5. After completion, download:
   - `results_hcfnet_credible/`
   - `results_external_credible/`
   - `results/`
   - `dlc_logs/`

# External Baseline Feasibility

## What is fixed

- `prepare_external_nie_data.py` now prepares true heterogeneous inputs for `ACM`, `DBLP`, and `Yelp`.
- `ACM` and `DBLP` use raw heterogeneous relations from `torch_geometric.datasets.HGBDataset`.
- `Yelp` uses the raw multi-type relations already stored in `Yelp.mat`.
- The prepared files are written to `external_prepared/<dataset>/`.
- `RGTN-NIE` was patched to accept custom `.pt/.pk` files with:
  - `edges`
  - `edge_types`
  - `features`
  - `semantic_features`
  - `labels`
  - `invalid_masks`
  - explicit `train_idx/val_idx/test_idx`
- `EASING` was patched to accept the same custom payload and to run on CPU in the local `rgtn310` environment.

## Local smoke-test status

- `RGTN-NIE` smoke test passed on `ACM` with:
  - env: `D:\App\Anaconda\envs\rgtn310\python.exe`
  - dataset: `external_prepared/ACM/acm_nie.pt`
  - mode: CPU
- `EASING` smoke test passed on `ACM` with the same environment and dataset.
- This means the main blocker is no longer data-format incompatibility. The next step is full training and evaluation, not reverse-engineering inputs.

## Recommended recent baselines

- `GENI` (KDD 2020)
  - already included inside `external/RGTN-NIE`
  - same NIE problem family
  - easiest extra baseline after `RGTN`
- `RGTN-NIE` (KDD 2021)
  - already local
  - now adapted to custom heterogeneous inputs
- `LICAP` (TNNLS 2024)
  - local repo: `external/LICAP`
  - built on top of RGTN-style NIE datasets
  - good next deep baseline after `RGTN`
- `EASING` (WWW 2025)
  - already local
  - now adapted to custom heterogeneous inputs
- `LENIE` (Knowledge-Based Systems 2025)
  - local repo: `external/LENIE`
  - not a standalone ranker; it is an LLM semantic augmentation pipeline for downstream NIE models
  - use only if we want a stronger semantic baseline and can afford extra preprocessing

## Optional but weaker-fit baselines

- `MAHE-IM`
  - now feasible in principle because we can export tripartite edge files:
    - `external_prepared/ACM/acm_mahe_edges.txt`
    - `external_prepared/DBLP/dblp_mahe_edges.txt`
    - `external_prepared/Yelp/yelp_mahe_edges.txt`
  - but it is an influence maximization method, not the cleanest match for your current NIE framing
  - treat it as auxiliary, not the main learned baseline

## Not recommended right now

- `MKNI`
  - I have not verified a public reproducible official codebase locally or online
  - do not put it into the main experimental table unless you implement it yourself
- `PINE`
  - paper exists, but I have not verified an official runnable implementation
  - same advice as above

## Commands to continue locally

- Prepare all custom NIE inputs:
  - `python prepare_external_nie_data.py`

- RGTN smoke/full run example on ACM:
  - `D:\App\Anaconda\envs\rgtn310\python.exe external\RGTN-NIE\two_branch\two_branch_batch_train.py --dataset CUSTOM_ACM_two --data_path d:\Graduate\code\gemini\external_prepared\ACM\acm_nie.pt --cross-num 1 --gpu -1 --epochs 50 --batch-size 1024 --num-workers 0 --save-path acm_checkpoint.pt`

- EASING smoke/full run example on ACM:
  - `D:\App\Anaconda\envs\rgtn310\python.exe external\EASING\easing\main_easing.py --dataset CUSTOM_ACM --data_path d:/Graduate/code/gemini --graph_data external_prepared/ACM/acm_nie.pt --semantic_data placeholder.pk --structure_data placeholder.pk --cross-num 1 --gpu -1 --epochs 50 --train_num 1.0 --samp_ssl 5 --unc_layers 1 --save-path acm_checkpoint.pt`

## Practical next order

- Run `RGTN` on `ACM`, `DBLP`, `Yelp`.
- Run `GENI` on the same prepared inputs.
- Run `EASING` on the same prepared inputs.
- If the numbers are stable, add `LICAP`.
- Only after that decide whether `LENIE` is worth the extra cost.

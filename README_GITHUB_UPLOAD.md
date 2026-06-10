# Gemini Code Bundle

This folder is a curated code-only bundle for uploading to GitHub.

## Included

- Core experiment drivers
- Figure/table build scripts
- Cloud/DLC helper scripts
- External baseline source trees needed by the pipeline:
  - `external/EASING`
  - `external/HeCo`
  - `external/LICAP`
  - `external/MAHE-IM`
  - `external/RGTN-NIE`
  - `external/ToupleGDD`
- `scripts/` helpers for runtime summary, DLC/ECS running, and Degree Discount / Adaptive Degree updates

## Not included

- Large datasets such as `ACM.mat`, `DBLP.mat`, `Yelp.mat`
- Result folders and cached outputs
- Model checkpoints, `.npy`, `.npz`, `.pt`, `.pkl`
- Most bulky raw data under external projects

## You still need separately

- `ACM.mat`
- `DBLP.mat`
- `Yelp.mat`
- Any required mounted data path such as `/mnt/data`

## Recommended GitHub use

Upload this folder as one repository tree so directory structure is preserved.

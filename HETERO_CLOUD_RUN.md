# Heterogeneous Cloud Run

This repo now evaluates diffusion on the full heterogeneous graph while still ranking target-type nodes.

## Required files

Put these files under the repo root after `git clone`:

- `ACM.mat`
- `DBLP.mat`
- `Yelp.mat`

Put these prepared heterogeneous graph files under:

- `external_prepared/ACM/acm_nie.pt`
- `external_prepared/DBLP/dblp_nie.pt`
- `external_prepared/Yelp/yelp_nie.pt`

If your cloud platform mounts data under `/mnt/data`, you can link them instead of copying:

```bash
ln -sfn /mnt/data/ACM.mat ACM.mat
ln -sfn /mnt/data/DBLP.mat DBLP.mat
ln -sfn /mnt/data/Yelp.mat Yelp.mat
mkdir -p external_prepared/ACM external_prepared/DBLP external_prepared/Yelp
ln -sfn /mnt/data/external_prepared/ACM/acm_nie.pt external_prepared/ACM/acm_nie.pt
ln -sfn /mnt/data/external_prepared/DBLP/dblp_nie.pt external_prepared/DBLP/dblp_nie.pt
ln -sfn /mnt/data/external_prepared/Yelp/yelp_nie.pt external_prepared/Yelp/yelp_nie.pt
```

## Full comparison run

```bash
python run_heco_baseline.py --datasets ACM DBLP Yelp
python run_external_baselines.py --datasets ACM DBLP Yelp --methods GENI RGTN EASING
python run_licap_baseline.py --datasets ACM DBLP Yelp --gpu 0 --pretrain-epochs 3 --downstream-epochs 3
python run_mahe_baseline.py --datasets ACM DBLP Yelp
python credible_experiment_runner.py --datasets ACM DBLP Yelp --epochs 180
python build_full_comparison_artifacts.py
python scripts/update_degree_discount_adaptive.py
python build_paper_curve_plots.py
python build_paper_tables.py
python build_combined_diffusion_panels.py
python scripts/summarize_all_runtimes.py
```

## Sensitivity run

```bash
python credible_sensitivity_runner.py --datasets ACM DBLP Yelp --epochs 20 --sim-runs 4 --t-steps 20
```

## Main outputs

- `results_external_credible/full_comparison.json`
- `results_external_credible/all_method_runtimes.json`
- `results_external_credible/all_method_runtimes.md`
- `results_external_credible/sensitivity/`
- `results_hcfnet_credible/*_hcf_meta.json`

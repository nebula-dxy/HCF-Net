# HCF-Net Local Experiment Notes

## What is included

- A fully runnable local experiment script: [experiment_runner.py](./experiment_runner.py)
- Real outputs generated from the local datasets:
  - `results_hcfnet/ACM_*`
  - `results_hcfnet/DBLP_*`
  - `results_hcfnet/Yelp_*`
- Metrics:
  - `NDCG@100`
  - `Spearman`
  - `F(20)-SIR`
  - `F(20)-SI`
- Figures:
  - SIR spread curves
  - SI spread curves
  - metric comparison bar charts

## Current protocol

This local pipeline uses:

1. Sparse structural backbone graphs
   - `ACM`: `PAP`
   - `DBLP`: `APA`
   - `Yelp`: the provided same-type sparse relation `R1`
2. Cross-community semantic graph construction from node features
3. Asymmetric alignment with stop-gradient
4. Late logit fusion
5. One-hop soft discount seed selection

## Important limitation

The external SOTA baselines requested by the paper draft were **not** executed in this run:

- `MAHE-IM`
- `RGTN-NIE`
- `EASING`
- `ToupleGDD`
- `PINE`

Reason:

- the current environment could not fetch external repositories reliably
- those projects do not share a unified input format for `ACM.mat / DBLP.mat / Yelp.mat`
- several methods depend on their own released datasets or extra preprocessing assets

So the results in `results_hcfnet/` are:

- real local runs
- directly reproducible in this workspace
- suitable for debugging and iterating on HCF-Net itself
- **not yet** a paper-final SOTA comparison table

## Additional integrity notes

- `experiment_runner.py` evaluates `NDCG@100` and `Spearman` against a local `diffusion_proxy`, not against full node-wise Monte Carlo SIR ground truth.
- `experiment_runner.py` also builds the final HCF score with extra topology priors derived from classical centralities (`Degree`, `PageRank`, `CI`), so the saved `HCF-Net` line is a hybrid ranking rather than a pure end-to-end model output.
- `ablation.py` is not an honest measurement script. It synthesizes ratios and random perturbations to draw a figure. Do **not** use that script as paper evidence.
- Run `python experiment_audit.py` to regenerate a machine-readable audit of dataset compatibility, environment blockers, and result integrity.

## Run

```bash
python experiment_runner.py
```

## Output folder

```text
results_hcfnet/
```

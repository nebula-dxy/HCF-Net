# HCF-Net

This repository now includes a compact heterogeneous influence-ranking pipeline in `hetero_fast/`.

Quick start:

```bash
python -m hetero_fast.run --datasets ACM DBLP Yelp --gpu 0 --with-sensitivity --truth-mc-runs 2
```

Main entry docs:

- `HETERO_FAST_RUN.md`

The compact pipeline:

- keeps the paper-style HCF logic
- uses Monte Carlo heterogeneous SI/SIR diffusion as truth
- re-evaluates external baselines under the same heterogeneous diffusion setting
- exports metrics, curves, runtime summaries, and sensitivity results

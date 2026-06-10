# Experiment Audit

## Summary

- Current local results exist and are reproducible in this workspace.
- Current local results are not yet a fair paper-final SOTA comparison against MAHE-IM / RGTN-NIE / EASING / ToupleGDD / PINE.
- One ablation script currently synthesizes numbers and should not be used as evidence.

## Dataset Snapshot

### ACM

- Nodes: 3025
- Feature dim: 1870
- Relation `PTP`: shape=(3025, 3025), nnz=9150593, density=0.999997
- Relation `PLP`: shape=(3025, 3025), nnz=2210761, density=0.241597
- Relation `PAP`: shape=(3025, 3025), nnz=29281, density=0.003200
- Note: HAN-style projected paper graph.
- Note: Raw typed author/paper/category edge list is not exposed in this .mat file.

### DBLP

- Nodes: 4057
- Feature dim: 334
- Relation `APA`: shape=(4057, 4057), nnz=11113, density=0.000675
- Relation `APTPA`: shape=(4057, 4057), nnz=6772278, density=0.411457
- Relation `APCPA`: shape=(4057, 4057), nnz=5000495, density=0.303811
- Note: HAN-style projected author graph.
- Note: Raw typed author/paper/term/venue edges expected by some external baselines are not present.

### Yelp

- Nodes: 16239
- Feature dim: 14295
- Relation `R0`: shape=(16239, 14284), nnz=198397, density=0.000855
- Relation `R1`: shape=(16239, 16239), nnz=158590, density=0.000601
- Relation `R2`: shape=(16239, 11), nnz=76875, density=0.430361
- Relation `R3`: shape=(14284, 511), nnz=40009, density=0.005481
- Relation `R4`: shape=(14284, 47), nnz=14267, density=0.021251
- Note: HIN package with multiple relation blocks.
- Note: Current local pipeline projects Yelp to a same-type graph plus derived features.

## Baseline Feasibility

### MAHE-IM

- Repo present: True
- Runnable here: False
- Dataset compatible: False
- Blocker: Needs raw typed a/p/c heterogeneous edges; current ACM.mat and DBLP.mat only expose projected graphs and do not share MAHE-IM's original node-id space.

### RGTN-NIE

- Repo present: True
- Runnable here: False
- Dataset compatible: False
- Blocker: README expects released NIE datasets plus working DGL 0.5.3; local ACM/DBLP/Yelp files are not that format and local DGL import is broken.

### EASING

- Repo present: True
- Runnable here: False
- Dataset compatible: False
- Blocker: README expects released .pk relation/semantic packages plus working DGL 0.6.1; current datasets and environment do not satisfy that contract.

### ToupleGDD

- Repo present: True
- Runnable here: True
- Dataset compatible: True
- Blocker: Can be adapted to plain projected graphs, but it is a generic influence-maximization RL baseline rather than an official attributed-network node-importance pipeline on ACM/DBLP/Yelp.

### PINE

- Repo present: False
- Runnable here: False
- Dataset compatible: False
- Blocker: Only the paper reference is available locally; no official code is present in this workspace.

## Integrity Checks

- [warning] Ranking target in experiment_runner.py
  Evidence: experiment_runner.py:291 uses compute_diffusion_proxy(...) instead of per-node Monte Carlo SIR ground truth.
  Impact: NDCG@100 and Spearman are currently proxy-aligned rather than directly aligned to true node-level diffusion influence.
- [warning] HCF-Net final score construction
  Evidence: experiment_runner.py:535-558 mixes HCF outputs with Degree/PageRank/CI inside build_topology_prior(...) and fusion candidates.
  Impact: The reported HCF-Net ranking is not a pure model output; it is a hybrid score that already absorbs baseline signals.
- [warning] External SOTA comparison
  Evidence: EXPERIMENT_NOTES.md states MAHE-IM / RGTN-NIE / EASING / ToupleGDD / PINE were not executed in the saved local run.
  Impact: Current summary.json is not a paper-final SOTA comparison table.
- [error] Ablation script authenticity
  Evidence: ablation.py:17-46 constructs metric ratios and injects random noise instead of reading measured experiment outputs.
  Impact: That script should not be used as evidence in a paper or rebuttal.

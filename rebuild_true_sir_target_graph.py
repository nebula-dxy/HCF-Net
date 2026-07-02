import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp

import credible_experiment_runner as credible
import experiment_runner as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_hcfnet_credible" / "true_sir"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORK_GRAPH = None
WORK_BETA = 0.0
WORK_GAMMA = 0.0
WORK_RUNS = 0
WORK_T_STEPS = 0


def build_target_graph(dataset: str) -> Tuple[nx.Graph, dict]:
    bundle = base.load_dataset(dataset)
    weighted = credible.feature_weighted_graph(bundle.adjacency, bundle.features, floor=0.15)
    graph = nx.from_scipy_sparse_array(weighted)
    cfg = credible.epidemic_config(dataset, weighted)
    return graph, cfg


def simulate_single_seed(graph: nx.Graph, beta: float, gamma: float, seed: int, t_steps: int) -> float:
    status = {n: 0 for n in graph.nodes()}
    infected = {seed}
    recovered = set()
    status[seed] = 1
    for _ in range(t_steps):
        new_infected = set()
        new_recovered = set()
        for node in infected:
            for neigh in graph.neighbors(node):
                if status[neigh] != 0:
                    continue
                weight = float(graph[node][neigh].get("weight", 1.0))
                if random.random() < min(0.98, beta * weight):
                    new_infected.add(neigh)
            if random.random() < gamma:
                new_recovered.add(node)
        for node in new_infected:
            status[node] = 1
        for node in new_recovered:
            status[node] = 2
        infected.update(new_infected)
        infected.difference_update(new_recovered)
        recovered.update(new_recovered)
    return float((len(infected) + len(recovered)) / graph.number_of_nodes())


def init_worker(graph: nx.Graph, beta: float, gamma: float, runs: int, t_steps: int) -> None:
    global WORK_GRAPH, WORK_BETA, WORK_GAMMA, WORK_RUNS, WORK_T_STEPS
    WORK_GRAPH = graph
    WORK_BETA = float(beta)
    WORK_GAMMA = float(gamma)
    WORK_RUNS = int(runs)
    WORK_T_STEPS = int(t_steps)


def score_seed(seed: int) -> Tuple[int, float]:
    vals: List[float] = []
    for run_idx in range(WORK_RUNS):
        random.seed(credible.SEED + 1000003 * int(seed) + run_idx)
        vals.append(simulate_single_seed(WORK_GRAPH, WORK_BETA, WORK_GAMMA, int(seed), WORK_T_STEPS))
    return int(seed), float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild true SIR scores on the target-type graph")
    parser.add_argument("--dataset", type=str, default="ACM")
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--t-steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    args = parser.parse_args()

    started = time.perf_counter()
    graph, cfg = build_target_graph(args.dataset)
    beta = float(np.clip(float(cfg["sir_beta"]) * args.beta_scale, 0.001, 0.25))
    gamma = float(cfg["sir_gamma"])
    n = graph.number_of_nodes()
    scores = np.zeros(n, dtype=np.float32)

    print(
        f"start target-graph true SIR: dataset={args.dataset} nodes={n} "
        f"runs={args.runs} t_steps={args.t_steps} workers={args.workers} beta={beta:.6f}"
    )
    with Pool(processes=max(1, int(args.workers)), initializer=init_worker, initargs=(graph, beta, gamma, args.runs, args.t_steps)) as pool:
        for finished, (node, score) in enumerate(pool.imap_unordered(score_seed, range(n), chunksize=16), start=1):
            scores[node] = score
            if finished % 100 == 0 or finished == n:
                print(f"done={finished}/{n} latest_node={node+1} score={score:.6f} elapsed={time.perf_counter()-started:.1f}s")

    out_path = OUT_DIR / f"{args.dataset}_true_sir_scores_t{args.t_steps}_runs{args.runs}_beta{args.beta_scale:g}.npy"
    meta_path = OUT_DIR / f"{args.dataset}_true_sir_scores_t{args.t_steps}_runs{args.runs}_beta{args.beta_scale:g}_meta.json"
    np.save(out_path, scores)
    meta = {
        "dataset": args.dataset,
        "definition": "single-seed true SIR score on target-type graph",
        "runs": int(args.runs),
        "t_steps": int(args.t_steps),
        "sir_beta": beta,
        "sir_gamma": gamma,
        "num_nodes": n,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "top10_nodes": np.argsort(scores)[::-1][:10].astype(int).tolist(),
        "top10_scores": scores[np.argsort(scores)[::-1][:10]].astype(float).tolist(),
        "total_runtime_sec": float(time.perf_counter() - started),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved", out_path)
    print("saved", meta_path)


if __name__ == "__main__":
    main()

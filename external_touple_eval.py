import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch

import experiment_runner as er


ROOT = Path(__file__).resolve().parent
TOUPLE_DIR = ROOT / "external" / "ToupleGDD"
RESULTS_DIR = ROOT / "results_hcfnet"
EDGE_DIR = ROOT / "external_inputs"


@dataclass
class ToupleConfig:
    budget: int
    epochs: int
    num_trial: int
    num_process: int
    batch_size: int
    init_embed_epochs_train: int
    init_embed_epochs_test: int
    pretrain_games: int


def get_config(dataset_name: str) -> ToupleConfig:
    if dataset_name == "ACM":
        return ToupleConfig(budget=20, epochs=4, num_trial=600, num_process=2, batch_size=4, init_embed_epochs_train=3, init_embed_epochs_test=0, pretrain_games=1000)
    if dataset_name == "DBLP":
        return ToupleConfig(budget=50, epochs=4, num_trial=600, num_process=2, batch_size=4, init_embed_epochs_train=2, init_embed_epochs_test=0, pretrain_games=1000)
    return ToupleConfig(budget=15, epochs=1, num_trial=80, num_process=1, batch_size=1, init_embed_epochs_train=0, init_embed_epochs_test=0, pretrain_games=120)


def ensure_touple_graph(bundle: er.DatasetBundle) -> Tuple[Path, np.ndarray]:
    edge_path = EDGE_DIR / f"{bundle.name}_touple_edges.txt"
    map_path = EDGE_DIR / f"{bundle.name}_touple_nodes.npy"
    if edge_path.exists() and map_path.exists():
        return edge_path, np.load(map_path)

    adj = sp.csr_matrix(bundle.adjacency)
    degree = np.asarray(adj.sum(axis=1)).reshape(-1)
    active_nodes = np.where(degree > 0)[0].astype(np.int64)
    active_adj = adj[active_nodes][:, active_nodes].tocsr()
    coo = sp.triu(active_adj, k=1).tocoo()

    with open(edge_path, "w", encoding="utf-8") as f:
        for u, v in zip(coo.row, coo.col):
            f.write(f"{int(u)} {int(v)}\n")

    np.save(map_path, active_nodes)
    return edge_path, active_nodes


def latest_new_run_dir(before: set[str]) -> Path:
    candidates = [p for p in TOUPLE_DIR.iterdir() if p.is_dir() and p.name not in before]
    if not candidates:
        raise RuntimeError("Could not locate ToupleGDD output directory after training.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def train_external_model(dataset_name: str, model_name: str, graph_path: Path, cfg: ToupleConfig) -> Path:
    before = {p.name for p in TOUPLE_DIR.iterdir() if p.is_dir()}
    model_file = f"{dataset_name.lower()}_{model_name.lower()}.ckpt"
    cmd = [
        sys.executable,
        "main.py",
        "--graph",
        str(graph_path),
        "--model",
        model_name,
        "--model_file",
        model_file,
        "--budget",
        str(cfg.budget),
        "--epoch",
        str(cfg.epochs),
        "--lr",
        "0.001",
        "--bs",
        str(cfg.batch_size),
        "--n_step",
        "1",
        "--cpu",
        "--num_process",
        str(cfg.num_process),
        "--num_trial",
        str(cfg.num_trial),
        "--init_embed_epochs_train",
        str(cfg.init_embed_epochs_train),
        "--init_embed_epochs_test",
        str(cfg.init_embed_epochs_test),
        "--pretrain_games",
        str(cfg.pretrain_games),
    ]
    print(f"[ToupleGDD] training {dataset_name} {model_name}")
    subprocess.run(cmd, cwd=TOUPLE_DIR, check=True)
    run_dir = latest_new_run_dir(before)
    checkpoint = run_dir / model_file
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint}")
    return checkpoint


def load_external_agent(model_name: str, checkpoint: Path):
    sys.path.insert(0, str(TOUPLE_DIR))
    import rl_agents  # type: ignore

    args = SimpleNamespace()
    args.model = model_name
    args.gamma = 0.99
    args.n_step = 1
    args.test = True
    args.double_dqn = True
    args.device = torch.device("cpu")
    args.bs = 1
    args.memory_size = 1
    args.reg_hidden = 32
    args.lr = 0.001
    args.T = 3
    args.embed_dim = 50 if model_name == "Tripling" else 64
    args.model_file = str(checkpoint)
    args.init_embed_epochs_train = 0
    args.init_embed_epochs_test = 0
    agent = rl_agents.Agent(args)
    return agent


def external_graph(graph_path: Path):
    sys.path.insert(0, str(TOUPLE_DIR))
    import utils.graph_utils as graph_utils  # type: ignore

    return graph_utils.read_graph(str(graph_path), ind=0, directed=True)


def infer_scores_and_seeds(agent, graph, budget: int) -> Tuple[np.ndarray, List[int]]:
    state = torch.zeros(graph.num_nodes, dtype=torch.float32)
    graph_input = agent.setup_graph_input([graph], state.unsqueeze(dim=0))
    with torch.no_grad():
        q_all = agent.model(graph_input).squeeze(dim=1).cpu().numpy()

    seeds: List[int] = []
    rollout_state = torch.zeros(graph.num_nodes, dtype=torch.float32)
    for _ in range(budget):
        action = int(agent.select_action(graph, rollout_state, epsilon=0.0, training=False).item())
        seeds.append(action)
        rollout_state[action] = 1.0
    return q_all, seeds


def seed_rank_scores(num_nodes: int, seeds: List[int], base_value: float = 0.0) -> np.ndarray:
    scores = np.full(num_nodes, float(base_value), dtype=np.float32)
    top = float(len(seeds) + 1)
    for idx, node in enumerate(seeds):
        scores[int(node)] = top - idx
    return scores


def evaluate_external(
    dataset_name: str,
    model_name: str,
    checkpoint: Path,
    q_scores: np.ndarray,
    seeds: List[int],
    active_nodes: np.ndarray,
) -> Dict[str, float]:
    bundle = er.load_dataset(dataset_name)
    communities = er.build_communities(bundle.adjacency)
    sem_graph = er.build_cross_community_semantic_graph(bundle.features, communities, k=12)
    _, _, rank_target = er.compute_mean_field_targets(bundle.name, bundle.adjacency, bundle.features, sem_graph)

    full_scores = np.full(bundle.adjacency.shape[0], float(np.min(q_scores) - 1.0), dtype=np.float32)
    full_scores[active_nodes] = q_scores
    mapped_seeds = [int(active_nodes[s]) for s in seeds]

    rows = er.evaluate_rankings(rank_target, er.minmax_scale(full_scores), bundle.test_idx, k=100)
    diffusion_cfg = er.get_diffusion_config(dataset_name)
    graph = er.nx.from_scipy_sparse_array(bundle.adjacency)
    sir = er.SIRSimulation(graph, beta=float(diffusion_cfg["sir_beta"]), gamma=float(diffusion_cfg["sir_gamma"]))
    si = er.SISimulation(graph, beta=float(diffusion_cfg["si_beta"]))
    rows["F(20)-SIR"] = float(er.average_curve(sir, mapped_seeds[: diffusion_cfg["seed_k"]], runs=8, t_steps=20)[-1])
    rows["F(20)-SI"] = float(er.average_curve(si, mapped_seeds[: diffusion_cfg["seed_k"]], runs=8, t_steps=20)[-1])
    rows["budget"] = int(diffusion_cfg["seed_k"])
    rows["checkpoint"] = str(checkpoint)
    return rows


def run_one(dataset_name: str, model_name: str) -> Dict[str, float]:
    bundle = er.load_dataset(dataset_name)
    edge_path, active_nodes = ensure_touple_graph(bundle)
    cfg = get_config(dataset_name)
    checkpoint = train_external_model(dataset_name, model_name, edge_path, cfg)
    agent = load_external_agent(model_name, checkpoint)
    graph = external_graph(edge_path)
    q_scores, seeds = infer_scores_and_seeds(agent, graph, cfg.budget)
    # Touple agents can emit nearly flat one-shot Q values while the actual
    # sequential rollout remains informative. In that case, persist the rollout
    # order as ranking scores so downstream unified evaluation uses the method's
    # real selected seed set instead of an almost-constant score vector.
    if float(np.std(q_scores)) < 1e-6:
        q_scores = seed_rank_scores(graph.num_nodes, seeds, base_value=float(np.min(q_scores) - 1.0))
    rows = evaluate_external(dataset_name, model_name, checkpoint, q_scores, seeds, active_nodes)

    out_path = RESULTS_DIR / f"{dataset_name}_{model_name}_external.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "model": model_name,
                "metrics": rows,
                "seeds_reindexed": seeds,
                "seeds_original": [int(active_nodes[s]) for s in seeds],
            },
            f,
            indent=2,
        )
    full_scores = np.full(bundle.adjacency.shape[0], float(np.min(q_scores) - 1.0), dtype=np.float32)
    full_scores[active_nodes] = q_scores
    np.save(RESULTS_DIR / f"{dataset_name}_{model_name}_q_scores.npy", full_scores)
    print(f"Saved external results to: {out_path}")
    return rows


def merge_into_summary(dataset_name: str, model_name: str, metrics: Dict[str, float]) -> None:
    metrics_path = RESULTS_DIR / f"{dataset_name}_metrics.json"
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows[f"Touple-{model_name}"] = {
        "Spearman": float(metrics["Spearman"]),
        "NDCG@100": float(metrics["NDCG@100"]),
        "F(20)-SIR": float(metrics["F(20)-SIR"]),
        "F(20)-SI": float(metrics["F(20)-SI"]),
    }
    metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    tasks = [
        ("ACM", "S2V_DQN"),
        ("ACM", "Tripling"),
        ("DBLP", "S2V_DQN"),
        ("DBLP", "Tripling"),
    ]
    summary: Dict[str, Dict[str, float]] = {}
    for dataset_name, model_name in tasks:
        metrics = run_one(dataset_name, model_name)
        summary[f"{dataset_name}:{model_name}"] = metrics
        merge_into_summary(dataset_name, model_name, metrics)

    out_path = RESULTS_DIR / "external_touple_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved external baseline summary to: {out_path}")


if __name__ == "__main__":
    main()

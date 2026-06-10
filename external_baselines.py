import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import networkx as nx
import scipy.sparse as sp

from experiment_runner import ROOT, ensure_csr, load_dataset


EXTERNAL_DIR = ROOT / "external"
EXTERNAL_INPUT_DIR = ROOT / "external_inputs"
EXTERNAL_INPUT_DIR.mkdir(exist_ok=True)


@dataclass
class BaselineStatus:
    name: str
    repo_path: str
    downloaded: bool
    runnable: bool
    requires: List[str]
    dataset_ready: bool
    blocker: str


def _write_edgelist(dataset_name: str, adj: sp.csr_matrix) -> str:
    path = EXTERNAL_INPUT_DIR / f"{dataset_name}_projected_edges.txt"
    graph = nx.from_scipy_sparse_array(ensure_csr(adj).sign())
    nx.write_edgelist(graph, path, data=False)
    return str(path)


def _write_status_file(statuses: Dict[str, BaselineStatus]) -> None:
    with open(EXTERNAL_INPUT_DIR / "external_status.json", "w", encoding="utf-8") as f:
        json.dump({k: asdict(v) for k, v in statuses.items()}, f, indent=2)


def prepare_external_inputs() -> Dict[str, str]:
    paths = {}
    for dataset_name in ["ACM", "DBLP", "Yelp"]:
        bundle = load_dataset(dataset_name)
        paths[dataset_name] = _write_edgelist(dataset_name, bundle.adjacency)
    with open(EXTERNAL_INPUT_DIR / "prepared_inputs.json", "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)
    return paths


def inspect_external_baselines() -> Dict[str, BaselineStatus]:
    statuses = {
        "RGTN-NIE": BaselineStatus(
            name="RGTN-NIE",
            repo_path=str(EXTERNAL_DIR / "RGTN-NIE"),
            downloaded=(EXTERNAL_DIR / "RGTN-NIE").exists(),
            runnable=False,
            requires=["dgl>=0.5", "raw heterogeneous graph pk files", "official split format"],
            dataset_ready=False,
            blocker="Current ACM/DBLP .mat files are projected meta-path graphs, not the raw heterogeneous relation package expected by RGTN-NIE.",
        ),
        "EASING": BaselineStatus(
            name="EASING",
            repo_path=str(EXTERNAL_DIR / "EASING"),
            downloaded=(EXTERNAL_DIR / "EASING").exists(),
            runnable=False,
            requires=["dgl>=0.6", "official pk dataset files", "split folders"],
            dataset_ready=False,
            blocker="The installed DGL wheel is not usable here, and the current datasets do not match EASING's required serialized relation/semantic feature package.",
        ),
        "ToupleGDD": BaselineStatus(
            name="ToupleGDD",
            repo_path=str(EXTERNAL_DIR / "ToupleGDD"),
            downloaded=(EXTERNAL_DIR / "ToupleGDD").exists(),
            runnable=False,
            requires=["torch_scatter", "PyG-compatible checkpoint/runtime"],
            dataset_ready=True,
            blocker="Projected graph edge lists can be exported, but torch_scatter is still missing so the RL model cannot run yet.",
        ),
        "MAHE-IM": BaselineStatus(
            name="MAHE-IM",
            repo_path=str(EXTERNAL_DIR / "MAHE-IM"),
            downloaded=(EXTERNAL_DIR / "MAHE-IM").exists(),
            runnable=False,
            requires=["gensim", "raw tripartite heterogeneous edges with a/p/c node types"],
            dataset_ready=False,
            blocker="ACM.mat and DBLP.mat do not contain the original author-paper-category tripartite edge list needed by MAHE-IM.",
        ),
        "PINE": BaselineStatus(
            name="PINE",
            repo_path="",
            downloaded=False,
            runnable=False,
            requires=["official code release or reproducible implementation details"],
            dataset_ready=False,
            blocker="I have not found a local official implementation to wire in yet.",
        ),
    }
    _write_status_file(statuses)
    return statuses


def main() -> None:
    inputs = prepare_external_inputs()
    statuses = inspect_external_baselines()
    print("Prepared projected graph inputs:")
    for name, path in inputs.items():
        print(f"  {name}: {path}")
    print("\nExternal baseline status:")
    for name, status in statuses.items():
        print(f"  {name}: downloaded={status.downloaded}, runnable={status.runnable}")
        print(f"    blocker: {status.blocker}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import gensim
import numpy as np
import tensorflow as tf
import torch
import experiment_runner as base
import credible_experiment_runner as credible

ROOT = Path(__file__).resolve().parent
MAHE_CODE = ROOT / "external" / "MAHE-IM" / "code"
import sys
sys.path.insert(0, str(MAHE_CODE))

from model import MetaPathGenerator, metapath2vec  # noqa: E402

tf.compat.v1.disable_eager_execution()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


METAPATH_SPECS = [
    ("aaa", "a", ("generate_random_aa", 2, 2), 0.5 / 3.0),
    ("aca", "ac", ("generate_random_aca", 2, 1), 0.5 / 3.0),
    ("apa", "ap", ("generate_random_apa", 2, 1), 0.5 / 3.0),
    ("aaaa", "a", ("generate_random_aa", 2, 3), (0.5 ** 2) / 4.0),
    ("acaa", "ac", ("generate_random_acaa", 2, 1), (0.5 ** 2) / 4.0),
    ("apaa", "ap", ("generate_random_apaa", 2, 1), (0.5 ** 2) / 4.0),
    ("apca", "apc", ("generate_random_apca", 2, 1), (0.5 ** 2) / 4.0),
    ("aaaaa", "a", ("generate_random_aa", 2, 4), (0.5 ** 3) / 5.0),
    ("apaca", "apc", ("generate_random_apaca", 2, 1), (0.5 ** 3) / 5.0),
    ("apapa", "ap", ("generate_random_apa", 2, 2), (0.5 ** 3) / 5.0),
    ("acaca", "ac", ("generate_random_aca", 2, 2), (0.5 ** 3) / 5.0),
    ("apcpa", "apc", ("generate_random_apcpa", 2, 1), (0.5 ** 3) / 5.0),
]


def normalize_pairs(pairs):
    return sorted(set((int(s), int(d)) for s, d in pairs if int(s) != int(d)))


def to_map(pairs):
    out = defaultdict(set)
    for s, d in pairs:
        out[int(s)].add(int(d))
    return out


def compose(left_mid, mid_right):
    left_mid_map = to_map(left_mid)
    mid_right_map = to_map(mid_right)
    out = set()
    for left, mids in left_mid_map.items():
        for mid in mids:
            for right in mid_right_map.get(mid, ()):
                if left != right:
                    out.add((left, right))
    return sorted(out)


def stringify_pairs(pairs, src_prefix, dst_prefix):
    return [f"{src_prefix}{s} {dst_prefix}{d}" for s, d in normalize_pairs(pairs)]


def load_local_relations(dataset: str):
    payload = base.safe_torch_load(ROOT / "external_prepared" / dataset / f"{dataset.lower()}_nie.pt")
    src_all = np.asarray(payload["edges"][0]).reshape(-1)
    dst_all = np.asarray(payload["edges"][1]).reshape(-1)
    edge_types = np.asarray(payload["edge_types"]).reshape(-1)
    relation_names = list(payload["relation_names"])

    type_nodes = defaultdict(set)
    rel_pairs = {}
    rel_meta = {}
    for rel_id, rel_name in enumerate(relation_names):
        src_type, _, dst_type = rel_name.split(":", 2)
        mask = edge_types == rel_id
        src = src_all[mask]
        dst = dst_all[mask]
        rel_pairs[rel_name] = list(zip(src.tolist(), dst.tolist()))
        rel_meta[rel_name] = (src_type, dst_type)
        type_nodes[src_type].update(src.tolist())
        type_nodes[dst_type].update(dst.tolist())

    type_min = {name: min(ids) for name, ids in type_nodes.items()}
    local_rel = {}
    for rel_name, pairs in rel_pairs.items():
        src_type, dst_type = rel_meta[rel_name]
        local_rel[rel_name] = [(s - type_min[src_type], d - type_min[dst_type]) for s, d in pairs]
    return payload, local_rel


def build_dataset_relations(dataset: str):
    _, rel = load_local_relations(dataset)
    if dataset == "ACM":
        a_a = rel.get("paper:cite:paper", []) + rel.get("paper:ref:paper", [])
        p_a = rel.get("author:to:paper", []) or [(d, s) for s, d in rel.get("paper:to:author", [])]
        a_c = rel.get("paper:to:subject", []) or [(d, s) for s, d in rel.get("subject:to:paper", [])]
        p_c = compose(p_a, a_c)
    elif dataset == "DBLP":
        a_to_p = rel.get("author:to:paper", []) or [(d, s) for s, d in rel.get("paper:to:author", [])]
        p_a = rel.get("paper:to:author", []) or [(d, s) for s, d in a_to_p]
        p_c = rel.get("paper:to:venue", []) or [(d, s) for s, d in rel.get("venue:to:paper", [])]
        a_a = compose(a_to_p, p_a)
        a_c = compose(a_to_p, p_c)
    elif dataset == "Yelp":
        a_to_p = rel.get("user:user_to_business:business", []) or [(d, s) for s, d in rel.get("business:rev_user_to_business:user", [])]
        p_a = rel.get("business:rev_user_to_business:user", []) or [(d, s) for s, d in a_to_p]
        p_c = rel.get("business:business_to_category:category", []) or [(d, s) for s, d in rel.get("category:rev_business_to_category:business", [])]
        a_a = rel.get("user:user_to_user:user", [])
        a_c = compose(a_to_p, p_c)
    else:
        raise ValueError(dataset)

    return (
        stringify_pairs(a_a, "a", "a"),
        stringify_pairs(p_a, "p", "a"),
        stringify_pairs(p_c, "p", "c"),
        stringify_pairs(a_c, "a", "c"),
    )


def make_workdir(dataset: str) -> Path:
    workdir = ROOT / "results" / "MAHE" / f"CUSTOM_{dataset}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def existing_workdir(dataset: str) -> Path:
    workdir = ROOT / "results" / "MAHE" / f"CUSTOM_{dataset}"
    if not workdir.exists():
        raise FileNotFoundError(f"MAHE workdir does not exist: {workdir}")
    return workdir


def train_metapath2vec(walk_path: Path, emb_path: Path, type_str: str, epoch: int, batch_size: int) -> None:
    args = argparse.Namespace(
        filename=str(walk_path),
        embed_dim=128,
        neighbour_size=5,
        epoch=epoch,
        typestr=type_str,
        batch_size=batch_size,
        neg_size=3,
        gpu="-1",
        l2=0.0,
        learning_rate=1e-2,
        outname=str(emb_path),
    )
    tf.compat.v1.reset_default_graph()
    model = metapath2vec(args)
    config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        model.fit(sess)


def load_similarities(emb_path: Path, topn: int = 10):
    kv = gensim.models.KeyedVectors.load_word2vec_format(str(emb_path))
    keys = list(kv.key_to_index.keys())
    sims = {}
    for key in keys:
        sims[key] = kv.most_similar(key, topn=topn)
    return keys, sims


def rank_from_embeddings(workdir: Path, relevancy: float):
    all_keys = []
    sim_tables = {}
    for stem, _, _, weight in METAPATH_SPECS:
        emb_path = workdir / f"{stem}_embedding.txt"
        if not emb_path.exists():
            continue
        keys, sims = load_similarities(emb_path)
        all_keys.append(keys)
        sim_tables[stem] = (weight, sims)

    if not all_keys:
        return []
    base_keys = min(all_keys, key=len)
    vec = []
    for key in base_keys:
        count_scores = {}
        for stem, (weight, sims) in sim_tables.items():
            for nbr, sim in sims.get(key, []):
                count_scores[nbr] = count_scores.get(nbr, 0.0) + weight * float(sim)
        ranked = [(node, score) for node, score in sorted(count_scores.items(), key=lambda x: x[1], reverse=True) if score > relevancy]
        vec.append(ranked[:10])

    freq = Counter()
    for rels in vec:
        for node, _ in rels:
            freq[node] += 1
    return [node for node, _ in freq.most_common()]


def ranking_to_pred(dataset: str, ranking):
    payload = base.safe_torch_load(ROOT / "external_prepared" / dataset / f"{dataset.lower()}_nie.pt")
    target_count = int(payload["target_count"])
    pred = np.zeros(target_count, dtype=np.float32)
    max_score = float(len(ranking) + 1)
    for idx, node in enumerate(ranking):
        if not node.startswith("a"):
            continue
        node_id = int(node[1:])
        if 0 <= node_id < target_count:
            pred[node_id] = max_score - idx
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAHE baseline on prepared heterogeneous datasets")
    parser.add_argument("--datasets", nargs="+", default=["ACM"])
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--relevancy", type=float, default=0.31)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spec-limit", type=int, default=0, help="Use only the first N metapath specs for faster approximate runs; 0 means all")
    parser.add_argument("--rerank-only", action="store_true", help="Reuse existing embeddings in results/MAHE/CUSTOM_<dataset> and only recompute ranking/prediction")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for dataset in args.datasets:
        started_at = time.perf_counter()
        if args.rerank_only:
            workdir = existing_workdir(dataset)
        else:
            workdir = make_workdir(dataset)
            a_a, p_a, p_c, a_c = build_dataset_relations(dataset)

            mpg = MetaPathGenerator()
            mpg.read_data(a_a, p_a, p_c, a_c)
            specs = METAPATH_SPECS if args.spec_limit <= 0 else METAPATH_SPECS[: args.spec_limit]
            for stem, type_str, (fn_name, numwalks, walklength), _ in specs:
                walk_path = workdir / f"{stem}.txt"
                emb_path = workdir / f"{stem}_embedding.txt"
                try:
                    getattr(mpg, fn_name)(str(walk_path), numwalks, walklength)
                except Exception as exc:
                    print(f"skip {stem}: {exc}")
                    continue
                if not walk_path.exists() or walk_path.stat().st_size == 0:
                    continue
                train_metapath2vec(walk_path, emb_path, type_str, args.epoch, args.batch_size)

        ranking = rank_from_embeddings(workdir, args.relevancy)
        pred = ranking_to_pred(dataset, ranking)
        art = credible.prepare_dataset(dataset)
        rows, sir_curves, si_curves = credible.evaluate_methods(art, {"MAHE-IM": pred})
        row = rows["MAHE-IM"]
        np.save(workdir / f"{dataset.lower()}_mahe_pred.npy", pred)
        with (workdir / "MAHE-IM_seed.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(ranking))
        with (workdir / f"{dataset.lower()}_mahe_result.pk").open("wb") as f:
            import pickle
            pickle.dump(
                {
                    "ndcg": np.array([float(row["NDCG@100"])], dtype=np.float32),
                    "spearman": np.array([float(row["Spearman"])], dtype=np.float32),
                    "rmse": np.array([0.0], dtype=np.float32),
                    "args": vars(args),
                },
                f,
            )
        with (workdir / f"{dataset.lower()}_mahe_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        with (workdir / f"{dataset.lower()}_sir_curve.json").open("w", encoding="utf-8") as f:
            json.dump(sir_curves["MAHE-IM"], f, indent=2)
        with (workdir / f"{dataset.lower()}_si_curve.json").open("w", encoding="utf-8") as f:
            json.dump(si_curves["MAHE-IM"], f, indent=2)
        with (workdir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "ranking_size": len(ranking),
                    "spec_limit": args.spec_limit,
                    "epoch": args.epoch,
                    "total_runtime_sec": float(time.perf_counter() - started_at),
                    "metrics": row,
                },
                f,
                indent=2,
            )
        print("saved", workdir)


if __name__ == "__main__":
    main()

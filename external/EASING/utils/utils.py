import dgl
import numpy as np
import pickle
import random
import torch
from sklearn.metrics import mean_absolute_error
from .metric import ndcg, spearman_sci
import pdb
import torch.nn as nn
from tqdm import tqdm


def convert_to_gpu(*data, device):
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    return tuple(res)


def _to_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(dtype)
    return torch.as_tensor(x, dtype=dtype)


def _load_payload(graph_data_path):
    if str(graph_data_path).endswith('.pt'):
        return torch.load(graph_data_path, map_location='cpu')
    with open(graph_data_path, 'rb') as f:
        return pickle.load(f)


def _edge_norm(hg):
    g = hg.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_model(model, model_path):
    print(f"load model {model_path}")
    model.load_state_dict(torch.load(model_path))


def count_parameters_in_KB(model):
    param_num = np.sum(np.prod(v.size()) for v in model.parameters()) / 1e3
    return param_num


def get_rank_metrics(predicts, labels, NDCG_k, spearman=False):
    if spearman:
        return ndcg(labels, predicts, NDCG_k), spearman_sci(labels, predicts)
    return ndcg(labels, predicts, NDCG_k)


def rank_evaluate(predicts, labels, NDCG_k, loss_func, spearman=False):
    with torch.no_grad():
        loss = loss_func(predicts, labels.reshape(-1))
        mae = mean_absolute_error(predicts.cpu().numpy(), labels.cpu().numpy())
    if spearman:
        ndcg_score, spear_score = get_rank_metrics(predicts, labels, NDCG_k, spearman)
        return loss, ndcg_score, spear_score, mae
    else:
        ndcg_score = get_rank_metrics(predicts, labels, NDCG_k, spearman)
        return loss, ndcg_score, mae


def load_split_data(split_data_path, num_split_idx):
    dataset_spilt = []
    labels_idx = []
    for i in range(num_split_idx):
        with open(f"{split_data_path}/split_dataset_idx{i}.pkl", "rb") as input:
            data = pickle.load(input)
            dataset_spilt.append(data['idx'])
            labels_idx.append(data['labels'])
    return dataset_spilt, labels_idx


def split_train_val_test(dataset_spilt, labels_idx, train_num, num_split_idx):
    dataset_index = [i for i in range(num_split_idx)]
    train_labels = np.array([], dtype=np.float32)
    train_idx = np.array([], dtype=np.int64)
    val_labels = np.array([], dtype=np.float32)
    val_idx = np.array([], dtype=np.int64)
    test_labels = np.array([], dtype=np.float32)
    test_idx = np.array([], dtype=np.int64)
    unlabeled_labels = np.array([], dtype=np.float32)
    unlabeled_idx = np.array([], dtype=np.int64)
    split_num = num_split_idx // 10
    for _ in range(split_num):
        subdataset = random.choice(dataset_index)
        test_idx = np.concatenate((test_idx, dataset_spilt[subdataset]))
        test_labels = np.concatenate((test_labels, labels_idx[subdataset]))
        test_labels = torch.tensor(test_labels)
        dataset_index.remove(subdataset)
    for _ in range(split_num):
        subdataset = random.choice(dataset_index)
        val_idx = np.concatenate((val_idx, dataset_spilt[subdataset]))
        val_labels = np.concatenate((val_labels, labels_idx[subdataset]))
        val_labels = torch.tensor(val_labels)
        dataset_index.remove(subdataset)

    num = int(train_num * split_num)
    for _ in range(num):
        subdataset = random.choice(dataset_index)
        train_idx = np.concatenate((train_idx, dataset_spilt[subdataset]))
        train_labels = np.concatenate((train_labels, labels_idx[subdataset]))
        train_labels = torch.tensor(train_labels)
        dataset_index.remove(subdataset)

    for i in range(len(dataset_index)):
        unlabeled_idx = np.concatenate((unlabeled_idx, dataset_spilt[dataset_index[i]]))
        unlabeled_labels = np.concatenate((unlabeled_labels, labels_idx[dataset_index[i]]))
        unlabeled_labels = torch.tensor(unlabeled_labels)
    return train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, unlabeled_idx, unlabeled_labels


def _load_generic_rel_data(
        graph_data_path,
        semantic_data_path,
        train_num=8.0,
        split_data_path='',
        num_split_idx=1000):
    data = _load_payload(graph_data_path)

    edges = data['edges']
    edge_types = _to_tensor(data['edge_types'], dtype=torch.long).view(-1)
    rel_num = int(edge_types.max().item()) + 1
    node_feat1 = _to_tensor(data['features'])

    if 'semantic_features' in data:
        node_feat2 = _to_tensor(data['semantic_features'])
    else:
        sem_raw = pickle.load(open(semantic_data_path, 'rb'))
        node_feat2 = _to_tensor(sem_raw)

    hg = dgl.graph(edges, num_nodes=int(data.get('num_nodes', node_feat1.shape[0])))
    edge_norm = _edge_norm(hg)

    if {'train_idx', 'val_idx', 'test_idx'}.issubset(data.keys()):
        all_labels = _to_tensor(data['labels']).view(-1)
        train_idx = np.asarray(data['train_idx']).reshape(-1).astype(np.int64)
        val_idx = np.asarray(data['val_idx']).reshape(-1).astype(np.int64)
        test_idx = np.asarray(data['test_idx']).reshape(-1).astype(np.int64)
        invalid_masks = np.asarray(data['invalid_masks']).reshape(-1)
        unlabeled_idx = np.where(invalid_masks != 0)[0]
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]
        test_labels = all_labels[test_idx]
        unlabeled_labels = all_labels[unlabeled_idx]
        print(len(test_idx), len(val_idx), len(train_idx))
        return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, unlabeled_idx, unlabeled_labels

    dataset_spilt, labels_idx = load_split_data(split_data_path, num_split_idx)
    train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, unlabeled_idx, unlabeled_labels = split_train_val_test(
        dataset_spilt, labels_idx, train_num, num_split_idx)
    print(len(test_idx), len(val_idx), len(train_idx))
    return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, unlabeled_idx, unlabeled_labels


def load_fb15k_rel_data(
        graph_data_path,
        semantic_data_path,
        train_num=8.0,
        split_data_path='',
        num_split_idx=1000):
    return _load_generic_rel_data(graph_data_path, semantic_data_path, train_num, split_data_path, num_split_idx)


def load_imdb_s_rel_subgraph_data(
        graph_data_path,
        structure_data_path,
        semantic_data_path,
        train_num=8.0,
        split_data_path='',
        num_split_idx=1000):
    return _load_generic_rel_data(graph_data_path, semantic_data_path, train_num, split_data_path, num_split_idx)


def load_tmdb_rel_data(
        graph_data_path,
        semantic_data_path,
        train_num=8.0,
        split_data_path='',
        num_split_idx=1000
):
    return _load_generic_rel_data(graph_data_path, semantic_data_path, train_num, split_data_path, num_split_idx)


def load_data(
        graph_data_path,
        structure_data_path,
        semantic_data_path,
        dataset_name,
        train_num=8.0,
        split_data_path='',
        num_split_idx=1000
):
    if dataset_name.startswith('FB15K'):
        return load_fb15k_rel_data(
            graph_data_path=graph_data_path,
            semantic_data_path=semantic_data_path,
            train_num=train_num,
            split_data_path=split_data_path,
            num_split_idx=num_split_idx
        )

    elif dataset_name.startswith('TMDB'):
        return load_tmdb_rel_data(
            graph_data_path=graph_data_path,
            semantic_data_path=semantic_data_path,
            train_num=train_num,
            split_data_path=split_data_path,
            num_split_idx=num_split_idx
        )

    elif dataset_name.startswith('IMDB'):
        return load_imdb_s_rel_subgraph_data(
            graph_data_path=graph_data_path,
            structure_data_path=structure_data_path,
            semantic_data_path=semantic_data_path,
            train_num=train_num,
            split_data_path=split_data_path,
            num_split_idx=num_split_idx
        )
    elif dataset_name.startswith('CUSTOM'):
        return _load_generic_rel_data(
            graph_data_path=graph_data_path,
            semantic_data_path=semantic_data_path,
            train_num=train_num,
            split_data_path=split_data_path,
            num_split_idx=num_split_idx
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {dataset_name}")


def get_centrality(graph):
    g = graph.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    theta = 1e-4
    centrality = torch.log(in_deg + theta)
    return centrality


def get_relative_entropy(graph, content_feat):
    if not torch.is_tensor(content_feat):
        content_feat = torch.as_tensor(content_feat, dtype=torch.float32)
    probs = torch.softmax(content_feat, dim=1)
    probs = torch.clamp(probs, min=1e-8)
    ent = -(probs * torch.log(probs)).sum(dim=1)
    ent = (ent - ent.mean()) / (ent.std() + 1e-8)
    return ent

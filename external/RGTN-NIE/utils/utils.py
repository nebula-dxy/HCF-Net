import dgl
import numpy as np
import pickle
import random
import torch
from sklearn.metrics import f1_score
from .metric import ndcg, spearman_sci
import pdb


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


def _direct_or_random_splits(data, num_nodes, invalid_masks, cross_validation_shift):
    if {'train_idx', 'val_idx', 'test_idx'}.issubset(data.keys()):
        train_idx = np.asarray(data['train_idx']).reshape(-1).astype(np.int64)
        val_idx = np.asarray(data['val_idx']).reshape(-1).astype(np.int64)
        test_idx = np.asarray(data['test_idx']).reshape(-1).astype(np.int64)
        return train_idx, val_idx, test_idx

    float_mask = np.ones(num_nodes) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))

    if cross_validation_shift == 0:
        test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        train_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')
    return train_idx, val_idx, test_idx


def _load_generic_rel_data(data_path, cross_validation_shift=0, dataset_name='CUSTOM_rel'):
    if str(data_path).endswith('.pt'):
        data = torch.load(data_path, map_location='cpu')
    else:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

    edges = data['edges']
    labels = _to_tensor(data['labels']).view(-1)
    invalid_masks = np.asarray(data['invalid_masks']).reshape(-1)
    edge_types = _to_tensor(data['edge_types'], dtype=torch.long).view(-1)
    rel_num = int(edge_types.max().item()) + 1

    if 'two' in dataset_name:
        node_feat1 = _to_tensor(data['features'])
        node_feat2 = _to_tensor(data['semantic_features'])
    elif 'concat' in dataset_name:
        node_feat1 = _to_tensor(data['features'])
        node_feat2 = _to_tensor(data['semantic_features'])
        node_feats = torch.cat([node_feat1, node_feat2], dim=1)
    elif 'semantic' in dataset_name:
        node_feats = _to_tensor(data['semantic_features'])
    else:
        node_feats = _to_tensor(data['features'])

    hg = dgl.graph(edges, num_nodes=int(data.get('num_nodes', labels.numel())))
    g = hg.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    edge_norm = g.edata['norm']

    labels = torch.log1p(labels)
    train_idx, val_idx, test_idx = _direct_or_random_splits(data, hg.number_of_nodes(), invalid_masks, cross_validation_shift)
    print(len(test_idx), len(val_idx), len(train_idx))
    if 'two' in dataset_name:
        return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
    return hg, edge_types, edge_norm, rel_num, node_feats, labels, train_idx, val_idx, test_idx


def set_random_seed(seed=0):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_model(model, model_path):
    """Load the model.
    :param model: model
    :param model_path: model path
    """
    print(f"load model {model_path}")
    model.load_state_dict(torch.load(model_path))


def count_parameters_in_KB(model):
    """
    count the size of trainable parameters in model (KB)
    :param model: model
    :return:
    """
    param_num = np.sum(np.prod(v.size()) for v in model.parameters()) / 1e3
    return param_num


def get_rank_metrics(predicts, labels, NDCG_k, spearman=False):
    """
    calculate NDCG@k metric
    :param predicts: Tensor, shape (N, 1)
    :param labels: Tensor, shape (N, 1)
    :return:
    """
    if spearman:
        return ndcg(labels, predicts, NDCG_k), spearman_sci(labels, predicts)
    return ndcg(labels, predicts, NDCG_k)


def rank_evaluate(predicts, labels, NDCG_k, loss_func, spearman=False):
    """
    evaluation used for validation or test
    :param predicts: Tensor, shape (N, 1)
    :param labels: Tensor, shape (N, 1)
    :param loss_func: loss function
    :return:
    """
    with torch.no_grad():
        loss = loss_func(predicts, labels)
    if spearman:
        ndcg_score, spear_score = get_rank_metrics(predicts, labels, NDCG_k, spearman)
        return loss, ndcg_score, spear_score
    else:
        ndcg_score = get_rank_metrics(predicts, labels, NDCG_k, spearman)
        return loss, ndcg_score


def load_fb15k_rel_data(data_path, cross_validation_shift=0, dataset_name='FB15k_rel'):
    return _load_generic_rel_data(data_path, cross_validation_shift, dataset_name)


def load_imdb_s_rel_data(data_path, cross_validation_shift=0, dataset_name='IMDB_S_rel'):
    return _load_generic_rel_data(data_path, cross_validation_shift, dataset_name)


def load_tmdb_rel_data(data_path, cross_validation_shift=0, dataset_name='TMDB_rel'):
    return _load_generic_rel_data(data_path, cross_validation_shift, dataset_name)


def load_data(data_path, dataset_name, cross_validation_shift=0):
    """
    load dataset based on the input dataset name
    :param data_path: str, data file path
    :param dataset_name: dataset name
    :param cross_validation_shift: int, shift of data split
    :return:
    """

    if dataset_name.startswith('FB15k'):
        return load_fb15k_rel_data(data_path=data_path, cross_validation_shift=cross_validation_shift, dataset_name=dataset_name)
    elif dataset_name.startswith('IMDB_S'):
        return load_imdb_s_rel_data(data_path, cross_validation_shift, dataset_name)
    elif dataset_name.startswith('TMDB'):
        return load_tmdb_rel_data(data_path, cross_validation_shift, dataset_name)
    elif dataset_name.startswith('CUSTOM'):
        return _load_generic_rel_data(data_path, cross_validation_shift, dataset_name)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset_name))


def get_centrality(graph):
    g = graph.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    theta = 1e-4
    centrality = torch.log(in_deg + theta)
    return centrality

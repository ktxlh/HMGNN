# _*_ coding:utf-8 _*_

from os.path import isfile
import os
import numpy as np
import random
import math

random.seed(123)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def make_mini_subset(labels, edges, features, size):
    n_edges = edges.shape[1]
    idxs = random.sample([i for i in range(n_edges)], min(size, n_edges))
    edges = np.stack([edges[0][idxs], edges[1][idxs]], axis=0)
    old_idx = list(set(edges.flatten().tolist()))  # set is sorted
    new_idx = [i for i in range(len(old_idx))]
    new_edges = np.zeros_like(edges)
    for old, new in zip(old_idx, new_idx):
        new_edges[edges == old] = new
    return labels[old_idx], new_edges, features[old_idx]

def filter_empty_node(labels, edges, features):
    EPS = 1e-5
    non_empty = features.std(axis=-1) > EPS
    j = 0
    new_edges = np.zeros_like(edges)
    idx_map = [old_idx for old_idx, empty in enumerate(non_empty) if not empty]
    idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(idx_map)}
    for i in range(edges.shape[1]):
        if edges[0,i] in idx_map.keys() and edges[1,i] in idx_map.keys():
            new_edges[0,j] = idx_map[edges[0,i]]
            new_edges[1,j] = idx_map[edges[1,i]]
            j += 1
    new_edges = new_edges[:, :j]
    return labels[non_empty], new_edges, features[non_empty]

def load_data(args):
    data_path = args.data_dir
    edges = np.load(os.path.join(data_path, "edges_mat.npy"))
    labels = np.load(os.path.join(data_path, "labels.npy"))
    if args.bert_in != '':
        assert args.feature_dim > 768 # conactentation of bert's pooled_output and other features
        # Convert node indices to match edges_mat
        with open(os.path.join(data_path, "news_id.txt")) as f:
            id2idx = {news_id.strip(): idx for idx, news_id in enumerate(f.readlines())}
        with open(os.path.join(args.bert_in, "ids-bert-baseline.txt"), 'r') as f:
            idx2id = [news_id.strip() for news_id in f.readlines()]
        features_old = np.load(os.path.join(args.bert_in, "features-bert-baseline.npy"))
        features = np.zeros_like(features_old)
        for idx in range(features.shape[0]):
            if idx2id[idx] in id2idx.keys():
                features[id2idx[idx2id[idx]]] = features_old[idx]
    else:
        # assert args.feature_dim == 768 or args.feature_dim == 1433 # pure bert's pooled_output or vocab
        features = np.load(os.path.join(data_path, "features.npy"))

    # print("# nodes before filter empty:", len(labels))
    # labels, edges, features = filter_empty_node(labels, edges, features)
    # print("# nodes after filter empty: ", len(labels))
    # labels, edges, features = make_mini_subset(labels, edges, features, 100)  # to overfit

    node_num = labels.shape[0]
    classify = labels.shape[1]
    assert classify == args.label_kinds

    ids = [i for i in range(node_num)]
    random.shuffle(ids)

    train_ratio = args.train_ratio
    test_ratio = args.train_ratio + args.test_ratio

    train_ids = ids[0: math.ceil(train_ratio * node_num)]
    test_ids = ids[math.ceil(train_ratio*node_num): math.ceil(test_ratio*node_num)]
    val_ids = ids[math.ceil(test_ratio*node_num):]

    train_mask = sample_mask(train_ids, node_num)
    val_mask = sample_mask(val_ids, node_num)
    test_mask = sample_mask(test_ids, node_num)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    row = list(edges[0,:])
    col = list(edges[1,:])
    weight = [1 for _ in range(len(row))]
    adj = [row, col, weight, node_num]
    adjs = [adj]

    return adjs, list(features), labels, y_train, y_test, y_val, train_mask, test_mask, val_mask


if __name__ == '__main__':
    import hparams
    FLAGS = hparams.create()
    FLAGS.data_dir = "." + FLAGS.data_dir
    load_data(FLAGS)

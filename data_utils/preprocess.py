"""
Construct the data for PolitiFact and PHEME:

[labels.py] dtype=int32, shape=(n_nodes, n_labels)
Each news is either 0 or 1.

[features.npy] dtype=float32, shape=(n_nodes, n_features)
"The features of users mainly come from multiple aspects and some categorical features are encoded via a one-hot encoder resulting in the final features being sparse and high-dimensional."
=> Use the same news features for HetGCN.

[edges_mat.npy] dtype=int32, shape=(2, n_edges)
Each news is a node. There is a link between two nodes if they are k-hop neighbors in our graph definition.
=> Process graph definition files first
"""
from collections import Counter
from os.path import join
import numpy as np
from tqdm import tqdm, trange
from data_utils.read_raw import read_politifact_input, read_pheme_input
from transformers import BertTokenizerFast

root = "/rwproject/kdd-db/20-rayw1/"

def read_files(edge_types, edge_path, feature_paths, L, feature_lines):
    def read_edges(edge_types, edge_path):
        def _add_edge(d, a, b):
            if a == b: return
            def _add(d, n0, n1):
                if n0 not in d.keys():
                    d[n0] = []
                d[n0].append(n1)
            _add(d, a, b)
            _add(d, b, a)
        edges = dict()  # typed_id -> list(neighbors' typed_ids)
        for (t0, t1), fname in edge_types.items():
            with open(join(edge_path, fname), 'r') as f:
                lines = f.readlines()
            for line in lines:
                ids = line.strip().split()
                _add_edge(edges, t0 + ids[0], t1 + ids[1])
        for k, v in edges.items():
            edges[k] = list(set(v))
        return edges
    def read_feature_input(feature_paths, L, feature_lines):
        labels = dict()  # news_id -> int
        features = dict()  # news_id -> np.array
        for p in feature_paths:
            with open(p, 'r') as f:
                lines = f.readlines()
            for i in range(len(lines) // L):
                # Label
                info = lines[i * L].strip().split()
                if len(info) == 3:
                    news_id = info[0] + info[1]
                    if news_id == 'nPADDING': continue
                    label = int(info[2])
                else: # len(info) == 2
                    news_id = info[0]
                    if news_id == 'nPADDING': continue
                    label = int(info[1])
                labels[news_id] = label
                # Feature
                feature = []
                for j in feature_lines:
                    feature.append(np.array(lines[i * L + j].strip().split(), dtype=np.float32))
                features[news_id] = np.concatenate(feature, axis=-1)
        return features, labels
    def read_text_and_convert(vocab_size=1433):
        batch_size = 256
        if 'politifact' in feature_paths[0].lower():
            inpt = read_politifact_input()
            max_seq_len = 490
        elif 'pheme' in feature_paths[0].lower():
            inpt = read_pheme_input()
            max_seq_len = 49
        else:
            raise NotImplementedError
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', model_max_length=max_seq_len)
        ids = ['n' + i[0] for i in inpt]
        text = [i[1] for i in inpt]
        labels = [i[2] for i in inpt]
        other_features = [i[3] for i in inpt]
        input_ids, input_features = [], []
        num_batches = (len(text) + batch_size - 1) // batch_size
        for i_batch in trange(num_batches, desc=f'embed'):
            t = text[i_batch * batch_size : min((i_batch + 1) * batch_size, len(text))]
            input_id = tokenizer(t, return_tensors="np", max_length=max_seq_len, padding='max_length', truncation=True)
            input_ids.append(input_id.input_ids)
            f = np.array(other_features[i_batch * batch_size : min((i_batch + 1) * batch_size, len(text))])
            f = np.tanh(f)
            input_features.append(f)
        input_ids = np.concatenate(input_ids, axis=0)
        input_features = np.concatenate(input_features, axis=0)
        vocab = Counter(input_ids.flatten().tolist())
        for special_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id]:
            del vocab[special_id]
        vocab = [(k, v) for k, v in vocab.items()]
        vocab.sort(key=lambda x:-x[1])
        vocab.insert(0, (tokenizer.unk_token_id, None))
        vocab = {vid: idx for idx, (vid, _) in enumerate(vocab[:vocab_size])}
        features = dict()
        input_feature_size = input_features.shape[1]
        skip_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
        for id, input_id, input_feature in zip(tqdm(ids, 'convert'), input_ids, input_features):
            feature = np.zeros(vocab_size + input_feature_size)
            feature[-input_feature_size:] = input_feature
            for vid in input_id:
                if vid not in skip_ids:
                    feature[vocab[vid if vid in vocab.keys() else tokenizer.unk_token_id]] = 1
            features[id] = feature
        new_labels = {id: label for id, label  in zip(ids, labels)}
        return features, new_labels
    edges = read_edges(edge_types, edge_path)
    # features, labels = read_feature_input(feature_paths, L, feature_lines)
    features, labels = read_text_and_convert()
    return edges, features, labels

def read_politifact():
    print("Reading PolitiFact...")
    out_path = root + "HMGNN/data/politifact/vocab_other"
    edge_path = "/rwproject/kdd-db/20-rayw1/FakeNewsNet/graph_def/politifact/"
    feature_path = "/rwproject/kdd-db/20-rayw1/FakeNewsNet/FNN_input/Politifact/transformer_200/normalized_news_nodes"
    k = 4
    n = 5
    return read_files(
        edge_types={
            ('n', 'n') : 'news-news edges.txt',
            ('n', 'p') : 'news-post edges.txt',
            ('p', 'u') : 'post-user edges.txt',
            # ('u', 'u') : 'user-user edges.txt',  # complete graph within each post -> redundant for news
        },
        edge_path=edge_path,
        feature_paths=[join(feature_path, "batch_0.txt"),],
        L=7, feature_lines=[1] # 2
    ), out_path, k, n
    
def read_pheme():
    print("Reading PHEME...")
    out_path = root + "HMGNN/data/pheme/vocab_other"
    edge_path = "/rwproject/kdd-db/20-rayw1/pheme-figshare/"
    feature_path = "/rwproject/kdd-db/20-rayw1/pheme-figshare/pheme_input/transformer_200/normalized_news_nodes"
    k = 4
    n = 5
    return read_files(
        edge_types={
            # ('n', 'n') : 'PhemeNewsNews.txt',  # complete graph within each event => Too dense & noisy
            ('n', 'p') : 'PhemeNewsPost.txt',
            ('n', 'u') : 'PhemeNewsUser.txt',
            ('p', 'p') : 'PhemePostPost.txt',  # forwarding tree. Not redundant.
            ('p', 'u') : 'PhemePostUser.txt',
            # ('u', 'u') : 'PhemeUserUser.txt',  # complete graph within each post => redundant for news
        },
        edge_path=edge_path,
        feature_paths=[join(feature_path, f"batch_{i}.txt") for i in range(2)],
        L=5, feature_lines=[1]
    ), out_path, k, n

def read_buzzfeed():
    print("Reading BuzzFeed...")
    out_path = root + "HMGNN/data/buzzfeed"
    edge_path = "/rwproject/kdd-db/20-rayw1/buzzfeed-kaggle"
    feature_path = "/rwproject/kdd-db/20-rayw1/buzzfeed-kaggle/buzzfeed_input/buzzfeed_200/normalized_news_nodes/"
    k = 20
    n = 5
    return read_files(
        edge_types = {
            ('s', 'n'): 'BuzzFeedSourceNews.txt',
            ('n', 'n'): 'BuzzFeedNewsNews.txt',
        },
        edge_path=edge_path,
        feature_paths=[join(feature_path, f"batch_{i}.txt") for i in range(1)],
        L=8, feature_lines=[1]
    ), out_path, k, n

def convert_id(labels):
    print(f"# news: {len(labels)}")
    return {k : id for id, k in enumerate(labels)}
    
def density(e, v):
    # https://math.stackexchange.com/questions/1526372/what-is-the-definition-of-the-density-of-a-graph/1526421
    return 2 * e / (v * (v - 1))

def get_k_hop_neighbors(edges, k, ids, n):
    """Run BFS on each news to get its k-hop neighbors"""
    def bfs(root):
        visited, neighbors = set(), set()
        bfs_queue = [(root, 0), ]
        while len(bfs_queue) > 0 and len(neighbors) < n:
            n0, level = bfs_queue.pop(0)
            if n0 in visited:
                continue
            visited.add(n0)
            if n0 != root and n0[0] == 'n':
                neighbors.add(n0)
            if level < k and n0 in edges.keys():
                for n1 in edges[n0]:
                    bfs_queue.append((n1, level + 1))
        return neighbors
    edges_mat = [[], []]
    for n0 in tqdm(ids.keys(), desc=f'get {k}-hop'):
        neighbors = bfs(n0)
        for n1 in neighbors:
            if n0 in ids.keys() and n1 in ids.keys():
                edges_mat[0].append(ids[n0])
                edges_mat[1].append(ids[n1])
    print("k={}, n={}, # edges: {}, density={:.8f}".format(k, n, len(edges_mat[0]), density(len(edges_mat[0]), len(ids))))
    return np.array(edges_mat, dtype=np.int32)

def convert_labels(labels, ids):
    labels_mat = np.zeros((len(ids), 2), dtype=np.int32)
    for n0, l in labels.items():
        labels_mat[ids[n0], l] = 1
    assert all(labels_mat.sum(axis=-1) == 1)
    return labels_mat

def convert_features(features, ids):
    v = [v for v in features.values()][0]
    features_mat = np.zeros((len(ids), len(v)), dtype=np.float32)
    for n0, v in features.items():
        features_mat[ids[n0]] = v
    return features_mat

def save_files(out_path, labels, features, edges_mat, ids):
    np.save(join(out_path, 'labels.npy'), labels)
    np.save(join(out_path, 'features.npy'), features)
    np.save(join(out_path, 'edges_mat.npy'), edges_mat)
    ids_str = ['' for _ in range(len(ids))]
    if 'buzzfeed' in out_path.lower():
        with open("/rwproject/kdd-db/20-rayw1/buzzfeed-kaggle/BuzzFeedNews.txt", 'r') as f:
            news_id_map = [l.strip() for l in f.readlines()]
        print([k for k in ids.keys()].sort())
        print(news_id_map.sort())
        raise NotImplementedError
        for k, v in ids.items():
            ids_str[v] = news_id_map[int(k[1:])]
    else:
        for k, v in ids.items():
            ids_str[v] = k[1:]
    with open(join(out_path, 'news_id.txt'), 'w') as f:
        f.write('\n'.join(ids_str) + '\n')
    with open(join(out_path, 'stats.txt'), 'w') as f:
        f.write('# Edges: {}\n'.format(edges_mat.shape[1]))
        f.write('# Nodes: {}\n'.format(features.shape[0]))
        f.write('Density: {:.8f}\n'.format(density(edges_mat.shape[1], features.shape[0])))
        f.write('feature_dim: {}\n'.format(features.shape[1]))

if __name__ == "__main__":
    for f in [read_pheme, read_politifact]:  # read_buzzfeed
        (edges, features, labels), out_path, k, n = f()
        ids = convert_id(labels)
        edges_mat = get_k_hop_neighbors(edges, k, ids, n)
        labels_mat = convert_labels(labels, ids)
        features_mat = convert_features(features, ids)
        save_files(out_path, labels_mat, features_mat, edges_mat, ids)
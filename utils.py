import math
import pickle
import random
import pandas as pd
import numpy as np

N_ADD = 0
N_DEL = 1
N_KEP = 2

E_DEL = 0
E_REF = 1
E_CTX = 2


class Graph(object):
    def __init__(self, input_ids, input_masks, node_types, g_0, g_1, g_2,
                 add_ids, add_labels, add_nums, target_ids):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.node_types = node_types
        self.g_0 = g_0
        self.g_1 = g_1
        self.g_2 = g_2
        self.add_ids = add_ids
        self.add_labels = add_labels
        self.add_nums = add_nums
        self.target_ids = target_ids


def load_data(path, tokenizer, max_tokens):
    commits, files, graphs, labels = pickle.load(open(path, "rb"))

    # commits = commits[:64]
    # files = files[:64]
    # graphs = graphs[:64]
    # labels = labels[:64]

    o_graphs = []
    for graph in graphs:
        texts, node_type, edge_index, edge_type, adds = graph
        target_ids = list(range(len(texts)))

        add_ids = [i for i in adds]
        add_labels = [adds[i]["label"] for i in adds]
        add_nums = [adds[i]["num"] for i in adds]

        from_0, from_1, from_2 = [], [], []
        to_0, to_1, to_2 = [], [], []
        e_size = len(edge_index)
        for i in range(e_size):
            from_0.append(edge_index[i][0])
            to_0.append(edge_index[i][1])
            if edge_type[i] == N_DEL:
                from_2.append(edge_index[i][0])
                to_2.append(edge_index[i][1])
            else:
                from_1.append(edge_index[i][0])
                to_1.append(edge_index[i][1])

        g_0, g_1, g_2 = [from_0, to_0], [from_1, to_1], [from_2, to_2]

        input_ids, input_masks = [], []
        for index, text in enumerate(texts):
            tokens = tokenizer.tokenize(text)
            if len(tokens) > max_tokens - 2:
                tokens = tokens[:(max_tokens - 2)]
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            ids = tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(ids)
            padding_length = max_tokens - len(ids)
            ids = ids + ([0] * padding_length)
            mask = mask + ([0] * padding_length)
            assert len(ids) == max_tokens
            assert len(mask) == max_tokens
            input_ids.append(ids)
            input_masks.append(mask)

        o_graph = Graph(input_ids, input_masks, node_type, g_0, g_1, g_2,
                        add_ids, add_labels, add_nums, target_ids)
        o_graphs.append(o_graph)
    return commits, files, o_graphs, labels


def train_batch(data, batch_size, seed):
    commits, files, graphs, labels = data
    size = len(commits)
    np.random.seed(seed)

    commits = np.array(commits)
    files = np.array(files)
    graphs = np.array(graphs)
    labels = np.array(labels)

    batches = []
    shuffled_commits, shuffled_files = commits, files
    shuffled_graphs, shuffled_labels = graphs, labels
    Y_pos = [i for i, label in enumerate(labels) if label == 1]
    Y_neg = [i for i, label in enumerate(labels) if label == 0]

    # Randomly pick batch_size / 2 from each of positive and negative labels
    n_batches = int(math.floor(size / float(batch_size))) + 1
    for k in range(n_batches):
        ids = sorted(random.sample(Y_pos, int(batch_size / 2)) +
                     random.sample(Y_neg, int(batch_size / 2)))
        batch_commits = shuffled_commits[ids].tolist()
        batch_files = shuffled_files[ids].tolist()
        batch_graph = integrate(shuffled_graphs[ids])
        batch_labels = shuffled_labels[ids].tolist()
        batch = (batch_commits, batch_files, batch_graph, batch_labels)
        batches.append(batch)
    return batches


def test_batch(data, batch_size):
    commits, files, graphs, labels = data
    size = len(commits)

    commits = np.array(commits)
    files = np.array(files)
    graphs = np.array(graphs)
    labels = np.array(labels)

    batches = []
    shuffled_commits, shuffled_files = commits, files
    shuffled_graphs, shuffled_labels = graphs, labels

    num_batches = int(math.floor(size / float(batch_size)))
    for k in range(num_batches):
        batch_commits = shuffled_commits[k * batch_size: k * batch_size + batch_size].tolist()
        batch_files = shuffled_files[k * batch_size: k * batch_size + batch_size].tolist()
        batch_graph = integrate(shuffled_graphs[k * batch_size: k * batch_size + batch_size])
        batch_labels = shuffled_labels[k * batch_size: k * batch_size + batch_size].tolist()
        batch = (batch_commits, batch_files, batch_graph, batch_labels)
        batches.append(batch)
    return batches


def integrate(graphs):
    o_input_ids, o_input_masks = [], []
    from_0, from_1, from_2 = [], [], []
    to_0, to_1, to_2 = [], [], []
    o_node_types, o_target_ids = [], []
    o_add_ids, o_add_labels, o_add_nums = [], [], []
    for g in graphs:
        size = len(o_input_ids)
        o_input_ids.extend(g.input_ids)
        o_input_masks.extend(g.input_masks)
        o_node_types.extend(g.node_types)

        from_0.extend(map(lambda x: x + size, g.g_0[0]))
        to_0.extend(map(lambda x: x + size, g.g_0[1]))
        from_1.extend(map(lambda x: x + size, g.g_1[0]))
        to_1.extend(map(lambda x: x + size, g.g_1[1]))
        from_2.extend(map(lambda x: x + size, g.g_2[0]))
        to_2.extend(map(lambda x: x + size, g.g_2[1]))
        o_target_ids.append(list(map(lambda x: x + size, g.target_ids)))

        o_add_ids.append(g.add_ids)
        o_add_labels.append(g.add_labels)
        o_add_nums.append(g.add_nums)

    o_g_0 = [from_0, to_0] if len(from_0) != 0 else None
    o_g_1 = [from_1, to_1] if len(from_1) != 0 else None
    o_g_2 = [from_2, to_2] if len(from_2) != 0 else None
    return Graph(o_input_ids, o_input_masks, o_node_types, o_g_0, o_g_1, o_g_2,
                 o_add_ids, o_add_labels, o_add_nums, o_target_ids)


# mode: 1-仅评估缺陷文件, 2-仅评估预测正确的缺陷文件
def eval_line(commits, files, labels, preds, probs, add_ids, add_tags, add_nums, attns, out, mode):
    line_result = []

    for commit, file, label, pred, prob, ids, tags, nums, scores in zip(commits, files, labels, preds, probs, add_ids,
                                                                        add_tags, add_nums, attns):
        if mode == 1:
            if label != 1:
                continue
        elif mode == 2:
            if not (pred == 1 and label == 1):
                continue
        scores = get_scores(ids, scores, prob)
        for num, score, tag in zip(nums, scores, tags):
            line_result.append([commit, file, num, score, label])
    pd.DataFrame(line_result, columns=['commit', 'file', 'id', 'score', 'label']).to_csv('data/{}.txt'.format(out),
                                                                                         sep='\t', index=False)


def get_scores(ids, scores, prob):
    _scores = [i * prob for i in scores]
    scores = [_scores[i] for i in ids]
    b = sorted(zip(scores, range(len(scores))))
    b.sort(key=lambda x: x[0])
    c = [x[1] for x in b]

    mid = math.ceil(len(scores) / 3)
    k = 1
    while k < mid:
        scores[c[-k]] += 0.5
        k += 1
    return scores

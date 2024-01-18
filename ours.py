import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FAConv


class Model(nn.Module):
    def __init__(self, params, bert):
        super(Model, self).__init__()
        self.in_dim = params.in_dim
        self.hid_dim = params.hid_dim
        self.out_dim = 1
        self.eps = params.eps
        self.dropout = params.dropout
        self.n_heads = 8

        # Sequential Representation Module
        self.embed = bert
        self.proj_seq = nn.Linear(768, self.in_dim)
        self.sequence_encoder = AHT(self.in_dim, self.dropout)

        # Structural Representation Module
        self.structure_encoder = FAGCN_MOE(self.in_dim, self.eps)

        # Defect Prediction Module
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid_dim, self.out_dim),
            nn.Sigmoid()
        )

    def forward(self, input_ids, input_masks, g_0, g_1, g_2, target_ids, add_ids, pertub):

        x_seq = self.embed(input_ids, input_masks)[1]
        x_seq = self.proj_seq(x_seq)

        x = F.relu(x_seq)

        x = self.sequence_encoder(x, target_ids, add_ids)
        g_feat1, scores = self.structure_encoder(x, g_0, g_1, g_2, target_ids, pertub)
        pred = self.mlp(g_feat1)
        pred = pred.squeeze(1)

        if pertub:
            g_feat2, _ = self.structure_encoder(x, g_0, g_1, g_2, target_ids, pertub)
            return pred, g_feat1, g_feat2
        else:
            return pred, scores


class FAGCN_MOE(nn.Module):
    def __init__(self, in_dim, eps):
        super(FAGCN_MOE, self).__init__()
        self.layer_num = 2
        self.n_heads = 8
        self.dropout = 0.5
        self.in_dim = in_dim
        self.eps = eps

        self.gnn = FAGCN_NET(self.in_dim, self.eps, self.dropout)
        self.gate_a = nn.Linear(self.in_dim, 2)
        self.gate_b = nn.Linear(self.in_dim, 2)
        self.lin = nn.Linear(self.in_dim * 2, self.in_dim)
        self.dp = nn.Dropout(self.dropout)
        self.fuse = nn.MultiheadAttention(self.in_dim, self.n_heads, self.dropout)

    def forward(self, x, g_0, g_1, g_2, target_ids, pertub):
        s_out, weights_s = self.gnn(x, g_0, pertub)
        a_out, weights_a = self.gnn(x, g_1, pertub)
        b_out, weights_b = self.gnn(x, g_2, pertub)

        s_out = s_out.unsqueeze(1)
        a_out = a_out.unsqueeze(1)
        b_out = b_out.unsqueeze(1)

        gate_a = self.gate_a(x).softmax(1).unsqueeze(1)
        experts = torch.cat([a_out, s_out], dim=1)
        gate_a_out = torch.matmul(gate_a, experts).squeeze(1)

        gate_b = self.gate_b(x).softmax(1).unsqueeze(1)
        experts = torch.cat([b_out, s_out], dim=1)
        gate_b_out = torch.matmul(gate_b, experts).squeeze(1)

        x = torch.cat((gate_a_out, gate_b_out), dim=1)
        x = self.lin(x)
        x = self.dp(F.relu(x))

        vectors, scores = [], []
        for item in target_ids:
            index = torch.tensor(item).to(x.device)
            target = torch.index_select(x, dim=0, index=index)
            x_file, weights = self.fuse(target, target, target)
            vectors.append(x_file[0])
            scores.append(weights[0])
        vectors = torch.stack(vectors)
        return vectors, scores


class FAGCN_NET(nn.Module):
    def __init__(self, in_dim, eps, dropout):
        super(FAGCN_NET, self).__init__()
        self.layer_num = 2
        self.eps = eps
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for i in range(self.layer_num):
            self.convs.append(FAConv(in_dim, eps, dropout))
        self.dp = nn.Dropout(self.dropout)

    def reset_parameters(self):
        for cov in self.convs:
            cov.reset_parameters()

    def forward(self, x, edges_index, pertub):
        if edges_index is None:
            return self.eps * x, None

        raw = x
        for i in range(self.layer_num):
            x, weights = self.convs[i](x, raw, edges_index, return_attention_weights=True)
            x = self.dp(F.relu(x))

            if pertub:
                random_noise = torch.rand_like(x).to(x.device)
                x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        return x, weights


# Extract global information
class AHT(nn.Module):
    def __init__(self, dim, dropout):
        super(AHT, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.n_heads = 8

        self.pool = MAB(self.dim, self.n_heads)

    def reset_parameters(self):
        self.pool.reset_parameters()

    def forward(self, x, target_ids, add_ids):
        size = len(target_ids)
        vectors = None
        for idx in range(size):
            t_ids_ = torch.tensor(target_ids[idx]).to(x.device)
            key = torch.index_select(x, dim=0, index=t_ids_)

            a_ids_ = torch.tensor(add_ids[idx]).to(x.device)
            query = torch.index_select(key, dim=0, index=a_ids_)

            x_hid = self.pool(query, key)

            num_nodes, dim = key.shape
            x_fuse = torch.zeros(num_nodes, dim).to(x.device)
            for i, j in enumerate(add_ids[idx]):
                x_fuse[j] = x_hid[i]
            x_fuse = x_fuse + key
            if idx == 0:
                vectors = x_fuse
            else:
                vectors = torch.cat((vectors, x_fuse), dim=0)
        return vectors


class MAB(nn.Module):
    def __init__(self, dim, n_heads):
        super(MAB, self).__init__()
        self.n_heads = n_heads
        self.dim_Q = dim
        self.dim_K = dim
        self.dim_V = dim

        self.proj_q = nn.Linear(self.dim_Q, self.dim_V)
        self.proj_k = nn.Linear(self.dim_K, self.dim_V)
        self.proj_v = nn.Linear(self.dim_K, self.dim_V)

        self.ln1 = nn.LayerNorm(self.dim_V)
        self.ln2 = nn.LayerNorm(self.dim_V)

        self.lin = nn.Linear(self.dim_V, self.dim_V)

    def reset_parameters(self):
        self.proj_q.reset_parameters()
        self.proj_k.reset_parameters()
        self.proj_v.reset_parameters()
        if self.layer_norm:
            self.ln1.reset_parameters()
            self.ln2.reset_parameters()
        self.lin.reset_parameters()
        pass

    def forward(self, query, key):
        Q = self.proj_q(query).unsqueeze(0)
        K = self.proj_k(key).unsqueeze(0)
        V = self.proj_v(key).unsqueeze(0)

        dim_split = self.dim_V // self.n_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), -1)

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        out = out.squeeze(0)
        out = self.ln1(out)

        out = out + F.relu(self.lin(out))
        out = self.ln2(out)

        return out


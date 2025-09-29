# File: state_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    单头 GAT 层：把 N 个节点的特征从 F_in 维映射到 F_out 维，
    只用节点自身特征（无显式邻接矩阵），相当于全图注意力。
    """

    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)

    def forward(self, h):
        """
        :param h: Tensor (N, in_dim)
        :return: h_prime: Tensor (N, out_dim)
        """
        Wh = self.W(h)            # (N, out_dim)
        N = Wh.size(0)

        # 计算 e_ij = a_src(Wh_i) + a_dst(Wh_j)，然后 softmax
        e_src = self.a_src(Wh)    # (N, 1)
        e_dst = self.a_dst(Wh)    # (N, 1)
        e = e_src + e_dst.T       # (N, N) 广播加

        attention = F.softmax(F.leaky_relu(e), dim=1)  # (N, N)
        h_prime = torch.matmul(attention, Wh)          # (N, out_dim)
        return F.elu(h_prime)


class MultiHeadGAT(nn.Module):
    """
    多头 GAT，把 N 个节点特征映射到 (N, head_dim * num_heads) 维
    """

    def __init__(self, in_dim, head_dim, num_heads=4):
        super(MultiHeadGAT, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([GATLayer(in_dim, head_dim) for _ in range(num_heads)])

    def forward(self, x):
        # 把每个头的输出 concat 在一起
        head_outs = [h(x) for h in self.heads]  # 每个 h(x) 形状 (N, head_dim)
        return torch.cat(head_outs, dim=1)      # (N, head_dim * num_heads)


class SimpleNodeAggregator(nn.Module):
    """
    把 N x feat_dim 的节点特征映射成一个 embed_dim 维向量：
      1) 先通过 2层 GAT(…) + ReLU
      2) 再对 N 个节点沿第一维做平均，得到 (feat)维向量
      3) 再通过一层线性变换映射到 embed_dim
    """

    def __init__(self, node_feat_dim, hidden_dim, embed_dim, num_heads=4):
        super(SimpleNodeAggregator, self).__init__()
        # MultiHeadGAT: 输入 node_feat_dim，输出 hidden_dim = head_dim * num_heads
        assert hidden_dim % num_heads == 0
        head_dim = hidden_dim // num_heads
        self.gat1 = MultiHeadGAT(node_feat_dim, head_dim, num_heads)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        """
        :param x: Tensor (N, node_feat_dim) 每行对应一个节点的特征向量
        :return: emb: Tensor (embed_dim,)
        """
        h = F.relu(self.gat1(x))    # (N, hidden_dim)
        h_mean = torch.mean(h, dim=0)  # (hidden_dim,)
        emb = self.fc2(h_mean)        # (embed_dim,)
        return emb

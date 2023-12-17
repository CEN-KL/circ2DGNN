import torch
import dgl
import numpy as np
from model import *
import math
import logging
import random
from sys import stdout
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
# np.set_printoptions(threshold=np.inf)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_aupr(pos_score, neg_score):
    scores = torch.cat([pos_score.cpu(), neg_score.cpu()]).squeeze().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)
    return aupr

def compute_auc_f1(pos_score, neg_score):
    scores = torch.cat([pos_score.cpu(), neg_score.cpu()]).squeeze().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    threshold = np.mean(scores)
    roc_auc = roc_auc_score(labels, scores)
    f1scores = np.zeros_like(scores)
    f1scores[scores >= threshold] = 1
    f1s = f1_score(labels, f1scores)
    return roc_auc, f1s

def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

# 计算节点间存在连接可能性的得分。
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h, etype):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        # section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']     

def construct_neg_graph(G, k, edge_type, mask_src):
    src_type, e_type, dst_type = edge_type
    n_src = G.num_nodes(src_type)
    n_dst = G.num_nodes(dst_type)
    src_G, dst_G = G.edges(etype = edge_type)
    mask_tmp = [set() for _ in range(n_src)]
    num_of_pos_edges = len(src_G)
    for i in range(num_of_pos_edges):
        mask_tmp[src_G[i].item()].add(dst_G[i].item())
    eids = np.arange(G.num_edges(etype=edge_type))
    neg_G = dgl.remove_edges(G, eids, etype=edge_type)
    for i in range(n_src):
        if len(mask_tmp[i]) == 0:
            continue
        v1 = mask_src[i]
        v2 = mask_tmp[i]
        neg_dst = set()
        while len(neg_dst) < k * len(v2):
            if len(neg_dst) + len(v2) >= n_dst:
                break
            x = torch.randint(0, n_dst, (1, )).item()
            if x not in v1 and x not in v2:
                neg_dst.add(x)
        for y in neg_dst:
            neg_G.add_edges(i, y, etype=edge_type)
    return neg_G
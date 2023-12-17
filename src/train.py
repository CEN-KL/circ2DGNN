#! /usr/bin/env python
# coding: utf-8

import pickle
import argparse
import pandas as pd
import datetime
from utils import *

'''
    to use fivefold cross validation, run in cmd line: 
    nohup python src/train.py --fivefold > /dev/null 2>&1 &

    to eval model on test set, run in cmd line: 
    nohup python src/train.py > /dev/null 2>&1 &

'''

# parse arg
parser = argparse.ArgumentParser(
    description="hyper parameters for circ2DGNN"
)
parser.add_argument("--n_epoch", 
                    type=int, 
                    default=70, 
                    help='number of epochs')
parser.add_argument("--n_inp", 
                    type=int, 
                    default=256, 
                    help='embedding size')
parser.add_argument("--n_hid", 
                    type=int, 
                    default=256, 
                    help='hidden layer size')
parser.add_argument("--clip", 
                    type=int, 
                    default=1.0, 
                    help='limit the range of the gradients')
parser.add_argument("--max_lr", 
                    type=float, 
                    default=1e-3, 
                    help='max learning rate for OneCycleLR')
parser.add_argument("--dropout", 
                    type=float, 
                    default=0.2, 
                    help='drop out')
parser.add_argument("--step_size", 
                    type=int, 
                    default=70, 
                    help='step size for stepLR')
parser.add_argument("--n_layers", 
                    type=int, 
                    default=2,
                    help='number of GNNLayers')
parser.add_argument("--n_heads", 
                    type=int, 
                    default=4,
                    help='number of attention heads')
parser.add_argument("--k_neg", 
                    type=int, 
                    default=1,
                    help='number of neg edges (k times of postive edges)')
parser.add_argument("--cuda", 
                    type=str, 
                    default='0',
                    help='cuda_idx')    
parser.add_argument("--comments", 
                    type=str, 
                    default='None',
                    help='comments for this train')    
parser.add_argument("--log_path", 
                    type=str, 
                    default='',
                    help='log file path')    
parser.add_argument("--log_file", 
                    type=str, 
                    default='',
                    help='log file name (dont append .log suffix)')    
parser.add_argument("--fivefold", 
                    action='store_true', 
                    help='use 5-fold cross validation or not')
parser.add_argument("--MLPPredictor", 
                    action='store_true', 
                    help='use HeteroMLPPredictor or not')

args = parser.parse_args()

# logging config
log_path = './logs/'
log_name = 'log_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.log'
if args.log_path:
    log_path = args.log_path
if args.log_file:
    log_name = args.log_file + '.log'
logging.basicConfig(filename=log_path + log_name, level=logging.DEBUG)

# logging.info hyper parameters
logging.info('===== Training Hyper Parameters =====')
logging.info('n_epoch: {}'.format(args.n_epoch))
logging.info('n_inp: {}'.format(args.n_inp))
logging.info('n_hid: {}'.format(args.n_hid))
logging.info('clip: {}'.format(args.clip))
logging.info('drop_out: {}'.format(args.dropout))
logging.info('n_layers: {}'.format(args.n_layers))
logging.info('n_heads: {}'.format(args.n_heads))
logging.info('k_neg: {}'.format(args.k_neg))
logging.info('cuda_idx: {}'.format(args.cuda))
logging.info('comments: {}'.format(args.comments))
logging.info('log file path: {}'.format(log_path + log_name))
if args.MLPPredictor:
    logging.info('use MLPPreidictor')
else:
    logging.info('use HeteroDotProductPredictor')
logging.info('=====================================')
logging.info('\nTraining starts at : {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# setup GPU device
cuda_idx = "cuda:" + args.cuda
device = torch.device(cuda_idx)

# for reproductivity
setup_seed(2023)

etype_to_pred = ('circ', 'circ_e_disease', 'disease')

node_dict = {}
edge_dict = {}

def train(model, G, G_test, neg_G_test):
    optimizer = torch.optim.AdamW(model.parameters())
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    scheduler = torch.optim.lr_scheduler.StepLR(
        # optimizer, total_steps=args.n_epoch, max_lr=args.max_lr
        optimizer, step_size=args.step_size
    )
    if args.MLPPredictor:
        pred = HeteroMLPPredictor(args.n_inp, args.n_inp)
    else:
        pred = HeteroDotProductPredictor()
    for epoch in range(1, args.n_epoch + 1):
        model.train()
        neg_G = construct_neg_graph(G, args.k_neg, etype_to_pred, mask_src)
        neg_G.to(device)
        h_after_gnn = model(G)
        pos_score, neg_score = pred(G, h_after_gnn, etype_to_pred), pred(neg_G, h_after_gnn, etype_to_pred)
        loss = compute_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()
        if epoch % 5 == 0:
            model.eval()
            h_after_gnn = model(G)
            pos_score, neg_score = pred(G, h_after_gnn, etype_to_pred), pred(neg_G, h_after_gnn, etype_to_pred)
            logging.info("In epoch {}, loss: {}".format(epoch, loss))

    with torch.no_grad():
        pos_score = pred(G_test, h_after_gnn, etype_to_pred)
        neg_score = pred(neg_G_test, h_after_gnn, etype_to_pred)
        auc, f1 = compute_auc_f1(pos_score, neg_score)
        aupr = compute_aupr(pos_score, neg_score)
        logging.info("AUC: {:.4f}  |  AUPR: {:.4f}  |  F1-score: {:.4f} ".format(auc, aupr, f1))

    return h_after_gnn, auc, f1, aupr

'''
# if you want to eval 5 folds on test set, run this
def fivefoldCV(G, G_test, neg_G_test):
'''
def fivefoldCV(G):
    etype_to_pred = ('circ', 'circ_e_disease', 'disease')
    u, v = G.edges(etype=etype_to_pred) 
    eids = np.arange(G.num_edges(etype=etype_to_pred))
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.2)
    train_size = G.num_edges(etype=etype_to_pred) - test_size
    start_idx = 0
    h_cv, auc_list, f1_list, aupr_list = [], [], [], []
    for i in range(5):
        logging.info('fold {} :'.format(i + 1))
        model = GNN(
            node_dict,
            edge_dict,
            n_inp=args.n_inp,
            n_hid=args.n_hid,
            n_out=args.n_inp,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            use_norm=True,
        ).to(device)
        if i != 4:
            test_pos_u, test_pos_v = u[eids[start_idx: start_idx + test_size]], v[eids[start_idx: start_idx + test_size]]
            train_pos_u, train_pos_v = torch.cat([u[:start_idx], u[start_idx + test_size:]]), torch.cat([v[:start_idx], v[start_idx + test_size:]])
        else:
            test_pos_u, test_pos_v = u[eids[start_idx:]], v[eids[start_idx:]]
            train_pos_u, train_pos_v = u[eids[:start_idx]], v[eids[:start_idx]]
        start_idx += test_size

        test_eids = G.edge_ids(test_pos_u, test_pos_v, etype=etype_to_pred)
        train_eids = G.edge_ids(train_pos_u, train_pos_v, etype=etype_to_pred)
        G_train_cv = dgl.remove_edges(G, test_eids, etype=etype_to_pred)
        G_val = dgl.remove_edges(G, train_eids, etype=etype_to_pred)
        neg_G_val = construct_neg_graph(G_val, args.k_neg, etype_to_pred, mask_src)
        h, auc, f1, aupr = train(model, G_train_cv, G_val, neg_G_val)
        h_cv.append(h)
        auc_list.append(auc)
        f1_list.append(f1)
        aupr_list.append(aupr)
        model.cpu()
        torch.cuda.empty_cache()
    logging.info('==========================')
    logging.info("AUC on 5 folds cross validation : {}".format(auc_list))
    logging.info("AUPR on 5 folds cross validation : {}".format(aupr_list))
    logging.info("F1-score on 5 folds cross validation : {}".format(f1_list))
    logging.info("avg AUC on 5 folds cross validation : {:.4f} ± {:.4f} ".format(np.mean(auc_list), np.std(auc_list)))
    logging.info("avg AUPR on 5 folds cross validation : {:.4f} ± {:.4f} ".format(np.mean(aupr_list), np.std(aupr_list)))
    logging.info("avg F1-score on 5 folds cross validation : {:.4f} ± {:.4f} ".format(np.mean(f1_list), np.std(f1_list)))
    logging.info('==========================')

    '''
    # for evaluating 5 folds on test set:
    pred = HeteroDotProductPredictor()
    with torch.no_grad():
        res_auc, res_f1, res_aupr = [], [], []
        for h in h_cv:
            pos_score = pred(G_test, h, etype_to_pred)
            neg_score = pred(neg_G_test, h, etype_to_pred)
            auc, f1 = compute_auc_f1(pos_score, neg_score)
            aupr = compute_aupr(pos_score, neg_score)
            res_auc.append(auc)
            res_f1.append(f1)
            res_aupr.append(aupr)
    logging.info("AUC for 5 folds on test set : {}".format(res_auc))
    logging.info("AUPR for 5 folds on test set : {}".format(res_aupr))
    logging.info("F1-score for 5 folds on test set : {}".format(res_f1))
    logging.info("avg AUC on test set: {:.4f} ± {:.4f} ".format(np.mean(res_auc), np.std(res_auc)))
    logging.info("avg AUPR on test set: {:.4f} ± {:.4f} ".format(np.mean(res_aupr), np.std(res_aupr)))
    logging.info("avg F1-score on test set: {:.4f} ± {:.4f} ".format(np.mean(res_f1), np.std(res_f1)))
    '''

def eval_test_set(G, G_test, neg_G_test):
    model = GNN(
            # G_train,
            node_dict,
            edge_dict,
            n_inp=args.n_inp,
            n_hid=args.n_hid,
            n_out=args.n_inp,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            use_norm=True,
        ).to(device)
    h, auc, f1, aupr = train(model, G, G_test, neg_G_test)

# read graph
graphs_path = './data/heterogeneous_graphs/heterographs.bin'
mask_src_path = './data/heterogeneous_graphs/mark_src.pkl'
g_list, _ = dgl.load_graphs(graphs_path)
G_tot, G_train, G_test, neg_G_test = g_list
with open(mask_src_path, 'rb') as f:
    mask_src = pickle.load(f)


for ntype in G_tot.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G_tot.etypes:
    edge_dict[etype] = len(edge_dict) 
    # G.edges[etype].data["id"] = ( torch.ones(G.num_edges(etype), dtype=torch.long) * edge_dict[etype] )

# initialize node input node feature
for ntype in G_train.ntypes:
    emb = nn.Parameter(
        torch.Tensor(G_train.num_nodes(ntype),  args.n_inp), requires_grad=False
    )
    nn.init.xavier_uniform_(emb)
    G_train.nodes[ntype].data["inp"] = emb


# G = G.to(device)
G_train = G_train.to(device)
'''
# for evaluating 5 folds on test set:
G_test = G_test.to(device)
neg_G_test = neg_G_test.to(device)
'''

if args.fivefold:
    logging.info('5-fold cross validation:')
    '''
    # for evaluating 5 folds on test set:
    fivefoldCV(G_train, G_test, neg_G_test)
    '''
    fivefoldCV(G_train)
else:
    G_test = G_test.to(device)
    neg_G_test = neg_G_test.to(device)
    logging.info('eval on test set:')
    eval_test_set(G_train, G_test, neg_G_test)
    

logging.info('\nTraining ends at : {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
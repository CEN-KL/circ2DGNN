import argparse
import pickle
from utils import *
from scipy.sparse import csc_matrix, load_npz, save_npz

setup_seed(2023)

# parse arg
parser = argparse.ArgumentParser(
    description="parameters for data split"
)
parser.add_argument("--k_neg", 
                    type=int, 
                    default=1,
                    help='number of neg edges (k times of postive edges)')

args = parser.parse_args()

# data loading
csc_path_file = './data/adj_matrix/'
csc_circ_disease     = load_npz(csc_path_file + '01_csc_circ_disease.npz')
csc_mi_disease       = load_npz(csc_path_file + '02_csc_mi_disease.npz')
csc_mi_circ          = load_npz(csc_path_file + '03_csc_mi_circ.npz')
csc_circ_compounds   = load_npz(csc_path_file + '04_csc_circ_compounds.npz')
csc_mi_compounds     = load_npz(csc_path_file + '05_csc_mi_compounds.npz')
csc_mi_gene          = load_npz(csc_path_file + '06_csc_mi_gene.npz')
csc_circ_gene        = load_npz(csc_path_file + '07_csc_circ_gene.npz')
csc_circ_mi          = load_npz(csc_path_file + '08_csc_circ_mi.npz')
csc_gene_mi          = load_npz(csc_path_file + '09_csc_gene_mi.npz')
csc_gene_rbp         = load_npz(csc_path_file + '10_csc_gene_rbp.npz')
csc_gene_tf          = load_npz(csc_path_file + '11_csc_gene_tf.npz')
csc_mi_rbp           = load_npz(csc_path_file + '12_csc_mi_rbp.npz')
csc_mi_tf            = load_npz(csc_path_file + '13_csc_mi_tf.npz')
csc_circ_seq         = load_npz(csc_path_file + '14_csc_circ_seq.npz')
csc_disease_semantic = load_npz(csc_path_file + '15_csc_disease_semantic.npz')

mean_seq_sim = csc_circ_seq.mean()
mean_dis_sim = csc_disease_semantic.mean()
csc_circ_seq.data[csc_circ_seq.data < mean_seq_sim] = 0
csc_disease_semantic.data[csc_disease_semantic.data < mean_dis_sim] = 0

G = dgl.heterograph(
    {
        ('circ', 'circ_e_disease',  'disease') : csc_circ_disease.nonzero(),
        ('circ', 'circ_e_cpd',          'cpd') : csc_circ_compounds.nonzero(),
        ('cpd',  'cpd_e_circ',         'circ') : csc_circ_compounds.transpose().nonzero(),
        ('circ', 'circ_e_gene',        'gene') : csc_circ_gene.nonzero(),
        ('gene',  'gene_e_circ',       'circ') : csc_circ_gene.transpose().nonzero(),
        ('circ', 'circ_e_mi',            'mi') : csc_circ_mi.nonzero(),

        ('circ',   'circ_seq',         'circ') : csc_circ_seq.nonzero(),
        ('disease', 'd_semantic',   'disease') : csc_disease_semantic.nonzero(),

        ('mi',   'mi_e_circ',          'circ') : csc_mi_circ.nonzero(),
        ('mi',   'mi_e_cpd',            'cpd') : csc_mi_compounds.nonzero(),
        ('cpd',  'cpd_e_mi',             'mi') : csc_mi_compounds.transpose().nonzero(),
        ('mi',   'mi_e_disease',    'disease') : csc_mi_disease .nonzero(),
        ('mi',   'mi_e_rbp',            'rbp') : csc_mi_rbp.nonzero(),
        ('rbp',  'rbp_e_mi',             'mi') : csc_mi_rbp.transpose().nonzero(),
        ('mi',   'mi_e_tf',              'tf') : csc_mi_tf.nonzero(),
        ('tf',   'tf_e_mi',              'mi') : csc_mi_tf.transpose().nonzero(),
        ('mi',   'mi_e_gene',          'gene') : csc_mi_gene.nonzero(),

        ('gene',   'gene_e_mi',          'mi') : csc_gene_mi.nonzero(),
        ('gene',   'gene_e_rbp',        'rbp') : csc_gene_rbp.nonzero(),
        ('rbp',   'rbp_e_gene',        'gene') : csc_gene_rbp.transpose().nonzero(),
        ('gene',   'gene_e_tf',          'tf') : csc_gene_tf.nonzero(),
        ('tf',     'tf_e_gene',        'gene') : csc_gene_tf.transpose().nonzero(),
    }
)

node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict) 
    G.edges[etype].data["id"] = ( torch.ones(G.num_edges(etype), dtype=torch.long) * edge_dict[etype] )

# split training set(G_train for 5-fold cv) and test set(G_test)
etype_to_pred = ('circ', 'circ_e_disease', 'disease')
u, v = G.edges(etype=etype_to_pred) 
eids = np.arange(G.num_edges(etype=etype_to_pred))
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = G.num_edges(etype=etype_to_pred) - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
test_eids = G.edge_ids(test_pos_u, test_pos_v, etype=etype_to_pred)
train_eids = G.edge_ids(train_pos_u, train_pos_v, etype=etype_to_pred)
G_train = dgl.remove_edges(G, test_eids, etype=etype_to_pred)
G_test = dgl.remove_edges(G, train_eids, etype=etype_to_pred)

'''
    mask_src
    type: dict
    key type: src node
    val type: dst node
    mask_src records positve edges in G and neg_G_test to avoid those edges being sampled in later negative edge generating
'''
# compute mask_src in training set
src_nodes = G.num_nodes(etype_to_pred[0])
dst_nodes = G.num_nodes(etype_to_pred[2])
mask_src = [set() for _ in range(src_nodes)]
num_of_pos_edges = u.shape[0]
for i in range(num_of_pos_edges):
    mask_src[u[i].item()].add(v[i].item())

# neg sampling for test set
neg_G = construct_neg_graph(G, 1, etype_to_pred, mask_src)
neg_u, neg_v = neg_G.edges(etype=etype_to_pred) 
neg_eids = np.arange(neg_G.num_edges(etype=etype_to_pred))
neg_eids = np.random.permutation(neg_eids)
neg_test_pos_u, neg_test_pos_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
neg_train_pos_u, neg_train_pos_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
neg_train_eids = neg_G.edge_ids(neg_train_pos_u, neg_train_pos_v, etype=etype_to_pred)
neg_G_test = dgl.remove_edges(neg_G, neg_train_eids, etype=etype_to_pred)

# update mask_src in test set
neg_u, neg_v = neg_G_test.edges(etype=etype_to_pred) 
num_of_neg_edges = neg_u.shape[0]
for i in range(num_of_neg_edges):
    mask_src[neg_u[i].item()].add(neg_v[i].item())

graph_pth = './data/heterogeneous_graphs/'

# save mark_src
with open(graph_pth + 'mark_src.pkl', 'wb') as f:
    pickle.dump(mask_src, f)

# save train set and test set
graph_labels = {"glabel": torch.tensor([0, 1, 2, 3])}
dgl.save_graphs(graph_pth + 'heterographs.bin', 
                [G, G_train, G_test, neg_G_test],
                graph_labels)

# create adjacent matrix for train/test set and save
# adjacent matrix for train set
A = csc_circ_disease.toarray()
A[test_pos_u, test_pos_v] = 0
csc_circ_disease_train = csc_matrix(A)

# adjacent matrix for test set
B = csc_circ_disease.toarray()
B[train_pos_u, train_pos_v] = 0
csc_circ_disease_test = csc_matrix(B)
save_npz('./data/adj_matrix/16_csc_circ_disease_train', csc_circ_disease_train)
save_npz('./data/adj_matrix/17_csc_circ_disease_test', csc_circ_disease_test)
import argparse 
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, load_npz, save_npz
import datetime
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(
    description="hyper parameters for word2vec"
)
parser.add_argument("--vector_size", type=int, default=100)
parser.add_argument("--window", type=int, default=5)
parser.add_argument("--min_count", type=int, default=2)
parser.add_argument("--workers", type=int, default=4)

args = parser.parse_args()


print('building disease semantic similarity')
print('\nScript starts at : {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# print hyper parameters
print('===== Training Hyper Parameters =====')
print('vector_size: {}'.format(args.vector_size))
print('window: {}'.format(args.window))
print('min_count: {}'.format(args.min_count))
print('workers: {}'.format(args.workers))
print('=====================================')


df_disease = pd.read_excel('./data/Entities/08_Disease_for_word2vec.xlsx')
sentences = [name.split() for name in df_disease['disease_name'].tolist()]
model = Word2Vec(sentences, vector_size=args.vector_size, window=args.window,
                min_count=args.min_count, workers=args.workers)

def calc_semantic_sim(i, j):
    sum_sim = 0
    cnt = 0
    for d1 in sentences[i]:
        for d2 in sentences[j]:
            if d1 in model.wv.key_to_index and d2 in model.wv.key_to_index:
                sim = model.wv.similarity(d1, d2)
                if sim > 0:
                    sum_sim += sim
            cnt += 1
    if cnt > 0:
        return sum_sim / cnt 
    else:
        return 0.0
    
n = df_disease.shape[0]
adj_disease_semantic = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        sim_ij = calc_semantic_sim(i, j) 
        adj_disease_semantic[i][j] = sim_ij
        adj_disease_semantic[j][i] = sim_ij
csc_disease_semantic = csc_matrix(adj_disease_semantic)
save_npz('./data/adj_matrix/15_csc_disease_semantic', csc_disease_semantic)
print('\nScript ends at : {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
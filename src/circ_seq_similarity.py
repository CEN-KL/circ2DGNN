import argparse 
import math
import numpy as np
import pandas as pd
import dgl
from scipy.sparse import csc_matrix, load_npz, save_npz
import datetime
from model import *
from utils import *
import openpyxl

print('building circRNA sequence similarity')
print('\nScript starts at : {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# 对于每个circRNA序列，输出一个 n_dim * n_dim * 3 维（一般取8 * 8 * 3维）的向量用于计算相似性
def calc_seq_vector(seq, n_dim = 8):
    seq_len = len(seq)
    x, y = [], []
    x_cur = y_cur = 0.5
    for nucleotide in seq:
        Fx = Fy = 0
        if nucleotide == 'A':
            Fx, Fy = 0, 0
        elif nucleotide == 'C':
            Fx, Fy = 0, 1
        elif nucleotide == 'T':
            Fx, Fy = 1, 0
        else:
            Fx, Fy = 1, 1
        x_cur = 0.5 * (x_cur + Fx)
        y_cur = 0.5 * (y_cur + Fy)
        x.append(x_cur)
        y.append(y_cur)
    
    vector_x = []
    vector_y = []
    vector_cnt = []
    position = []    # position[i] 表示第i个核苷酸所在的格子序号

    for i in range(seq_len):
        position.append(int(x[i] * n_dim) + int(y[i] * n_dim) * n_dim)
    
    for i in range(n_dim * n_dim):
        x_sum = y_sum = cnt = 0
        for j in range(seq_len):
            if i == position[j]:
                x_sum += x[j]
                y_sum += y[j]
                cnt += 1
        vector_x.append(x_sum)
        vector_y.append(y_sum)
        vector_cnt.append(cnt)

    seq_vector = []
    for i in range(n_dim * n_dim):
        cnt = vector_cnt[i]
        if cnt == 0:
            seq_vector.append(0)
        else:
            z_score = (cnt - np.average(vector_cnt)) / np.std(vector_cnt)
            seq_vector.append(z_score)
    seq_vector.extend(vector_x)
    seq_vector.extend(vector_y)
    return seq_vector


# 根据两个circRNA的序列表示向量，计算相似性
def calc_similarity(v1, v2):
    s1 = pd.Series(v1)
    s2 = pd.Series(v2)
    return s1.corr(s2)

df_circRNA = pd.read_excel('./data/Entities/01_circRNA.xlsx')
# print(df_circRNA.shape)

seq_vectors = []
circ_lack_of_seq = set()
for i, r in df_circRNA.iterrows():
    seq = r['seq']
    if isinstance(seq, float):
        circ_lack_of_seq.add(i)
        seq_vectors.append([])
        print(r['circBase_ID'])
        continue
    seq_vectors.append(calc_seq_vector(seq))

n = len(seq_vectors)
# print(n)

adj_circ_seq = np.zeros((n, n))
for i in range(n):
    if i % 100 == 0:
        print(f'now running with i = {i}')
    if i == 2454 or i == 2774:
        continue 
    for j in range(i + 1, n):
        if j == 2454 or j == 2774:
            continue
        sim_ij = calc_similarity(seq_vectors[i], seq_vectors[j])
        adj_circ_seq[i][j] = sim_ij
        adj_circ_seq[j][i] = sim_ij
csc_circ_seq = csc_matrix(adj_circ_seq)
save_npz('./data/adj_matrix/14_csc_circ_seq', csc_circ_seq)
print('\nScript ends at : {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
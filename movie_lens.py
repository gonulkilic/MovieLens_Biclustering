#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:21:49 2019

@author: gonul
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster.bicluster import SpectralBiclustering

with open("u.data", "r") as f:
    lines_data = f.readlines()

data=[]
for line in lines_data:
    column_seperated = line[0:-1].split('\t')
    int_column_seperated = list(map(int,column_seperated))
    data.append(int_column_seperated)
    
data=np.array(data)

users_list = np.unique(data[:,0])
item_list = np.unique(data[:,1])
data_matrix = np.zeros((len(users_list), len(item_list)))
for user in users_list:
    row_index = data[:,0] == user
    item_id = data[row_index,1]
    rate=data[row_index,2]
    data_matrix[user-1, item_id-1]= rate
    
n_clusters=(3,3)
num_biclusters = 3

model= SpectralBiclustering(n_clusters=n_clusters, method='log', n_components=6,
                            n_best=3, svd_method='randomized', n_svd_vecs=None,
                            mini_batch=False, init='k-means++', n_init=10, n_jobs=None, 
                            random_state=0)
model.fit(data_matrix)

sample_mat = data_matrix
fit_data=data_matrix[np.argsort(model.row_labels_)]    
fit_data=fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(data_matrix, cmap=plt.cm.Blues)
plt.title("Original dataset")
plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering;rearranged nto show biclusters")

plt.matshow(np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) +1 ), cmap=plt.cm.Blues)
plt.title("Checkboard structure of rearranged data")

plt.show()



num_users = sample_mat.shape[0]

num_distinct_rates= 5
theta = np.zeros((num_users, num_distinct_rates))
for m in range(1, num_distinct_rates+1):
    theta[:,m-1] = np.sum(sample_mat == m, axis=1)




def mmd_sim_u(u1,u2):
    Iu1 = np.sum(sample_mat[u1]>0)
    Iu2 = np.sum(sample_mat[u2]>0)
    denum = np.sum((theta[u1] - theta[u2])**2 - 1/Iu1 - 1/Iu2)
    return 1./(1+denum)
    

def jacard_u(u1,u2):
    num = (sample_mat[u1] * sample_mat[u2]) > 0
    denum = (sample_mat[u1] + sample_mat[u2]) > 0
    score = np.sum(num)/np.sum(denum)    
    return score

def cos_sim_u(u1,u2):
    return np.dot(sample_mat[u1], sample_mat[u2]) / (np.linalg.norm(sample_mat[u1])*np.linalg.norm(sample_mat[u2]))


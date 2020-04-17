#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri, 25 May 2018 20:29:09

@author: luohao
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set (torch tensor)
probFea: all feature vectors of the gallery set (torch tensor)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""

import numpy as np
import torch
import tqdm
#from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, lil_matrix


def mem_saving_divide(a, b):
    for i in range(len(a)):
        a[i] = a[i] / b
    return a


### 这里是减小内存开销的关键 （idx仅需要存储 [all_num, k1+1]，节约一个4 Byte大方阵，减少内存占用 28 GB，并大幅加快推理速度）
def mem_saving_argsort(a, top_k=10):
    assert a.ndim==2
    top_k = min(a.shape[1], top_k)
    idx = np.zeros((a.shape[0],top_k), dtype=np.int32)
    for i in tqdm.tqdm(range(len(a))):
        idx[i] = np.argsort(a[i])[:top_k]
    return idx

#################################
def compute_distmat_using_gpu(probFea, galFea, memory_save=True, mini_batch=5000):
    print('Computing distance using GPU ...')
    feat = torch.cat([probFea, galFea]).cuda()
    all_num = probFea.size(0) + galFea.size(0)
    if memory_save:
        distmat = torch.zeros((all_num, all_num), dtype=torch.float16)  # 14 GB memory on Round2
        i = 0
        while True:
            it = i + mini_batch
            # print('i, it', i, it)
            if it < feat.size()[0]:
                distmat[i:it, :] = torch.pow(torch.cdist(feat[i:it, :], feat), 2)
            else:
                distmat[i:, :] = torch.pow(torch.cdist(feat[i:, :], feat), 2)
                break
            i = it
    else:
        ### new API
        distmat = torch.pow(torch.cdist(feat, feat), 2)

    # print('Copy distmat to original_dist ...')
    original_dist = distmat.numpy()  # 14 GB memory
    del distmat
    del feat
    return original_dist

# torch version with GPU to accelerate distance computation
def re_ranking(original_dist, query_num, gallery_num, k1, k2, lambda_value, local_distmat=None, theta_value=0.5, only_local=False):
    assert k2 <= k1 + 1
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    all_num = query_num + gallery_num
    if only_local:
        original_dist = local_distmat
    else:
        if not local_distmat is None:
            original_dist = original_dist * theta_value + local_distmat * (1 - theta_value)
    gallery_num = original_dist.shape[0]

    ### memory optimization
    print('Division ...')
    # memory inefficient
    #original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    m = np.max(original_dist, axis=0)
    original_dist = mem_saving_divide(original_dist, m)
    #print('Transpose ...')
    original_dist = original_dist.T    # ultra fast
    ###
    print('Argsort to get initial_rank ...')
    #initial_rank = np.argsort(original_dist)  # .astype(np.int32)
    initial_rank = mem_saving_argsort(original_dist, top_k=k1+1)  # Time: 8.5 min

    print('Allocate V ...')
    #V = np.zeros_like(original_dist, dtype=np.float16) # 14 GB memory
    #V = csr_matrix(original_dist.shape, dtype=np.float32)
    V = lil_matrix(original_dist.shape, dtype=np.float32)

    print('Start re_ranking ...')
    for i in tqdm.tqdm(range(all_num)): # Time: 18 s
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num, :]
    if k2 != 1:
        #V_qe = np.zeros_like(V, dtype=np.float16) # 14 GB memory
        V_qe = lil_matrix(V.shape, dtype=np.float32)
        for i in tqdm.tqdm(range(all_num)): # Time: 12.6 min
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in tqdm.tqdm(range(gallery_num)):   # Time: 20 s
        #invIndex.append(np.where(V[:, i] != 0)[0])
        invIndex.append(V[:, i].nonzero()[0])

    ##### To save memory, don't allocate another matrix, re-use memory of original_dist instead
    #jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    print('V(_qe)', V.shape)



    for i in tqdm.tqdm(range(query_num)):    # Time: 1 min
        temp_min = np.zeros(shape=[gallery_num], dtype=np.float16)
        #indNonZero = np.where(V[i, :] != 0)[0]
        indNonZero = V[i, :].nonzero()[0]
        indImages = [invIndex[ind] for ind in indNonZero]

        print('indNonZero', len(indNonZero), indNonZero, 'indImages', len(indImages), indImages)

        for j in range(len(indNonZero)):
            #print(j, 'indImages[j]', indImages[j], indNonZero[j], V[i, indNonZero[j]], V[indImages[j], indNonZero[j]].todense().reshape(-1).shape)
            temp_min[indImages[j]] = temp_min[indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]].todense().reshape(-1))
        #######
        #jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
        jaccard_dist_i = 1 - temp_min / (2 - temp_min)
        original_dist[i] = jaccard_dist_i * (1 - lambda_value) + original_dist[i] * lambda_value
        #######

    #final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    #del original_dist
    del V
    #del jaccard_dist
    final_dist = original_dist[:query_num, query_num:]
    return final_dist








def re_ranking_old(probFea, galFea, k1, k2, lambda_value, local_distmat=None, theta_value=0.5, only_local=False,
               MemorySave=True, Minibatch = 5000):
    assert k2 <= k1 + 1
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        print('Computing original distance using GPU ...')
        feat = torch.cat([probFea, galFea]).cuda()

        if MemorySave:
            distmat = torch.zeros((all_num, all_num), dtype=torch.float16) # 14 GB memory on Round2
            i = 0
            while True:
                it = i + Minibatch
                #print('i, it', i, it)
                if it < feat.size()[0]:
                    distmat[i:it, :] = torch.pow(torch.cdist(feat[i:it, :], feat), 2)
                else:
                    distmat[i:, :] = torch.pow(torch.cdist(feat[i:, :], feat), 2)
                    break
                i = it
        else:
            ### new API
            distmat = torch.pow(torch.cdist(feat, feat), 2)

        #print('Copy distmat to original_dist ...')
        original_dist = distmat.numpy() # 14 GB memory
        del distmat
        del feat
        if not local_distmat is None:
            original_dist = original_dist * theta_value + local_distmat * (1 - theta_value)
    gallery_num = original_dist.shape[0]

    ### memory optimization
    print('Division ...')
    # memory inefficient
    #original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    m = np.max(original_dist, axis=0)
    original_dist = mem_saving_divide(original_dist, m)
    #print('Transpose ...')
    original_dist = original_dist.T    # ultra fast
    ###
    print('Argsort to get initial_rank ...')
    #initial_rank = np.argsort(original_dist)  # .astype(np.int32)
    initial_rank = mem_saving_argsort(original_dist, top_k=k1+1)  # Time: 8.5 min

    print('Allocate V ...')
    V = np.zeros_like(original_dist, dtype=np.float16) # 14 GB memory

    print('Start re_ranking ...')
    for i in tqdm.tqdm(range(all_num)): # Time: 18 s
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num, :]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16) # 14 GB memory
        for i in tqdm.tqdm(range(all_num)): # Time: 12.6 min
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in tqdm.tqdm(range(gallery_num)):   # Time: 20 s
        invIndex.append(np.where(V[:, i] != 0)[0])

    ##### To save memory, don't allocate another matrix, re-use memory of original_dist instead
    #jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in tqdm.tqdm(range(query_num)):    # Time: 1 min
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        #######
        #jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
        jaccard_dist_i = 1 - temp_min / (2 - temp_min)
        original_dist[i] = jaccard_dist_i * (1 - lambda_value) + original_dist[i] * lambda_value
        #######

    #final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    #del original_dist
    del V
    #del jaccard_dist
    final_dist = original_dist[:query_num, query_num:]
    return final_dist
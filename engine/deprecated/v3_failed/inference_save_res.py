# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import numpy as np
import torch.nn.functional as F
from data.datasets.eval_reid import evaluate
from data.prefetcher import data_prefetcher
from utils.re_ranking import re_ranking, re_ranking_v3
from utils.distance import low_memory_local_dist, local_dist


def inference(
        cfg,
        model,
        test_dataloader,
        num_query,
        thetas
):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    model.eval()

    feats, pids, camids = [], [], []
    local_feats = []
    test_prefetcher = data_prefetcher(test_dataloader)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        with torch.no_grad():
            feat = model(img)

        if isinstance(feat, tuple):
            feats.append(feat[0])
            local_feats.append(feat[1])
        else:
            feats.append(feat)
        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))

        batch = test_prefetcher.next()

    feats = torch.cat(feats, dim=0)
    if len(local_feats) > 0:
        local_feats = torch.cat(local_feats, dim=0)
    if cfg.TEST.NORM:
        feats = F.normalize(feats, p=2, dim=1)
        if len(local_feats) > 0:
            local_feats = F.normalize(local_feats, p=2, dim=1)
    # query
    qf = feats[:num_query]
    if len(local_feats) > 0:
        lqf = local_feats[:num_query]

    # gallery
    gf = feats[num_query:]
    if len(local_feats) > 0:
        lgf = local_feats[num_query:]

    if len(local_feats) > 0:
        # calculate the local distance
        lqf = lqf.permute(0, 2, 1)
        lgf = lgf.permute(0, 2, 1)
        #logger.info('Computing local_qg_distmat ...')
        local_qg_distmat = low_memory_local_dist(lqf.cpu().numpy(), lgf.cpu().numpy(), aligned=True)
        #logger.info('Computing local_qq_distmat ...')
        local_qq_distmat = low_memory_local_dist(lqf.cpu().numpy(), lqf.cpu().numpy(), aligned=True)
        #logger.info('Computing local_gg_distmat ...')
        local_gg_distmat = low_memory_local_dist(lgf.cpu().numpy(), lgf.cpu().numpy(), aligned=True)
        local_distmat = np.concatenate(
            [np.concatenate([local_qq_distmat, local_qg_distmat], axis=1),
             np.concatenate([local_qg_distmat.T, local_gg_distmat], axis=1)],
            axis=0)
    else:
        local_distmat = None

    use_rerank = True
    if use_rerank:
        #thetas = [0.4, 0.5, 0.9, 0.95, 1.0]
        scores, indices, dist_mats = [], [], []
        logger.info("use reranking")
        for theta in thetas:
            distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.3, local_distmat=local_distmat, theta_value=theta)
            score = distmat
            index = np.argsort(score, axis=1)  # from small to large

            scores.append(score)
            indices.append(index)
            dist_mats.append(distmat)

        return scores, indices, dist_mats
    else:
        logger.info("No reranking")
        distmat = -torch.mm(qf, gf.t()).cpu().numpy()

        score = distmat
        index = np.argsort(score, axis=1)  # from small to large

        return score, index

def compute_local_distmat(lqf, lgf):
    lqf = lqf.permute(0, 2, 1)
    lgf = lgf.permute(0, 2, 1)
    local_qg_distmat = low_memory_local_dist(lqf.cpu().numpy(), lgf.cpu().numpy(), aligned=True)
    local_qq_distmat = low_memory_local_dist(lqf.cpu().numpy(), lqf.cpu().numpy(), aligned=True)
    local_gg_distmat = low_memory_local_dist(lgf.cpu().numpy(), lgf.cpu().numpy(), aligned=True)
    local_distmat = np.concatenate(
        [np.concatenate([local_qq_distmat, local_qg_distmat], axis=1),
         np.concatenate([local_qg_distmat.T, local_gg_distmat], axis=1)],
        axis=0)
    return local_distmat

def inference_flipped(
        cfg,
        model,
        test_dataloader,
        num_query,
        thetas,
        use_local_feature=True # 是否使用local特征
):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    model.eval()

    feats, feats_flipped, pids, camids = [], [], [], []
    local_feats, local_feats_flipped = [], []
    local_feats2, local_feats2_flipped = [], []
    test_prefetcher = data_prefetcher(test_dataloader)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        with torch.no_grad():
            feat = model(img)
            feat_flipped = model(torch.flip(img, [3]))

        if isinstance(feat, tuple):
            feats.append(feat[0])
            feats_flipped.append(feat_flipped[0])

            if len(feat[1]) > 1:
                local_feats.append(feat[1][0])
                local_feats2.append(feat[1][1])

                local_feats_flipped.append(feat_flipped[1][0])
                local_feats2_flipped.append(feat_flipped[1][1])
            else:
                local_feats.append(feat[1])
                local_feats_flipped.append(feat_flipped[1])



        else:
            feats.append(feat)
            feats_flipped.append(feat_flipped)

        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))

        batch = test_prefetcher.next()

    feats = torch.cat(feats, dim=0)
    feats_flipped = torch.cat(feats_flipped, dim=0)
    if len(local_feats) > 0:
        local_feats = torch.cat(local_feats, dim=0)
        local_feats_flipped = torch.cat(local_feats_flipped, dim=0)

    if len(local_feats2) > 0:
        local_feats2 = torch.cat(local_feats2, dim=0)
        local_feats2_flipped = torch.cat(local_feats2_flipped, dim=0)

    if cfg.TEST.NORM:
        feats = F.normalize(feats, p=2, dim=1)
        feats_flipped = F.normalize(feats_flipped, p=2, dim=1)

    # query
    qf = feats[:num_query]
    qf_flipped = feats_flipped[:num_query]
    if len(local_feats) > 0:
        lqf = local_feats[:num_query]
        lqf_flipped = local_feats_flipped[:num_query]
    if len(local_feats2) > 0:
        lqf2 = local_feats2[:num_query]
        lqf2_flipped = local_feats2_flipped[:num_query]

    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])
    # gallery
    gf = feats[num_query:]
    gf_flipped = feats_flipped[num_query:]
    if len(local_feats) > 0:
        lgf = local_feats[num_query:]
        lgf_flipped = local_feats_flipped[num_query:]
    if len(local_feats2) > 0:
        lgf2 = local_feats2[num_query:]
        lgf2_flipped = local_feats2_flipped[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    # cosine distance
    # distmat = torch.mm(qf, gf.t()).cpu().numpy()

    if len(local_feats) > 0:
        local_distmat = compute_local_distmat(lqf, lgf)
        local_distmat_flipped = compute_local_distmat(lqf_flipped, lgf_flipped)
    else:
        local_distmat = None
        local_distmat_flipped = None

    if len(local_feats2) > 0:
        local_distmat2 = compute_local_distmat(lqf2, lgf2)
        local_distmat2_flipped = compute_local_distmat(lqf2_flipped, lgf2_flipped)
    else:
        local_distmat2 = None
        local_distmat2_flipped = None

    # use reranking
    logger.info("use reranking")
    theta = 0.95
    scores, indices, dist_mats = [],[],[]
    for theta2 in thetas:
        distmat = re_ranking_v3(qf, gf, k1=6, k2=2, lambda_value=0.3, local_distmat=local_distmat, theta_value=theta,
                                local_distmat2=local_distmat2, theta_value2=theta2,
                             only_local=False)
        distmat_flip = re_ranking_v3(qf_flipped, gf_flipped, k1=6, k2=2, lambda_value=0.3, local_distmat=local_distmat_flipped, theta_value=theta,
                                     local_distmat2=local_distmat2_flipped, theta_value2=theta2,
                                  only_local=False)
        # 合并距离
        distmat = (distmat + distmat_flip) / 2

        score = distmat
        index = np.argsort(score, axis=1)  # from small to large

        scores.append(score)
        indices.append(index)
        dist_mats.append(distmat)
    return scores, indices, dist_mats

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
from utils.re_ranking import re_ranking
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

    g_feats, l_feats, gf_feats, lf_feats, pids, camids = [], [], [], [], [], []
    val_prefetcher = data_prefetcher(test_dataloader)
    batch = val_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch

        with torch.no_grad():
            g_feat, l_feat = model(img)
            gf_feat, lf_feat = model(torch.flip(img, [3]))

        g_feats.append(g_feat.data.cpu())
        l_feats.append(l_feat.data.cpu())
        gf_feats.append(gf_feat.data.cpu())
        lf_feats.append(lf_feat.data.cpu())

        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))

        batch = val_prefetcher.next()

    g_feats = torch.cat(g_feats, dim=0)
    l_feats = torch.cat(l_feats, dim=0)
    gf_feats = torch.cat(gf_feats, dim=0)
    lf_feats = torch.cat(lf_feats, dim=0)

    if cfg.TEST.NORM:
        g_feats = F.normalize(g_feats, p=2, dim=1)
        gf_feats = F.normalize(gf_feats, p=2, dim=1)

    # query
    qf = g_feats[:num_query]
    lqf = l_feats[:num_query]
    qff = gf_feats[:num_query]
    lqff = lf_feats[:num_query]
    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])

    # gallery
    gf = g_feats[num_query:]
    lgf = l_feats[num_query:]
    gff = gf_feats[num_query:]
    lgff = lf_feats[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    # calculate the global distance
    scores, indices, dist_mats = [], [], []

    #use_local_feature = True
    if use_local_feature:
        logger.info("--------computing local features ...--------")
        lqf = lqf.permute(0, 2, 1)
        lgf = lgf.permute(0, 2, 1)
        local_distmat = low_memory_local_dist(lqf.numpy(), lgf.numpy(), aligned=True)
        local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(), aligned=True)
        local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(), aligned=True)
        local_dist = np.concatenate(
            [np.concatenate([local_qq_distmat, local_distmat], axis=1),
             np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
            axis=0)

        logger.info("--------computing flipped local features ...--------")
        lqff = lqff.permute(0, 2, 1)
        lgff = lgff.permute(0, 2, 1)
        local_distmat = low_memory_local_dist(lqff.numpy(), lgff.numpy(), aligned=True)
        local_qq_distmat = low_memory_local_dist(lqff.numpy(), lqff.numpy(), aligned=True)
        local_gg_distmat = low_memory_local_dist(lgff.numpy(), lgff.numpy(), aligned=True)
        local_dist_flip = np.concatenate(
            [np.concatenate([local_qq_distmat, local_distmat], axis=1),
             np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
            axis=0)
    else:
        local_dist = None
        local_dist_flip = None

    logger.info("use reranking")
    for theta in thetas:
        distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.3, local_distmat=local_dist, theta_value=theta,
                             only_local=False)
        distmat_flip = re_ranking(qff, gff, k1=6, k2=2, lambda_value=0.3, local_distmat=local_dist_flip, theta_value=theta,
                                  only_local=False)
        # 合并距离
        distmat = (distmat + distmat_flip) / 2

        score = distmat
        index = np.argsort(score, axis=1)  # from small to large

        scores.append(score)
        indices.append(index)
        dist_mats.append(distmat)
    return scores, indices, dist_mats

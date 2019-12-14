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
from utils.distance import low_memory_local_dist
import h5py
import time



top_k = 20


def inference_no_rerank(
        cfg,
        model,
        test_dataloader,
        num_query
):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    model.eval()

    feats, pids, camids = [], [], []
    test_prefetcher = data_prefetcher(test_dataloader, cfg)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        with torch.no_grad():
            feat = model(img)
            #feat = model(torch.flip(img, [3]))

        if isinstance(feat, tuple):
            feats.append(feat[0])
            #local_feats.append(feat[1])
        else:
            feats.append(feat)
        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))

        batch = test_prefetcher.next()

    feats = torch.cat(feats, dim=0)
    if cfg.TEST.NORM:
        feats = F.normalize(feats, p=2, dim=1)

    # query
    qf = feats[:num_query]

    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])

    # gallery
    gf = feats[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    distmat = -torch.mm(qf, gf.t()).cpu().numpy()

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
    logger.info(f"Score: {(mAP + cmc[0]) / 2.:.1%}")

    index = np.argsort(distmat, axis=1)  # from small to large

    new_gallery_index = np.unique(index[:, :top_k].reshape(-1))

    print('new_gallery_index', len(new_gallery_index))


    return new_gallery_index

    #return distmat, index

def inference(
        cfg,
        model,
        test_dataloader,
        num_query
):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    model.eval()

    feats, pids, camids = [], [], []
    #local_feats = []
    test_prefetcher = data_prefetcher(test_dataloader, cfg)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        with torch.no_grad():
            feat = model(img)
            #feat = model(torch.flip(img, [3]))

        if isinstance(feat, tuple):
            feats.append(feat[0])
            #local_feats.append(feat[1])
        else:
            feats.append(feat)
        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))

        batch = test_prefetcher.next()

    feats = torch.cat(feats, dim=0)
    if cfg.TEST.NORM:
        feats = F.normalize(feats, p=2, dim=1)


    # query
    qf = feats[:num_query]

    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])

    # gallery
    gf = feats[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    local_distmat = None
    # use reranking
    logger.info("use reranking")
    #distmat = re_ranking(qf, gf, k1=14, k2=4, lambda_value=0.4)

    search_param = False
    if search_param:
        best_score = 0
        best_param = []
        for k1 in range(5, 9):
            for k2 in range(1, k1):
                for l in np.linspace(0, 0.5, 11):
                    distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=l)
                    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
                    score = (cmc[0] + mAP) / 2
                    #logger.info(f"mAP: {mAP:.1%}")
                    print('k1, k2, l', k1, k2, np.around(l,2), 'r1, mAP, score', np.around(cmc[0], 4), np.around(mAP, 4), np.around(score, 4))
                    if score > best_score:
                        best_score = score
                        best_param = [k1, k2, l]
        print('Best Param', best_param)
        distmat = re_ranking(qf, gf, k1=best_param[0], k2=best_param[1], lambda_value=best_param[2], local_distmat=local_distmat, only_local=False)
    else:
        distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.3, local_distmat=local_distmat, only_local=False, theta_value=0.9) #(current best)
        #distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.4) # try

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
    logger.info(f"Score: {(mAP + cmc[0]) / 2.:.1%}")


def inference_aligned(
        cfg,
        model,
        test_dataloader,
        num_query
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
            #feat = model(torch.flip(img, [3]))

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
        # 局部特征是三维的，不做归一化 (对结果没影响)
        #if len(local_feats) > 0:
        #    local_feats = F.normalize(local_feats, p=2, dim=1)

    # query
    qf = feats[:num_query]
    if len(local_feats) > 0:
        lqf = local_feats[:num_query]

    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])

    # gallery
    gf = feats[num_query:]
    if len(local_feats) > 0:
        lgf = local_feats[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])


    if len(local_feats) > 0:
    #if True:
        # calculate the local distance
        lqf = lqf.permute(0, 2, 1)
        lgf = lgf.permute(0, 2, 1)
        local_qg_distmat = low_memory_local_dist(lqf.cpu().numpy(), lgf.cpu().numpy(), aligned=True)
        local_qq_distmat = low_memory_local_dist(lqf.cpu().numpy(), lqf.cpu().numpy(), aligned=True)
        local_gg_distmat = low_memory_local_dist(lgf.cpu().numpy(), lgf.cpu().numpy(), aligned=True)
        local_distmat = np.concatenate(
            [np.concatenate([local_qq_distmat, local_qg_distmat], axis=1),
             np.concatenate([local_qg_distmat.T, local_gg_distmat], axis=1)],
            axis=0)

    else:
        local_distmat = None


    # use reranking
    logger.info("use reranking")
    #distmat = re_ranking(qf, gf, k1=14, k2=4, lambda_value=0.4)

    search_param = False
    search_theta = True
    if search_param:
        best_score = 0
        best_param = []
        for k1 in range(5, 9):
            for k2 in range(1, k1):
                for l in np.linspace(0, 0.5, 11):
                    distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=l)
                    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
                    score = (cmc[0] + mAP) / 2
                    #logger.info(f"mAP: {mAP:.1%}")
                    print('k1, k2, l', k1, k2, np.around(l,2), 'r1, mAP, score', np.around(cmc[0], 4), np.around(mAP, 4), np.around(score, 4))
                    if score > best_score:
                        best_score = score
                        best_param = [k1, k2, l]
        print('Best Param', best_param)
        distmat = re_ranking(qf, gf, k1=best_param[0], k2=best_param[1], lambda_value=best_param[2], local_distmat=local_distmat, only_local=False)
    elif search_theta:
        best_score = 0
        for theta in np.linspace(0, 1.0, 11):
            distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.3, local_distmat=local_distmat, theta_value=theta,
                                 only_local=False)  # (current best)
            cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
            score = (cmc[0] + mAP) / 2
            print('theta', theta, 'r1, mAP, score', np.around(cmc[0], 4), np.around(mAP, 4),
                  np.around(score, 4))
            if score > best_score:
                best_score = score
                best_param = theta
        print('Best Param', best_param)
        distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.3, local_distmat=local_distmat, theta_value=best_param,
                             only_local=False)  # (current best)
    else:
        distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.3, local_distmat=local_distmat, only_local=False, theta_value=0.9) #(current best)
        #distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.4) # try

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
    logger.info(f"Score: {(mAP + cmc[0]) / 2.:.1%}")



def compute_distmat(cfg, num_query, feats, local_feats, theta,
                    use_local_feature, use_rerank):
    """
    Given a pair of global feature and local feature, compute distmat.
    :param cfg:
    :param num_query:
    :param pids:
    :param camids:
    :param feats:
    :param feats_flipped:
    :param local_feats:
    :param local_feats_flipped:
    :param theta:
    :param use_local_feature:
    :return:
    """
    feats = torch.cat(feats, dim=0)
    if len(local_feats) > 0 and use_local_feature:
        local_feats = torch.cat(local_feats, dim=0)

    if cfg.TEST.NORM:
        feats = F.normalize(feats, p=2, dim=1)

    ## torch to numpy to save memory in re_ranking
    #feats = feats.numpy()
    #feats_flipped = feats_flipped.numpy()
    ###########

    # query
    qf = feats[:num_query]
    if len(local_feats) > 0 and use_local_feature:
        lqf = local_feats[:num_query]

    # gallery
    gf = feats[num_query:]
    if len(local_feats) > 0:
        lgf = local_feats[num_query:]
    #g_pids = np.asarray(pids[num_query:])
    #g_camids = np.asarray(camids[num_query:])


    if len(local_feats) > 0 and use_local_feature:
        # if True:
        # calculate the local distance
        lqf = lqf.permute(0, 2, 1)
        lgf = lgf.permute(0, 2, 1)
        local_qg_distmat = low_memory_local_dist(lqf.numpy(), lgf.numpy(), aligned=True)
        local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(), aligned=True)
        local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(), aligned=True)
        local_distmat = np.concatenate(
            [np.concatenate([local_qq_distmat, local_qg_distmat], axis=1),
             np.concatenate([local_qg_distmat.T, local_gg_distmat], axis=1)],
            axis=0)
    else:
        local_distmat = None
        local_distmat_flipped = None

    if use_rerank:
        distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.3, local_distmat=local_distmat, theta_value=theta,
                         only_local=False)  # (current best)
        del qf, gf
    else:
        distmat = -torch.mm(qf, gf.t()).cpu().numpy()

    return distmat

############
## Using
def inference_aligned_flipped(
        cfg,
        model,
        test_dataloader,
        num_query,
        use_local_feature,
        use_rerank,
        use_cross_feature
):
    """
    inference an aligned net with flipping and two pairs of global feature and local feature
    :param cfg:
    :param model:
    :param test_dataloader:
    :param num_query:
    :return:
    """
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing aligned with flipping")

    model.eval()

    pids, camids = [], []
    gfs, bn_gfs, lfs, bn_lfs = [], [], [], []

    test_prefetcher = data_prefetcher(test_dataloader, cfg)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        with torch.no_grad():
            gf, bn_gf, lf, bn_lf = model(img)
            gff, bn_gff, lff, bn_lff = model(torch.flip(img, [3]))

            # combine flipped feature here to save computation
            gf = (gf + gff) / 2
            bn_gf = (bn_gf + bn_gff) / 2
            lf = (lf + lff) / 2
            bn_lf = (bn_lf + bn_lff) / 2

        # 4 features
        gfs.append(gf.cpu())
        bn_gfs.append(bn_gf.cpu())

        if use_local_feature:
            lfs.append(lf.cpu())
            bn_lfs.append(bn_lf.cpu())


        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))

        batch = test_prefetcher.next()

    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    logger.info(f"use_cross_feature = {use_cross_feature}, use_local_feature = {use_local_feature}, use_rerank = {use_rerank}")

    if use_cross_feature:
        logger.info("Computing distmat with bn_gf (+ lf)")
        distmat2 = compute_distmat(cfg, num_query, bn_gfs, lfs, theta=0.45,
                                   use_local_feature=use_local_feature, use_rerank=use_rerank)
        distmat = distmat2
        #distmat = (distmat1 + distmat2) / 2
    else:
        logger.info("Computing distmat with gf + bn_lf")
        distmat1 = compute_distmat(cfg, num_query, gfs, bn_lfs, theta=0.95,
                                   use_local_feature=use_local_feature, use_rerank=use_rerank)
        distmat = distmat1
        #distmat1 = None
        #distmat2 = None

    #distmat = original_distmat
    #distmat[:, new_gallery_index] = distmat1 - 100

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
    logger.info(f"Score: {(mAP + cmc[0]) / 2.:.1%}")



def inference_with_distmat(
        cfg,
        test_dataloader,
        num_query,
        distmat,
):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    pids, camids = [], []
    test_prefetcher = data_prefetcher(test_dataloader, cfg)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))
        batch = test_prefetcher.next()

    # query
    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])
    # gallery
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    #distmat = re_ranking(qf, gf, k1=14, k2=4, lambda_value=0.4)

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
    logger.info(f"Score: {(mAP + cmc[0]) / 2.:.1%}")


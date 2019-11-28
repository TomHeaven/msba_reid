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
    local_feats = []
    test_prefetcher = data_prefetcher(test_dataloader)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        with torch.no_grad():
            #feat = model(img)
            feat = model(torch.flip(img, [3]))

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


def inference_aligned(
        cfg,
        model,
        test_dataloader,
        num_query,
):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    model.eval()

    g_feats, l_feats, pids, camids = [], [], [], []
    val_prefetcher = data_prefetcher(test_dataloader)
    batch = val_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        with torch.no_grad():
            g_feat, l_feat = model(img)
            #g_feat, l_feat = model(torch.flip(img, [3])) # better
        g_feats.append(g_feat.data.cpu())
        l_feats.append(l_feat.data.cpu())
        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))

        batch = val_prefetcher.next()

    g_feats = torch.cat(g_feats, dim=0)
    l_feats = torch.cat(l_feats, dim=0)

    if cfg.TEST.NORM:
        g_feats = F.normalize(g_feats, p=2, dim=1)

    # query
    qf = g_feats[:num_query]
    lqf = l_feats[:num_query]
    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])

    # gallery
    gf = g_feats[num_query:]
    lgf = l_feats[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    # calculate the global distance
    if True:
        logger.info("--------use re-ranking--------")
        lqf = lqf.permute(0, 2, 1)
        lgf = lgf.permute(0, 2, 1)
        local_distmat = low_memory_local_dist(lqf.numpy(), lgf.numpy(), aligned=True)
        local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(), aligned=True)
        local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(), aligned=True)
        local_dist = np.concatenate(
            [np.concatenate([local_qq_distmat, local_distmat], axis=1),
             np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
            axis=0)

        distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.3, local_distmat=local_dist, theta_value=0.5,
                             only_local=False)
        ## theta hyer-patameters
        # for theta in np.arange(0,1.1,0.1):
        #     distmat = re_ranking(qf,gf,k1=6,k2=2,lambda_value=0.3,local_distmat=local_dist,theta=theta,only_local=False)
        #     cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        #     logger.info(f"mAP: {mAP:.1%}")
        #     for r in [1, 5, 10]:
        #         logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
        #     logger.info("Theta:{}; Score: {}".format(theta, (mAP+cmc[0])/2.))

    #score = distmat
    #index = np.argsort(score, axis=1)  # from small to large


    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
    logger.info(f"Score: {(mAP + cmc[0]) / 2.:.1%}")



def compute_distmat(cfg, num_query, pids, camids, feats, feats_flipped, local_feats, local_feats_flipped, theta):
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
    :return:
    """
    feats = torch.cat(feats, dim=0)
    feats_flipped = torch.cat(feats_flipped, dim=0)
    if len(local_feats) > 0:
        local_feats = torch.cat(local_feats, dim=0)
        local_feats_flipped = torch.cat(local_feats_flipped, dim=0)

    # print('feats_flipped', len(feats_flipped), feats_flipped[0])
    if cfg.TEST.NORM:
        feats = F.normalize(feats, p=2, dim=1)
        feats_flipped = F.normalize(feats_flipped, p=2, dim=1)

    # query
    qf = feats[:num_query]
    qf_flipped = feats_flipped[:num_query]
    if len(local_feats) > 0:
        lqf = local_feats[:num_query]
        lqf_flipped = local_feats_flipped[:num_query]

    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])
    # gallery
    gf = feats[num_query:]
    gf_flipped = feats_flipped[num_query:]
    if len(local_feats) > 0:
        lgf = local_feats[num_query:]
        lgf_flipped = local_feats_flipped[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])


    if len(local_feats) > 0:
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

        # flipped
        lqf = lqf_flipped.permute(0, 2, 1)
        lgf = lgf_flipped.permute(0, 2, 1)
        local_qg_distmat = low_memory_local_dist(lqf.numpy(), lgf.numpy(), aligned=True)
        local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(), aligned=True)
        local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(), aligned=True)
        local_distmat_flipped = np.concatenate(
            [np.concatenate([local_qq_distmat, local_qg_distmat], axis=1),
             np.concatenate([local_qg_distmat.T, local_gg_distmat], axis=1)],
            axis=0)

    else:
        local_distmat = None
        local_distmat_flipped = None

    distmat = re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.3, local_distmat=local_distmat, theta_value=theta,
                         only_local=False)  # (current best)

    distmat_flipped = re_ranking(qf_flipped, gf_flipped, k1=6, k2=2, lambda_value=0.3,
                                 local_distmat=local_distmat_flipped, theta_value=theta,
                                 only_local=False)  # (current best)

    distmat = (distmat + distmat_flipped) / 2

    #if True:
    #    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    #    score = (cmc[0] + mAP) / 2
    #return qf, gf, local_distmat, qf_flipped, gf_flipped, local_distmat_flipped

    return distmat

def inference_aligned_flipped(
        cfg,
        model,
        test_dataloader,
        num_query
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
    gfs_flipped, bn_gfs_flipped, lfs_flipped, bn_lfs_flipped = [], [], [], []

    test_prefetcher = data_prefetcher(test_dataloader, cfg)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        with torch.no_grad():
            gf, bn_gf, lf, bn_lf = model(img)
            gff, bn_gff, lff, bn_lff = model(torch.flip(img, [3]))

        # 4 features
        gfs.append(gf.cpu())
        bn_gfs.append(bn_gf.cpu())
        lfs.append(lf.cpu())
        bn_lfs.append(bn_lf.cpu())
        # 4 features flipped
        gfs_flipped.append(gff.cpu())
        bn_gfs_flipped.append(bn_gff.cpu())
        lfs_flipped.append(lff.cpu())
        bn_lfs_flipped.append(bn_lff.cpu())

        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))

        batch = test_prefetcher.next()

    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    logger.info("Computing distmat with gf + bn_lf")
    distmat1 = compute_distmat(cfg, num_query, pids, camids, gfs, gfs_flipped, bn_lfs, bn_lfs_flipped, theta=0.95)
    logger.info("Computing distmat with bn_gf + lf")
    distmat2 = compute_distmat(cfg, num_query, pids, camids, bn_gfs, bn_gfs_flipped, lfs, lfs_flipped, theta=0.45)

    search_alpha = True
    if search_alpha:
        best_score = 0
        #for theta in np.linspace(0.9, 1.0, 11):
        for alpha in np.linspace(0, 1.0, 21):
            distmat = distmat1 * alpha + distmat2 * (1 - alpha)
            cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
            score = (cmc[0] + mAP) / 2
            print('alpha', np.around(alpha, 2), 'r1, mAP, score', np.around(cmc[0], 4), np.around(mAP, 4),
                  np.around(score, 4))
            if score > best_score:
                best_score = score
                best_param = alpha
                best_distmat = distmat

             # saving
            strtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            target_alpha = 0.5
            if abs(alpha - target_alpha) < 1e-4:
                # saving dist_mats
                f = h5py.File('dist_mats/val_%s_%s_4f_a%.3f.h5' % (cfg.MODEL.NAME, strtime, alpha), 'w')
                f.create_dataset('dist_mat', data=distmat, compression='gzip')
                f.create_dataset('dist_mat1', data=distmat1, compression='gzip')
                f.create_dataset('dist_mat2', data=distmat2, compression='gzip')
                f.close()

        print('Best Param', best_param)
        distmat = best_distmat
    else:
        distmat = (distmat1 + distmat2) / 2

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


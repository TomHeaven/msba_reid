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
from utils.re_ranking import re_ranking, mem_saving_argsort
from utils.distance import low_memory_local_dist, local_dist

from engine.inference import compute_distmat


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
            ret = model(img)
            ret_flip = model(torch.flip(img, [3]))
            if len(ret) == 4:
                gf, bn_gf, lf, bn_lf = ret
                gff, bn_gff, lff, bn_lff = ret_flip
            elif len(ret) == 2:
                gf, bn_gf = ret
                gff, bn_gff = ret_flip
                lf, bn_lf = None, None
                lff, bn_lff = None, None
            elif ret is not tuple:
                gf = bn_gf = ret
                gff = bn_gff = ret_flip
                lf, bn_lf = None, None
                lff, bn_lff = None, None
            else:
                #print('ret', ret.size())
                raise Exception("Unknown model returns, length = ", len(ret))

        # 4 features
        gfs.append(gf.cpu())
        bn_gfs.append(bn_gf.cpu())

        if use_local_feature:
            lfs.append(lf.cpu())
            bn_lfs.append(bn_lf.cpu())

        # 4 features flipped
        gfs_flipped.append(gff.cpu())
        bn_gfs_flipped.append(bn_gff.cpu())

        if use_local_feature:
            lfs_flipped.append(lff.cpu())
            bn_lfs_flipped.append(bn_lff.cpu())

        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))

        batch = test_prefetcher.next()


    logger.info(f"use_local_feature = {use_local_feature}, use_rerank = {use_rerank}")

    logger.info("Computing distmat with bn_gf (+ lf)")
    distmat2 = compute_distmat(cfg, num_query, bn_gfs, bn_gfs_flipped, lfs, lfs_flipped, theta=0.45,
                               use_local_feature=use_local_feature, use_rerank=use_rerank)

    logger.info("Computing distmat with gf (+ bn_lf)")
    distmat1 = compute_distmat(cfg, num_query, gfs, gfs_flipped, bn_lfs, bn_lfs_flipped, theta=0.95,
                                   use_local_feature=use_local_feature, use_rerank=use_rerank)
    theta = 0.55
    distmat = distmat1 * (1 - theta) + distmat2 * theta

    score = distmat
    index = np.argsort(score, axis=1)  # from small to large
    #index = mem_saving_argsort(score) # better for large matrix ?

    return distmat, index, distmat1, distmat2


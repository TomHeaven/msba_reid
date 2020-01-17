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

from .inference import compute_distmat


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
    bn_gfs = []
    bn_gfs_flipped = []

    test_prefetcher = data_prefetcher(test_dataloader, cfg)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        with torch.no_grad():
            ret = model(img)
            ret_flip = model(torch.flip(img, [3]))

            if len(ret) == 2:
                gf, bn_gf = ret
                gff, bn_gff = ret_flip
            elif ret is not tuple:
                bn_gf = ret
                bn_gff = ret_flip
            else:
                # print('ret', ret.size())
                raise Exception("Unknown model returns, length = ", len(ret))

        # 4 features
        bn_gfs.append(bn_gf.cpu())

        # 4 features flipped
        bn_gfs_flipped.append(bn_gff.cpu())

        pids.extend(pid.cpu().numpy())
        camids.extend(np.asarray(camid))

        batch = test_prefetcher.next()

    distmat1 = None
    logger.info(f"use_cross_feature = {use_cross_feature}, use_local_feature = {use_local_feature}, use_rerank = {use_rerank}")
    if use_cross_feature:
        logger.info("Computing distmat with bn_gf (+ lf)")
        distmat2 = compute_distmat(cfg, num_query, bn_gfs, bn_gfs_flipped, None, None, theta=0.45,
                               use_local_feature=use_local_feature, use_rerank=use_rerank)
        distmat = distmat2
        #distmat = (distmat1 + distmat2) / 2
    else:
        raise Exception('wrong feature!')

    score = distmat
    index = np.argsort(score, axis=1)  # from small to large
    #index = mem_saving_argsort(score) # better for large matrix ?

    return distmat, index, distmat1, distmat2


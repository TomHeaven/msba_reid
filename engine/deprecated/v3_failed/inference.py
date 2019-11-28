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

    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])
    # gallery
    gf = feats[num_query:]
    if len(local_feats) > 0:
        lgf = local_feats[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    # cosine distance
    distmat = torch.mm(qf, gf.t()).cpu().numpy()

    local_distmat = torch.mm(lqf, lgf.t()).cpu().numpy()

    # euclidean distance
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())
    # distmat = distmat.numpy()
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"global feature mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

    cmc, mAP = evaluate(-local_distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"local feature mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

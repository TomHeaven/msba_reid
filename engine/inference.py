# encoding: utf-8
"""
@author:  Hanlin Tan & Huaxin Xiao & Xiaoyu Zhang
@contact: hanlin_tan@nudt.edu.cn
"""
import logging

import torch
import numpy as np
import torch.nn.functional as F
from data.datasets.eval_reid import evaluate
from data.prefetcher import data_prefetcher
from utils.re_ranking import re_ranking, compute_distmat_using_gpu
from utils.distance import low_memory_local_dist, fast_local_dist
import h5py
import time
import os, cv2



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



def compute_distmat(cfg, num_query, feats, feats_flipped, local_feats, local_feats_flipped, theta,
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
    feats_flipped = torch.cat(feats_flipped, dim=0)
    if local_feats is not None and len(local_feats) > 0 and use_local_feature:
        local_feats = torch.cat(local_feats, dim=0)
        local_feats_flipped = torch.cat(local_feats_flipped, dim=0)

    # print('feats_flipped', len(feats_flipped), feats_flipped[0])
    if cfg.TEST.NORM:
        feats = F.normalize(feats, p=2, dim=1)
        feats_flipped = F.normalize(feats_flipped, p=2, dim=1)

    ## torch to numpy to save memory in re_ranking
    #feats = feats.numpy()
    #feats_flipped = feats_flipped.numpy()
    ###########

    # query
    qf = feats[:num_query]
    qf_flipped = feats_flipped[:num_query]
    if local_feats is not None and len(local_feats) > 0 and use_local_feature:
        lqf = local_feats[:num_query]
        lqf_flipped = local_feats_flipped[:num_query]

    #q_pids = np.asarray(pids[:num_query])
    #q_camids = np.asarray(camids[:num_query])

    # gallery
    gf = feats[num_query:]
    gf_flipped = feats_flipped[num_query:]
    if local_feats is not None and len(local_feats) > 0:
        lgf = local_feats[num_query:]
        lgf_flipped = local_feats_flipped[num_query:]
    #g_pids = np.asarray(pids[num_query:])
    #g_camids = np.asarray(camids[num_query:])

    local_distmat1 = None
    local_distmat2 = None

    aligned = True

    if use_rerank:
        #print('len(local_feats)', len(local_feats))

        if local_feats is not None and len(local_feats) > 0 and use_local_feature:
            # if True:
            # calculate the local distance
            lqf = lqf.permute(0, 2, 1)
            lgf = lgf.permute(0, 2, 1)
            local_distmat1  = fast_local_dist(lqf.numpy(), lgf.numpy(), aligned=aligned)
            del lqf, lgf

        query_num = qf.size(0)
        gallery_num = gf.size(0)

        distmat = compute_distmat_using_gpu(qf, gf, local_distmat1, theta=theta)
        distmat1 = re_ranking(distmat, query_num, gallery_num, k1=12, k2=2, lambda_value=0.3,
                              #local_distmat=local_distmat1,
                              #theta_value=theta,
                              only_local=False)
        del distmat, local_distmat1

        if  local_feats is not None and len(local_feats) > 0 and use_local_feature:
            # flipped
            lqf = lqf_flipped.permute(0, 2, 1)
            lgf = lgf_flipped.permute(0, 2, 1)
            local_distmat2 = fast_local_dist(lqf.numpy(), lgf.numpy(), aligned=aligned)
            del lqf, lgf, lqf_flipped, lgf_flipped

        distmat = compute_distmat_using_gpu(qf_flipped, gf_flipped, local_distmat2, theta=theta)
        distmat2 = re_ranking(distmat, query_num, gallery_num, k1=12, k2=2, lambda_value=0.3, only_local=False)  # (current best)
        del distmat, local_distmat2
        distmat = (distmat1 + distmat2) / 2

    else:
        distmat1 = -torch.mm(qf, gf.t()).cpu().numpy()
        distmat2= -torch.mm(qf_flipped, gf_flipped.t()).cpu().numpy()

        distmat = (distmat1 + distmat2) / 2

    #if True:
    #    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    #    score = (cmc[0] + mAP) / 2
    #return qf, gf, local_distmat, qf_flipped, gf_flipped, local_distmat_flipped

    return distmat

############
def inference_vis(
        cfg,
        model,
        test_dataloader,
        num_query,
        dataset
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

    # Visualize feature maps
    activation = {}
    def hook(module, input, output):
        setattr(module, "_value_hook", output)

    for n, m in model.named_modules():
        #print('name', n)
        if n == "base.layer4.2.relu":
            m.register_forward_hook(hook)
    ####

    test_prefetcher = data_prefetcher(test_dataloader, cfg)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        with torch.no_grad():
            ret = model(img)

            break

    for n, m in model.named_modules():
        if n == "base.layer4.2.relu":
            in_output = m._value_hook
    
    #act = activation['relu'].squeeze()
    print('in_output ', in_output.size())
    in_output = in_output.permute(0, 2, 3, 1)
    act = np.zeros((in_output.size(0), in_output.size(1), in_output.size(2)))
    act = in_output.sum(dim=3).cpu().numpy()
    print('act', act.shape)
    act = act / act.max()

    if not os.path.isdir('vis'):
        os.mkdir('vis')
    img = img.permute(0, 2, 3, 1).cpu().numpy()

    query_names = dataset.query

    for idx in range(act.shape[0]):
        print('query_names', query_names[idx][0])

        img = cv2.imread(query_names[idx][0])
        a = cv2.resize(act[idx], (img.shape[1], img.shape[0]))
        heatmapshow = cv2.applyColorMap(np.uint8(a * 255.0), cv2.COLORMAP_JET)
        res = cv2.addWeighted(heatmapshow, 0.5, img, 0.5, 0)
        cv2.imwrite('vis/fea_%d.png' % idx, res)
        #break

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
                # print('ret', ret.size())
                raise Exception("Unknown model returns, length = ", len(ret))
            
        # 4 features
        gfs.append(gf.cpu())
        bn_gfs.append(bn_gf.cpu())

        if use_local_feature:
            if use_cross_feature:
                lfs.append(lf.cpu())
            else:
                bn_lfs.append(bn_lf.cpu())

        # 4 features flipped
        gfs_flipped.append(gff.cpu())
        bn_gfs_flipped.append(bn_gff.cpu())

        if use_local_feature:
            if use_cross_feature:
                lfs_flipped.append(lff.cpu())
            else:
                bn_lfs_flipped.append(bn_lff.cpu())

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
        distmat2 = compute_distmat(cfg, num_query, bn_gfs, bn_gfs_flipped, lfs, lfs_flipped, theta=0.45,
                                   use_local_feature=use_local_feature, use_rerank=use_rerank)
        distmat = distmat2
        #distmat = (distmat1 + distmat2) / 2
    else:
        logger.info("Computing distmat with gf + bn_lf")
        distmat1 = compute_distmat(cfg, num_query, gfs, gfs_flipped, bn_lfs, bn_lfs_flipped, theta=0.95,
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

def inference_ssg(
        cfg,
        model,
        test_dataloader,
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
    bn_gfs = []

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
                # print('ret', ret.size())
                raise Exception("Unknown model returns, length = ", len(ret))

        bn_gfs.append(bn_gf.cpu() + bn_gff.cpu())
        pids.extend(pid.cpu().numpy())
        #camids.extend(np.asarray(camid))
        batch = test_prefetcher.next()

    feats = torch.cat(bn_gfs, dim=0)
    feats = F.normalize(feats, p=2, dim=1)
    return feats



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


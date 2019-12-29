# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir
import json
import time

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import get_test_dataloader
from engine.inference_save_res import inference_aligned_flipped
from modeling import build_model
from utils.logger import setup_logger
import h5py

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument('-cfg',
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument('--test_phase', action='store_true', help="use cpu")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # set pretrian = False to avoid loading weight repeatedly
    cfg.MODEL.PRETRAIN = False
    cfg.DATASETS.PRELOAD_IMAGE = False
    cfg.freeze()

    logger = setup_logger("reid_baseline", False, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True

    model = build_model(cfg, 0)
    model = model.cuda()
    model.load_params_wo_fc(torch.load(cfg.TEST.WEIGHT))


    test_dataloader, num_query, dataset = get_test_dataloader(cfg, test_phase=True)

    use_local_feature = False
    use_rerank = True
    use_cross_feature = True

    #print('dataset.gallery', dataset.gallery[:100])
    #return

    distmat, index, distmat1, distmat2 = inference_aligned_flipped(cfg, model, test_dataloader, num_query,
                                                                   use_local_feature, use_rerank, use_cross_feature)

    suffix = 'flip'
    if use_local_feature:
        suffix += '_aligned'
    if use_rerank:
        suffix += '_rerank'
    if use_cross_feature:
        suffix += '_cross'


    # saving results
    if args.test_phase:
        query_path = [t[0] for t in dataset.query]
        gallery_path = [t[0] for t in dataset.gallery]
        logger.info("-------------Write resutls to json file----------")

        results = {}
        top_k = 200
        for i in range(len(query_path)):
            topk_res = []
            for j in range(top_k):
                img_path = gallery_path[index[i, j]]
                # print(img_path)
                topk_res.append(img_path.split('/')[-1].split('_')[-1])
            results[query_path[i].split('/')[-1].split('_')[-1]] = topk_res

        # 写入结果
        if not os.path.isdir('submit'):
            os.mkdir('submit')

        strtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        json.dump(results, open('submit/reid_%s_%s_%s.json' % (cfg.MODEL.NAME, strtime, suffix), 'w'))

        # saving dist_mats
        mat_path = '/Volumes/Data/比赛/行人重识别2019/dist_mats'
        if not os.path.isdir(mat_path):
            os.mkdir(mat_path)
        f = h5py.File('%s/test_%s_%s_%s.h5' % (mat_path, cfg.MODEL.NAME, strtime, suffix), 'w')
        f.create_dataset('dist_mat', data=distmat, compression='gzip')

        if distmat1 is not None:
            f.create_dataset('dist_mat1', data=distmat1, compression='gzip')
        if distmat2 is not None:
            f.create_dataset('dist_mat2', data=distmat2, compression='gzip')
        f.close()

if __name__ == '__main__':
    main()


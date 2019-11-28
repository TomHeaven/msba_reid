# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import h5py
import numpy as np

sys.path.append('.')
from config import cfg
from data import get_test_dataloader
from engine.inference import inference_with_distmat
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument('-cfg',
                        "--config_file", default="", help="path to config file", type=str
                        )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # set pretrian = False to avoid loading weight repeatedly
    cfg.MODEL.PRETRAIN = False
    cfg.freeze()

    logger = setup_logger("reid_baseline", False, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))


    test_dataloader, num_query, _ = get_test_dataloader(cfg, test_phase=False)

    distmat_paths = [cfg.TEST.DISTMAT1, cfg.TEST.DISTMAT2, cfg.TEST.DISTMAT3,
                     cfg.TEST.DISTMAT4, cfg.TEST.DISTMAT5, cfg.TEST.DISTMAT6]
    # 加载dist_mats
    dist_mats = []

    cnt = 0
    thresh = 2
    for distmat_path in distmat_paths:
        if os.path.isfile(distmat_path):
            f = h5py.File(distmat_path, 'r')
            #mat = f['dist_mat'][()]

            if cnt < thresh:
                mat = f['dist_mat1'][()]
            else:
                mat = f['dist_mat'][()]

            mat = mat[np.newaxis, ...]
            dist_mats.append(mat)
            f.close()
            cnt += 1

    logger.info(f'Average {cnt} results')
    dist_mat = np.concatenate(dist_mats, axis=0).mean(axis=0)

    inference_with_distmat(cfg, test_dataloader, num_query, dist_mat)

if __name__ == '__main__':
    main()


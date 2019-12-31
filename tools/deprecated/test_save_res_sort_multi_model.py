# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import numpy as np
import h5py
import json
import time

sys.path.append('.')
from config import cfg
from data import get_test_dataloader
from utils.logger import setup_logger

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


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
    cfg.freeze()

    logger = setup_logger("reid_baseline", False, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test_dataloader, num_query, dataset = get_test_dataloader(cfg, test_phase=True)

    original_filenames = []
    for img_path, _, _ in dataset.query:
        original_filenames.append(img_path.split('/')[-1].split('_')[1])
    query_idx = argsort(original_filenames )

    print('fixed query order', [dataset.query[i][0] for i in query_idx[:10]])

    original_filenames = []
    for img_path, _, _ in dataset.gallery:
        original_filenames.append(img_path.split('/')[-1].split('_')[1])
    gallery_idx = argsort( original_filenames)
    print('fixed gallery order', [dataset.gallery[i][0] for i in gallery_idx[:10]])

    distmat_paths = [cfg.TEST.DISTMAT1, cfg.TEST.DISTMAT2, cfg.TEST.DISTMAT3,
                     cfg.TEST.DISTMAT4, cfg.TEST.DISTMAT5, cfg.TEST.DISTMAT6,
                     cfg.TEST.DISTMAT7, cfg.TEST.DISTMAT8, cfg.TEST.DISTMAT9,
                     cfg.TEST.DISTMAT10, cfg.TEST.DISTMAT11, cfg.TEST.DISTMAT12,
                     cfg.TEST.DISTMAT13, cfg.TEST.DISTMAT14, cfg.TEST.DISTMAT15,
                     cfg.TEST.DISTMAT16, cfg.TEST.DISTMAT17, cfg.TEST.DISTMAT18,
                     ]

    cnt = 0
    for distmat_path in distmat_paths:
        if os.path.isfile(distmat_path):
            f = h5py.File(distmat_path, 'r')
            mat = f['dist_mat'][()]

            if not distmat_path.endswith('baseline_v5.1_distmat.h5'):
                mat = mat[query_idx]
                mat = mat[:, gallery_idx]

                f2 = h5py.File('%s_sorted.h5' % distmat_path, 'w')
                f2.create_dataset('dist_mat', data=mat, compression='gzip')
                f2.close()

                cnt += 1

            #mat = mat[np.newaxis, ...]
            #dist_mats.append(mat)
            f.close()

        else:
            logger.info(f'Invalid checkpoint path {distmat_path}')
    logger.info(f'Sort {cnt} results')


if __name__ == '__main__':
    main()


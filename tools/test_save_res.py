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
from engine.inference_save_res import inference, inference_flipped
from modeling import build_model
from utils.logger import setup_logger
import h5py


def retain_top1(index, index_original):
    assert len(index) == len(index_original) and index.shape[1] == index_original.shape[1]

    for i in range(len(index)):
        # 在 index[i]中查找index_original对应的top1 ID，如果找到就交换
        b_found = False
        for j in range(index.shape[1]):
            if index[i][j] == index_original[i][0]:
                tmp = index[i][j]
                index[i][j] = index[i][0]
                index[i][0] = tmp

                b_found = True
                break

        # 如果没找到，就直接插入到顶部
        if not b_found:
            index[i][0] = index_original[i][0]
    return index


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

    cudnn.benchmark = True

    model = build_model(cfg, 0)
    model = model.cuda()
    model.load_params_wo_fc(torch.load(cfg.TEST.WEIGHT))

    test_dataloader, num_query, dataset = get_test_dataloader(cfg, test_phase=True)


    if cfg.MODEL.NAME.endswith('abd'):
        target_theta = 0.45  # abd网络的参数
    else:
        target_theta = 0.95  # 非abd网络的参数

    # thetas = [0.45, 0.5, 0.9, 0.95]
    thetas = [target_theta]

    use_flip = True

    if use_flip:
        scores, indices, dist_mats = inference_flipped(cfg, model, test_dataloader, num_query, thetas)
    else:
        scores, indices, dist_mats = inference(cfg, model, test_dataloader, num_query, thetas)

    print('distmats', len(dist_mats), 'thetas', len(thetas))


    # saving results
    if args.test_phase:
        query_path = [t[0] for t in dataset.query]
        gallery_path = [t[0] for t in dataset.gallery]
        logger.info("-------------Write resutls to json file----------")

        for idx, (theta, score, index) in enumerate(zip(thetas, scores, indices)):
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
            strtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            if use_flip:
                json.dump(results, open('submit/reid_%s_%s_(r, t %.3f, flip).json' % (cfg.MODEL.NAME, strtime, theta), 'w'))
            else:
                json.dump(results, open('submit/reid_%s_%s_(r, t %.3f).json' % (cfg.MODEL.NAME, strtime, theta), 'w'))


            if abs(theta - target_theta) < 1e-4:
                # saving dist_mats
                f = h5py.File('dist_mats/test_%s_%s_t_%.2f_flip.h5' % (cfg.MODEL.NAME, strtime, theta), 'w')
                f.create_dataset('dist_mat', data=dist_mats[idx], compression='gzip')
                f.close()




if __name__ == '__main__':
    main()


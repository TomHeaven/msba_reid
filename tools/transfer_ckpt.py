# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import get_test_dataloader
from engine.inference_rerank import inference, inference_aligned, inference_flipped
from modeling import build_model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument('-cfg',
        "--config_file",
        default="",
        help="path to config file", type=str
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

    cudnn.benchmark = True

    model = build_model(cfg, 0)

    sd = model.state_dict()
    xsd = torch.load(cfg.TEST.WEIGHT)

    print('sd', sd.keys())
    print('xsd', xsd.keys())

    for key in xsd.keys():
        if key.startswith('base') or key.startswith('bn') or key.startswith('conv1'):
            new_key = 'base.' + key
        elif key.startswith('shallow_cam'):
            new_key = 'base.base.' + key
        else:
            new_key = key

        sd[new_key] = xsd[key]

    torch.save(sd, cfg.TEST.WEIGHT + '.updated')




if __name__ == '__main__':
    main()


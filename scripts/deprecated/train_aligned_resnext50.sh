#!/usr/bin/env bash
GPUS='0,1'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/train.py -cfg='configs/softmax_triplet_resnext.yml' \
DATASETS.NAMES '("competition1910",)'  \
DATASETS.TEST_NAMES 'competition1910' \
INPUT.SIZE_TRAIN '[256, 128]' \
INPUT.SIZE_TEST '[256, 128]' \
SOLVER.IMS_PER_BATCH '63' \
MODEL.NAME 'aligned_resnext50' \
MODEL.WITH_IBN 'True' \
MODEL.BACKBONE 'aligned_resnext50' \
MODEL.VERSION 'aligned_resnext50_ibn_bs63' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \
MODEL.PRETRAIN_PATH '/Users/tomheaven/.cache/torch/checkpoints/resnext50_ibn_a.pth.tar' \

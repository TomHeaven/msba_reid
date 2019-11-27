#!/usr/bin/env bash
GPUS='0,1'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("competition1910",)'  \
DATASETS.TEST_NAMES 'competition1910' \
INPUT.SIZE_TRAIN '[256, 128]' \
INPUT.SIZE_TEST '[256, 128]' \
MODEL.NAME 'aligned_se_resnet101_ibn' \
MODEL.WITH_IBN 'True' \
MODEL.BACKBONE 'aligned_se_resnet101' \
MODEL.VERSION 'aligned_se_resnet101_ibn_bs63' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \
MODEL.PRETRAIN_PATH '/Users/tomheaven/.cache/torch/checkpoints/se_resnet101_ibn_a.pth.tar' \

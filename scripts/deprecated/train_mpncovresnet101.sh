#!/usr/bin/env bash
GPUS='0,1'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("competition1910",)'  \
DATASETS.TEST_NAMES 'competition1910' \
INPUT.SIZE_TRAIN '[256, 128]' \
INPUT.SIZE_TEST '[256, 128]' \
SOLVER.IMS_PER_BATCH '63' \
MODEL.NAME 'baseline' \
MODEL.WITH_IBN 'False' \
MODEL.BACKBONE 'mpncov_resnet101' \
MODEL.VERSION 'mpncov_resnet101_bs63' \
MODEL.PRETRAIN_PATH '/Users/tomheaven/.cache/torch/checkpoints/mpncovresnet101-ade9737a.pth' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \

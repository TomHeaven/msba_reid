#!/usr/bin/env bash
GPUS='0,1'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("competition1910",)'  \
DATASETS.TEST_NAMES 'competition1910' \
INPUT.SIZE_TRAIN '[256, 128]' \
INPUT.SIZE_TEST '[256, 128]' \
SOLVER.IMS_PER_BATCH '84' \
MODEL.NAME 'baseline' \
MODEL.WITH_IBN 'True' \
MODEL.BACKBONE 'resnet101' \
MODEL.VERSION 'resnet101_ibn_bs63' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \
MODEL.PRETRAIN_PATH '/media/tomheaven/MacOS/Users/tomheaven/.cache/torch/checkpoints/resnet101_ibn_a.pth.tar' \
#MODEL.CHECKPOINT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/sota/aligned_resnet101_ibn_bs63/ckpts/model_epoch90.pth' \

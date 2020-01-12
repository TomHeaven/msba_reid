#!/usr/bin/env bash
GPUS='0,1'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("competition1910",)'  \
DATASETS.TEST_NAMES 'competition1910' \
INPUT.SIZE_TRAIN '[256, 128]' \
INPUT.SIZE_TEST '[256, 128]' \
SOLVER.IMS_PER_BATCH '78' \
MODEL.NAME 'mgn_resnet50' \
MODEL.WITH_IBN 'True' \
MODEL.BACKBONE 'mgn_resnet50' \
MODEL.VERSION 'mgn_resnet50_ibn_bs63' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \
MODEL.PRETRAIN_PATH '/Users/tomheaven/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
#MODEL.CHECKPOINT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/mgn_resnet50_ibn_bs63_v6/ckpts/model_epoch70.pth'
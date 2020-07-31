#!/usr/bin/env bash
GPUS='0,1'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("cuhk03",)' \
DATASETS.TEST_NAMES 'cuhk03' \
SOLVER.IMS_PER_BATCH '64' \
MODEL.NAME 'resnet50_ibn' \
MODEL.WITH_IBN 'True' \
MODEL.BACKBONE 'resnet50' \
MODEL.VERSION 'resnet50_ibn_bs64' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \
MODEL.PRETRAIN_PATH ${HOME}'/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
MODEL.USE_PARTS '2' \
INPUT.SIZE_TRAIN '[384, 128]' \
INPUT.SIZE_TEST '[384, 128]' 
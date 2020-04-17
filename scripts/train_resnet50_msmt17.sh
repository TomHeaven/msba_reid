#!/usr/bin/env bash
GPUS='0,1'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("msmt17",)' \
DATASETS.TEST_NAMES 'msmt17' \
SOLVER.IMS_PER_BATCH '64' \
DATALOADER.NUM_INSTANCE '4' \
MODEL.NAME 'resnet50_ibn' \
MODEL.WITH_IBN 'True' \
MODEL.BACKBONE 'resnet50' \
MODEL.VERSION 'resnet50_ibn_bs64' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \
MODEL.PRETRAIN_PATH '/home/tomheaven/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
MODEL.USE_PARTS '2' \
INPUT.SIZE_TRAIN '[384, 192]' \
INPUT.SIZE_TEST '[384, 192]' 
#MODEL.CHECKPOINT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/sota/aligned_resnet101_ibn_bs63/ckpts/model_epoch90.pth' \

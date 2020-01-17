#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python3 tools/val_xiao.py -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'msmt17' \
MODEL.NAME 'resnet50_ibn' \
MODEL.BACKBONE 'resnet50' \
MODEL.WITH_IBN 'True' \
TEST.WEIGHT 'logs/msmt17/resnet50_ibn_xiao_bs64/ckpts/model_best.pth' \
TEST.IMS_PER_BATCH '128'

#TEST.WEIGHT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/aligned_resnet101_ibn_bs63/ckpts/model_best.pth'


#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='1'
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/transfer_ckpt.py -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
MODEL.NAME 'aligned_resnext101_ibn_abd' \
MODEL.BACKBONE 'aligned_resnext101_abd' \
MODEL.WITH_IBN 'True' \
MODEL.PRETRAIN 'False' \
TEST.WEIGHT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/xiao/aligned_resnext101_Ibn_bs63_xiao/ckpts/model_epoch120_xiao.pth'

#TEST.WEIGHT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/aligned_resnet101_ibn_bs63_xiao/ckpts/model_best_xiao.pth'


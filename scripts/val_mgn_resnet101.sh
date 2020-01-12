#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python3 tools/val.py -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
MODEL.NAME 'mgn_resnet101_ibn' \
MODEL.BACKBONE 'mgn_resnet101' \
MODEL.WITH_IBN 'True' \
TEST.WEIGHT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/mgn_resnet101_ibn_bs63/ckpts/model_best.pth' \
TEST.IMS_PER_BATCH '16'

#TEST.WEIGHT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/aligned_resnet101_ibn_bs63/ckpts/model_best.pth'


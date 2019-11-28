#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='1'
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/val_multi_model.py -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
TEST.DISTMAT1 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/sota/val_aligned_resnet101_ibn_20191128_094009_4f_a0.500.h5' \
TEST.DISTMAT2 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/sota/val_aligned_resnext101_ibn_abd_20191128_095319_4f_a0.500.h5' \
TEST.DISTMAT3 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/sota/val_aligned_resnext101_ibn_20191128_094958_4f_a0.500.h5' \

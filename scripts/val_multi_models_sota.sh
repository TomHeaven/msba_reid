#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='1'
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/val_multi_model.py -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
TEST.DISTMAT1 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/sota/val_aligned_resnet101_ibn_20191126_221556_t0.95_flip.h5' \
TEST.DISTMAT2 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/sota/val_aligned_resnet101_ibn_20191119_141143_t0.95_flip.h5' \
TEST.DISTMAT3 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/bak/val_aligned_resnext101_ibn_abd_20191121_085105_t0.95_flip.h5'  \
TEST.DISTMAT4 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/bak/val_aligned_resnet101_ibn_abd_20191120_224335_t0.95_flip.h5' \
TEST.DISTMAT5 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/xiao/val_aligned_resnet101_ibn_abd_20191121_175835_t0.45_flip.h5'
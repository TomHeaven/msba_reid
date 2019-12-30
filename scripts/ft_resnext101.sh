#!/usr/bin/env bash
GPUS='0,1'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("competition1910",)'  \
DATASETS.TEST_NAMES 'competition1910' \
INPUT.SIZE_TRAIN '[384, 192]' \
INPUT.SIZE_TEST '[384, 192]' \
SOLVER.IMS_PER_BATCH '66' \
MODEL.NAME 'resnext101_ibn' \
MODEL.WITH_IBN 'True' \
MODEL.BACKBONE 'resnext101' \
MODEL.VERSION 'resnext101_ibn_bs63' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \
MODEL.FINE_TUNE 'True' \
MODEL.CHECKPOINT 'logs/competition1910/sota/resnext101_ibn_bs63/ckpts/model_best.pth' \
SOLVER.BASE_LR '4e-6' \
SOLVER.WARMUP_ITERS '0' \
SOLVER.STEPS '[100,200]' \
SOLVER.EVAL_PERIOD '5' \
SOLVER.MAX_EPOCHS '10'

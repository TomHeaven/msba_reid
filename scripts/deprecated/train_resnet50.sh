GPUS='0,1'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("competition1910",)'  \
DATASETS.TEST_NAMES 'competition1910' \
INPUT.SIZE_TRAIN '[256, 128]' \
INPUT.SIZE_TEST '[256, 128]' \
SOLVER.IMS_PER_BATCH '63' \
MODEL.NAME 'resnet50_ibn' \
MODEL.WITH_IBN 'False' \
MODEL.BACKBONE 'resnet50' \
MODEL.VERSION 'baseline_bs63' \
MODEL.PRETRAIN_PATH '/Users/tomheaven/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \

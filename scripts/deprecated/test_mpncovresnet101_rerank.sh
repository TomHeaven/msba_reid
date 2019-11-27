GPUS='1'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/test_rerank.py -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
MODEL.NAME 'baseline' \
MODEL.BACKBONE 'mpncov_resnet101' \
MODEL.WITH_IBN 'False' \
TEST.WEIGHT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/mpncov_resnet101_bs63/ckpts/model_best.pth'

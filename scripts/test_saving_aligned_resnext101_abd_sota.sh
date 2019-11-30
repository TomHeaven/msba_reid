
export CUDA_VISIBLE_DEVICES='0'
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/test_save_res.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
MODEL.NAME 'aligned_resnext101_ibn_abd_sota' \
MODEL.BACKBONE 'aligned_resnext101_abd' \
MODEL.WITH_IBN 'True' \
TEST.WEIGHT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/sota/aligned_resnext101_ibn_abd_bs63/ckpts/model_best.pth'



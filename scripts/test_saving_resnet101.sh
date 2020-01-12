export CUDA_VISIBLE_DEVICES='0'

python3 tools/test_save_res.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
INPUT.SIZE_TRAIN '[384, 128]' \
INPUT.SIZE_TEST '[384, 128]' \
MODEL.NAME 'resnet101_ibn' \
MODEL.BACKBONE 'resnet101' \
MODEL.WITH_IBN 'True' \
TEST.WEIGHT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/resnet101_ibn_bs63/ckpts/model_best.pth' \
DATASETS.PRELOAD_IMAGE 'False'



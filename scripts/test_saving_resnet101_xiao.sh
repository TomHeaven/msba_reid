export CUDA_VISIBLE_DEVICES='1'

python3 tools/test_save_res_xiao.py --test_phase --fine_tune -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
INPUT.SIZE_TRAIN '[384, 192]' \
INPUT.SIZE_TEST '[384, 192]' \
MODEL.NAME 'resnet101_ibn_xiao' \
MODEL.BACKBONE 'resnet101' \
MODEL.WITH_IBN 'True' \
TEST.WEIGHT 'logs/competition1910/resnet101_ibn_bs63/ckpts/model_best.pth' \
DATASETS.PRELOAD_IMAGE 'False' \
TEST.IMS_PER_BATCH 64



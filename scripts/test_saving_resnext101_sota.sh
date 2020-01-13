export CUDA_VISIBLE_DEVICES='0'

python3 tools/test_save_res.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
INPUT.SIZE_TRAIN '[384, 192]' \
INPUT.SIZE_TEST '[384, 192]' \
MODEL.NAME 'resnext101_ibn' \
MODEL.BACKBONE 'resnext101' \
MODEL.WITH_IBN 'True' \
TEST.WEIGHT 'logs/competition1910/sota/resnext101_ibn_bs63/ckpts/model_best.pth' \
DATASETS.PRELOAD_IMAGE 'False' \
TEST.IMS_PER_BATCH 128



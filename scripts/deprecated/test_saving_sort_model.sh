
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/test_save_res_sort_multi_model.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
TEST.DISTMAT1 '/Volumes/Data/比赛/行人重识别2019/dist_mats/sota/test_mgn_resnet50_ibn_20191222_004057_flip_rerank_cross.h5' \












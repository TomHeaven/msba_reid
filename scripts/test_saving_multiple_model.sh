
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/test_save_res_multi_model.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
TEST.DISTMAT1 '/Volumes/Data/比赛/行人重识别2019/dist_mats/sota/baseline_v5.1_distmat.h5' \
TEST.DISTMAT2 '/Volumes/Data/比赛/行人重识别2019/dist_mats/sota/test_resnet101_ibn_20191218_230227_flip_rerank_both.h5_sorted.h5' \
TEST.DISTMAT3 '/Volumes/Data/比赛/行人重识别2019/dist_mats/sota/test_resnext101_ibn_20191219_003243_flip_rerank_both.h5_sorted.h5' \
TEST.DISTMAT4 '/Volumes/Data/比赛/行人重识别2019/dist_mats/sota/test_mgn_resnet50_ibn_20191222_004057_flip_rerank_cross.h5_sorted.h5' \












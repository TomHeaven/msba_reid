
export CUDA_VISIBLE_DEVICES='1'
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/test_save_res_multi_model.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
TEST.DISTMAT1 '/Volumes/Data/比赛/行人重识别2019/dist_mats/test_aligned_resnet101_ibn_20191211_162250_flip_rerank_cross.h5' \
TEST.DISTMAT2 '/Volumes/Data/比赛/行人重识别2019/dist_mats/test_aligned_resnext101_ibn_20191212_102528_flip_rerank_cross.h5' \










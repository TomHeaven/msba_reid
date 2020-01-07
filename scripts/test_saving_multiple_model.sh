GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/test_save_res_multi_model.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
TEST.DISTMAT1 'dist_mats/test_resnet101_ibn_xiao_ft_flip_rerank_cross.h5' \
TEST.DISTMAT2 'dist_mats/adjusted/test_resnext101_ibn_ft_flip_rerank_cross.h5' \
TEST.DISTMAT3 'dist_mats/adjusted/test_resnet101_ibn_ft_flip_rerank_cross.h5' 












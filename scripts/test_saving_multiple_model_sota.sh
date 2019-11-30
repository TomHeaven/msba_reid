
export CUDA_VISIBLE_DEVICES='1'
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/test_save_res_multi_model.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
TEST.DISTMAT1 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/test_aligned_resnext101_ibn_abd_sota_20191128_202657_4f.h5' \
TEST.DISTMAT2 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/test_aligned_resnet101_ibn_sota_20191128_210147_4f.h5' \
TEST.DISTMAT3 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/test_aligned_resnext101_ibn_sota_20191128_202332_4f.h5' \
TEST.DISTMAT4 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/test_aligned_resnet101_ibn_abd_alldata_20191128_222628_4f.h5' \
TEST.DISTMAT5 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/test_aligned_resnet101_ibn_abd_myparam_20191129_015559_4f.h5' \
TEST.DISTMAT6 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/test_aligned_resnext101_ibn_abd_myparam_20191129_020238_4f.h5' \
TEST.DISTMAT7 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/test_aligned_resnext101_ibn_sota_new_20191129_065820_4f.h5' \
TEST.DISTMAT8 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/test_aligned_resnet101_ibn_abd_xiao_20191129_021601_4f.h5' \

export CUDA_VISIBLE_DEVICES='1'
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/test_save_res_multi_model.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
TEST.DISTMAT1 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/sota/bak/test_aligned_resnet101_ibn_20191127_173704_4f.h5' \
TEST.DISTMAT2 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/sota/bak/test_aligned_resnext101_ibn_20191127_195043_4f.h5' \
TEST.DISTMAT3 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/sota/bak/test_aligned_resnext101_ibn_abd_20191127_194955_4f.h5' \
TEST.DISTMAT4 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/bak/test_aligned_resnet101_ibn_abd_20191120_225307_t0.95_flip.h5' \
TEST.DISTMAT5 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/bak/test_aligned_resnext101_ibn_abd_20191121_051307_t0.95_flip.h5' \







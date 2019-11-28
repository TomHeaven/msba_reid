
export CUDA_VISIBLE_DEVICES='1'
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/test_save_res_multi_model.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
TEST.DISTMAT1 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/all_data/test_aligned_resnet101_ibn_20191127_192159_4f.h5' \
TEST.DISTMAT2 '/Users/tomheaven/NutstoreFiles/我的坚果云/PycharmProjects/reid_baseline/dist_mats/sota/test_aligned_resnet101_ibn_20191127_173704_4f.h5' \




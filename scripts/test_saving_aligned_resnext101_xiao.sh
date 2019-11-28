
export CUDA_VISIBLE_DEVICES='1'
GPUS='0'

CUDA_VISIBLE_DEVICES=$GPUS python3 tools/test_save_res.py --test_phase -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'competition1910' \
MODEL.NAME 'aligned_resnext101_ibn_xiao' \
MODEL.BACKBONE 'aligned_resnext101' \
MODEL.WITH_IBN 'True' \
INPUT.PIXEL_MEAN '[0.406, 0.456, 0.485]' \
INPUT.PIXEL_STD '[0.225, 0.224, 0.229]'  \
TEST.WEIGHT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/xiao/aligned_resnext101_ibn_bs63/ckpts/model_epoch120.pth'


#TEST.WEIGHT '/Volumes/Data/比赛/行人重识别2019/logs/competition1910/aligned_resnet101_ibn_bs63/ckpts/model_best.pth'



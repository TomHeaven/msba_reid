# MSBA: Multiple Scales, Branches and Attention Network With Bag of Tricks for Person Re-Identification
`Hanlin Tan ; Huaxin Xiao ; Xiaoyu Zhang ; Bin Dai ; Shiming Lai ; Yu Liu ; Maojun Zhang`

`National Unverisity of Defense Technology, China`

This repository contains code for our paper [MSBA: Multiple Scales, Branches and Attention Network With Bag of Tricks for Person Re-Identification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9052718), in which we have achieved the state-of-the-art performance on Market1501, DukeMTMC-reid, CUHK03(Detected) and MSMT17 datasets up to 2020.02. Please note the "state-of-the-art" is under restriction of no inference tricks and no additional information other than image contents. Namely, the results in the paper is without re-ranking or flipping. 

The model proposed by the paper is named MMNet (Muti-scale, Multi-branch Network).

# Performance 

Performance on the four datasets are available:




# Training
+ Clone the repo:
```shell
git clone --depth 1 https://github.com/TomHeaven/reid2019
cd reid2019
```
+ Compile Cython codes:
```shell
cd csrc/eval_cylib; make
```
+ Prepare datasets at `data` folder and set it path by updating `_C.DATASETS.ROOT_DIR` of `config/defaults.py` .
+ Prepare pretrained IBN-resnet50 weights from this [repo](https://github.com/XingangPan/IBN-Net) and place it at `~/.torch/checkpoints/resnet50-ibn-a.pth`.
+ Run 
```shell
sh scripts/train_resnet50_market1501.sh，
```
to train and evaluate on market1501 dataset at resolution 384x128. Scripts for other datasets are available at `scripts`. The trained weights and logs are available at `logs` by default. 



# Inference


+ Inference at CUHK03(Detected) dataset using MMNet-b under resolution 384x128
```shell
python3 tools/val.py -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'cuhk03' \
MODEL.NAME 'resnet50_ibn' \
MODEL.BACKBONE 'resnet50' \
MODEL.WITH_IBN 'True' \
TEST.WEIGHT 'logs/384x128_MMNet-b/cuhk03/resnet50_ibn_bs64/ckpts/model_best.pth' \
TEST.IMS_PER_BATCH '128' \
INPUT.SIZE_TRAIN '[384, 128]' \
INPUT.SIZE_TEST '[384, 128]' 
```
The parameter TEST.WEIGHT specify weight path.

# Switch between MMNet-a and MMNet-b
MMNet-b achieves better performance on CUHK03(Detected) dataset. Switch to MMNet-b by
```
cp modeling/baseline_parts_b.py  modeling/baseline_parts.py 
```
and back to MMNet-a by
```
cp modeling/baseline_parts_a.py  modeling/baseline_parts.py 
```
The difference is only in construction of loss function and affects training only. That is to say, you don't have to do this if you do inference only.

# Reference

If you find our paper or repo helpful to your work, please consider cite our paper
```
@ARTICLE{9052718, 
author={H. {Tan} and H. {Xiao} and X. {Zhang} and B. {Dai} and S. {Lai} and Y. {Liu} and M. {Zhang}}, 
journal={IEEE Access}, 
title={MSBA: Multiple Scales, Branches and Attention Network With Bag of Tricks for Person Re-Identification}, year={2020}, volume={8}, 
pages={63632-63642},
}
```
or star this repo. :)







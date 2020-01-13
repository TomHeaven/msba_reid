# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .transforms import *


def build_transforms(cfg, is_train=True):
    res = []
    if is_train:
        res.append(T.Resize(cfg.INPUT.SIZE_TRAIN))
        if cfg.INPUT.DO_FLIP:       
            res.append(T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB))
        if cfg.INPUT.DO_PAD:        
            res.extend([T.Pad(cfg.INPUT.PADDING, padding_mode=cfg.INPUT.PADDING_MODE),
                        T.RandomCrop(cfg.INPUT.SIZE_TRAIN)])
        if cfg.INPUT.DO_LIGHTING:   
            res.append(T.ColorJitter(cfg.INPUT.MAX_LIGHTING, cfg.INPUT.MAX_LIGHTING))
        # res.append(T.ToTensor())  # to slow
        if cfg.INPUT.DO_RE:
            #### Modify By Tan ####
            #  res.append(RandomErasing(probability=cfg.INPUT.RE_PROB)) # origin

            ## use BGR values in RGB order (ImageDataset use PIL to read image and convert to RGB order)
            res.append(RandomErasing(probability=cfg.INPUT.RE_PROB, mean=(cfg.INPUT.PIXEL_MEAN[2]*255.0,
                                                                          cfg.INPUT.PIXEL_MEAN[1]*255.0,
                                                                          cfg.INPUT.PIXEL_MEAN[0]*255.0)))
            #######################
    else:
        res.append(T.Resize(cfg.INPUT.SIZE_TEST))
        # res.append(T.ToTensor())
    return T.Compose(res)


# def build_transforms(cfg):
#     "Utility func to easily create a list of flip, rotate, `zoom`, warp, lighting transforms."
#     res = []
#     if cfg.INPUT.DO_FLIP:   res.append(flip_lr(p=cfg.INPUT.FLIP_PROB))
#     if cfg.INPUT.DO_PAD:    res.extend(rand_pad(padding=cfg.INPUT.PADDING,
#                                                 size=cfg.INPUT.SIZE_TRAIN,
#                                                 mode=cfg.INPUT.PADDING_MODE))
#     if cfg.INPUT.DO_LIGHTING:
#         res.append(brightness(change=(0.5*(1-cfg.INPUT.MAX_LIGHTING), 0.5*(1+cfg.INPUT.MAX_LIGHTING)), p=cfg.INPUT.P_LIGHTING))
#         res.append(contrast(scale=(1-cfg.INPUT.MAX_LIGHTING, 1/(1-cfg.INPUT.MAX_LIGHTING)), p=cfg.INPUT.P_LIGHTING))
#     res.append(RandomErasing())
#     #       train                   , valid
#     return (res, [crop_pad()])
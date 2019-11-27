# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from .baseline import Baseline
from .baseline_v3 import Baseline as Baseline_v3
from .losses import reidLoss


def build_model(cfg, num_classes) -> nn.Module:
    model = Baseline(
        cfg.MODEL.BACKBONE, 
        num_classes, 
        cfg.MODEL.LAST_STRIDE, 
        cfg.MODEL.WITH_IBN, 
        cfg.MODEL.GCB, 
        cfg.MODEL.STAGE_WITH_GCB, 
        cfg.MODEL.PRETRAIN, 
        cfg.MODEL.PRETRAIN_PATH)
    return model


def build_model_v3(cfg, num_classes) -> nn.Module:
    model = Baseline_v3(
        cfg.MODEL.BACKBONE,
        num_classes,
        cfg.MODEL.LAST_STRIDE,
        cfg.MODEL.WITH_IBN,
        cfg.MODEL.GCB,
        cfg.MODEL.STAGE_WITH_GCB,
        pretrain=cfg.MODEL.PRETRAIN,
        model_path=cfg.MODEL.PRETRAIN_PATH)
    return model
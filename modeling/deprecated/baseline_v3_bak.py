# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F

from .backbones import *
from .losses.cosface import AddMarginProduct
from .utils import *

from .backbones.aligned.HorizontalMaxPool2D import HorizontalMaxPool2d
from .backbones.components.shallow_cam import ShallowCAM

import pdb

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self,
                 backbone,
                 num_classes,
                 last_stride,
                 with_ibn,
                 gcb,
                 stage_with_gcb,
                 loss={'softmax'},
                 aligned=True,
                 pretrain=True,
                 model_path=''):
        super().__init__()

        try:
            if backbone == 'aligned_resnet101_abd':
                #self.base = ResNet_ABD.from_name('resnet101', last_stride, with_ibn, gcb, stage_with_gcb)
                self.base = ResNet.from_name('resnet101', last_stride, with_ibn, gcb, stage_with_gcb, False)
        except:
            print(f'not support {backbone} backbone')

        if pretrain:
            self.base.load_pretrain(model_path)

        self.shallow_cam = ShallowCAM(256)

        self.loss = loss
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.feat_dim = self.in_planes # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()

        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.bn.apply(weights_init_kaiming)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv1.apply(weights_init_kaiming)

    def forward(self, x, label=None):

        ## new added by hxx
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x_layer1 = self.base.layer1(x)
        x_layer1 = x_intermediate = self.shallow_cam(x_layer1)
        x = self.base.layer2(x_layer1)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        fmap_dict = defaultdict(list)
        fmap_dict['intermediate'].append(x_intermediate)
        fmap_dict = {k: tuple(v) for k, v in fmap_dict.items()}

        # calculate local feature
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()

        # calculate global feature
        x = self.gap(x)  # (b, 2048, 1, 1)
        x = x.view(-1, x.size()[1])
        f = self.bottleneck(x)

        if not self.training:
            return f,lf

        y = self.classifier(f)

        if self.aligned:
            return y, x, lf, fmap_dict
        return y, x, fmap_dict


    def load_params_wo_fc(self, state_dict):
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     k = '.'.join(k.split('.')[1:])
        #     new_state_dict[k] = v
        # state_dict = new_state_dict
        state_dict.pop('classifier.weight')
        # state_dict.pop('classifier.bias')
        res = self.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(['classifier.weight']), 'issue loading pretrained weights'

    # def load_param(self, trained_path):
    #     param_dict = torch.load(trained_path)
    #     for k, v in param_dict.items():
    #         if 'classifier' in k:
    #             continue
    #         self.state_dict()[k].copy_(param_dict[k])

    #     for k, v in param_dict.items():
    #         if 'classifier' in k:
    #             continue
    #         assert torch.sum(self.state_dict()[k]) == torch.sum(param_dict[k]), '{} error!'.format(v)

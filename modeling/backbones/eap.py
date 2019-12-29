# -*-encoding: utf-8-*-

import random
import copy as cp
import torch
from torch import nn
import torchvision.models as models
from .resnet import ResNet, Bottleneck
from .resnext_ibn_a import resnext101_ibn_a
#from .layers.SE_Resnet import SEResnet
#from .layers.SE_module import SELayer


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class FeatureDrop(nn.Module):

    def __init__(self, h_ratio, w_ratio):
        super(FeatureDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


class FeatureFusion(nn.Module):

    def __init__(self, basename, input_size, output_size):
        super(FeatureFusion, self).__init__()
        if basename == 'resnet50':
            self.base = ResNet.from_name('resnet50', last_stride=1, with_ibn=True, gcb=False, stage_with_gcb=None,
                                         with_abd=False)
        elif basename == 'resnet101':
            self.base = ResNet.from_name('resnet101', last_stride=1, with_ibn=True, gcb=False, stage_with_gcb=None,
                                         with_abd=False)
        elif basename == 'resnext101':
            self.base = resnext101_ibn_a(4, 32, last_stride=1, with_abd=False)
        else:
            raise Exception("Unknown base ", basename)

        self.mix_conv = nn.Sequential(nn.Conv2d(input_size, output_size, kernel_size=1, stride=1, bias=False))
        self.mix_conv.apply(weights_init_kaiming)

    def forward(self, x, reid_feature):
        base_feature = self.base(x)
        x = torch.cat([reid_feature, base_feature], dim=1)
        x = self.mix_conv(x)
        return x


class LBNNeck(nn.Module):

    def __init__(self, in_planes, num_classes):
        super(LBNNeck, self).__init__()
        self.BN = nn.BatchNorm1d(in_planes)
        self.BN.bias.requires_grad_(False)
        self.fc = nn.Linear(in_planes, num_classes, bias=False)
        self.BN.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        triplet_feat = nn.functional.normalize(x)
        test_feat = self.BN(triplet_feat)

        if self.training:
            cl_feat = self.fc(test_feat)
            return cl_feat, triplet_feat
        return test_feat


class EAP(nn.Module):
    in_planes = 2048

    def __init__(self, model_path, basename, num_classes, last_stride, width_ratio=1.0, height_ratio=0.33):
        super(EAP, self).__init__()

        self.layer_lis = [[3, 4, 6, 3], [3, 4, 23, 3]]

        if basename == 'resnet50':
            self.base = ResNet.from_name('resnet50', last_stride, with_ibn=True, gcb=False, stage_with_gcb=None, with_abd=False,
                                         with_layer3=True)
        elif basename == 'resnet101':
            self.base = ResNet.from_name('resnet101', last_stride, with_ibn=True, gcb=False, stage_with_gcb=None, with_abd=False)
        elif basename == 'resnext101':
            self.base = resnext101_ibn_a(4, 32, last_stride, with_abd=False)
        else:
            raise Exception("Unknown base ", basename)

        self.ffm = FeatureFusion(2048, 1024)

        # load pretrained weights
        if model_path != '':
            print('pretrain path', model_path)
            self.base.load_pretrain(model_path)

        self.branch1_layer4 = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.branch1_layer4.load_state_dict(self.base.layer4.state_dict())

        self.branch2_layer4 = cp.copy(self.branch1_layer4)

        self.branch2_layer4.load_state_dict(self.base.layer4.state_dict())
        self.feature_drop = FeatureDrop(height_ratio, width_ratio)

        self.max_gap_global = nn.AdaptiveMaxPool2d(1)
        self.avg_gap_global = nn.AdaptiveAvgPool2d(1)
        self.max_gap_branch1 = nn.AdaptiveMaxPool2d(1)
        self.avg_gap_branch2 = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        self.lbn1 = LBNNeck(self.in_planes, self.num_classes)
        self.lbn2 = LBNNeck(self.in_planes, self.num_classes)
        self.lbn3 = LBNNeck(self.in_planes, self.num_classes)

    def load_pretrain(self, model_path=''):
        #self.base.load_pretrain(model_path)
        pass

    def forward(self, x):
        # x3, x
        reid_feat, feat1 = self.base(x)

        ffm_feat = self.ffm(x, reid_feat)
        feat2 = self.branch1_layer4(ffm_feat)
        feat3 = self.branch2_layer4(ffm_feat)
        if self.training and random.randint(0, 1):
             feat3 = self.feature_drop(feat3)

        # 全局分支
        feat1 = self.max_gap_global(feat1) + self.avg_gap_global(feat1)
        feat2 = self.max_gap_branch1(feat2)
        feat3 = self.avg_gap_branch2(feat3)

        if self.training:
            cl_1, triplet_1 = self.lbn1(feat1)
            cl_2, triplet_2 = self.lbn2(feat2)
            cl_3, triplet_3 = self.lbn3(feat3)
            return [cl_1, cl_2, cl_3], [triplet_1, triplet_2, triplet_3]

        test_out = torch.cat([self.lbn1(feat1), self.lbn2(feat2) + self.lbn3(feat3)], dim=1)
        return test_out
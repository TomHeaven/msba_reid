# mgn_v6_4：相对于v6，减少了分支的deepcopy特征，仅deepcopy layer4的特征。
# 参考了eap的实现方式，不在横切特征，二是寻求增强特征的范化性能
import copy
import torch
import torch.nn as nn
from ..utils import *
#from torchvision.models.resnet import resnet50, Bottleneck

from .resnet import ResNet, Bottleneck
from .resnext_ibn_a import resnext101_ibn_a
import random

DEBUG = False

def get_reduction(in_planes, out_planes):
    bn = nn.BatchNorm2d(in_planes)
    bn.apply(weights_init_kaiming) # Tom added
    relu = nn.ReLU(inplace=True)
    conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
    conv1.apply(weights_init_kaiming) # Tom added
    return nn.Sequential(bn, relu, conv1)


def get_bottleneck_classifier(in_planes, out_planes):
    bottleneck = nn.BatchNorm1d(in_planes)
    bottleneck.bias.requires_grad_(False)  # no shift
    classifier = nn.Linear(in_planes, out_planes, bias=False)

    bottleneck.apply(weights_init_kaiming)
    classifier.apply(weights_init_classifier)
    return bottleneck, classifier

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

class MGN(nn.Module):
    def __init__(self, model_path, num_classes, basename, last_stride, with_ibn, gcb, stage_with_gcb, with_abd, feats = 256, **kwargs):
        super(MGN, self).__init__()

        if basename == 'resnet50':
            self.base = ResNet.from_name('resnet50', last_stride, with_ibn, gcb, stage_with_gcb, with_abd)
        elif basename == 'resnet101':
            self.base = ResNet.from_name('resnet101', last_stride, with_ibn, gcb, stage_with_gcb, with_abd)
        elif basename == 'resnext101':
            self.base = resnext101_ibn_a(4, 32, last_stride, with_abd, **kwargs)
        else:
            raise Exception("Unknown base ", basename)
        #resnet = resnet50(pretrained=True)

        # load pretrained weights
        if model_path != '':
            print('pretrain path', model_path)
            self.base.load_pretrain(model_path)

        self.backbone = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
        )

        # deepcopy，不共享权重
        self.p0 = self.base.layer4
        self.p1 = copy.deepcopy(self.base.layer4)
        self.p2 = copy.deepcopy(self.base.layer4)

        # p1, p2, p3 size: bs x 2048 x 16 x 8
        self.maxpool0 = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool0 = nn.AdaptiveAvgPool2d((1, 1))

        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

        feature_num = 2048

        self.bottleneck0, self.classifier0 = get_bottleneck_classifier(feature_num, num_classes)
        self.bottleneck1, self.classifier1 = get_bottleneck_classifier(feature_num, num_classes)
        self.bottleneck2, self.classifier2 = get_bottleneck_classifier(feature_num, num_classes)

        self.feature_drop = FeatureDrop(h_ratio=0.33, w_ratio=1.0)

    def load_pretrain(self, model_path=''):
        pass

    def forward(self, x):
        x = self.backbone(x)

        p0 = self.p0(x)
        p1 = self.p1(x)
        p2 = self.p2(x)

        if DEBUG:
            print('p1', p1.size())

        # 分支0：全局特征 max + avg
        fea0 = self.maxpool0(p0) + self.avgpool0(p0) # output: bs x 2048 x 1 x 1
        # 分支1：全局特征 max
        fea1 = self.maxpool1(p1)  # output: bs x 2048 x 1 x 1
        # 分支2：全局特征 avg + random feature drop
        fea2 = self.avgpool2(p2)  # output: bs x 2048 x 1 x 1
        if self.training and random.randint(0, 1):
            fea2 = self.feature_drop(fea2)

        bs = fea0.shape[0]
        fea0 = fea0.view(bs, -1)
        fea1 = fea1.view(bs, -1)
        fea2 = fea2.view(bs, -1)

        bn_fea0 = self.bottleneck0(fea0)
        bn_fea1 = self.bottleneck1(fea1)
        bn_fea2 = self.bottleneck2(fea2)

        predict = torch.cat([bn_fea0, bn_fea1 + bn_fea2], dim=1) # bs x 4096

        if DEBUG:
            print('predict', predict.size())

        if self.training:
            cls0 = self.classifier0(bn_fea0)
            cls1 = self.classifier1(bn_fea1)
            cls2 = self.classifier2(bn_fea2)

            return predict, (fea0, fea1, fea2), (cls0, cls1, cls2)
        else:
            return predict

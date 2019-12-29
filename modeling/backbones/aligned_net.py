from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from ..utils import *

from .aligned.HorizontalMaxPool2D import HorizontalMaxPool2d
from .resnet import ResNet
from .resnext_ibn_a import resnext101_ibn_a
from .densenet_ibn import densenet169_ibn_a
from .se_resnet_ibn import se_resnet101_ibn_a

from modeling.backbones.representation import CovpoolLayer, SqrtmLayer, TriuvecLayer

__all__ = ['aligned_net', 'Aligned_Densenet169','Aligned_SE_Resnet101']

from torch.utils import model_zoo
from ops import *
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class Aligned_Net(nn.Module):
    expansion = 4

    def __init__(self, basename, last_stride, with_ibn, gcb, stage_with_gcb, aligned=False, with_abd=False,
                 with_mpncov=False, **kwargs):
        super(Aligned_Net, self).__init__()
        self.with_abd = with_abd

        if basename == 'resnet50':
            self.base = ResNet.from_name('resnet50', last_stride, with_ibn, gcb, stage_with_gcb, with_abd)
        elif basename == 'resnet101':
            self.base = ResNet.from_name('resnet101', last_stride, with_ibn, gcb, stage_with_gcb, with_abd)
        elif basename == 'resnext101':
            self.base = resnext101_ibn_a(4, 32, last_stride, with_abd, **kwargs)
        else:
            raise Exception("Unknown base ", basename)

        #self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(self.feat_dim)
            self.bn.apply(weights_init_kaiming) # Tom added
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(self.feat_dim, 128, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv1.apply(weights_init_kaiming) # Tom added

        self.with_mpncov = with_mpncov
        if self.with_mpncov:
            self.layer_reduce = nn.Conv2d(512 * self.expansion, 256, kernel_size=1, stride=1, padding=0,
                                          bias=False)
            self.layer_reduce_bn = nn.BatchNorm2d(256)
            self.layer_reduce_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.base(x)
        if self.with_abd and self.training:
            x, f_dict = x

        if self.aligned:
            if not self.training:
                lf = self.horizon_pool(x)
                lf = lf.view(lf.size()[0:3])
                ##  # torch.Size([32, 2048, 16])
                #lf = F.adaptive_max_pool1d(input=lf, output_size=(1))
                ##
                lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
                #print('lf.size', lf.size())

            #print('1 x', x.size())  # torch.Size([32, 2048, 16, 8])
            bn_lf = self.bn(x)
            bn_lf = self.relu(bn_lf)
            bn_lf = self.horizon_pool(bn_lf)
            bn_lf = self.conv1(bn_lf)

            bn_lf = bn_lf.view(bn_lf.size()[0:3])
            bn_lf = bn_lf / torch.pow(bn_lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()

        if self.with_mpncov:
            x = self.layer_reduce(x)
            x = self.layer_reduce_bn(x)
            x = self.layer_reduce_relu(x)

            # print('2 x', x.size())
            x = CovpoolLayer(x)
            # print('3 x', x.size())
            x = SqrtmLayer(x, 5)
            #print('4 x', x.size()) # torch.Size([64, 256, 256])
            #f = F.adaptive_max_pool2d(x, output_size=(64, 32))
            x = F.adaptive_avg_pool2d(x, output_size=(64, 32))
            f = x.view(x.size(0), -1)
        else:
            x = F.avg_pool2d(x, x.size()[2:])
            f = x.view(x.size(0), -1)

        # 非训练，直接返回特征
        if not self.training:
            return f, lf, bn_lf

        # 训练
        if self.with_abd:
            if self.aligned:
                return f, bn_lf, f_dict
            return f, f_dict
        else:
            if self.aligned:
                return f, bn_lf
            return f

    def load_pretrain(self, model_path=''):
        self.base.load_pretrain(model_path)


def aligned_net(modelname, last_stride, with_ibn, gcb, stage_with_gcb, aligned=False, with_abd=False, with_mpncov=False, **kwargs):
    return Aligned_Net(modelname, last_stride, with_ibn, gcb, stage_with_gcb, aligned=aligned, with_abd=with_abd,
                       with_mpncov=with_mpncov, **kwargs)


#TODO: 支持 abd
class Aligned_Densenet169(nn.Module):
    def __init__(self, last_stride, with_ibn, gcb, stage_with_gcb, aligned=False, with_abd=False, **kwargs):
        super(Aligned_Densenet169, self).__init__()
        assert with_abd == False
        self.with_abd = with_abd
        #self.loss = loss
        #resnet101 = torchvision.models.resnet101(pretrained=False)
        #self.base = nn.Sequential(*list(resnet101.children())[:-2])
        #resnet101 = ResNet.from_name('resnet101', last_stride, with_ibn, gcb, stage_with_gcb)
        self.base = densenet169_ibn_a()
        #self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 1664 # 2048, feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(self.feat_dim)
            self.bn.apply(weights_init_kaiming)  # Tom added
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(self.feat_dim, 128, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv1.apply(weights_init_kaiming)  # Tom added

    def forward(self, x):
        x = self.base(x)
        if self.with_abd:
            x, f_dict = x

        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            # print('1 x', x.size())  # torch.Size([32, 2048, 16, 8])
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)

        # 非训练，直接返回特征
        if not self.training:
            return f, lf

        # 训练
        if self.with_abd:
            if self.aligned:
                return f, lf, f_dict
            return f, f_dict
        else:
            if self.aligned:
                return f, lf
            return f

    def load_pretrain(self, model_path=''):
        self.base.load_pretrain(model_path)

#TODO: 支持 abd
class Aligned_SE_Resnet101(nn.Module):
    def __init__(self, last_stride, with_ibn, gcb, stage_with_gcb, aligned=False, with_abd=False, **kwargs):
        super(Aligned_SE_Resnet101, self).__init__()
        assert with_abd == False
        self.with_abd = with_abd
        #resnet101 = torchvision.models.resnet101(pretrained=False)
        #self.base = nn.Sequential(*list(resnet101.children())[:-2])
        #resnet101 = ResNet.from_name('resnet101', last_stride, with_ibn, gcb, stage_with_gcb)
        self.base = se_resnet101_ibn_a()
        #self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.bn.apply(weights_init_kaiming)  # Tom added
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv1.apply(weights_init_kaiming)  # Tom added

    def forward(self, x):
        x = self.base(x)
        if self.with_abd:
            x, f_dict = x

        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            # print('1 x', x.size())  # torch.Size([32, 2048, 16, 8])
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)

        # 非训练，直接返回特征
        if not self.training:
            return f, lf

        # 训练
        if self.with_abd:
            if self.aligned:
                return f, lf, f_dict
            return f, f_dict
        else:
            if self.aligned:
                return f, lf
            return f

    def load_pretrain(self, model_path=''):
        self.base.load_pretrain(model_path)

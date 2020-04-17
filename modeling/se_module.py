import torch
from torch import nn
import torch.nn.functional as F

from .utils import *

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, ft_flag=False):
        super(SELayer, self).__init__()
        self.ft_flag = ft_flag
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        init_params(self.fc)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.ft_flag:
            return x * y.expand_as(x)
        else:
            return x * y.expand_as(x), self.avg_pool(x * (1-y).expand_as(x)).view(b, c)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# class MultiScaleLayer(nn.Module):
#     def __init__(self, in_planes, num_classes):
#         super(MultiScaleLayer, self).__init__()
#         self.scale1 = BasicConv2d(in_planes//4, in_planes//4, (1,3), 1, padding=(0,1))
#         self.scale2 = BasicConv2d(in_planes//4, in_planes//4, (3,1), 1, padding=(1,0))
#         self.scale3 = BasicConv2d(in_planes//4, in_planes//4, (1,5), 1, padding=(0,2))
#         self.scale4 = BasicConv2d(in_planes//4, in_planes//4, (5,1), 1, padding=(2,0))
#         init_params(self.scale1)
#         init_params(self.scale2)
#         init_params(self.scale3)
#         init_params(self.scale4)

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.bottleneck = nn.BatchNorm1d(in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.classifier = nn.Linear(in_planes, num_classes, bias=False)
#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)

#     def forward(self, x):
#         x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
#         x1 = self.scale1(x1)
#         x2 = self.scale2(x2)
#         x3 = self.scale3(x3)
#         x4 = self.scale4(x4)
#         x = torch.cat((x1, x2, x3, x4), dim=1)

#         x = self.gap(x)
#         x = x.view(-1, x.size()[1])
#         x = self.bottleneck(x)  # normalize for angular softmax
#         return self.classifier(x)

class MultiScaleLayer_v2(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(MultiScaleLayer_v2, self).__init__()
        self.scale1 = BasicConv2d(in_planes//4, in_planes//4, (1,3), 1, padding=(0,1))
        self.scale2 = BasicConv2d(in_planes//4, in_planes//4, (3,1), 1, padding=(1,0))
        self.scale3 = BasicConv2d(in_planes//4, in_planes//4, (1,5), 1, padding=(0,2))
        self.scale4 = BasicConv2d(in_planes//4, in_planes//4, (5,1), 1, padding=(2,0))
        init_params(self.scale1)
        init_params(self.scale2)
        init_params(self.scale3)
        init_params(self.scale4)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(in_planes, num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = self.scale1(x1)
        x2 = self.scale2(x2)
        x3 = self.scale3(x3)
        x4 = self.scale4(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.gap(x)
        x = x.view(-1, x.size()[1])
        x = self.bottleneck(x)  # normalize for angular softmax
        x = self.classifier(x)
        return x


class SELayer_Local(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_Local, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc_avg = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        init_params(self.fc_avg)

        self.fc_max = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        init_params(self.fc_max)

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc_avg(y_avg).view(b, c, 1, 1)
        y_avg = x * y_avg.expand_as(x)

        y_max = self.max_pool(x).view(b, c)
        y_max = self.fc_max(y_max).view(b, c, 1, 1)
        y_max = x * y_max.expand_as(x)

        return y_avg + y_max


# class SideLayer(nn.Module):
#     def __init__(self, in_planes, out_planes, out_up_planes, num_classes):
#         super(SideLayer, self).__init__()
#         self.reduction = BasicConv2d(in_planes, out_planes, kernel_size=1, stride=1)
#         self.adapt = BasicConv2d(out_up_planes, out_planes, kernel_size=1, stride=1)
#         init_params(self.reduction)
#         init_params(self.adapt)

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.gmp = nn.AdaptiveMaxPool2d(1)

#         self.bottleneck = nn.BatchNorm1d(out_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.classifier = nn.Linear(out_planes, num_classes, bias=False)
#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)


#     def forward(self, x, high_res_x):
#         x = self.reduction(x)
#         x = F.interpolate(x, size=(high_res_x.shape[2], high_res_x.shape[3]), mode='bilinear', align_corners=True)

#         x = torch.cat([x, high_res_x], dim=1)
#         x = self.adapt(x)
#         x_triptle = torch.squeeze(self.gap(x) + self.gmp(x))

#         x_feat = self.bottleneck(x_triptle)
#         x_score = self.classifier(x_feat)
#         if self.training:
#             return x_score, x_triptle, x
#         else:
#             return x_score, x_feat, x


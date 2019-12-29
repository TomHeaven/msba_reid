import copy
import torch
import torch.nn as nn
from modeling.utils import *
#from torchvision.models.resnet import resnet50, Bottleneck

from modeling.backbones.resnet import ResNet, Bottleneck
from modeling.backbones.resnext_ibn_a import resnext101_ibn_a



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
            self.base.layer4[0],
        )

        res_conv4_branch0 = nn.Sequential(*self.base.layer4[1:])
        res_conv4_branch1 = nn.Sequential(
            #Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_conv4_branch1.load_state_dict(res_conv4_branch0.state_dict())
        #self.res_p_conv5 = res_p_conv5

        # deepcopy，不共享权重
        self.p1 = res_conv4_branch0
        self.p2 = res_conv4_branch1

        # p1, p2, p3 size: bs x 2048 x 16 x 8
        self.maxpool0 = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool1 = nn.AdaptiveMaxPool2d((2, 1))

        feature_num = 2048
        reduction_num = feature_num // 2

        self.reduction1_0 = get_reduction(feature_num, reduction_num)
        self.reduction1_1 = get_reduction(feature_num, reduction_num)

        self.bottleneck0, self.classfier0 = get_bottleneck_classifier(feature_num, num_classes)
        self.bottleneck1_0, self.classfier1_0 = get_bottleneck_classifier(reduction_num , num_classes)
        self.bottleneck1_1, self.classfier1_1 = get_bottleneck_classifier(reduction_num , num_classes)

    def load_pretrain(self, model_path=''):
        pass
        #self.base.load_pretrain(model_path)
        #self.res_p_conv5.load_state_dict(self.base.layer4.state_dict())

    def forward(self, x):
        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)

        if DEBUG:
            print('p1', p1.size())
            print('p2', p2.size())

        fea0 = self.maxpool0(p1) # output: bs x 2048 x 1 x 1
        fea1 = self.maxpool1(p2) # output: bs x 2048 x 2 x 1

        fea1_0 = fea1[:, :, 0:1, :]  # output: bs x 2048 x 1 x 1
        fea1_1 = fea1[:, :, 1:2, :]  # output: bs x 2048 x 1 x 1

        fea0 = fea0.view(fea0.shape[0], -1)
        fea1_0 = self.reduction1_0(fea1_0).view(fea1_0.shape[0], -1) # bs x 1024
        fea1_1 = self.reduction1_1(fea1_1).view(fea1_1.shape[0], -1) # bs x 1024


        bn_fea0 = self.bottleneck0(fea0)
        bn_fea1_0 = self.bottleneck1_0(fea1_0)
        bn_fea1_1 = self.bottleneck1_1(fea1_1)

        predict = torch.cat([bn_fea0, bn_fea1_0, bn_fea1_1], dim=1) # bs x 4096

        if DEBUG:
            print('predict', predict.size()) # bs x 2048

        if self.training:
            cls0 = self.classfier0(bn_fea0)
            cls1_0 = self.classfier1_0(bn_fea1_0)
            cls1_1 = self.classfier1_1(bn_fea1_1)

            return predict, fea0, fea1_0, fea1_1, cls0, cls1_0, cls1_1
        else:
            return predict

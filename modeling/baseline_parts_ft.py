# encoding: utf-8
"""
@author:  Hanlin Tan & Huaxin Xiao & Xiaoyu Zhang
@contact: hanlin_tan@nudt.edu.cn
"""
import torch
from torch import nn

from .backbones import *
from .utils import *
from .se_module import SELayer, MultiScaleLayer_v2, SELayer_Local, BasicConv2d

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, 
                 backbone, 
                 num_classes, 
                 last_stride, 
                 with_ibn, 
                 gcb, 
                 stage_with_gcb,
                 use_parts=2,
                 pretrain=True, 
                 model_path=''):
        super().__init__()
        try:
            if backbone.startswith('resnet'):
                self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
            elif backbone.startswith('resnext'):
                self.base = ResNext.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
        except:
            print(f'not support {backbone} backbone')

        #if pretrain:
        #    self.base.load_pretrain(model_path)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.use_parts = use_parts

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)

        # use attention
        self.att_layers = [256, 512, 1024]
        self._init_attention()
        self.bottleneck_1 = nn.BatchNorm1d(sum(self.att_layers))
        self.bottleneck_1.bias.requires_grad_(False)  # no shift
        # self.classifier_1 = nn.Linear(sum(self.att_layers), self.num_classes, bias=False)

        self.bottleneck_1.apply(weights_init_kaiming)
        # self.classifier_1.apply(weights_init_classifier)

        # use multi scales
        # self.branch1 = MultiScaleLayer_v2(512, num_classes)
        # self.branch2 = MultiScaleLayer_v2(1024, num_classes)

        # use local features
        self.local_reduction = BasicConv2d(2048, 512, kernel_size=1, stride=1)
        init_params(self.local_reduction)

        self.local_att = SELayer_Local(512)

        self.local_bottlenecks = nn.ModuleList()
        # self.local_classifiers = nn.ModuleList()
        for i in range(self.use_parts):
            l_bottleneck = nn.BatchNorm1d(512)
            l_bottleneck.bias.requires_grad_(False)
            l_bottleneck.apply(weights_init_kaiming)
            self.local_bottlenecks.append(l_bottleneck)

            # l_classifier = nn.Linear(512, self.num_classes, bias=False)
            # l_classifier.apply(weights_init_classifier)
            # self.local_classifiers.append(l_classifier)

    def _init_attention(self):
        self.att_modules = nn.ModuleList()
        for layer in self.att_layers:
            att_module = SELayer(layer, ft_flag=True)
            self.att_modules.append(att_module)

    def forward(self, x, label=None):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.att_modules[0](x)

        x = self.base.layer2(x)
        x = self.att_modules[1](x)
        # cls_score_br1 = self.branch1(x)

        x = self.base.layer3(x)
        x = self.att_modules[2](x)
        # cls_score_br2 = self.branch2(x)

        x = self.base.layer4(x)
        
        # global features	
        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(-1, global_feat.size()[1])
        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        # local features
        lf_xent, local_feat = [], []

        local_x = self.local_reduction(x)
        margin = local_x.size(2) // self.use_parts
        for i in range(self.use_parts):
            x_out = self.local_att(local_x[:, :, margin*i:margin*(i+1), :])
            x_out = self.gap(x_out)
            x_out_feat = x_out.view(-1, x_out.size()[1])
            x_feat = self.local_bottlenecks[i](x_out_feat)
            local_feat.append(x_feat)

        local_feat.append(feat)
        if self.training:
            return tuple(local_feat)
        else:
            return torch.cat(local_feat, dim=1)

    def load_params_wo_fc(self, state_dict):
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     k = '.'.join(k.split('.')[1:])
        #     new_state_dict[k] = v
        # state_dict = new_state_dict
        # state_dict.pop('classifier.weight')
        res = self.load_state_dict(state_dict, strict=False)
        print(res.missing_keys)
        # assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'

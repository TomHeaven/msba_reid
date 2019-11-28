# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from modeling.backbones import *
from modeling.losses.cosface import AddMarginProduct
from modeling.utils import *

__all__ =['Baseline', 'BASE_RESNET50', 'BASE_MPNCOV_RESNET101', 'BASE_ALIGNED_RESNET50', 'BASE_ALIGNED_RESNET101',
          'BASE_ALIGNED_RESNEXT101', 'BASE_ALIGNED_RESNEXT50', 'BASE_ALIGNED_SE_RESNET101', 'BASE_ALIGNED_DENSENET169',
          'BASE_ALIGNED_RESNET50_ABD', 'BASE_ALIGNED_RESNET101_ABD', 'BASE_ALIGNED_RESNEXT101_ABD',
          'BASE_ALIGNED_MPNCOV_RESNET50', 'BASE_ALIGNED_MPNCOV_RESNET101', 'BASE_ALIGNED_MPNCOV_RESNEXT101']

BASE_RESNET50 = 0
BASE_MPNCOV_RESNET101 = 1
BASE_ALIGNED_RESNET50 = 2
BASE_ALIGNED_RESNET101 = 3
BASE_ALIGNED_RESNEXT101 = 4
BASE_ALIGNED_RESNEXT50 = 13
BASE_ALIGNED_SE_RESNET101 = 5
BASE_ALIGNED_DENSENET169 = 6

BASE_ALIGNED_RESNET50_ABD = 7
BASE_ALIGNED_RESNET101_ABD = 8
BASE_ALIGNED_RESNEXT101_ABD = 9

BASE_ALIGNED_MPNCOV_RESNET50 = 10
BASE_ALIGNED_MPNCOV_RESNET101 = 11
BASE_ALIGNED_MPNCOV_RESNEXT101 = 12


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, 
                 backbone, 
                 num_classes, 
                 last_stride, 
                 with_ibn, 
                 gcb, 
                 stage_with_gcb,
                 pretrain=True, 
                 model_path=''):
        super().__init__()
        #try:
        if True:
            if backbone.startswith('resnet'):
                self.base_type = BASE_RESNET50
                self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
            elif backbone == 'mpncov_resnet101':
                #self.use_mpn_cov = True
                self.base_type = BASE_MPNCOV_RESNET101
                print('backbone', backbone)
                self.base = mpncovresnet101(False, last_stride, with_ibn, gcb, stage_with_gcb)
            ########################################
            ## 加 aligned
            elif backbone == 'aligned_resnet50':
                self.base_type = BASE_ALIGNED_RESNET50
                self.base = aligned_net('resnet50',last_stride, with_ibn, gcb, stage_with_gcb, aligned=True)
            elif backbone == 'aligned_resnet101':
                self.base_type = BASE_ALIGNED_RESNET101
                self.base = aligned_net('resnet101',last_stride, with_ibn, gcb, stage_with_gcb, aligned=True)
            elif backbone == 'aligned_resnext101':
                self.base_type = BASE_ALIGNED_RESNEXT101
                self.base = aligned_net('resnext101',last_stride, with_ibn, gcb, stage_with_gcb, aligned=True)
            elif backbone == 'aligned_resnext50':
                self.base_type = BASE_ALIGNED_RESNEXT50
                self.base = aligned_net('resnext50', last_stride, with_ibn, gcb, stage_with_gcb, aligned=True)
            elif backbone == 'aligned_se_resnet101':
                self.base_type = BASE_ALIGNED_SE_RESNET101
                self.base = Aligned_SE_Resnet101(last_stride, with_ibn, gcb, stage_with_gcb, aligned=True)
            elif backbone == 'aligned_densenet169':
                self.base_type = BASE_ALIGNED_DENSENET169
                self.base = Aligned_Densenet169(last_stride, with_ibn, gcb, stage_with_gcb, aligned=True)
                self.in_planes = 1664
            ########################################
            ## 加 abd
            elif backbone == 'aligned_resnet50_abd':
                self.base_type = BASE_ALIGNED_RESNET50_ABD
                self.base = aligned_net_v1('resnet50',last_stride, with_ibn, gcb, stage_with_gcb, aligned=True, with_abd=True)
            elif backbone == 'aligned_resnet101_abd':
                self.base_type = BASE_ALIGNED_RESNET101_ABD
                #self.base = aligned_net('resnet101',last_stride, with_ibn, gcb, stage_with_gcb, aligned=True, with_abd=True)
                self.base = aligned_net_v1('resnet101', last_stride, with_ibn, gcb, stage_with_gcb, aligned=True, with_abd=True)
            elif backbone == 'aligned_resnext101_abd':
                self.base_type = BASE_ALIGNED_RESNEXT101_ABD
                #self.base = aligned_net('resnext101', last_stride, with_ibn, gcb, stage_with_gcb,aligned=True, with_abd=True)
                self.base = aligned_net_v1('resnext101', last_stride, with_ibn, gcb, stage_with_gcb, aligned=True,
                                           with_abd=True)
            ########################################
            ## aligned + mpncov
            elif backbone == 'aligned_mpncov_resnet50':
                self.base_type = BASE_ALIGNED_MPNCOV_RESNET50
                self.base = aligned_net('resnet50', last_stride, with_ibn, gcb, stage_with_gcb, aligned=True, with_mpncov=True)
            elif backbone == 'aligned_mpncov_resnet101':
                self.base_type = BASE_ALIGNED_MPNCOV_RESNET101
                self.base = aligned_net('resnet101', last_stride, with_ibn, gcb, stage_with_gcb, aligned=True, with_mpncov=True)
            elif backbone == 'aligned_mpncov_resnext101':
                self.base_type = BASE_ALIGNED_MPNCOV_RESNEXT101
                self.base = aligned_net('resnext101', last_stride, with_ibn, gcb, stage_with_gcb, aligned=True, with_mpncov=True)
            else:
                raise Exception('Unknown backbone', backbone)
        #except:
        #    print(f'not support {backbone} backbone')

        if pretrain:
            self.base.load_pretrain(model_path)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        # self.classifier = AddMarginProduct(self.in_planes, self.num_classes, s=20, m=0.3)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):

        if self.base_type in [BASE_MPNCOV_RESNET101]:
            global_feat = self.base(x)  # mpncov doesnot need self.gap to reduce features
        elif self.base_type in [BASE_ALIGNED_RESNET50, BASE_ALIGNED_RESNET101, BASE_ALIGNED_RESNEXT101,
                                BASE_ALIGNED_RESNEXT50,
                                BASE_ALIGNED_DENSENET169, BASE_ALIGNED_SE_RESNET101,
                                BASE_ALIGNED_MPNCOV_RESNET50, BASE_ALIGNED_MPNCOV_RESNET101,
                                BASE_ALIGNED_MPNCOV_RESNEXT101]:
            global_feat, local_feat = self.base(x)
            # local_feat = local_feat.view(local_feat.shape[0], -1)
            # print('1 global_feat', global_feat.size()) # torch.Size([31, 2048])
            # print('1 local_feat', local_feat.size())   # train: local_feat torch.Size([32, 128, 16]) / test: torch.Size([64, 2048, 16])
        elif self.base_type in [BASE_ALIGNED_RESNET101_ABD, BASE_ALIGNED_RESNET50_ABD, BASE_ALIGNED_RESNEXT101_ABD]:
            if self.training:
                global_feat, local_feat, f_dict = self.base(x)
            else:
                global_feat, local_feat = self.base(x)
        else:
            global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)

        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        bn_feat = self.bottleneck(global_feat)  # normalize for angular softmax
        if self.training:
            cls_score = self.classifier(bn_feat)
            if self.base_type in [BASE_ALIGNED_RESNET50, BASE_ALIGNED_RESNET101, BASE_ALIGNED_RESNEXT101, BASE_ALIGNED_RESNEXT50,
                                  BASE_ALIGNED_DENSENET169, BASE_ALIGNED_SE_RESNET101,
                                  BASE_ALIGNED_MPNCOV_RESNET50, BASE_ALIGNED_MPNCOV_RESNET101, BASE_ALIGNED_MPNCOV_RESNEXT101]:
                return cls_score, global_feat, local_feat
            elif self.base_type in [BASE_ALIGNED_RESNET101_ABD, BASE_ALIGNED_RESNET50_ABD, BASE_ALIGNED_RESNEXT101_ABD]:
                return cls_score, global_feat, local_feat, f_dict
            else:
                return cls_score, global_feat
        else:
            if self.base_type in [BASE_ALIGNED_RESNET50, BASE_ALIGNED_RESNET101, BASE_ALIGNED_RESNEXT101, BASE_ALIGNED_RESNEXT50,
                                  BASE_ALIGNED_DENSENET169, BASE_ALIGNED_SE_RESNET101,
                                  BASE_ALIGNED_MPNCOV_RESNET50, BASE_ALIGNED_MPNCOV_RESNET101,
                                  BASE_ALIGNED_MPNCOV_RESNEXT101
                                  ]:
                return global_feat, local_feat  #非abd时返回 g + bn_l
            elif self.base_type in [BASE_ALIGNED_RESNET101_ABD, BASE_ALIGNED_RESNET50_ABD, BASE_ALIGNED_RESNEXT101_ABD]:
                return bn_feat, local_feat # abd时返回 bn_g + l
            else:
                return bn_feat # 几乎不使用

    def load_params_wo_fc(self, state_dict):
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     k = '.'.join(k.split('.')[1:])
        #     new_state_dict[k] = v
        # state_dict = new_state_dict
        state_dict.pop('classifier.weight')
        res = self.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'

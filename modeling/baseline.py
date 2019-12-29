# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from .backbones import *
from .losses.cosface import AddMarginProduct
from .utils import *
#from .backbones.resnext_ibn_a import resnext101_ibn_a

__all__ =['Baseline',
          'BASE_RESNET50', 'BASE_RESNET101', 'BASE_RESNEXT101',
          'BASE_RESNET101_ABD', 'BASE_RESNEXT101_ABD',
          'BASE_MPNCOV_RESNET101', 'BASE_ALIGNED_RESNET50', 'BASE_ALIGNED_RESNET101',
          'BASE_ALIGNED_RESNEXT101', 'BASE_ALIGNED_RESNEXT50', 'BASE_ALIGNED_SE_RESNET101', 'BASE_ALIGNED_DENSENET169',
          'BASE_ALIGNED_RESNET50_ABD', 'BASE_ALIGNED_RESNET101_ABD', 'BASE_ALIGNED_RESNEXT101_ABD',
          'BASE_ALIGNED_MPNCOV_RESNET50', 'BASE_ALIGNED_MPNCOV_RESNET101', 'BASE_ALIGNED_MPNCOV_RESNEXT101',
          'MGN_RESNET50','MGN_RESNET101', 'MGN_RESNEXT101']

BASE_RESNET50 = 0
BASE_RESNET101 = 20
BASE_RESNEXT101 = 21
BASE_RESNET101_ABD = 22
BASE_RESNEXT101_ABD = 23

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


MGN_RESNET50 = 30
MGN_RESNET101 = 31
MGN_RESNEXT101 = 32


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
            #print('backbone', backbone)
            if backbone == 'resnet50':
                self.base_type = BASE_RESNET50
                self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb, with_abd=False)
            elif backbone == 'resnet101':
                self.base_type = BASE_RESNET101
                self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb, with_abd=False)
            elif backbone == 'resnext101':
                self.base_type = BASE_RESNEXT101
                self.base = resnext101_ibn_a(4, 32, last_stride, with_abd=False)
            elif backbone == 'resnet101_abd':
                self.base_type = BASE_RESNET101_ABD
                self.base = ResNet.from_name('resnet101', last_stride, with_ibn, gcb, stage_with_gcb, with_abd=True)
            elif backbone == 'resnext101_abd':
                self.base_type = BASE_RESNEXT101_ABD
                self.base = resnext101_ibn_a(4, 32, last_stride, with_abd=True)
            elif backbone == 'mpncov_resnet101':
                #self.use_mpn_cov = True
                self.base_type = BASE_MPNCOV_RESNET101
                self.base = mpncovresnet101(False, last_stride, with_ibn, gcb, stage_with_gcb)
            ########################################
            ## mgn
            elif backbone == 'mgn_resnet50':
                self.base_type = MGN_RESNET101
                self.base = MGN(model_path, num_classes, 'resnet50', last_stride, with_ibn, gcb, stage_with_gcb, with_abd=False)
            elif backbone == 'mgn_resnet101':
                self.base_type = MGN_RESNET101
                self.base = MGN(model_path, num_classes, 'resnet101', last_stride, with_ibn, gcb, stage_with_gcb, with_abd=False)
            elif backbone == 'mgn_resnext101':
                self.base_type = MGN_RESNEXT101
                self.base = MGN(model_path, num_classes, 'resnet101', last_stride, with_ibn, gcb, stage_with_gcb, with_abd=False)
            #########
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
                self.base = aligned_net('resnet50',last_stride, with_ibn, gcb, stage_with_gcb, aligned=True, with_abd=True)
            elif backbone == 'aligned_resnet101_abd':
                self.base_type = BASE_ALIGNED_RESNET101_ABD
                #self.base = aligned_net('resnet101',last_stride, with_ibn, gcb, stage_with_gcb, aligned=True, with_abd=True)
                self.base = aligned_net('resnet101', last_stride, with_ibn, gcb, stage_with_gcb, aligned=True, with_abd=True)
            elif backbone == 'aligned_resnext101_abd':
                self.base_type = BASE_ALIGNED_RESNEXT101_ABD
                #self.base = aligned_net('resnext101', last_stride, with_ibn, gcb, stage_with_gcb,aligned=True, with_abd=True)
                self.base = aligned_net('resnext101', last_stride, with_ibn, gcb, stage_with_gcb, aligned=True,
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

        if self.base_type not in [MGN_RESNET50, MGN_RESNET101, MGN_RESNEXT101]:
        #if True:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.num_classes = num_classes

            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):

        if self.base_type in [BASE_MPNCOV_RESNET101]:
        ### global only (mpncov)
            global_feat = self.base(x)  # mpncov doesnot need self.gap to reduce features
        elif self.base_type in [MGN_RESNET50, MGN_RESNET101, MGN_RESNEXT101]:
            # mgn is independent
            return self.base(x)
            # use mgn as global feature
            #global_feat = self.base(x)
        elif self.base_type in [BASE_ALIGNED_RESNET50, BASE_ALIGNED_RESNET101, BASE_ALIGNED_RESNEXT101,
        ### aligned
                                BASE_ALIGNED_RESNEXT50,
                                BASE_ALIGNED_DENSENET169, BASE_ALIGNED_SE_RESNET101,
                                BASE_ALIGNED_MPNCOV_RESNET50, BASE_ALIGNED_MPNCOV_RESNET101,
                                BASE_ALIGNED_MPNCOV_RESNEXT101]:
            if self.training:
                global_feat, bn_local_feat = self.base(x)
            else:
                global_feat, local_feat, bn_local_feat = self.base(x)

            # print('1 global_feat', global_feat.size()) # torch.Size([31, 2048])
            # train: local_feat torch.Size([32, 128, 16]) / test: bn_local_feat torch.Size([64, 2048, 16])
            # print('1 bn_local_feat', bn_local_feat.size())
        elif self.base_type in [BASE_ALIGNED_RESNET101_ABD, BASE_ALIGNED_RESNET50_ABD, BASE_ALIGNED_RESNEXT101_ABD]:
        ### aligned abd
            if self.training:
                global_feat, bn_local_feat, f_dict = self.base(x)
            else:
                global_feat, local_feat, bn_local_feat = self.base(x)
        elif self.base_type in [BASE_RESNET101_ABD, BASE_RESNEXT101_ABD]:
        ### abd
            if self.training:
                global_feat, f_dict = self.base(x)
            else:
                global_feat = self.base(x)
            global_feat = self.gap(global_feat)  # (b, 2048, 1, 1)
        else:
        ### global only
            x = self.base(x)
            global_feat = self.gap(x) # (b, 2048, 1, 1)

        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        bn_global_feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(bn_global_feat)
            if self.base_type in [BASE_ALIGNED_RESNET50, BASE_ALIGNED_RESNET101, BASE_ALIGNED_RESNEXT101, BASE_ALIGNED_RESNEXT50,
                                  BASE_ALIGNED_DENSENET169, BASE_ALIGNED_SE_RESNET101,
                                  BASE_ALIGNED_MPNCOV_RESNET50, BASE_ALIGNED_MPNCOV_RESNET101, BASE_ALIGNED_MPNCOV_RESNEXT101]:
                ### aligned
                return cls_score, global_feat, bn_local_feat
            elif self.base_type in [BASE_ALIGNED_RESNET101_ABD, BASE_ALIGNED_RESNET50_ABD, BASE_ALIGNED_RESNEXT101_ABD]:
                ### aligned abd
                return cls_score, global_feat, bn_local_feat, f_dict
            elif self.base_type in [BASE_RESNET101_ABD, BASE_RESNEXT101_ABD]:
                return cls_score, global_feat, f_dict
            else:
                ### global only
                return cls_score, global_feat
        else:
            if self.base_type in [BASE_ALIGNED_RESNET50,  BASE_ALIGNED_RESNET101, BASE_ALIGNED_RESNEXT101, BASE_ALIGNED_RESNEXT50,
                                  BASE_ALIGNED_DENSENET169, BASE_ALIGNED_SE_RESNET101,
                                  BASE_ALIGNED_MPNCOV_RESNET50, BASE_ALIGNED_MPNCOV_RESNET101,
                                  BASE_ALIGNED_MPNCOV_RESNEXT101,
                                  BASE_ALIGNED_RESNET101_ABD, BASE_ALIGNED_RESNET50_ABD, BASE_ALIGNED_RESNEXT101_ABD
                                  ]:
                ### aligned (+-abd)
                return global_feat, bn_global_feat, local_feat, bn_local_feat
            else:
                ### global only
                return global_feat, bn_global_feat

    def load_params_wo_fc(self, state_dict):
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     k = '.'.join(k.split('.')[1:])
        #     new_state_dict[k] = v
        # state_dict = new_state_dict

        #if 'classifier.weight' in state_dict:
        #    state_dict.pop('classifier.weight')

        new_state_dict = state_dict.copy()
        for k in state_dict.keys():
            if k =='classifier.weight' or k.startswith('base.fc_id_') or k.startswith('base.classfier') or k.startswith('base.classifier'):
                new_state_dict.pop(k)
        state_dict = new_state_dict

        res = self.load_state_dict(state_dict, strict=False)
        #assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'
        print('missing_keys', res.missing_keys)



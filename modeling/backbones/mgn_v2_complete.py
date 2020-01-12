import copy
import torch
import torch.nn as nn
#from torchvision.models.resnet import resnet50, Bottleneck

from .resnet import ResNet, Bottleneck
from .resnext_ibn_a import resnext101_ibn_a

#num_classes = 751  # change this depend on your dataset
#num_classes = 4768 + 1

DEBUG = False

class MGN(nn.Module):
    def __init__(self, num_classes, basename, last_stride, with_ibn, gcb, stage_with_gcb, with_abd, feats = 256, **kwargs):
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

        self.backbone = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3[0],
        )

        res_conv4 = nn.Sequential(*self.base.layer3[1:])

        res_g_conv5 = self.base.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        #res_p_conv5.load_state_dict(self.base.layer4.state_dict())
        self.res_p_conv5 = res_p_conv5

        # deepcopy，不共享权重
        self.p1 = nn.Sequential(res_conv4, res_g_conv5)
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        # 共享权重
        #self.p1 = nn.Sequential(res_conv4, res_g_conv5)
        #self.p2 = nn.Sequential(res_conv4,res_p_conv5)
        #self.p3 = nn.Sequential(res_conv4, res_p_conv5)

        # p1, p2, p3 size: bs x 2048 x 16 x 8

        self.maxpool_zg_p1 = nn.AdaptiveMaxPool2d((1, 1))  #nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.AdaptiveMaxPool2d((1, 1))  #nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.AdaptiveMaxPool2d((1, 1))  #nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.AdaptiveMaxPool2d((2, 1))    #nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.AdaptiveMaxPool2d((3, 1))    #nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction_0 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_1_0 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_1_1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_2_0 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_2_1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_2_2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        if self.training:
            self.fc_id_2048_0 = nn.Linear(feats, num_classes)
            self.fc_id_2048_1 = nn.Linear(feats, num_classes)
            self.fc_id_2048_2 = nn.Linear(feats, num_classes)

            self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
            self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
            self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
            self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
            self.fc_id_256_2_2 = nn.Linear(feats, num_classes)


    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def load_pretrain(self, model_path=''):
        self.base.load_pretrain(model_path)
        self.res_p_conv5.load_state_dict(self.base.layer4.state_dict())

        ## init fc layers
        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    def forward(self, x):
        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        if DEBUG:
            print('p1', p1.size())
            print('p2', p2.size())
            print('p3', p3.size())

        zg_p1 = self.maxpool_zg_p1(p1) # output: bs x 2048 x 1 x 1
        zg_p2 = self.maxpool_zg_p2(p2) # output: bs x 2048 x 1 x 1
        zg_p3 = self.maxpool_zg_p3(p3) # output: bs x 2048 x 1 x 1

        zp2 = self.maxpool_zp2(p2)

        if DEBUG:
            print('zp2', zp2.size())

        z0_p2 = zp2[:, :, 0:1, :]  # output: bs x 2048 x 1 x 1
        z1_p2 = zp2[:, :, 1:2, :]  # output: bs x 2048 x 1 x 1

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2) # bs x 256
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_1_0(z0_p2).squeeze(dim=3).squeeze(dim=2) # bs x 256
        f1_p2 = self.reduction_1_1(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_2_0(z0_p3).squeeze(dim=3).squeeze(dim=2) # bs x 256
        f1_p3 = self.reduction_2_1(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_2_2(z2_p3).squeeze(dim=3).squeeze(dim=2)


        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        if DEBUG:
            print('predict', predict.size()) # bs x 2048

        if self.training:
            l_p1 = self.fc_id_2048_0(fg_p1)
            l_p2 = self.fc_id_2048_1(fg_p2)
            l_p3 = self.fc_id_2048_2(fg_p3)

            l0_p2 = self.fc_id_256_1_0(f0_p2)
            l1_p2 = self.fc_id_256_1_1(f1_p2)
            l0_p3 = self.fc_id_256_2_0(f0_p3)  #
            l1_p3 = self.fc_id_256_2_1(f1_p3)
            l2_p3 = self.fc_id_256_2_2(f2_p3)

            return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
        else:
            return predict

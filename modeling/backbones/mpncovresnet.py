import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from modeling.backbones.resnet import Bottleneck
from modeling.backbones.representation import CovpoolLayer, SqrtmLayer, TriuvecLayer
import torch.nn.functional as F
__all__ = ['MPNCOVResNet','mpncovresnet50', 'mpncovresnet101']


model_urls = {
    'mpncovresnet50': 'http://jtxie.com/models/mpncovresnet50-15991845.pth',
    'mpncovresnet101': 'http://jtxie.com/models/mpncovresnet101-ade9737a.pth'
}



class MPNCOVResNet(nn.Module):

    def __init__(self, block, layers, last_stride, with_ibn, gcb, stage_with_gcb, num_classes=1000):
        self.inplanes = 64
        super(MPNCOVResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], with_ibn=with_ibn,
                         gcb=gcb if stage_with_gcb[0] else None)
        self.layer2 = self._make_layer(block, 128, layers[1], with_ibn=with_ibn,
                         gcb=gcb if stage_with_gcb[0] else None, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], with_ibn=with_ibn,
                         gcb=gcb if stage_with_gcb[0] else None, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], with_ibn=with_ibn,
                         gcb=gcb if stage_with_gcb[0] else None, stride=last_stride) # mpncov_resnet last_stride本来就是1
        self.layer_reduce = nn.Conv2d(512 * block.expansion, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(int(256*(256+1)/2), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, with_ibn, gcb, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, with_ibn, gcb, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #print('1 x', x.size())
        # 1x1 Conv. for dimension reduction
        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)

        #print('2 x', x.size())

        x = CovpoolLayer(x)
        #print('3 x', x.size())
        x = SqrtmLayer(x, 5)
        #print('4 x', x.size())
        #x = TriuvecLayer(x)
        #print('5 x', x.size())

        x = F.adaptive_max_pool2d(x, output_size=(64, 32))

        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

    def load_pretrain(self, model_path):
        print('mpncov_resnet pretrained weight path', model_path)
        pretrained_dict = torch.load(model_path)
        own_dict = self.state_dict()
        # print('pretrained dict : ', pretrained_dict)
        # print('own_dict : ', own_dict)
        for key in own_dict.keys():
            if key in pretrained_dict.keys():
                if pretrained_dict[key].size() == own_dict[key].size():
                    print('copied key : ', key)
                    # print('pretained_dict[key].size : ', pretrained_dict[key].size())
                    # print('own_dict[key].size : ', own_dict[key].size())
                    own_dict[key] = pretrained_dict[key].clone()

        self.load_state_dict(own_dict)



def mpncovresnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MPNCOVResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['mpncovresnet50']))
        pretrained_dict = model_zoo.load_url(model_urls['mpncovresnet50'])
        own_dict = model.state_dict()
        # print('pretrained dict : ', pretrained_dict)
        # print('own_dict : ', own_dict)
        for key in own_dict.keys():
            if key in pretrained_dict.keys():
                if pretrained_dict[key].size() == own_dict[key].size():
                    print('copied key : ', key)
                    # print('pretained_dict[key].size : ', pretrained_dict[key].size())
                    # print('own_dict[key].size : ', own_dict[key].size())
                    own_dict[key] = pretrained_dict[key].clone()

        model.load_state_dict(own_dict)
    return model


def mpncovresnet101(pretrained, last_stride, with_ibn, gcb, stage_with_gcb):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MPNCOVResNet(Bottleneck, [3, 4, 23, 3], last_stride, with_ibn, gcb, stage_with_gcb)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['mpncovresnet101']))
        pretrained_dict = model_zoo.load_url(model_urls['mpncovresnet101'])
        own_dict = model.state_dict()
        # print('pretrained dict : ', pretrained_dict)
        # print('own_dict : ', own_dict)
        for key in own_dict.keys():
            if key in pretrained_dict.keys():
                if pretrained_dict[key].size() == own_dict[key].size():
                    print('copied key : ', key)
                    # print('pretained_dict[key].size : ', pretrained_dict[key].size())
                    # print('own_dict[key].size : ', own_dict[key].size())
                    own_dict[key] = pretrained_dict[key].clone()

        model.load_state_dict(own_dict)
    return model


if __name__ == '__main__':
    x = torch.zeros(1, 3, 512, 512)
    net = mpncovresnet101()
    y = net(x)


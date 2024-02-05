import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch.nn.functional as F

from pyfed.utils.augmentation import (
    GaussianFeatureAugmentationClientBatch,
    GaussianFeatureAugmentationClientMomentum,
    GaussianFeatureAugmentationFederated,
    RandomFeatureAugmentation,
    GaussianFeatureAugmentationClientMemory
)


def gaussian_feature_augmentation(aug_method=None):
    if aug_method is None:
        return nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()
    elif aug_method == 'client_batch':
        aug = GaussianFeatureAugmentationClientBatch()
        return aug, aug, aug, aug
    elif aug_method == 'client_memory':
        aug1 = GaussianFeatureAugmentationClientMemory(num_features=64, memory_size=32)
        aug2 = GaussianFeatureAugmentationClientMemory(num_features=192, memory_size=32)
        aug3 = GaussianFeatureAugmentationClientMemory(num_features=384, memory_size=32)
        aug4 = GaussianFeatureAugmentationClientMemory(num_features=256, memory_size=32)
        return aug1, aug2, aug3, aug4
    elif aug_method == 'client_momentum':
        aug1 = GaussianFeatureAugmentationClientMomentum(momentum=0.5, num_features=64)
        aug2 = GaussianFeatureAugmentationClientMomentum(momentum=0.5, num_features=192)
        aug3 = GaussianFeatureAugmentationClientMomentum(momentum=0.5, num_features=384)
        aug4 = GaussianFeatureAugmentationClientMomentum(momentum=0.5, num_features=256)
        return aug1, aug2, aug3, aug4
    elif aug_method == 'federated_copy':
        pass
    elif aug_method == 'federated_grad':
        aug1 = GaussianFeatureAugmentationFederated(momentum=0.5, num_features=64)
        aug2 = GaussianFeatureAugmentationFederated(momentum=0.5, num_features=192)
        aug3 = GaussianFeatureAugmentationFederated(momentum=0.5, num_features=384)
        aug4 = GaussianFeatureAugmentationFederated(momentum=0.5, num_features=256)
        return aug1, aug2, aug3, aug4
    elif aug_method == 'random':
        aug = RandomFeatureAugmentation()
        return aug, aug, aug, aug


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self, block, layers, type, aug_method=None, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.type = type
        self.aug1, self.aug2, self.aug3, self.aug4, self.aug5, self.aug6 = gaussian_feature_augmentation(aug_method)

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion
        self.linear = nn.Linear(self._out_features, num_classes)

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.aug1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.aug2(x)
        f1 = self.layer1(x)
        #print('f1', f1.shape)   [10, 64, 8, 8]
        x = self.aug3(f1)
        f2 = self.layer2(x)
        #print('f2', f2.shape)  [10, 128, 4, 4]
        x = self.aug4(f2)
        f3 = self.layer3(x)
        #print('f3', f3.shape)  [10, 256, 2, 2]
        x = self.aug5(f3)
        f4 = self.layer4(x)
        #print('f4',f4.shape)   [10, 512, 1, 1]
        x = self.aug6(f4)
        v = self.global_avgpool(x)
        #print('global',v.shape)    [10, 512, 1, 1]
        v = v.view(v.size(0), -1)
        #print('v',v.shape) [10, 512]
        v = self.linear(v)
        #print('linear',v.shape)
        if self.type == 'base':
            return v
        else:
            return f1, f2, f3, f4, v


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""
"""
Standard residual networks
"""


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def resnet18(type, pretrained=True, aug_method=None):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], type=type, aug_method=aug_method)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

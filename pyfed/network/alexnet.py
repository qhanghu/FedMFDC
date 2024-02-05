import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from pyfed.utils.augmentation import (
    GaussianFeatureAugmentationClientBatch,
    GaussianFeatureAugmentationFederated,
    RandomFeatureAugmentation,
    GaussianFeatureAugmentationClientMemory,
)
from pyfed.utils.mixup import mixup_process, to_one_hot


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, return_feature = False, **kwargs):
        super(AlexNet, self).__init__()
        self.return_feature = return_feature

        self.layer1 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True))
            ])
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer2 = nn.Sequential(
            OrderedDict([
                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True))
            ])
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer3 = nn.Sequential(
            OrderedDict([
                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True))
            ])
        )
        self.layer4 = nn.Sequential(
            OrderedDict([
                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True))
            ])
        )
        self.layer5 = nn.Sequential(
            OrderedDict([
                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True))
            ])
        )
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        """
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 1024)),
                ('bn6', nn.BatchNorm1d(1024)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(1024, 1024)),
                ('bn7', nn.BatchNorm1d(1024)),
                ('relu7', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(1024, num_classes)),
            ])
        )
        """
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x1 = self.pool1(x)
        #print('x1', x.shape)    [32, 64, 31, 31]

        x = self.layer2(x1)
        x = self.pool2(x)
        #print('x2', x.shape)   [32, 192, 15, 15]

        x = self.layer3(x)
        #print('x3', x.shape)   [32, 384, 15, 15]

        x = self.layer4(x)
        #print('x4', x.shape)   [32, 256, 15, 15]

        x = self.layer5(x)
        x5 = self.pool5(x)
        #print('x5', x.shape)   [32, 256, 7, 7]

        f0 = self.avgpool(x5)
        #print('f0',f0.shape)   #[32, 256, 6, 6]
        f1 = torch.flatten(f0, 1)
        #print('f1',f1.shape)    #[32, 9216]
        #x = self.classifier(f)
        f2 = self.fc1(f1)
        f = self.relu6(self.bn6(f2))
        f3 = self.fc2(f)
        f = self.relu7(self.bn7(f3))
        x = self.fc3(f)

        if self.return_feature == False:
            return x
        else:
            return x5, x



def gaussian_feature_augmentation(aug_method=None):
    if aug_method == 'fedfa-r':
        aug = RandomFeatureAugmentation()
        return aug, aug, aug, aug, aug
    elif aug_method == 'fedfa-c':
        aug = GaussianFeatureAugmentationClientBatch()
        return aug, aug, aug, aug, aug
    elif aug_method == 'fedfa-m':
        aug1 = GaussianFeatureAugmentationClientMemory(num_features=64, memory_size=32)
        aug2 = GaussianFeatureAugmentationClientMemory(num_features=192, memory_size=32)
        aug3 = GaussianFeatureAugmentationClientMemory(num_features=384, memory_size=32)
        aug4 = GaussianFeatureAugmentationClientMemory(num_features=256, memory_size=32)
        aug5 = GaussianFeatureAugmentationClientMemory(num_features=256, memory_size=32)
        return aug1, aug2, aug3, aug4, aug5
    elif aug_method == 'fedfa':
        alpha = 0.99
        aug1 = GaussianFeatureAugmentationFederated(momentum1=alpha, momentum2=alpha, num_features=64)
        aug2 = GaussianFeatureAugmentationFederated(momentum1=alpha, momentum2=alpha, num_features=192)
        aug3 = GaussianFeatureAugmentationFederated(momentum1=alpha, momentum2=alpha, num_features=384)
        aug4 = GaussianFeatureAugmentationFederated(momentum1=alpha, momentum2=alpha, num_features=256)
        aug5 = GaussianFeatureAugmentationFederated(momentum1=alpha, momentum2=alpha, num_features=256)
        return aug1, aug2, aug3, aug4, aug5


class AlexNetFedFa(AlexNet):
    """
    used for DomainNet and Office-Caltech10
    """

    def __init__(self, num_classes=10, **kwargs):
        super(AlexNetFedFa, self).__init__(num_classes, **kwargs)

        self.aug1, self.aug2, self.aug3, self.aug4, self.aug5 = \
            gaussian_feature_augmentation(kwargs['aug_method'])

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.aug1(x)

        x = self.layer2(x)
        x = self.pool2(x)
        x = self.aug2(x)

        x = self.layer3(x)
        x = self.aug3(x)

        x = self.layer4(x)
        x = self.aug4(x)

        x = self.layer5(x)
        x = self.pool5(x)
        x = self.aug5(x)

        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        x = self.classifier(f)

        return x


class AlexNetManifoldMixup(AlexNet):
    def __init__(self, num_classes=10, **kwargs):
        super(AlexNetManifoldMixup, self).__init__(num_classes, **kwargs)

    def forward(self, x, target=None, mixup_hidden=False, mixup=False, mixup_alpha=None, num_classes=10):
        if mixup_hidden:
            layer_mix = random.randint(0, 2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None

        if mixup_alpha is not None:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = torch.autograd.Variable(lam)

        if target is not None:
            target_reweighted = to_one_hot(target, num_classes).cuda()

        x = self.layer1(x)
        x = self.pool1(x)

        if layer_mix == 0:
            x, target_reweighted = mixup_process(x, target_reweighted, lam=lam)

        x = self.layer2(x)
        x = self.pool2(x)

        if layer_mix == 1:
            x, target_reweighted = mixup_process(x, target_reweighted, lam=lam)

        x = self.layer3(x)

        if layer_mix == 2:
            x, target_reweighted = mixup_process(x, target_reweighted, lam=lam)

        x = self.layer4(x)

        x = self.layer5(x)

        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        x = self.classifier(f)

        if mixup or mixup_hidden:
            return x, target_reweighted

        return x

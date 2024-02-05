from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

from pyfed.utils.augmentation import (
    GaussianFeatureAugmentationClientBatch,
    GaussianFeatureAugmentationFederated,
    RandomFeatureAugmentation,
    GaussianFeatureAugmentationClientMemory
)
from torch.distributions.uniform import Uniform

def gaussian_feature_augmentation(aug_method=None):
    if aug_method == 'fedfa-r':
        aug = RandomFeatureAugmentation()
        return aug, aug, aug, aug, aug
    elif aug_method == 'fedfa-c':
        aug = GaussianFeatureAugmentationClientBatch()
        return aug, aug, aug, aug, aug
    elif aug_method == 'fedfa-m':
        aug1 = GaussianFeatureAugmentationClientMemory(num_features=32)
        aug2 = GaussianFeatureAugmentationClientMemory(num_features=64)
        aug3 = GaussianFeatureAugmentationClientMemory(num_features=128)
        aug4 = GaussianFeatureAugmentationClientMemory(num_features=256)
        aug5 = GaussianFeatureAugmentationClientMemory(num_features=512)
        return aug1, aug2, aug3, aug4, aug5
    elif aug_method == 'fedfa':
        aug1 = GaussianFeatureAugmentationFederated(momentum1=0.99, momentum2=0.99, num_features=32)
        aug2 = GaussianFeatureAugmentationFederated(momentum1=0.99, momentum2=0.99, num_features=64)
        aug3 = GaussianFeatureAugmentationFederated(momentum1=0.99, momentum2=0.99, num_features=128)
        aug4 = GaussianFeatureAugmentationFederated(momentum1=0.99, momentum2=0.99, num_features=256)
        aug5 = GaussianFeatureAugmentationFederated(momentum1=0.99, momentum2=0.99, num_features=512)
        return aug1, aug2, aug3, aug4, aug5


def _block(in_channels, features, name, affine=True, track_running_stats=True):
    bn_func = nn.BatchNorm2d

    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "_conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "_bn1", bn_func(num_features=features, affine=affine,
                                        track_running_stats=track_running_stats)),
                (name + "_relu1", nn.ReLU(inplace=True)),
                (
                    name + "_conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "_bn2", bn_func(num_features=features, affine=affine,
                                        track_running_stats=track_running_stats)),
                (name + "_relu2", nn.ReLU(inplace=True)),
            ]
        )
    )

def _block_UA_MT(in_channels, features, name, affine=True, dropout_p=0.0, track_running_stats=True):
    bn_func = nn.BatchNorm2d

    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "_conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "_bn1", bn_func(num_features=features, affine=affine,
                                        track_running_stats=track_running_stats)),
                (name + "_relu1", nn.ReLU(inplace=True)),
                (name + "_dropout", nn.Dropout(dropout_p)),
                (
                    name + "_conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "_bn2", bn_func(num_features=features, affine=affine,
                                        track_running_stats=track_running_stats)),
                (name + "_relu2", nn.ReLU(inplace=True)),
            ]
        )
    )


class UNetFedfa(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True, aug_method=None):
        super(UNetFedfa, self).__init__()

        self.aug1, self.aug2, self.aug3, self.aug4, self.aug5 = gaussian_feature_augmentation(aug_method)

        features = init_features
        self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
                                 track_running_stats=track_running_stats)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = _block(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)
        enc1_ = self.aug1(enc1_)

        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)
        enc2_ = self.aug2(enc2_)

        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)
        enc3_ = self.aug3(enc3_)

        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)
        enc4_ = self.aug4(enc4_)

        bottleneck = self.bottleneck(enc4_)
        bottleneck = self.aug5(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)

        return dec1


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True, return_feature = False,aug_method=None):
        super(UNet, self).__init__()
        self.return_feature = return_feature

        features = init_features
        self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
                                 track_running_stats=track_running_stats)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = _block(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)
        #print('enc1_ ',enc1_.shape) [16, 32, 192, 192]

        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)
        #print('enc2_ ', enc2_.shape)   [16, 64, 96, 96]

        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)
        #print('enc3_ ', enc3_.shape)   [16, 128, 48, 48]

        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)
        #print('enc4_ ', enc4_.shape)   [16, 256, 24, 24]

        bottleneck = self.bottleneck(enc4_)
        #print('bottleneck ', bottleneck.shape) [16, 512, 24, 24]

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        #print('dec4 ', dec4.shape) [16, 256, 48, 48]

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        #print('dec3 ', dec3.shape) [16, 128, 96, 96]

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        #print('dec2 ', dec2.shape) [16, 64, 192, 192]

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #print('dec1 ', dec1.shape) [16, 32, 384, 384]

        v = self.conv(dec1)

        if self.return_feature == False:
            return v
        else:
            return enc1_, v

class UNet_UA_MT(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True, return_feature = False,aug_method=None):
        super(UNet_UA_MT, self).__init__()
        self.return_feature = return_feature

        features = init_features
        self.encoder1 = _block_UA_MT(in_channels, features, name="enc1", affine=affine, dropout_p=0.01,
                               track_running_stats=track_running_stats)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block_UA_MT(features, features * 2, name="enc2", affine=affine, dropout_p=0.01,
                               track_running_stats=track_running_stats)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block_UA_MT(features * 2, features * 4, name="enc3", affine=affine, dropout_p=0.01,
                               track_running_stats=track_running_stats)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block_UA_MT(features * 4, features * 8, name="enc4", affine=affine, dropout_p=0.01,
                               track_running_stats=track_running_stats)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block_UA_MT(features * 8, features * 16, name="bottleneck", affine=affine, dropout_p=0.01,
                                 track_running_stats=track_running_stats)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block_UA_MT((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = _block_UA_MT((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block_UA_MT((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = _block_UA_MT(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)
        #print('enc1_ ',enc1_.shape) [16, 32, 192, 192]

        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)
        #print('enc2_ ', enc2_.shape)   [16, 64, 96, 96]

        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)
        #print('enc3_ ', enc3_.shape)   [16, 128, 48, 48]

        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)
        #print('enc4_ ', enc4_.shape)   [16, 256, 24, 24]

        bottleneck = self.bottleneck(enc4_)
        #print('bottleneck ', bottleneck.shape) [16, 512, 24, 24]

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        #print('dec4 ', dec4.shape) [16, 256, 48, 48]

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        #print('dec3 ', dec3.shape) [16, 128, 96, 96]

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        #print('dec2 ', dec2.shape) [16, 64, 192, 192]

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #print('dec1 ', dec1.shape) [16, 32, 384, 384]

        v = self.conv(dec1)

        if self.return_feature == False:
            return v
        else:
            return enc1_, v

class FCDiscriminator(nn.Module):

    def __init__(self, num_classes=2, ndf=64, n_channel=3):
        super(FCDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=4, padding=0)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=4, padding=0)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=4, padding=0)
        self.classifier = nn.Linear(ndf*8, 2)
        self.avgpool = nn.AvgPool2d((3, 3))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.sigmoid = nn.Sigmoid()

    def forward(self, map, feature):
        map_feature = self.conv0(map)
        image_feature = self.conv1(feature)
        x = torch.add(map_feature, image_feature)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x

class UNet_SASSNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32,
                 affine=True, track_running_stats=True, return_feature = False,aug_method=None):
        super(UNet_SASSNet, self).__init__()
        self.return_feature = return_feature

        features = init_features
        self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
                                 track_running_stats=track_running_stats)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = _block(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.conv_2 = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)
        #print('enc1_ ',enc1_.shape) [16, 32, 192, 192]

        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)
        #print('enc2_ ', enc2_.shape)   [16, 64, 96, 96]

        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)
        #print('enc3_ ', enc3_.shape)   [16, 128, 48, 48]

        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)
        #print('enc4_ ', enc4_.shape)   [16, 256, 24, 24]

        bottleneck = self.bottleneck(enc4_)
        #print('bottleneck ', bottleneck.shape) [16, 512, 24, 24]

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        #print('dec4 ', dec4.shape) [16, 256, 48, 48]

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        #print('dec3 ', dec3.shape) [16, 128, 96, 96]

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        #print('dec2 ', dec2.shape) [16, 64, 192, 192]

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #print('dec1 ', dec1.shape) [16, 32, 384, 384]

        v_seg = self.conv(dec1)
        out = self.conv_2(dec1)
        out_tach = self.tanh(out)

        if self.return_feature == False:
            return out_tach, v_seg
        else:
            return enc1_, v_seg

def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

class Decoder_(nn.Module):
    def __init__(self, init_features=32, track_running_stats=True, affine=True, out_channels=3):
        super(Decoder_, self).__init__()
        features = init_features
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = _block(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, feature):
        enc1 = feature[0]
        enc2 = feature[1]
        enc3 = feature[2]
        enc4 = feature[3]
        bottleneck = feature[4]

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        v = self.conv(dec1)

        return v


class UNet_CCT(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True, return_feature = False,aug_method=None):
        super(UNet_CCT, self).__init__()
        self.return_feature = return_feature

        features = init_features
        self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
                                 track_running_stats=track_running_stats)

        self.main_decoder = Decoder_()
        self.aux_decoder1 = Decoder_()
        self.aux_decoder2 = Decoder_()
        self.aux_decoder3 = Decoder_()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)
        #print('enc1_ ',enc1_.shape) [16, 32, 192, 192]

        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)
        #print('enc2_ ', enc2_.shape)   [16, 64, 96, 96]

        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)
        #print('enc3_ ', enc3_.shape)   [16, 128, 48, 48]

        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)
        #print('enc4_ ', enc4_.shape)   [16, 256, 24, 24]

        bottleneck = self.bottleneck(enc4_)
        #print('bottleneck ', bottleneck.shape) [16, 512, 24, 24]

        feature_ = [enc1, enc2, enc3, enc4, bottleneck]

        main_seg = self.main_decoder(feature_)
        aux1_feature = [FeatureNoise()(i) for i in feature_]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout(i) for i in feature_]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout(i) for i in feature_]
        aux_seg3 = self.aux_decoder3(aux3_feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3

class UNet_MC_Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True, return_feature = False,aug_method=None):
        super(UNet_MC_Net, self).__init__()
        self.return_feature = return_feature

        features = init_features
        self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
                                 track_running_stats=track_running_stats)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.up4_new = nn.Sequential(
            nn.Conv2d(features * 16, features * 8, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.up3_new = nn.Sequential(
            nn.Conv2d(features * 8, features * 4, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.up2_new = nn.Sequential(
            nn.Conv2d(features * 4, features * 2, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.up1_new = nn.Sequential(
            nn.Conv2d(features * 2, features , kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.decoder1 = _block(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)

        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)

        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)

        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)

        bottleneck = self.bottleneck(enc4_)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        v1 = self.conv(dec1)

        dec4 = self.up4_new(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.up3_new(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.up2_new(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1_new(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        v2 = self.conv(dec1)

        return v1, v2

if __name__ == '__main__':
    model = UNet(input_shape=(3, 384, 384))

    print(model.state_dict().keys())

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class FederatedDistributionUncertainty(nn.Module):
    def __init__(self, p=0.5, eps=1e-6, num_features=None, momentum=1,
                 track_stats=False, with_instancenorm=False):
        super(FederatedDistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0
        self.num_features = num_features
        self.momentum = momentum
        self.track_stats = track_stats
        self.with_instancenorm = with_instancenorm

        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)

        # What to track: buffers = savable params with no grads.
        self.register_buffer('running_var_mean', torch.ones(self.num_features))
        self.register_buffer('running_var_std', torch.ones(self.num_features))
        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_std', torch.ones(self.num_features))

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def var(self, x):
        t = x.var(dim=0, keepdim=False) + self.eps
        return t

    def momentum_updating_running_mean_and_std(self, mean, std):
        with torch.no_grad():
            self.running_mean = self.running_mean * self.momentum + mean.mean(dim=0, keepdim=False) * (1 - self.momentum)
            self.running_std = self.running_std * self.momentum + std.mean(dim=0, keepdim=False) * (1 - self.momentum)

    def momentum_updating_running_var(self, var_mean, var_std):
        with torch.no_grad():
            self.running_var_mean = self.running_var_mean * self.momentum + var_mean * (1 - self.momentum)
            self.running_var_std = self.running_var_std * self.momentum + var_std * (1 - self.momentum)

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        # Here `std` is actually `var`. I wrote this intentionally
        # for easier coding in comm.py
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps)

        self.momentum_updating_running_mean_and_std(mean, std)

        std = std.sqrt()
        var_mean, var_std = self.var(mean), self.var(std)

        self.momentum_updating_running_var(var_mean, var_std)

        running_var_mean = self.running_var_mean.sqrt().repeat(x.shape[0], 1)
        running_var_std = self.running_var_std.sqrt().repeat(x.shape[0], 1)

        beta = self._reparameterize(mean, running_var_mean).reshape(x.shape[0], x.shape[1], 1, 1)
        gamma = self._reparameterize(std, running_var_std).reshape(x.shape[0], x.shape[1], 1, 1)

        x = gamma * self.instance_norm(x) + beta

        return x


class FederatedDistributionUncertaintyMeanGrad(nn.Module):
    def __init__(self, p=0.5, eps=1e-6, num_features=None, momentum=1,
                 track_stats=False, with_instancenorm=False, lr=1e-3):
        super(FederatedDistributionUncertaintyMeanGrad, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0
        self.num_features = num_features
        self.momentum = momentum
        self.track_stats = track_stats
        self.with_instancenorm = with_instancenorm
        self.lr = lr

        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)

        # What to track: buffers = savable params with no grads.
        self.register_buffer('running_mean_grad', torch.ones(self.num_features))
        self.register_buffer('running_var_grad', torch.ones(self.num_features))

        self.register_buffer('running_var_mean', torch.ones(self.num_features))
        self.register_buffer('running_var_std', torch.ones(self.num_features))
        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_std', torch.ones(self.num_features))

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def var(self, x):
        t = x.var(dim=0, keepdim=False) + self.eps
        return t

    def momentum_updating_running_mean_and_std(self, mean, std):
        with torch.no_grad():
            self.running_mean = self.running_mean * self.momentum + mean.mean(dim=0, keepdim=False) * (1 - self.momentum)
            self.running_std = self.running_std * self.momentum + std.mean(dim=0, keepdim=False) * (1 - self.momentum)

    def momentum_updating_running_var(self, var_mean, var_std):
        with torch.no_grad():
            self.running_var_mean = self.running_var_mean * self.momentum + var_mean * (1 - self.momentum)
            self.running_var_std = self.running_var_std * self.momentum + var_std * (1 - self.momentum)

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        # Here `std` is actually `var`. I wrote this intentionally
        # for easier coding in comm.py
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps)

        self.momentum_updating_running_mean_and_std(mean, std)

        std = std.sqrt()
        var_mean, var_std = self.var(mean), self.var(std)

        self.momentum_updating_running_var(var_mean, var_std)

        running_var_mean = self.running_var_mean.sqrt().repeat(x.shape[0], 1)
        running_var_std = self.running_var_std.sqrt().repeat(x.shape[0], 1)

        # beta = self._reparameterize(mean, running_var_mean).reshape(x.shape[0], x.shape[1], 1, 1)
        beta = mean - self.lr * self.running_mean_grad + self.lr * self._reparameterize(mean, running_var_mean)
        beta = beta.reshape(x.shape[0], x.shape[1], 1, 1)

        gamma = self._reparameterize(std, running_var_std).reshape(x.shape[0], x.shape[1], 1, 1)
        # gamma = std - self.lr * self.running_var_grad + self.lr * self._reparameterize(std, running_var_std)
        gamma = gamma.reshape(x.shape[0], x.shape[1], 1, 1)

        x = gamma * self.instance_norm(x) + beta

        return x

# class GMVAE(nn.Module):
#     def __init__(self, p=0.5, eps=1e-6, num_features=None, momentum=1,
#                  track_stats=False, with_instancenorm=False):
#         super(GMVAE, self).__init__()
#         self.eps = eps
#         self.p = p
#         self.factor = 1.0
#         self.num_features = num_features
#         self.momentum = momentum
#         self.track_stats = track_stats
#         self.with_instancenorm = with_instancenorm
#
#         self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
#
#         # What to track: buffers = savable params with no grads.
#         self.register_buffer('running_mean', torch.zeros(self.num_features))
#         self.register_buffer('running_std', torch.ones(self.num_features))
#
#         self.register_buffer('running_mean_mixture', torch.zeros(6, self.num_features))
#         self.register_buffer('running_std_mixture', torch.ones(6, self.num_features))
#
#     def _reparameterize(self, mu, std):
#         epsilon = torch.randn_like(std) * self.factor
#         return mu + epsilon * std
#
#     def var(self, x):
#         t = x.var(dim=0, keepdim=False) + self.eps
#         return t
#
#     def mean(self, x):
#         t = x.mean(dim=0, keepdim=False)
#         return t
#
#     def momentum_updating_running_var(self, mean, std):
#         with torch.no_grad():
#             self.running_mean = self.running_mean * self.momentum + mu_mean * (1 - self.momentum)
#             self.running_var_mean = self.running_var_mean * self.momentum + var_mean * (1 - self.momentum)
#             self.running_mu_std = self.running_mu_std * self.momentum + mu_std * (1 - self.momentum)
#             self.running_var_std = self.running_var_std * self.momentum + var_std * (1 - self.momentum)
#
#     def forward(self, x):
#         # if (not self.training) or (np.random.random()) > self.p:
#         #     if self.with_instancenorm:
#         #         x = self.instance_norm(x)
#         #     return x,
#
#         mean = x.mean(dim=[2, 3], keepdim=False)
#         # Here `std` is actually `var`. I wrote this intentionally
#         # for easier coding in comm.py
#         std = (x.var(dim=[2, 3], keepdim=False) + self.eps)
#
#         # self.momentum_updating_running_mean_and_std(mean, std)
#
#         std = std.sqrt()
#         var_mean, var_std = self.var(mean), self.var(std)
#         mu_mean, mu_std = self.mean(mean), self.mean(std)
#
#         self.momentum_updating_running_var(mu_mean, var_mean, mu_std, var_std)
#
#         var_mean = var_mean.sqrt().repeat(x.shape[0], 1)
#         var_std = var_std.sqrt().repeat(x.shape[0], 1)
#
#         mu_sample = self._reparameterize(mean, var_mean).reshape(x.shape[0], x.shape[1], 1, 1)
#         var_sample = self._reparameterize(std, var_std).reshape(x.shape[0], x.shape[1], 1, 1)
#
#         # x = gamma * self.instance_norm(x) + beta
#
#         kl_loss_mu = log_normal(mu_sample, mu_mean, var_mean) - \
#                      log_normal_mixture(mu_sample, self.running_mu_mean_mixture, self.running_var_mean_mixture)
#
#         kl_loss_std = log_normal(var_sample, mu_std, var_std) - \
#                      log_normal_mixture(var_sample, self.running_mu_std_mixture, self.running_var_std_mixture)
#
#         return kl_loss_mu, kl_loss_std
#
# def log_normal(x, m, v):
#     log_prob = (-0.5 * (torch.log(v) + (x - m).pow(2) / v)).sum(-1)
#     return log_prob
#
# def log_normal_mixture(z, m, v, mask=None):
#     m = m.unsqueeze(0).expand(z.shape[0], -1, -1)
#     v = v.unsqueeze(0).expand(z.shape[0], -1, -1)
#     batch, mix, dim = m.size()
#     z = z.view(batch, 1, dim).expand(batch, mix, dim)
#     indiv_log_prob = log_normal(z, m, v) + torch.ones_like(mask) * (-1e6) * (1. - mask)
#     log_prob = log_mean_exp(indiv_log_prob, mask)
#     return log_prob
#
# def log_mean_exp(x, mask):
#     return log_sum_exp(x, mask) - torch.log(mask.sum(1))
#
# def log_sum_exp(x, mask):
#     max_x = torch.max(x, 1)[0]
#     new_x = x - max_x.unsqueeze(1).expand_as(x)
#     return max_x + (new_x.exp().sum(1)).log()

class FederatedDistributionUncertaintyforExp(nn.Module):
    def __init__(self, p=0.5, eps=1e-6, num_features=None, momentum=1,
                 track_stats=False, with_instancenorm=False):
        super(FederatedDistributionUncertaintyforExp, self).__init__()

class DistributionUncertainty(nn.Module):
    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.zeros_like(std)
        epsilon += torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

class DTTAnorm(nn.Module):
    def __init__(self):
        super(DTTAnorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x):
        x_ = self.conv1(x)
        scale = (torch.randn([1, 16, 1, 1]) * 0.05 + 0.2).to(x_.device)
        x_ = torch.exp(-(x_ ** 2) / (scale ** 2))
        x_ = self.conv2(x_)
        x_ = torch.exp(-(x_ ** 2) / (scale ** 2))
        x_ = self.conv3(x_)
        return x_ + x

# borrow from https://github.com/iantsen/hecktor/blob/5ebc6774d178139b52abebbe926b4cafabe3e29d/src/layers.py#L19
class FastSmoothSENorm(nn.Module):
    class SEWeights(nn.Module):
        def __init__(self, in_channels, reduction=2):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            b, c, d, h, w = x.size()
            out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)  # output_shape: in_channels x (1, 1, 1)
            out = F.relu(self.conv1(out))
            out = self.conv2(out)
            return out

    def __init__(self, in_channels, reduction=2):
        super(FastSmoothSENorm, self).__init__()
        # self.norm = nn.InstanceNorm3d(in_channels, affine=False)
        self.norm = nn.BatchNorm3d(in_channels, affine=True, track_running_stats=True)
        self.gamma = self.SEWeights(in_channels, reduction)
        self.beta = self.SEWeights(in_channels, reduction)

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma(x))
        beta = torch.tanh(self.beta(x))
        x = self.norm(x)
        return gamma * x + beta

class FastSmoothSeNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(FastSmoothSeNormConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = FastSmoothSENorm(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.norm(x)
        return x

class RESseNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(RESseNormConv3d, self).__init__()
        self.conv1 = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, **kwargs)

        if in_channels != out_channels:
            self.res_conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = self.res_conv(x) if self.res_conv else x
        x = self.conv1(x)
        x += residual
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        # self.norm = nn.BatchNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x

if __name__ == '__main__':
    exit()

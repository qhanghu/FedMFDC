import torch
import torch.nn as nn
import numpy as np


class GaussianFeatureAugmentationFederated(nn.Module):
    def __init__(self, p=0.5, eps=1e-6, momentum1=0.99, momentum2=0.99, lr=1e-4, num_features=None):
        super(GaussianFeatureAugmentationFederated, self).__init__()
        self.p = p
        self.lr = lr
        self.eps = eps
        self.factor = 1
        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.num_features = num_features

        self.register_buffer('running_var_mean_bmic', torch.ones(self.num_features))
        self.register_buffer('running_var_std_bmic', torch.ones(self.num_features))
        self.register_buffer('running_mean_bmic', torch.zeros(self.num_features))
        self.register_buffer('running_std_bmic', torch.ones(self.num_features))

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def var(self, x):
        t = x.var(dim=0, keepdim=False) + self.eps
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps)

        self.momentum_updating_running_mean_and_std(mean, std)

        std = std.sqrt()
        var_mu = self.var(mean)
        var_std = self.var(std)

        # %%%%%%%%%%%%%
        # running_var_mean_bmic = x.shape[1] * self.running_var_mean_bmic
        # alpha_mu = running_var_mean_bmic / (sum(self.running_var_mean_bmic) + self.eps)
        #
        # running_var_std_bmic = x.shape[1] * self.running_var_std_bmic
        # alpha_std = running_var_std_bmic / (sum(self.running_var_std_bmic) + self.eps)
        #
        # var_mu = (alpha_mu + 1) * var_mu
        # var_std = (alpha_std + 1) * var_std

        # %%%%%%%%%
        running_var_mean_bmic = 1 / (1 + 1 / (self.running_var_mean_bmic + self.eps))
        gamma_mu = x.shape[1] * running_var_mean_bmic / sum(running_var_mean_bmic)

        running_var_std_bmic = 1 / (1 + 1 / (self.running_var_std_bmic + self.eps))
        gamma_std = x.shape[1] * running_var_std_bmic / sum(running_var_std_bmic)

        var_mu = (gamma_mu + 1) * var_mu
        var_std = (gamma_std + 1) * var_std

        var_mu = var_mu.sqrt().repeat(x.shape[0], 1)
        var_std = var_std.sqrt().repeat(x.shape[0], 1)

        beta = self._reparameterize(mean, var_mu)
        gamma = self._reparameterize(std, var_std)

        # %%%%%%%%%%%%%
        # self.momentum_updating_running_var(var_mu, var_std)
        #
        # running_var_mean = self.running_var_mean_bmic.sqrt().repeat(x.shape[0], 1)
        # running_var_std = self.running_var_std_bmic.sqrt().repeat(x.shape[0], 1)

        # beta = self._reparameterize(mean, running_var_mean)
        # gamma = self._reparameterize(std, running_var_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

    def momentum_updating_running_mean_and_std(self, mean, std):
        with torch.no_grad():
            self.running_mean_bmic = self.running_mean_bmic * self.momentum1 + \
                                     mean.mean(dim=0, keepdim=False) * (1 - self.momentum1)
            self.running_std_bmic = self.running_std_bmic * self.momentum1 + \
                                    std.mean(dim=0, keepdim=False) * (1 - self.momentum1)

    def momentum_updating_running_var(self, var_mean, var_std):
        with torch.no_grad():
            self.running_var_mean_bmic = self.running_var_mean_bmic * self.momentum2 + var_mean * (1 - self.momentum2)
            self.running_var_std_bmic = self.running_var_std_bmic * self.momentum2 + var_std * (1 - self.momentum2)


class GaussianFeatureAugmentationClientMomentum(nn.Module):
    def __init__(self, p=0.5, eps=1e-6, momentum=0.5, num_features=None):
        super(GaussianFeatureAugmentationClientMomentum, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0
        self.momentum = momentum
        self.num_features = num_features

        self.register_buffer('running_var_mean_bmic', torch.ones(self.num_features))
        self.register_buffer('running_var_std_bmic', torch.ones(self.num_features))

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def var(self, x):
        t = x.var(dim=0, keepdim=False) + self.eps
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        var_mu = self.var(mean)
        var_std = self.var(std)

        self.momentum_updating_running_var(var_mu, var_std)

        # running_var_mean_bmic = self.running_var_mean_bmic * self.momentum + var_mu * (1 - self.momentum)
        # running_var_std_bmic = self.running_var_std_bmic * self.momentum + var_std * (1 - self.momentum)

        if (np.random.random()) > self.p:
            running_sqrtvar_mean_bmic = self.running_var_mean_bmic.sqrt().repeat(x.shape[0], 1)
            running_sqrtvar_std_bmic = self.running_var_std_bmic.sqrt().repeat(x.shape[0], 1)

            beta = self._reparameterize(mean, running_sqrtvar_mean_bmic)
            gamma = self._reparameterize(std, running_sqrtvar_std_bmic)
        else:
            beta = self._reparameterize(mean, var_mu.sqrt().repeat(x.shape[0], 1))
            gamma = self._reparameterize(std, var_std.sqrt().repeat(x.shape[0], 1))

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

    def momentum_updating_running_var(self, var_mean, var_std):
        with torch.no_grad():
            self.running_var_mean_bmic = self.running_var_mean_bmic * self.momentum + var_mean * (1 - self.momentum)
            self.running_var_std_bmic = self.running_var_std_bmic * self.momentum + var_std * (1 - self.momentum)


class GaussianFeatureAugmentationClientMemory(nn.Module):
    def __init__(self, p=0.5, eps=1e-6, memory_size=16, num_features=None):
        super(GaussianFeatureAugmentationClientMemory, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 0.5
        self.memory_size = memory_size
        self.num_features = num_features

        self.register_buffer('mean_queue_bmic', torch.zeros(self.memory_size, self.num_features))
        self.register_buffer('std_queue_bmic', torch.zeros(self.memory_size, self.num_features))
        self.register_buffer('queue_ptr_bmic', torch.zeros(1, dtype=torch.long))

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x, batch_size=None):
        if batch_size is None:
            batch_size = x.shape[0]
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(batch_size, 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        if not torch.all(self.mean_queue_bmic):
            mean_ = torch.cat((mean, self.mean_queue_bmic), dim=0)
            std_ = torch.cat((std, self.std_queue_bmic), dim=0)

            sqrtvar_mu = self.sqrtvar(mean_, mean.shape[0])
            sqrtvar_std = self.sqrtvar(std_, std.shape[0])
        else:
            sqrtvar_mu = self.sqrtvar(mean)
            sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        # update memory
        with torch.no_grad():
            self.mean_queue_bmic[self.queue_ptr_bmic:self.queue_ptr_bmic + x.shape[0], :] = mean
            self.std_queue_bmic[self.queue_ptr_bmic:self.queue_ptr_bmic + x.shape[0], :] = std
            self.queue_ptr_bmic[0] = (self.queue_ptr_bmic[0] + mean.shape[0]) % self.memory_size

        return x


class GaussianFeatureAugmentationClientBatch(nn.Module):
    def __init__(self, p=0.5, eps=1e-6):
        super(GaussianFeatureAugmentationClientBatch, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 0.5

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
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


class RandomFeatureAugmentation(nn.Module):
    def __init__(self, p=0.5, eps=1e-6):
        super(RandomFeatureAugmentation, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 0.3

    def _reparameterize(self, mu):
        epsilon = torch.randn_like(mu) * self.factor
        return mu + epsilon

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        beta = self._reparameterize(mean)
        gamma = self._reparameterize(std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x
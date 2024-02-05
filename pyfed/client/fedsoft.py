import copy

import torch

from pyfed.manager.helper.build_dataset import build_dataset
from pyfed.manager.helper.build_loss import build_loss, build_metric
from pyfed.manager.helper.build_optimizer import build_optimizer
from pyfed.utils.kl import surprise_soft_bins
import os
import torch.nn.functional as F
from pyfed.utils import ramps

def get_current_consistency_weight(epoch, max):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.6 * ramps.sigmoid_rampup(epoch, max)

def data_normal_2d(orign_data, dim="col"):
    if dim == "col":
        dim = 1
    else:
        dim = 0
    d_min = torch.min(orign_data, dim=dim)[0]
    d_max = torch.max(orign_data, dim=dim)[0]
    dst = d_max - d_min + 1e-6
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data, d_min).true_divide(dst)
    return norm_data


def get_counts(self, x, n_features, num_bins, tau):
    # x = torch.nn.AvgPool2d(32,32)(x)
    """ See Neural Decision Forests paper for details on the soft binning procedure. """
    # Conv: [B, C, H, W] --> [C, BxHxW], FC: [B, C] --> [C, B] .to(self.device)
    x = x.transpose(0, 1).flatten(1)
    x = data_normal_2d(x)
    # x = self._norm_inputs(x)
    x = x.reshape((n_features, -1, 1))  # reshape for make-shift batch outer prod via matmul

    cut_points = torch.linspace(0, num_bins - 2, num_bins - 1).to(self.device) / (num_bins - 2)
    D = cut_points.shape[0]
    cut_points, _ = torch.sort(cut_points)
    W = []
    b = []
    for f in range(n_features):
        w_ = torch.reshape(torch.linspace(1, D + 1, D + 1), [1, -1]).to(self.device)
        b_ = torch.cumsum(torch.cat([torch.zeros([1]).to(self.device), -cut_points], 0), 0).to(self.device)
        W.append(w_)
        b.append(b_)
    W = torch.vstack(W).reshape((n_features, 1, D + 1)).to(self.device)  # reshape for matmul later
    b = torch.vstack(b).reshape((n_features, 1, D + 1)).to(self.device)  # reshape for matmul later
    # Calculate "logits" per sample via batch outer-product.
    # x:[n_features, n_samples, 1] x W:[n_features, 1, n_bins] = [n_features, n_samples, n_bins]
    z = torch.matmul(x, W) + b
    if self.config.bin_type == 1:
        # Calculate soft allocations per sample ("soft" --> sum to 1)
        sft_cs = torch.nn.Softmax(dim=2)(z / tau)  # [n_features, n_samples, n_bins]
        # Sum over samples to get total soft counts ("soft" --> real number)
        total_sft_cs = sft_cs.sum(1)
    elif self.config.bin_type == 2:
        total_sft_cs = z.sum(1)

    return total_sft_cs / (x.size(1))


class SoftClient(object):
    def __init__(self, config, site, server_model):
        self.site = site
        self.config = config
        self.model = copy.deepcopy(server_model)
        self.device = torch.device('cuda:{}'.format(config.TRAIN_GPU) if torch.cuda.is_available() else 'cpu')
        self.curr_iter = 0
        self.round = 0

        self.dis_1 = torch.zeros(self.config.feature, self.config.BIN)
        self.total_dis_1 = torch.zeros(self.config.feature, self.config.BIN)
        self.other_dis_1 = torch.zeros(self.config.feature, self.config.BIN)

        self.iter_round = 0
        self._setup()

    @property
    def name(self):
        return self.site

    def _setup(self):
        self.loss_fn = build_loss(self.config)
        self.metric_fn = build_metric(self.config)
        self.optimizer = build_optimizer(self.config, self.model.parameters())
        self.train_labeled_loader, self.train_unlabeled_loader, self.valid_loader, self.test_loader = build_dataset(
            self.config, self.site)
        if self.config.TEST == True:
            checkpoint = torch.load(os.path.join(self.config.DIR_LOAD, 'model_best.pth'))
            self.model.load_state_dict(checkpoint['server'])
            print('model load')

    def get_target(self, other1, round):
        self.other_dis_1 = other1
        self.iter_round = round

    def get_dis_1(self):
        return self.total_dis_1

    def train(self, server_model=None):
        self.model.to(self.device)
        self.model.train()

        self.total_dis_1 = torch.zeros(self.config.feature, self.config.BIN)
        self.total_dis_1 = self.total_dis_1.to(self.device)

        loss_all = 0

        outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        labels = torch.tensor([], dtype=torch.float32, device=self.device)
        for _ in range(len(self.train_labeled_loader)):
            image, label = next(iter(self.train_labeled_loader))
            image_weak, image_strong = next(iter(self.train_unlabeled_loader))
            image, label = image.to(self.device), label.to(self.device)
            image_weak, image_strong = image_weak.to(self.device), image_strong.to(self.device)
            self.optimizer.zero_grad()

            f1, output = self.model(image)
            fw, logits_u_w = self.model(image_weak)

            self.dis_1 = get_counts(self, f1, n_features=self.config.feature, num_bins=self.config.BIN, tau=0.01)
            self.dis_1 = self.dis_1.to(self.device)
            self.total_dis_1 += self.dis_1

            self.dis_w = get_counts(self, fw, n_features=self.config.feature, num_bins=self.config.BIN, tau=0.01)
            self.dis_w = self.dis_w.to(self.device)
            self.total_dis_1 += self.dis_w
            self.total_dis_1 = self.total_dis_1.detach()

            loss_label = self.loss_fn(output, label)
            consistency_weight = get_current_consistency_weight(self.curr_iter, 200 * len(self.train_labeled_loader))

            if self.iter_round > self.config.begin:
                local = torch.nn.Softmax(dim=1)(self.dis_w / 0.01)
                last = torch.nn.Softmax(dim=1)(self.other_dis_1.detach() / 0.01)
                score_1 = surprise_soft_bins(local, last)
                loss_dis = score_1.mean()
            if self.iter_round <= self.config.begin:
                loss = loss_label
            else:
                loss = loss_label + consistency_weight * loss_dis
            loss.backward()

            loss_all += loss.item()

            outputs = torch.cat([outputs, output.detach()], dim=0)
            labels = torch.cat([labels, label.detach()], dim=0)

            self.optimizer.step()
            self.curr_iter += 1

        acc = self.metric_fn(outputs, labels)
        loss_1 = loss_all / len(self.train_labeled_loader)
        self.total_dis_1 /= len(self.train_labeled_loader)
        self.round += 1
        self.model.to('cpu')
        return loss_1, acc

    @torch.no_grad()
    def val(self, model=None):
        # personalized validation
        if model is None:
            model = self.model

        model.to(self.device)
        model.eval()
        loss_all = 0
        test_acc = 0.

        outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        labels = torch.tensor([], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for step, (image, label) in enumerate(self.valid_loader):
                image, label = image.to(self.device), label.to(self.device)
                f, output = model(image)
                # f0, f1, f2, f3, output = model(image)
                loss = self.loss_fn(output, label)
                loss_all += loss.item()

                outputs = torch.cat([outputs, output.detach()], dim=0)
                labels = torch.cat([labels, label.detach()], dim=0)

                # test_acc += DiceLoss().dice_coef(output, label).item()

        loss = loss_all / len(self.valid_loader)
        acc = self.metric_fn(outputs, labels)
        # acc = test_acc / len(self.valid_loader)
        model.to('cpu')
        return loss, acc

    @torch.no_grad()
    def test(self, model=None):
        # personalized testing
        if model is None:
            model = self.model

        model.to(self.device)
        model.eval()
        loss_all = 0

        outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        labels = torch.tensor([], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for step, (image, label) in enumerate(self.test_loader):
                image, label = image.to(self.device), label.to(self.device)
                f, output = model(image)
                # f0, f1, f2, f3, output = model(image)
                loss = self.loss_fn(output, label)
                loss_all += loss.item()

                outputs = torch.cat([outputs, output.detach()], dim=0)
                labels = torch.cat([labels, label.detach()], dim=0)
                # test_acc += DiceLoss().dice_coef(output, label).item()

        loss = loss_all / len(self.test_loader)
        acc = self.metric_fn(outputs, labels)
        # acc = test_acc / len(self.test_loader)
        model.to('cpu')
        return loss, acc

    def server_to_client(self, server_model):
        for key in server_model.state_dict().keys():
            self.model.state_dict()[key].data.copy_(server_model.state_dict()[key])

    def client_to_server(self):
        return {'model': self.model,
                'optimizer': self.optimizer,
                'data_len': len(self.train_labeled_loader)}

    def save(self):
        pass

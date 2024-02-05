import copy

import torch

from pyfed.manager.helper.build_dataset import build_dataset
from pyfed.manager.helper.build_loss import build_loss, build_metric
from pyfed.manager.helper.build_optimizer import build_optimizer


class BaseClient(object):
    def __init__(self, config, site, server_model):
        self.site = site
        self.config = config
        self.model = copy.deepcopy(server_model)
        self.device = torch.device('cuda:{}'.format(config.TRAIN_GPU) if torch.cuda.is_available() else 'cpu')
        self.curr_iter = 0
        self.round = 0
        self._setup()

    @property
    def name(self):
        return self.site

    def _setup(self):
        self.loss_fn = build_loss(self.config)
        self.metric_fn = build_metric(self.config)
        self.optimizer = build_optimizer(self.config, self.model.parameters())
        self.train_labeled_loader, self.train_unlabeled_loader, self.valid_loader, self.test_loader = build_dataset(self.config, self.site)

    def train(self, server_model=None):
        self.model.to(self.device)
        self.model.train()
        loss_all = 0

        outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        labels = torch.tensor([], dtype=torch.float32, device=self.device)

        for _ in range(len(self.train_labeled_loader)):
            image, label = next(iter(self.train_labeled_loader))
            self.optimizer.zero_grad()

            image, label = image.to(self.device), label.to(self.device)
            output = self.model(image)
            loss = self.loss_fn(output, label)
            loss_all += loss.item()

            outputs = torch.cat([outputs, output.detach()], dim=0)
            labels = torch.cat([labels, label.detach()], dim=0)

            loss.backward()
            self.optimizer.step()

            self.curr_iter += 1

        acc = self.metric_fn(outputs, labels)
        loss = loss_all / len(self.train_labeled_loader)
        self.round += 1

        self.model.to('cpu')
        return loss, acc

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
                output = model(image)
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
                output = model(image)
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

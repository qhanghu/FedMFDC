import os
import shutil

import numpy as np
import torch
import wandb

from pyfed.manager.comm import Comm
from pyfed.manager.helper.build_model import build_model
from pyfed.client import (
    ClientCls,
    ClientSoft
)
from pyfed.utils.log import print_log
from pyfed.dataset.dataset_cls import _define_data_loader
from pyfed.manager.helper.build_dataset import build_cls_dataset
import matplotlib.pyplot as plt
import pathlib

class Manager(object):
    def __init__(self, config):
        self.config = config
        self.best_acc = 0
        self.best_epoch = 0

        self._setup()

    def _setup(self):
        """setup three core components: server, a bunch of cliets, and a communicator"""
        self.server_model = build_model(self.config)
        #self._build_data_partitioner()
        self._build_clients()

        self.comm = Comm(self.server_model, self.config.COMM_TYPE)

        total_model_params = sum(p.numel() for p in self.server_model.parameters())
        total_model_params_learnable = sum(p.numel() for p in self.server_model.parameters() if p.requires_grad)
        print_log('# params: {} {}(learnable)'.format(total_model_params, total_model_params_learnable))

    def _build_clients(self):
        if self.config.CLIENT == 'base':
            client_class = ClientCls
        elif self.config.CLIENT == 'soft':
            client_class = ClientSoft

        print_log('Client type: {}'.format(self.config.CLIENT))
        self.clients = [client_class(self.config, site, self.server_model)
                        for site in range(self.config.NUM_SITES)]
        self.client_weights = [1. / len(self.clients) for _ in range(len(self.clients))]

    def _build_data_partitioner(self):
        self.dataset = build_cls_dataset(self.config)
        _, self.data_partitioner = _define_data_loader(
            self.config,
            dataset=self.dataset["train"],
            localdata_id=0,
            is_train=True,
            data_partitioner=None,
        )

    def select_clients(self, round, ratio=0.2):
        num_clients = int(ratio * len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices.tolist()

    def train(self):
        metrics = {}
        best_val_acc, best_val_round = 0, 0
        self.device = torch.device('cuda:{}'.format(self.config.TRAIN_GPU) if torch.cuda.is_available() else 'cpu')
        if self.config.CLIENT == 'soft':
            other_total_1 = torch.zeros(64, self.config.BIN)
            other_total_1 = other_total_1.to(self.device)
            other_total_2 = torch.zeros(128, self.config.BIN)
            other_total_2 = other_total_2.to(self.device)
            other_total_3 = torch.zeros(256, self.config.BIN)
            other_total_3 = other_total_3.to(self.device)
            other_total_4 = torch.zeros(512, self.config.BIN)
            other_total_4 = other_total_4.to(self.device)

        for iter_round in range(0, self.config.TRAIN_ROUNDS):
            active_clients_indices = self.select_clients(iter_round, ratio=self.config.ACTIVE_RATIO)
            active_clients_list = []
            for ci, client_index in enumerate(active_clients_indices):
                active_clients_list.append(client_index)
            print('active_clients_list: ', active_clients_list)

            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Round: {}/{}, Epoch: {}/{} ============".format(
                    iter_round, self.config.TRAIN_ROUNDS, iter_local, self.config.TRAIN_EPOCH_PER_ROUND))

                # for ci, client in enumerate(self.clients):
                for ci, client_index in enumerate(active_clients_indices):
                    client = self.clients[client_index]
                    if self.config.CLIENT == 'base':
                        train_loss, top1, top5 = client.train()
                    elif self.config.CLIENT == 'soft':
                        if iter_round == 0:
                            train_loss, top1, top5 = client.train()
                        else:
                            client.get_info(other_total_1, other_total_2, other_total_3, other_total_4, iter_round, iter_local)
                            train_loss, top1, top5 = client.train()
                    print_log('site-{:<10s}| train loss: {:.4f} | train top1: {:.4f} | train top5: {:.4f}'.format(
                        client.name, train_loss, top1, top5))

                    metrics['train_loss_' + client.name] = train_loss
                    metrics['train_top1_' + client.name] = top1
                    metrics['train_top5_' + client.name] = top5

                if self.config.CLIENT == 'soft':
                    other_total_4 = torch.zeros(512, self.config.BIN)
                    other_total_4 = other_total_4.to(self.device)
                    self.config.DIR_IMAGE = pathlib.Path(self.config.DIR_IMAGE)
                    plt.figure(figsize=(17.5, 10))
                    for ci, client_index in enumerate(active_clients_indices):
                        client = self.clients[client_index]
                        other_total_4 += client.get_dis_4()
                        dis_4 = client.get_dis_4()
                        dis_4 = dis_4.cpu()
                        dis_4 = torch.nn.Softmax(dim=1)(dis_4 / 0.01)
                        x4 = np.linspace(0, self.config.BIN - 1, self.config.BIN)
                        y4 = dis_4.sum(0)
                        plt.plot(x4, y4, label='feature_4_site_{}'.format(client_index))
                    total_dis = other_total_4
                    total_dis = total_dis.cpu()
                    total_dis = torch.nn.Softmax(dim=1)(total_dis / 0.01)
                    x = np.linspace(0, self.config.BIN - 1, self.config.BIN)
                    y = total_dis.sum(0)
                    plt.plot(x, y, label='feature_4_site_total')
                    plt.grid()
                    plt.legend()
                    plt.savefig(
                        self.config.DIR_IMAGE / 'feature_4_round_{}_local_{}.png'.format(iter_round, iter_local),
                        bbox_inches='tight')
                    plt.close()

            # client to server
            client_models = [self.clients[client_index].client_to_server()['model'] for client_index in active_clients_list]
            #client_models = [client.client_to_server()['model'] for client in self.clients]
            self.client_weights = [1. / len(active_clients_list) for _ in range(len(active_clients_list))]
            client_weights = self.client_weights

            self.server_model = self.comm(client_models, client_weights, self.server_model)

            # run global validation and testing
            print_log('============== {} =============='.format('Global Validation'))
            val_loss, val_top1, val_top5 = self.clients[0].val(self.server_model)
            test_loss, test_top1, test_top5 = self.clients[0].test(self.server_model)

            metrics['g_val_loss'] = val_loss
            metrics['g_val_top1'] = val_top1
            metrics['g_val_top5'] = val_top5

            metrics['g_test_top1'] = test_top1
            metrics['g_test_top5'] = test_top5

            print_log('val loss: {:.4f} | val top1: {:.4f} | val top5: {:.4f} | '
                      'test top1: {:.4f} | test top5: {:.4f}'.format(val_loss, val_top1, val_top5, test_top1,
                                                                     test_top5))

            if val_top1 > best_val_acc:
                best_val_acc = val_top1
                best_val_round = iter_round

            print_log('curr val acc {:.4f} | best val round: {} | best val acc: {:.4f}'.format(
                val_top1, best_val_round, best_val_acc))

            # server to client
            for ci, client in enumerate(self.clients):
                client.server_to_client(self.server_model)

            self.save(iter_round, val_top1)

            wandb.log(metrics)

    def save(self, iter_round, val_acc):
        better = False
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = iter_round
            better = True

        # save server status
        save_dicts = {'server': self.server_model.state_dict(),
                      'best_epoch': self.best_epoch,
                      'best_acc': self.best_acc,
                      'round': iter_round}

        torch.save(save_dicts, os.path.join(self.config.DIR_CKPT, 'model_latest.pth'))
        if better:
            shutil.copyfile(os.path.join(self.config.DIR_CKPT, 'model_latest.pth'),
                            os.path.join(self.config.DIR_CKPT, 'model_best.pth'))
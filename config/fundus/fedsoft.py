from .base import BaseConfig
import os

class Config(BaseConfig):
    def __init__(self, server='bit', exp_name='fedavg'):
        super(Config, self).__init__(server, exp_name)
        self.CLIENT = 'SoftClient'
        self.COMM_TYPE = 'FedAvg'
        self.BIN = 10
        self.PROPORTION_SOFT = 0.6
        self.bin_type = 2
        self.begin = 0

        self.feature = 32

        self.NETWORK_PARAMS = {'out_channels':3,
                               'return_feature': True}
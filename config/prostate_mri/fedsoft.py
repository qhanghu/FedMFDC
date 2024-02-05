from .base import BaseConfig
import os

class Config(BaseConfig):
    def __init__(self, server='bit', exp_name='fedavg'):
        super(Config, self).__init__(server, exp_name)
        self.CLIENT = 'SoftClient'
        self.COMM_TYPE = 'FedAvg'
        self.BIN = 12
        self.PROPORTION_SOFT = 0.6
        self.bin_type = 2   # 1:initial 2:new
        self.begin = 0

        self.feature = 32

        self.NETWORK_PARAMS = {'return_feature': True}
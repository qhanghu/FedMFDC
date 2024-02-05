from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, server='bit', exp_name='fedavg'):
        super(Config, self).__init__(server, exp_name)

        self.CLIENT = 'BaseClient'
        self.COMM_TYPE = 'FedAvg'

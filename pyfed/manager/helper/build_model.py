from pyfed.network.unet import UNet, UNetFedfa, UNet_UA_MT, UNet_SASSNet, UNet_CCT, UNet_MC_Net
from pyfed.network.alexnet import AlexNet, AlexNetManifoldMixup, AlexNetFedFa
from pyfed.network.resnet import resnet18
from pyfed.network.vgg import vgg
import torchvision.models as models
from pyfed.client import (
    BaseClient,
    UAMTClient,
    SASSNetClient,
    DANClient,
    EMClient,
    CCTClient,
    MCNETClient,
    UdaClient,
    FixMatchClient,
    SoftClient,
    FedProxClient,
    FedHarmoClient,
    FedBNClient,
    FedSAMClient,
    FedDynClient,
    MixUpClient,
    ManifoldMixUpClient
)


def build_model(config):
    if config.NETWORK == 'unet':
        model = UNet(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'unet_UA_MT':
        model = UNet_UA_MT(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'unet_SASSNet':
        model = UNet_SASSNet(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'unet_CCT':
        model = UNet_CCT(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'unet_MC_Net':
        model = UNet_MC_Net(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'unetfedfa':
        model = UNetFedfa(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'alexnet':
        model = AlexNet(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'alexnetfedfa':
        model = AlexNetFedFa(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'alexnetmanifoldmixup':
        model = AlexNetManifoldMixup(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'resnet18':
        model = resnet18(config.CLIENT)
    elif config.NETWORK == 'vgg':
        model = vgg()

    return model


def build_client(config):
    assert config.CLIENT in [
        'BaseClient',
        'UAMTClient',
        'SASSNetClient',
        'DANClient',
        'EMClient',
        'CCTClient',
        'MCNETClient',
        'UdaClient',
        'FixMatchClient',
        'SoftClient',
        'FedProxClient',
        'FedHarmoClient',
        'FedBNClient',
        'FedSAMClient',
        'FedDynClient',
        'MixUpClient',
        'ManifoldMixUpClient'
    ]
    client_class = eval(config.CLIENT)

    return client_class
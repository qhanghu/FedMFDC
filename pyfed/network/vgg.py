import torchvision.models as models
import torch.nn as nn

def vgg():
    model = models.vgg11(pretrained=True)
    model.add_module('linear',nn.Linear(1000,10))
    return model
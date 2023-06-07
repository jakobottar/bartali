from collections import OrderedDict

import torch
from torch import nn
from torchvision import models


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class EncodeProject(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        match backbone:
            case "resnet18" | "18":
                self.convnet = models.resnet18(weights="DEFAULT")
                self.encoder_dim = 512
            case "resnet50" | "50":
                self.convnet = models.resnet50(weights="DEFAULT")
                self.encoder_dim = 2048
            case "resnet152" | "152":
                self.convnet = models.resnet152(weights="DEFAULT")
                self.encoder_dim = 2048
            case _:
                raise NotImplementedError(f"Cound not load model {backbone}.")

        self.convnet = torch.nn.Sequential(*(list(self.convnet.children())[:-1]))

        self.proj_dim = 128
        projection_layers = [
            ("fc1", nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ("bn1", nn.BatchNorm1d(self.encoder_dim)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(self.encoder_dim, 128, bias=False)),
            ("bn2", BatchNorm1dNoBias(128)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))
        self.flatten = nn.Flatten()

    def forward(self, x, out="z"):
        h = self.convnet(x)
        h = self.flatten(h)
        if out == "h":
            return h
        return self.projection(h)

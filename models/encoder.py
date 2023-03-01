from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torch import nn

import models


class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        return torch.flatten(feat, start_dim=self.dim)


class ResNetEncoder(torchvision.models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""

    def __init__(self, block, layers, cifar_head=False):
        super().__init__(block, layers)
        self.cifar_head = cifar_head
        if cifar_head:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = self._norm_layer(64)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar_head:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ResNet18(ResNetEncoder):
    def __init__(self, cifar_head=True):
        super().__init__(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], cifar_head=cifar_head
        )


class ResNet50(ResNetEncoder):
    def __init__(self, cifar_head=True):
        super().__init__(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], cifar_head=cifar_head
        )


class ResNet152(ResNetEncoder):
    def __init__(self, cifar_head=True):
        super().__init__(
            torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], cifar_head=cifar_head
        )


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class EncodeProject(nn.Module):
    def __init__(self, backbone, cifar_head=True):
        super().__init__()

        match backbone:
            case "resnet18" | "18":
                self.convnet = ResNet18(cifar_head=cifar_head)
                self.encoder_dim = 512
            case "resnet50" | "50":
                self.convnet = ResNet50(cifar_head=cifar_head)
                self.encoder_dim = 2048
            case "resnet152" | "152":
                self.convnet = ResNet152(cifar_head=cifar_head)
                self.encoder_dim = 2048
            case _:
                raise NotImplementedError(f"Cound not load model {backbone}.")

        self.proj_dim = 128
        projection_layers = [
            ("fc1", nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ("bn1", nn.BatchNorm1d(self.encoder_dim)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(self.encoder_dim, 128, bias=False)),
            ("bn2", BatchNorm1dNoBias(128)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))

    def forward(self, x, out="z"):
        h = self.convnet(x)
        if out == "h":
            return h
        return self.projection(h)


class TwoHeadedEncoder(nn.Module):
    def __init__(self, backbone, n_classes, cifar_head=True):
        super().__init__()

        match backbone:
            case "resnet18" | "18":
                self.convnet = ResNet18(cifar_head=cifar_head)
                self.encoder_dim = 512
            case "resnet50" | "50":
                self.convnet = ResNet50(cifar_head=cifar_head)
                self.encoder_dim = 2048
            case "resnet152" | "152":
                self.convnet = ResNet152(cifar_head=cifar_head)
                self.encoder_dim = 2048
            case _:
                raise NotImplementedError(f"Cound not load model {backbone}.")

        projection_layers = [
            (
                "fc1",
                nn.Linear(self.encoder_dim, self.encoder_dim, bias=False),
            ),
            ("bn1", nn.BatchNorm1d(self.encoder_dim)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(self.encoder_dim, 128, bias=False)),
            ("bn2", BatchNorm1dNoBias(128)),
        ]
        self.projhead = nn.Sequential(OrderedDict(projection_layers))

        eval_layers = [
            (
                "fc1",
                nn.Linear(self.encoder_dim, n_classes),
            ),
        ]
        self.evalhead = nn.Sequential(OrderedDict(eval_layers))

    def forward(self, x):
        x = self.convnet(x)
        p = self.projhead(x)
        e = self.evalhead(x)
        return p, e

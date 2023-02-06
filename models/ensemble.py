# pylint: disable=missing-function-docstring, redefined-builtin, too-many-arguments
"""
Network Definitions
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# from torch.autograd import Function
from torchvision import models


class EnsembleNet(nn.Module):
    """N networks, with N outputs.

    Args:
        backbone (string): The backbone CNN, supports `resnet50`, `resnet101`,
        and `densenet121` (default: `resnet50`)

        N (int): number of networks in ensemble (default: `2`)

        num_classes (int): number of output classes, passed to backbone networks (default: `1000`)

        unit_norm (bool): should the output vectors be unit-normalized (default: `True`)"""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_branches: int = 2,
        num_classes: int = 1000,
        unit_norm: bool = False,
        pretrained: bool = False,
        early_out: bool = False,
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.num_classes = num_classes
        self.early_out = early_out

        if backbone == "resnet34":
            bb_net = models.resnet34
        elif backbone == "resnet50":
            bb_net = models.resnet50
        elif backbone == "resnet101":
            bb_net = models.resnet101
        elif backbone == "densenet121":
            bb_net = models.densenet121
        else:
            raise NotImplementedError

        self.ensemble_bb = nn.ModuleList(
            [
                bb_net(
                    num_classes=num_classes,
                    weights="DEFAULT" if pretrained else None,
                )
                for _ in range(num_branches)
            ]
        )

        print(self.ensemble_bb[0])

    def set_early_out(self, early_out: bool):
        self.early_out = early_out

    def forward(self, input: Tensor) -> Tensor:
        out = torch.empty(
            (self.num_branches, input.shape[0], self.num_classes), device=input.device
        )
        for i in range(self.num_branches):
            out[i] = self.ensemble_bb[i](input)

        return out

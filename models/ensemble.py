# pylint: disable=missing-function-docstring, redefined-builtin, too-many-arguments
"""
Network Definitions
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# from torch.autograd import Function
from torchvision import models


# class UnitNorm(nn.Module):
#     """
#     Unit normalize a vector
#     """

#     def forward(self, input: Tensor) -> Tensor:
#         """forward"""
#         norm = torch.linalg.norm(input, dim=1, ord=2)
#         return torch.diag(1 / norm) @ input


# class GRF(Function):
#     """
#     Gradient Reversal Layer from:
#     Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
#     Forward pass is the identity function. In the backward pass,
#     the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
#     """

#     @staticmethod
#     def forward(ctx, x, lambda_):
#         ctx.lambda_ = lambda_
#         return x.clone()

#     @staticmethod
#     def backward(ctx, grads):
#         lambda_ = ctx.lambda_
#         lambda_ = grads.new_tensor(lambda_)
#         dx = -lambda_ * grads
#         return dx, None


# class GRL(nn.Module):
#     """
#     Implements gradient reversal layer
#     """

#     def __init__(self, lambda_=1):
#         super().__init__()
#         self.lambda_ = lambda_

#     def forward(self, input):
#         return GRF.apply(input, self.lambda_)

#     def set_lambda(self, lambda_):
#         self.lambda_ = lambda_


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
                    num_classes=num_classes, weights="DEFAULT" if pretrained else None
                )._modules["features"]
                for _ in range(num_branches)
            ]
        )
        self.ensemble_classifier = nn.ModuleList(
            [
                nn.Linear(in_features=1024, out_features=num_classes, bias=True)
                for _ in range(num_branches)
            ]
        )

    def set_early_out(self, early_out: bool):
        self.early_out = early_out

    def forward(self, input: Tensor) -> Tensor:
        out = torch.empty(
            (self.num_branches, input.shape[0], self.num_classes), device=input.device
        )
        features = torch.empty(
            (self.num_branches, input.shape[0], 1024), device=input.device
        )
        for i in range(self.num_branches):
            feat = self.ensemble_bb[i](input)
            feat = F.relu(feat, inplace=True)
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            feat = torch.flatten(feat, 1)
            features[i] = feat
            pred = self.ensemble_classifier[i](feat)
            out[i] = pred

        if self.early_out:
            return out, features
        return out


# class GNet(nn.Module):
#     """Simple fully connected neural network with one hidden layer
#     to serve as the adversarial component between the backbones of `PairResNet`

#         Args:
#             channels (int): number of in and out channels. (default: 1000)

#             width (int): hidden layer width (default: 1024)"""

#     def __init__(self, channels: int = 1000, width: int = 1024) -> None:
#         super().__init__()

#         self.stack = nn.Sequential(
#             nn.Linear(in_features=channels, out_features=width),
#             nn.ReLU(),
#             nn.Linear(in_features=width, out_features=width),
#             nn.ReLU(),
#             nn.Linear(in_features=width, out_features=width),
#             nn.ReLU(),
#             nn.Linear(in_features=width, out_features=channels),
#             nn.ReLU(),
#         )

#         self.unit_norm = UnitNorm()

#     def forward(self, input: Tensor) -> Tensor:
#         out = self.stack(input)
#         out = self.unit_norm(out)
#         return out

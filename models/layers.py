""" layer implementations """

import math

import torch
import torch.nn.functional as F
from torch import nn


class OrthogonallyAttenuatedLinear(nn.Module):
    """
    TODO: add info
    """

    def __init__(self, in_features, out_features, sigma_scalar: float = 4.0):
        super().__init__()

        self.size_in, self.size_out = in_features, out_features
        self.sigma_scalar = sigma_scalar

        weights = torch.Tensor(out_features, in_features)
        bias = torch.Tensor(out_features)
        center = torch.Tensor(out_features, in_features)
        sigma = torch.Tensor(out_features)

        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)
        self.center = nn.Parameter(center)
        self.sigma = nn.Parameter(sigma)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, nonlinearity="relu")  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        nn.init.normal_(self.center, mean=0.0, std=0.1)  # center init
        nn.init.constant_(
            self.sigma, math.sqrt(in_features) / sigma_scalar
        )  # sigma init

    def forward(self, x):
        pw = F.linear(x, self.weights, self.bias)  # project to w direction
        wn = F.normalize(self.weights, dim=1)  # normalized w

        x = torch.reshape(x, (-1, self.size_in, 1))
        c = torch.reshape(self.center.t(), (1, self.size_in, self.size_out))
        x = x - c
        pwc = torch.sum(torch.mul(x, wn.t()), dim=1)  # project x - c onto wn
        pwL2 = torch.square(pwc)
        xn = torch.square(
            torch.linalg.norm(x, dim=1, ord=2)
        )  # squared L2 norm of x - c
        poL2 = xn - pwL2
        om = torch.exp(-torch.divide(poL2, torch.square(self.sigma)))

        return torch.mul(pw, om)

    def extra_repr(self) -> str:
        return f"in_features={self.size_in}, out_features={self.size_out}, sigma_scalar={self.sigma_scalar}"

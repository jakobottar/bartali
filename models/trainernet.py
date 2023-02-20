"""
replacement for ResNet class, with built-in trainers and testers
"""
import re
from typing import OrderedDict, Tuple

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import LARS


class Trainer:
    def __init__(self, configs) -> None:
        self.model = nn.Module()
        self.loss = nn.Module()
        self.optimizer = None
        self.scheduler = None
        self.configs = configs
        self.device = "cpu"
        self.mode = "standard"
        self.epoch = 0

    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()

    def to(self, device):
        if self.mode == "ddp":
            raise Exception("Model running in DistributedDataParallel mode!")

        self.device = device
        self.model.to(device)
        return self

    def to_ddp(self, rank):
        self.mode = "ddp"
        self.device = self.configs.gpus[rank]
        self.model = DDP(
            self.model.to(self.device),
            device_ids=[self.device],
            find_unused_parameters=self.configs.find_unused_parameters,
        )
        self.set_up_loss()  # re-set up loss, in case of ddp dependencies
        return self

    def get_model(self) -> nn.Module:
        return self.model

    def get_ckpt(self) -> OrderedDict:
        if self.mode == "standard":
            return self.model.state_dict()
        else:  # self.mode == "ddp"
            return self.model.module.state_dict()

    def _load_ckpt(self, model, model_state_dict):
        # TODO: what if we load a non-ddp model?
        model_dict = OrderedDict()
        pattern = re.compile("module.")
        for k, v in model_state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, "", k)] = v
            else:
                model_dict = model_state_dict

        model.load_state_dict(model_dict)
        return model

    def load_ckpt(self, model_state_dict) -> None:
        self.model = self._load_ckpt(model_state_dict)

    def set_up_optimizers(self) -> None:
        def exclude_from_wd_and_adaptation(name):
            if "bn" in name:
                return True
            if self.configs.optimizer == "lars" and "bias" in name:
                return True

        parameters = [
            {
                "params": [
                    p
                    for name, p in self.model.named_parameters()
                    if not exclude_from_wd_and_adaptation(name)
                ],
                "weight_decay": self.configs.weight_decay,
                "layer_adaptation": True,
            },
            {
                "params": [
                    p
                    for name, p in self.model.named_parameters()
                    if exclude_from_wd_and_adaptation(name)
                ],
                "weight_decay": 0.0,
                "layer_adaptation": False,
            },
        ]

        match self.configs.optimizer:
            case "adam":
                self.optimizer = torch.optim.Adam(
                    parameters,
                    lr=self.configs.lr,
                    weight_decay=self.configs.weight_decay,
                )
            case "lars":
                self.optimizer = torch.optim.SGD(
                    parameters,
                    lr=self.configs.lr,
                    momentum=0.9,
                )
                lars_optimizer = LARS(self.optimizer)
            case _:
                raise NotImplementedError(
                    f"Could not find scheduler {self.configs.lr_schedule}."
                )

        match self.configs.lr_schedule:
            case "exponential":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.configs.lr_gamma
                )
            case "cosine-anneal":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, self.configs.epochs
                )
            case "constant":
                self.scheduler = None
            case _:
                raise NotImplementedError(
                    f"Could not find scheduler {self.configs.lr_schedule}."
                )

        if self.configs.optimizer == "lars":
            self.optimizer = lars_optimizer

    def set_up_loss(self) -> None:
        raise NotImplementedError("'set_up_loss' not reimplemented in child class.")

    def step(self, value, target) -> Tuple[Tensor, Tensor]:

        pred = self.model(value)
        loss = self.loss(pred, target)

        return (pred, loss)

    def train_step(self, value, target) -> Tuple[Tensor, Tensor]:

        pred, loss = self.step(value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return pred, loss

    def test_step(self, value, target) -> Tuple[Tensor, Tensor]:
        return self.step(value, target)

    def train_epoch(self) -> dict:
        raise NotImplementedError("'train_epoch' not reimplemented in child class.")

    def test_epoch(self) -> dict:
        raise NotImplementedError("'test_epoch' not reimplemented in child class.")

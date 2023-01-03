"""
replacement for ResNet class, with built-in trainers and testers
"""
from typing import Tuple, OrderedDict
import re

import torch
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(self, configs) -> None:
        self.model = nn.Module()
        self.loss = nn.Module()
        self.optimizer = None
        self.scheduler = None
        self.configs = configs
        self.device = "cpu"
        self.mode = "standard"

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
        self.model = DDP(self.model.to(self.device), device_ids=[self.device])
        return self

    def get_ckpt(self) -> OrderedDict:
        if self.mode == "standard":
            return self.model.state_dict()
        else:  # self.mode == "ddp"
            return self.model.module.state_dict()

    def load_ckpt(self, model_state_dict) -> None:
        # TODO: what if we load a non-ddp model?
        model_dict = OrderedDict()
        pattern = re.compile("module.")
        for k, v in model_state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, "", k)] = v
            else:
                model_dict = model_state_dict
        self.model.load_state_dict(model_dict)

    def set_up_optimizers(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.configs.lr,
            weight_decay=self.configs.weight_decay,
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
            case _:
                raise NotImplementedError(
                    f"Could not find scheduler {self.configs.lr_schedule}."
                )

    def set_up_loss(self) -> None:
        raise NotImplementedError("'set_up_loss' not reimplemented in child class.")

    def step(self, value, target) -> Tuple[Tensor, Tensor]:

        pred = self.model(value)
        loss = self.loss(pred, target)

        return (pred, loss)

    def train_step(self, value, target) -> Tuple[Tensor, Tensor]:

        self.optimizer.zero_grad()
        pred, loss = self.step(value, target)
        loss.backward()
        self.optimizer.step()

        return pred, loss

    def test_step(self, value, target) -> Tuple[Tensor, Tensor]:
        return self.step(value, target)

    def train_epoch(self, dataloader) -> None:
        raise NotImplementedError("'train_epoch' not reimplemented in child class.")

    def test_epoch(self, dataloader) -> None:
        raise NotImplementedError("'test_epoch' not reimplemented in child class.")

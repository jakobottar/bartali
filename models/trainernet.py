"""
replacement for ResNet class, with built-in trainers and testers
"""
from typing import Tuple, OrderedDict
from tqdm import tqdm
import shutil

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms


class Trainer:
    def __init__(self) -> None:
        self.model = nn.Module()
        self.loss = nn.Module()
        self.optimizer = None
        self.scheduler = None
        self.device = "cpu"

    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()

    def to(self, device) -> None:
        self.device = device
        self.model.to(device)

    def get_ckpt(self) -> OrderedDict:
        return self.model.state_dict()

    def load_ckpt(self, model_state_dict) -> None:
        self.model.load_state_dict(model_state_dict)

    def set_up_optimizers(self, configs) -> None:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay
        )
        match configs.lr_schedule:
            case "exponential":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=configs.lr_gamma
                )
            case "cosine-anneal":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, configs.epochs
                )
            case _:
                raise NotImplementedError(
                    f"Could not find scheduler {configs.lr_schedule}."
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

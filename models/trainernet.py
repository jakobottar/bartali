"""
replacement for ResNet class, with built-in trainers and testers
"""
from typing import Tuple
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
        self.device = "cpu"

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.device = device
        self.model.to(device)

    def set_up_optimizers(self, lr=0.01, lr_gamma=0.97) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_gamma
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


class ResNet(Trainer):
    def __init__(self, model="resnet18") -> None:
        super().__init__()

        match model:
            case "resnet18" | "18":
                self.model = models.resnet18(weights="DEFAULT")
            case "resnet50" | "50":
                self.model = models.resnet50(weights="DEFAULT")
            case _:
                raise NotImplementedError(f"Cound not load model {model}.")

    def set_up_loss(self):
        self.loss = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, verbose = True) -> dict:
        self.train()

        loader = iter(dataloader)
        if verbose:
            loader = tqdm(loader, ncols=shutil.get_terminal_size().columns)

        train_loss = 0
        for _, (value, target) in enumerate(loader):
            value, target = value.to(self.device), target.to(self.device)
            _, loss = self.train_step(value, target)

            train_loss += loss.item()
            if verbose:
                loader.set_description(f"train loss: {loss.item():.4f}")

        return {
            "train_loss": train_loss / len(dataloader)
        }

    def test_epoch(self, dataloader, verbose = True) -> list:
        self.eval()

        loader = iter(dataloader)
        if verbose:
            loader = tqdm(loader, ncols=shutil.get_terminal_size().columns)

        test_loss = 0
        for _, (value, target) in enumerate(loader):
            value, target = value.to(self.device), target.to(self.device)
            _, loss = self.test_step(value, target)

            test_loss += loss.item()
            if verbose:
                loader.set_description(f"test loss: {loss.item():.4f}")

        return {
            "test_loss": test_loss / len(dataloader)
        }

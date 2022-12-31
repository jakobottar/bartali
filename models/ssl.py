from tqdm import tqdm
import shutil

import torch
from torchvision import models

from .trainernet import Trainer
from .losses import NTXent
from .encoder import EncodeProject


class SimCLR(Trainer):
    def __init__(self, backbone="resnet18") -> None:
        super().__init__()

        self.model = EncodeProject(backbone=backbone, cifar_head=True)

        match backbone:
            case "resnet18" | "18":
                self.model = models.resnet18(weights="DEFAULT")
            case "resnet50" | "50":
                self.model = models.resnet50(weights="DEFAULT")
            case _:
                raise NotImplementedError(f"Cound not load model {backbone}.")

    def set_up_loss(self, configs):
        self.loss = NTXent(tau=configs.tau, multiplier=int(configs.multiplier))

    def step(self, value, target):

        pred = self.model(value)
        loss = self.loss(pred)

        return (pred, loss)

    def train_epoch(self, dataloader, verbose=True) -> dict:
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

        return {"train_loss": train_loss / len(dataloader)}

    def test_epoch(self, dataloader, verbose=True) -> list:
        self.eval()

        loader = iter(dataloader)
        if verbose:
            loader = tqdm(loader, ncols=shutil.get_terminal_size().columns)

        test_loss = 0
        with torch.no_grad():
            for _, (value, target) in enumerate(loader):
                value, target = value.to(self.device), target.to(self.device)
                _, loss = self.test_step(value, target)

                test_loss += loss.item()
                if verbose:
                    loader.set_description(f"test loss: {loss.item():.4f}")

        return {"test_loss": test_loss / len(dataloader)}

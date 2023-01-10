import shutil
import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .trainernet import Trainer


class ResNet(Trainer):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        match configs.arch:
            case "resnet18" | "18":
                self.model = models.resnet18(weights="DEFAULT")
            case "resnet50" | "50":
                self.model = models.resnet50(weights="DEFAULT")
            case _:
                raise NotImplementedError(f"Cound not load model {configs.arch}.")

        self.set_up_optimizers()
        self.set_up_loss()

    def set_up_loss(self):
        self.loss = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, verbose=True) -> dict:
        self.train()

        loader = iter(dataloader)
        if verbose:
            loader = tqdm(loader, ncols=shutil.get_terminal_size().columns)

        train_loss, correct = 0.0, 0.0
        data_time, iter_time = 0.0, 0.0
        start_time = time.time()
        for _, (value, target) in enumerate(loader):
            value, target = value.to(self.device), target.to(self.device)
            data_time += time.time() - start_time

            # do training step
            pred, loss = self.train_step(value, target)
            iter_time += time.time() - start_time

            # get loss
            train_loss += loss.item()
            # get accuracy
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if verbose:
                loader.set_description(f"train loss: {loss.item():.4f}")

        return {
            "train_loss": train_loss / len(dataloader),
            "train_acc": correct / (len(dataloader.dataset) / len(self.configs.gpus)),
            "data_time": data_time / len(dataloader),
            "iter_time": iter_time / len(dataloader),
        }

    def test_epoch(self, dataloader, verbose=True) -> list:
        self.eval()

        loader = iter(dataloader)
        if verbose:
            loader = tqdm(loader, ncols=shutil.get_terminal_size().columns)

        test_loss, correct = 0.0, 0.0
        with torch.no_grad():
            for _, (value, target) in enumerate(loader):
                value, target = value.to(self.device), target.to(self.device)
                # do testing step
                pred, loss = self.test_step(value, target)
                # get loss
                test_loss += loss.item()
                # get accuracy
                pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()

                if verbose:
                    loader.set_description(f"test loss: {loss.item():.4f}")

        return {
            "test_loss": test_loss / len(dataloader),
            "test_acc": correct / (len(dataloader.dataset) / len(self.configs.gpus)),
        }

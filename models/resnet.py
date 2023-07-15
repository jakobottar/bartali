import shutil
import time

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from tqdm import tqdm

from .trainernet import Trainer


class ResNet(Trainer):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        match configs.arch:
            case "resnet18" | "18":
                self.model = models.resnet18(weights="DEFAULT")
                encoder_dim = 512
            case "resnet50" | "50":
                self.model = models.resnet50(weights="DEFAULT")
                encoder_dim = 2048
            case "resnet152" | "152":
                self.model = models.resnet152(weights="DEFAULT")
                encoder_dim = 2048
            case _:
                raise NotImplementedError(f"Cound not load model {configs.arch}.")

        n_classes = 12
        self.model.fc = nn.Linear(encoder_dim, n_classes)

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
        start_time = time.perf_counter()
        for _, (value, target) in enumerate(loader):
            value, target = value.to(self.device), target.to(self.device)
            data_time += time.perf_counter() - start_time

            # do training step
            pred, loss = self.train_step(value, target)
            iter_time += time.perf_counter() - start_time

            # get loss
            train_loss += loss.item()
            # get accuracy
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if verbose:
                loader.set_description(f"train loss: {loss.item():.4f}")

        self.epoch += 1
        if self.scheduler:
            self.scheduler.step()
        return {
            "train_loss": train_loss / (len(dataloader) / len(self.configs.gpus)),
            "train_acc": correct / (len(dataloader.dataset) / len(self.configs.gpus)),
            "data_time": data_time / (len(dataloader) / len(self.configs.gpus)),
            "iter_time": iter_time / (len(dataloader) / len(self.configs.gpus)),
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
            "val_loss": test_loss / (len(dataloader) / len(self.configs.gpus)),
            "val_acc": correct / (len(dataloader.dataset) / len(self.configs.gpus)),
        }

import shutil
import time
from tqdm import tqdm

import torch
import numpy as np
import scipy

from .trainernet import Trainer
from .losses import NTXent
from .encoder import EncodeProject


class SimCLR(Trainer):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        self.epoch = 0
        self.model = EncodeProject(
            backbone=configs.arch, cifar_head=(configs.dataset == "cifar")
        )
        self.reset_parameters()
        self.set_up_optimizers()
        self.set_up_loss()

    def reset_parameters(self):
        def conv2d_weight_truncated_normal_init(p):
            fan_in = p.shape[1]
            stddev = np.sqrt(1.0 / fan_in) / 0.87962566103423978
            r = scipy.stats.truncnorm.rvs(-2, 2, loc=0, scale=1.0, size=p.shape)
            r = stddev * r
            with torch.no_grad():
                p.copy_(torch.FloatTensor(r))

        def linear_normal_init(p):
            with torch.no_grad():
                p.normal_(std=0.01)

        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                conv2d_weight_truncated_normal_init(m.weight)
            elif isinstance(m, torch.nn.Linear):
                linear_normal_init(m.weight)

    def set_up_loss(self):
        self.loss = NTXent(
            tau=self.configs.tau,
            multiplier=int(self.configs.multiplier),
            distributed=(self.mode == "ddp"),
        )

    def step(self, value, target):

        pred = self.model(value)
        loss = self.loss(pred)

        return (pred, loss)

    # TODO: implement encode step

    def train_epoch(self, dataloader, verbose=True) -> dict:
        self.train()

        # dataloader.sampler.set_epoch(self.epoch)
        loader = iter(dataloader)
        if verbose:
            loader = tqdm(loader, ncols=shutil.get_terminal_size().columns)

        train_loss = 0
        data_time, iter_time = 0.0, 0.0
        start_time = time.time()
        for _, (value, target) in enumerate(loader):
            value, target = value.to(self.device), target.to(self.device)
            data_time += time.time() - start_time

            # do training step
            _, loss = self.train_step(value, target)
            iter_time += time.time() - start_time

            # get loss
            train_loss += loss.item()
            if verbose:
                loader.set_description(f"train loss: {loss.item():.4f}")

        self.epoch += 1
        self.scheduler.step()
        return {
            "train_loss": train_loss / len(dataloader),
            "data_time": data_time / len(dataloader),
            "iter_time": iter_time / len(dataloader),
        }

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

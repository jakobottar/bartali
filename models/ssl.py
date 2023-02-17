import re
import shutil
import time
from typing import OrderedDict

import numpy as np
import scipy
import sklearn.metrics as sk
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import utils

from .encoder import EncodeProject
from .trainernet import Trainer


class SimCLR(Trainer):
    def __init__(self, configs) -> None:
        super().__init__(configs)
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
        self.loss = utils.NTXent(
            tau=self.configs.tau,
            multiplier=int(self.configs.multiplier),
            distributed=(self.mode == "ddp"),
        )

    def step(self, value, target):

        pred = self.model(value)
        loss = self.loss(pred)

        return (pred, loss)

    def train_epoch(self, dataloader, verbose=True) -> dict:
        self.train()

        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(self.epoch)

        loader = iter(dataloader)
        if verbose:
            loader = tqdm(loader, ncols=shutil.get_terminal_size().columns)

        train_loss = 0
        data_time, iter_time = 0.0, 0.0
        start_time = time.perf_counter()
        for _, (value, target) in enumerate(loader):
            value, target = value.to(self.device), target.to(self.device)
            data_time += time.perf_counter() - start_time

            # do training step
            _, loss = self.train_step(value, target)
            iter_time += time.perf_counter() - start_time

            # get loss
            train_loss += loss.item()
            if verbose:
                loader.set_description(f"train loss: {loss.item():.4f}")

        self.epoch += 1
        if self.scheduler:
            self.scheduler.step()
        return {
            "train_loss": train_loss / (len(dataloader) / len(self.configs.gpus)),
            "data_time": data_time / (len(dataloader) / len(self.configs.gpus)),
            "iter_time": iter_time / (len(dataloader) / len(self.configs.gpus)),
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

        return {"test_loss": test_loss / (len(dataloader) / len(self.configs.gpus))}


class EvalSimCLR(Trainer):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        # Load a trained SIMCLR model
        self.encoder = EncodeProject(
            backbone=self.configs.arch, cifar_head=(self.configs.dataset == "cifar")
        )
        self.load_encoder_ckpt(torch.load(self.configs.chkpt_file))
        self.encoder.to(self.device)
        self.encoder.eval()

        # push torch.ones through the model to get input size
        match self.configs.dataset:
            case "cifar":
                hdim = self.encoder(
                    torch.ones(10, 3, 32, 32).to(self.device), out="h"
                ).shape[1]
                n_classes = 10
            case "nfs":
                hdim = self.encoder(
                    torch.ones(10, 3, 256, 256).to(self.device), out="h"
                ).shape[1]
                n_classes = 16
            case _:
                raise NotImplementedError(
                    f"Cound not load dataset {self.configs.dataset}."
                )

        # build a linear layer to choose classes
        model = nn.Linear(hdim, n_classes).to(self.device)
        model.weight.data.zero_()
        model.bias.data.zero_()
        self.model = model

        self.set_up_optimizers()
        self.set_up_loss()

        self.mean_train_var = utils.RunningAverage()

    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        return self

    def to_ddp(self, rank):
        super().to_ddp(rank)
        self.encoder.to(self.device)
        return self

    def encode_step(self, value):
        return self.encoder(value, out="h")

    def step(self, value, target):
        if self.configs.dataset == "nfs":
            value = self.encode_step(value)
        return super().step(value, target)

    # TODO: fix for use with all mags
    def precompute_encodings(self, dataloader: DataLoader, shuffle: bool = True):
        print("===> preprocessing encodings...")
        encodings, labels = [], []

        with torch.no_grad():
            for _, (value, target) in enumerate(dataloader):
                value = value.to(self.device)
                enc = self.encode_step(value)
                encodings.extend(list(enc.cpu()))
                labels.extend(list(target))

        encodings = torch.stack(encodings)
        labels = torch.stack(labels)
        # print(f"encodings:{encodings.shape}")
        # print(f"labels:{labels.shape}")

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(encodings),
            torch.LongTensor(labels),
        )

        sampler = DistributedSampler(dataset, shuffle=shuffle)

        dataloader = DataLoader(
            dataset,
            num_workers=self.configs.workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=self.configs.batch_size,
        )

        return dataloader

    def set_up_loss(self) -> None:
        self.loss = nn.CrossEntropyLoss()

    def load_encoder_ckpt(self, model_state_dict) -> None:
        # TODO: what if we load a non-ddp model?
        model_dict = OrderedDict()
        pattern = re.compile("module.")
        for k, v in model_state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, "", k)] = v
            else:
                model_dict = model_state_dict
        self.encoder.load_state_dict(model_dict)

    def ood_metric(self, values: Tensor, train: bool = False):
        if train:
            self.mean_train_var.average_in(value=torch.var(values))
            return
        else:
            var = torch.var(values)
            # return self.mean_train_var.mean - var
            return var.item()

    def train_epoch(self, dataloader) -> dict:
        self.train()

        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(self.epoch)

        loader = iter(dataloader)

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

        self.epoch += 1
        if self.scheduler:
            self.scheduler.step()
        return {
            "train_loss": train_loss / (len(dataloader) / len(self.configs.gpus)),
            "train_acc": correct / (len(dataloader.dataset) / len(self.configs.gpus)),
            "data_time": data_time / (len(dataloader) / len(self.configs.gpus)),
            "iter_time": iter_time / (len(dataloader) / len(self.configs.gpus)),
        }

    def test_epoch(self, test_loader, ood_loader) -> list:
        self.eval()

        iter_test_loader = iter(test_loader)
        iter_ood_loader = iter(ood_loader)

        test_loss, correct = 0, 0
        var_right, var_wrong, var_ood = [], [], []
        with torch.no_grad():
            if self.configs.dataset == "nfs":
                # process test dataset
                for _, (values, target) in enumerate(iter_test_loader):
                    preds = torch.zeros(
                        (len(values), target.shape[0]), device=self.device
                    )  # space for predicted values
                    encodings = torch.zeros(
                        (len(values), target.shape[0], 2048), device=self.device
                    )  # space for encodings
                    for i, value in enumerate(values):
                        value, target = value.to(self.device), target.to(self.device)

                        # do test step
                        pred, loss = self.test_step(value, target)
                        encodings[i] = self.encode_step(value)

                        # get loss
                        test_loss += loss.item()
                        # get prediction
                        preds[i] = torch.argmax(F.softmax(pred, dim=1), dim=1)

                    encodings = encodings.transpose(0, 1)
                    preds = torch.mode(preds, dim=0).values
                    r = preds.eq(target.view_as(preds))  # idx of correct preds
                    for i, is_correct in enumerate(r):
                        if is_correct:
                            var_right.append(self.ood_metric(encodings[i]))
                        else:
                            var_wrong.append(self.ood_metric(encodings[i]))

                    correct += r.sum().item()

                # process ood dataset
                for _, values in enumerate(iter_ood_loader):
                    encodings = torch.zeros(
                        (len(values), values[0].shape[0], 2048), device=self.device
                    )
                    for i, value in enumerate(values):
                        value = value.to(self.device)

                        # do test step
                        encodings[i] = self.encode_step(value)

                    encodings = encodings.transpose(0, 1)
                    var_ood = [self.ood_metric(chunk) for chunk in encodings]
            else:
                raise NotImplementedError

        in_labels = np.concatenate(
            [
                np.ones((len(var_right),), dtype=int),
                np.zeros((len(var_wrong),), dtype=int),
            ]
        )
        out_labels = np.concatenate(
            [
                np.ones((len(var_right),), dtype=int),
                np.zeros((len(var_ood),), dtype=int),
            ]
        )
        in_values = np.concatenate([var_right, var_wrong])
        out_values = np.concatenate([var_right, var_ood])

        auroc_in = sk.roc_auc_score(in_labels, in_values)
        auroc_out = sk.roc_auc_score(out_labels, out_values)

        return {
            "val_loss": test_loss / (len(test_loader) / len(self.configs.gpus)),
            "val_acc": correct / (len(test_loader.dataset) / len(self.configs.gpus)),
            "val_var_right": np.mean(var_right),
            "val_var_wrong": np.mean(var_wrong),
            "val_var_ood": np.mean(var_ood),
            "val_var_all": (np.mean(var_right) + np.mean(var_wrong)) / 2,
            "val_auroc_in": auroc_in,
            "val_auroc_out": auroc_out,
        }

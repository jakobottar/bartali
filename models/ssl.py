import re
import shutil
import time
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm

import utils

from .encoder import EncodeProject
from .layers import OrthogonallyAttenuatedLinear
from .trainernet import Trainer


class SimCLR(Trainer):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        self.model = EncodeProject(backbone=configs.arch)

        # self.reset_parameters()
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

        return {"val_loss": test_loss / (len(dataloader) / len(self.configs.gpus))}


class EvalSimCLR(Trainer):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        # Load a trained SIMCLR model
        self.encoder = EncodeProject(backbone=self.configs.arch)
        self.load_encoder_ckpt(torch.load(self.configs.chkpt_file, map_location="cpu"))
        self.encoder.to(self.device)
        self.freeze_encoder()

        # push torch.ones through the model to get input size
        match self.configs.dataset:
            case "cifar":
                n_classes = 10
            case "nfs":
                n_classes = 16
            case _:
                raise NotImplementedError(
                    f"Cound not load dataset {self.configs.dataset}."
                )

        # build a non-linear classification head
        if configs.mode == "standard":
            model = [
                ("bn", nn.BatchNorm1d(self.encoder.encoder_dim, affine=False)),
                ("fc1", nn.Linear(self.encoder.encoder_dim, self.encoder.encoder_dim)),
                ("relu1", nn.ReLU()),
                ("fc2", nn.Linear(self.encoder.encoder_dim, n_classes)),
            ]

        elif configs.mode == "oa":
            model = [
                ("bn", nn.BatchNorm1d(self.encoder.encoder_dim, affine=False)),
                (
                    "fc1",
                    OrthogonallyAttenuatedLinear(
                        self.encoder.encoder_dim,
                        self.encoder.encoder_dim,
                        sigma_scalar=2.0,
                    ),
                ),
                ("relu1", nn.ReLU()),
                ("fc2", nn.Linear(self.encoder.encoder_dim, n_classes)),
            ]

        model = nn.Sequential(OrderedDict(model)).to(self.device)

        self.model = model

        self.set_up_optimizers()
        self.set_up_loss()

        self.mean_train_var = utils.RunningAverage()

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.optimizer.add_param_group(
            {"params": self.encoder.parameters(), "lr": 1e-4}
        )

    def eval(self) -> None:
        self.model.eval()
        self.encoder.eval()

    def train(self) -> None:
        self.model.train()

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

    def test_epoch(self, test_loader) -> list:
        self.eval()

        iter_test_loader = iter(test_loader)

        test_loss, correct = 0, 0
        pred_classes, correct_classes = [], []
        with torch.no_grad():
            if self.configs.dataset == "nfs":
                # process test dataset
                for values, target in iter_test_loader:
                    preds = torch.zeros(
                        (len(values), target.shape[0]), device=self.device
                    )  # space for predicted values
                    for i, value in enumerate(values):
                        value, target = value.to(self.device), target.to(self.device)

                        # do test step
                        pred, loss = self.test_step(value, target)

                        # get loss
                        test_loss += loss.item()

                        # get prediction
                        preds[i] = torch.argmax(F.softmax(pred, dim=1), dim=1)

                    preds = torch.mode(preds, dim=0).values
                    correct += preds.eq(target.view_as(preds)).sum().item()

                    # gather preds and targets for confusion matrix
                    pred_classes.extend(preds.cpu().numpy())
                    correct_classes.extend(target.cpu().numpy())

            else:
                raise NotImplementedError

        return {
            "val_loss": test_loss / (len(test_loader) / len(self.configs.gpus)),
            "val_acc": correct / (len(test_loader.dataset) / len(self.configs.gpus)),
        }

    def __ood_predict(self, loader):
        self.eval()

        max_softmax = torch.empty(0)
        with torch.no_grad():
            for values, target in loader:
                # space for predicted values
                # TODO: automatically set this shape
                logits = torch.zeros((values[0].shape[0], 16), device=self.device)

                for value in values:
                    value = value.to(self.device)
                    target = torch.zeros_like(target, device=self.device)

                    # do test step
                    pred, _ = self.test_step(value, target)

                    # get prediction
                    logits += F.softmax(pred, dim=1) / len(values)

                logits = torch.max(logits, dim=1)
                max_softmax = torch.cat((max_softmax, logits.values.cpu()))

        return max_softmax

    def ood_epoch(self, test_loader, ood_loader) -> dict:
        metric = BinaryAUROC(thresholds=None)

        # evaluate test data (in distribution)
        in_ms = self.__ood_predict(test_loader)
        # evaluate ood data (out of distribution)
        out_ms = self.__ood_predict(ood_loader)

        # compute AUROC
        gt = torch.cat((torch.ones_like(in_ms), torch.zeros_like(out_ms)))
        pred = torch.cat((in_ms, out_ms))
        auroc = metric(pred, gt)

        return {
            "auroc": auroc,
        }

    def create_confusion_matrix(self, loader) -> ConfusionMatrixDisplay:
        self.eval()

        iter_loader = iter(loader)

        pred_classes, correct_classes = [], []
        with torch.no_grad():
            for values, target in iter_loader:
                preds = torch.zeros(
                    (len(values), target.shape[0]), device=self.device
                )  # space for predicted values
                for i, value in enumerate(values):
                    value, target = value.to(self.device), target.to(self.device)

                    # do test step
                    pred, _ = self.test_step(value, target)

                    # get prediction
                    preds[i] = torch.argmax(F.softmax(pred, dim=1), dim=1)

                # majority vote
                preds = torch.mode(preds, dim=0).values

                # gather preds and targets for confusion matrix
                pred_classes.extend(preds.cpu().numpy())
                correct_classes.extend(target.cpu().numpy())

        confusion = confusion_matrix(pred_classes, correct_classes, normalize="true")
        return ConfusionMatrixDisplay(confusion, display_labels=utils.ROUTES)

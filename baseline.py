"""
baseline resnet model
"""
import argparse
import os
import random
import shutil
import time

import mlflow
import namegenerator
import pyxis.torch as pxt
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import models
import utils


class TransformTorchDataset(pxt.TorchDataset):
    def __init__(self, dirpath, transform=None):
        super().__init__(dirpath)
        self.transform = transform

    def __getitem__(self, key):
        data = self.db[key]
        for k in data.keys():
            data[k] = torch.from_numpy(data[k])

        if self.transform:
            data["image"] = self.transform(data["image"].to(torch.float))

        return data["image"], data["label"]


def worker(configs):
    """
    single worker function
    """

    # prepare the dataloader
    image_size = 256
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(brightness=0.5),
            transforms.RandomResizedCrop(image_size),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = TransformTorchDataset(
        f"/scratch_nvme/jakobj/multimag/lmdb/{configs.dataset}_train_{configs.fold_num}",
        transform=transform,
    )
    test_dataset = TransformTorchDataset(
        f"/scratch_nvme/jakobj/multimag/lmdb/{configs.dataset}_val_{configs.fold_num}",
        transform=transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=True,
        num_workers=configs.workers,
        batch_size=configs.batch_size,
    )

    test_dataloader = DataLoader(
        test_dataset,
        pin_memory=True,
        num_workers=configs.workers,
        batch_size=configs.batch_size,
    )

    # set up model
    resnet = models.ResNet(configs).to("cuda")

    cudnn.benchmark = True

    mlflow.set_tracking_uri("http://tularosa.sci.utah.edu:5000")
    mlflow.set_experiment("bartali-artifacts")
    mlflow.start_run(run_name=configs.name)
    mlflow.log_params(configs.as_dict())
    best_metric = -9999

    for epoch in range(1, configs.epochs + 1):
        print(f"epoch {epoch} of {configs.epochs} ", end="")
        start_time = time.time()

        train_stats = resnet.train_epoch(train_dataloader, verbose=False)
        test_stats = resnet.test_epoch(test_dataloader, verbose=False)

        metrics = {**train_stats, **test_stats}
        mlflow.log_metrics(metrics, step=epoch)
        mlflow.log_metric(
            "learning_rate", resnet.scheduler.get_last_lr()[0], step=epoch
        )
        if metrics["val_acc"] > best_metric:
            torch.save(resnet.get_ckpt(), f"{configs.root}/best.pth")
        print(f"{time.time() - start_time:.2f} sec")

        resnet.scheduler.step()

    mlflow.pytorch.log_model(
        resnet.get_model(), "model", pip_requirements="requirements.txt"
    )
    mlflow.log_artifacts(configs.root)

    print("done!")


if __name__ == "__main__":
    # parse args/config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="config file location"
    )
    parser.add_argument("-e", "--epochs", type=int, default=None)
    parser.add_argument("--fold-num", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--chkpt-file", type=str, default=None)
    parser.add_argument("--drop-classes", type=str, default=None, nargs="+")
    args, _ = parser.parse_known_args()
    configs = utils.parse_config_file_and_overrule(args.config, args)

    if configs.seed != -1:
        random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        cudnn.deterministic = True

    if configs.name == "random":
        configs.name = "baseline-" + namegenerator.gen()
    else:
        configs.name = "baseline-" + configs.name

    print(f"Run name: {configs.name}")
    try:
        os.mkdir(f"runs/{configs.name}")
    except FileExistsError as error:
        pass
    configs.root = f"{configs.root}/{configs.name}"
    shutil.copy(args.config, os.path.join(configs.root, "config.yaml"))

    worker(configs=configs)

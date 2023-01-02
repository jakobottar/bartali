"""
simclr model
"""
import os
import yaml
import argparse
import namegenerator
from collections import namedtuple
import mlflow

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import models

if __name__ == "__main__":
    # parse args/config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="config file location"
    )
    args, _ = parser.parse_known_args()

    # load yaml config file
    with open(args.config, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    configs = namedtuple("ConfigStruct", config_dict.keys())(*config_dict.values())

    if configs.name == "random":
        NAME = "simclr-" + namegenerator.gen()
    else:
        NAME = "simclr-" + configs.name

    print(f"Run name: {NAME}")
    try:
        os.mkdir(f"runs/{NAME}")
    except FileExistsError as error:
        pass

    # set up data
    match configs.dataset:
        case "cifar":
            transform = transforms.Compose(
                [
                    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                    transforms.ToTensor(),
                ]
            )
            train_dataset = datasets.CIFAR10(
                "/scratch/jakobj/cifar", train=True, download=True, transform=transform
            )

            test_dataset = datasets.CIFAR10(
                "/scratch/jakobj/cifar", train=True, download=True, transform=transform
            )

        case _:
            raise NotImplementedError(f"Cound not load dataset {configs.dataset}.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=configs.workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=configs.workers,
    )

    # set up model
    simclr = models.SimCLR(backbone="resnet18")
    simclr.set_up_optimizers(configs)
    simclr.set_up_loss(configs)
    simclr.to(f"cuda:{configs.gpus[0]}")

    mlflow.set_tracking_uri("http://tularosa.sci.utah.edu:5000")
    mlflow.set_experiment("bartali")
    with mlflow.start_run(run_name=NAME):

        for epoch in range(1, configs.epochs + 1):
            print(f"epoch {epoch} of {configs.epochs}")
            train_stats = simclr.train_epoch(train_dataloader)
            mlflow.log_metrics(train_stats, step=epoch)
            test_stats = simclr.test_epoch(test_dataloader)
            mlflow.log_metrics(test_stats, step=epoch)

            mlflow.log_metric(
                "learning_rate", simclr.scheduler.get_last_lr()[0], step=epoch
            )
            simclr.scheduler.step()

            torch.save(simclr.get_ckpt(), f"runs/{NAME}/model.pth")

        print("done!")

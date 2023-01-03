"""
baseline resnet model
"""
import os
import argparse
import namegenerator
import mlflow


import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import models
import utils


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_dataloaders(rank: int, world_size: int, dataset: str, batch_size: int):

    match dataset:
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
            raise NotImplementedError(f"Cound not load dataset {dataset}.")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        num_workers=0,
        sampler=train_sampler,
    )

    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        num_workers=0,
        sampler=test_sampler,
    )

    return train_dataloader, test_dataloader


def main(rank, world_size, configs):
    print(f"model launched on gpu {rank}")
    # setup the process groups
    setup(rank, world_size)
    head = rank == 0

    # prepare the dataloader
    train_dataloader, test_dataloader = prepare_dataloaders(
        rank, world_size, dataset=configs.dataset, batch_size=configs.batch_size
    )

    # set up model
    resnet = models.ResNet(configs).to_ddp(rank)
    resnet.set_up_optimizers()
    resnet.set_up_loss()

    # TODO: set up head to gather/log metrics
    mlflow.set_tracking_uri("http://tularosa.sci.utah.edu:5000")
    mlflow.set_experiment("bartali")
    with mlflow.start_run(run_name=configs.name):

        for epoch in range(1, configs.epochs + 1):
            if head:
                print(f"epoch {epoch} of {configs.epochs}")

            train_stats = resnet.train_epoch(train_dataloader, verbose=False)
            mlflow.log_metrics(train_stats, step=epoch)
            test_stats = resnet.test_epoch(test_dataloader, verbose=False)
            mlflow.log_metrics(test_stats, step=epoch)

            mlflow.log_metric(
                "learning_rate", resnet.scheduler.get_last_lr()[0], step=epoch
            )

            if head:
                resnet.scheduler.step()

            # TODO: save checkpoint

        print("done!")

    cleanup()


if __name__ == "__main__":
    # parse args/config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="config file location"
    )
    args, _ = parser.parse_known_args()

    configs = utils.parse_config_file(args.config)

    if configs.name == "random":
        configs.name = "baseline-" + namegenerator.gen()
    else:
        configs.name = "baseline-" + configs.name

    print(f"Run name: {configs.name}")
    try:
        os.mkdir(f"runs/{configs.name}")
    except FileExistsError as error:
        pass

    world_size = torch.cuda.device_count()

    # TODO: something is sitting on gpu0 for all procs, what is up with that?
    mp.spawn(main, args=(world_size, configs), nprocs=world_size, join=True)

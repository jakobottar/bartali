import os
import yaml
import argparse
import namegenerator
from collections import namedtuple
import mlflow
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import shutil
import sys
import tempfile
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import models
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_dataloader(
    rank, world_size, batch_size=32, pin_memory=False, num_workers=0
):
    transform = transforms.Compose(
        [
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.CIFAR10(
        "/scratch/jakobj/cifar", train=True, download=False, transform=transform
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )

    return dataloader


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def main(rank, world_size):
    print(f"model launched on gpu {rank}")
    # setup the process groups
    setup(rank, world_size)  # prepare the dataloader
    dataloader = prepare_dataloader(rank, world_size)

    model = models.resnet18(weights="DEFAULT").to(rank)
    model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-9)

    for epoch in range(10):
        dataloader.sampler.set_epoch(epoch)

        if rank == 0:
            total_loss = 0

        for _, (value, target) in enumerate(dataloader):
            value, target = value.to(rank), target.to(rank)

            optimizer.zero_grad()
            pred = model(value)
            loss = loss_fn(pred, target)
            if rank == 0:
                total_loss += loss.item()
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(total_loss / (len(dataloader)))

    cleanup()


if __name__ == "__main__":

    datasets.CIFAR10("/scratch/jakobj/cifar", train=True, download=True)

    world_size = torch.cuda.device_count()

    # TODO: something is sitting on gpu0 for all procs, what is up with that?
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

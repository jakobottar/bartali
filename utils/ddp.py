import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def setup(rank, world_size, port="29500"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_dataloaders(rank: int, world_size: int, configs):

    match configs.dataset:
        case "cifar":
            transform = transforms.Compose(
                [
                    # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                    transforms.ToTensor(),
                ]
            )
            train_dataset = datasets.CIFAR10(
                configs.dataset_location,
                train=True,
                download=False,
                transform=transform,
            )

            test_dataset = datasets.CIFAR10(
                configs.dataset_location,
                train=False,
                download=False,
                transform=transform,
            )

        case _:
            raise NotImplementedError(f"Cound not load dataset {configs.dataset}.")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs.batch_size,
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
        batch_size=configs.batch_size,
        pin_memory=False,
        drop_last=False,
        num_workers=0,
        sampler=test_sampler,
    )

    return train_dataloader, test_dataloader

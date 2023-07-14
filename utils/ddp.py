"""
ddp utils
"""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from .data import MagImageDataset, MultiplyBatchSampler


def setup(rank, world_size, port="1234"):
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}"
    )


def cleanup():
    dist.destroy_process_group()


def get_color_distortion(s=0.5):
    # s is the strength of color distortion.
    # given from https://arxiv.org/pdf/2002.05709.pdf
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


# lambdas cannot be pickled, so we have to define a function
class Clamp(object):
    def __call__(self, x):
        return torch.clamp(x, 0, 1)


def prepare_dataloaders(rank: int, world_size: int, configs):
    match configs.dataset:
        case "nova":
            image_size = 256
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.ColorJitter(brightness=0.5),
                    transforms.RandomResizedCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            train_dataset = MagImageDataset(
                configs.dataset_location,
                split="train_nova",
                transform=transform,
                get_all_views=False,
                ignore_views=("magnification" not in configs.transforms),
                drop_classes=configs.drop_classes,
                fold=configs.fold_num,
                fraction=configs.fraction,
            )

            test_dataset = MagImageDataset(
                configs.dataset_location,
                split="test_nova",
                transform=transform,
                get_all_views=configs.multi_mag_majority_vote,
                ignore_views=("magnification" not in configs.transforms),
                fold=configs.fold_num,
            )

        case "full":
            image_size = 256
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.ColorJitter(brightness=0.5),
                    transforms.RandomResizedCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            train_dataset = MagImageDataset(
                configs.dataset_location,
                split="train_full",
                transform=transform,
                get_all_views=False,
                ignore_views=("magnification" not in configs.transforms),
                drop_classes=configs.drop_classes,
                fold=configs.fold_num,
            )

            test_dataset = MagImageDataset(
                configs.dataset_location,
                split="test_full",
                transform=transform,
                get_all_views=configs.multi_mag_majority_vote,
                ignore_views=("magnification" not in configs.transforms),
                fold=configs.fold_num,
            )

        case _:
            raise NotImplementedError(f"Cound not load dataset {configs.dataset}.")

    assert configs.batch_size % world_size == 0
    configs.batch_size //= world_size

    if rank == 0:
        print(f"===> {world_size} gpus total; batch_size={configs.batch_size} per gpu")

    batch_sampler = MultiplyBatchSampler
    batch_sampler.multiplier = configs.multiplier

    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=True,
        num_workers=configs.workers,
        batch_sampler=batch_sampler(train_sampler, configs.batch_size, drop_last=False),
    )

    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_dataloader = DataLoader(
        test_dataset,
        pin_memory=True,
        num_workers=configs.workers,
        batch_sampler=batch_sampler(test_sampler, configs.batch_size, drop_last=False),
    )

    return train_dataloader, test_dataloader

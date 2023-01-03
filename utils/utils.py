from collections import namedtuple
from dataclasses import dataclass
import yaml


@dataclass
class ConfigStruct:
    arch: str  # resnet18 or resnet50, backbone model architecture
    name: str  # run name
    dataset: str  # cifar, dataset
    dataset_location: str  # dataset filepath
    batch_size: int  # int, batch size
    workers: int  # int, dataloader worker threads
    lr_schedule: float  # cosine-anneal or exponential, learning rate schedule
    epochs: int  # int, num training epochs
    lr: float  # float, learning rate
    weight_decay: float  # float, optimizer weight decay
    seed: int  # random seed, -1 for random
    gpus: list  # int or list, gpu(s) to use


def parse_config_file(filename: str):
    # load yaml config file
    with open(filename, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    return ConfigStruct(*config_dict.values())

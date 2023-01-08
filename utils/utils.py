from dataclasses import dataclass, asdict
import yaml


@dataclass
class ConfigStruct:
    arch: str = "resnet18"  # resnet18 or resnet50, backbone model architecture
    name: str = "random"  # run name
    dataset: str = "cifar"  # cifar, dataset
    dataset_location: str = "./data/"  # dataset filepath
    optimizer: str = "adam"  # optimizer
    batch_size: int = 8  # int, batch size
    workers: int = 0  # int, dataloader worker threads
    lr_schedule: str = "cosine-anneal"  # learning rate schedule
    epochs: int = 2  # int, num training epochs
    lr: float = 1.0  # float, learning rate
    tau: float = 1.0  # float, NTXent parameter
    multiplier: int = 2  # int, NTXent parameter
    weight_decay: float = 1e-9  # float, optimizer weight decay
    find_unused_parameters: bool = False  # should DDP find unused parameters
    seed: int = -1  # random seed, -1 for random
    gpus: tuple = (0,)  # tuple, gpu(s) to use
    port: str = "29500"  # DDP port

    as_dict = asdict


def parse_config_file(filename: str):
    # load yaml config file
    with open(filename, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    configs = ConfigStruct()
    for key, val in config_dict.items():
        setattr(configs, key, val)

    return configs


# TODO: cleaner way of doing this?
def roll_objects(objects: list, method="mean"):
    rolled = dict()

    for dev in objects:
        for stats in dev.values():
            for key, val in stats.items():
                if key not in rolled.keys():
                    rolled[key] = 0
                rolled[key] += val

    if method == "mean":
        for key in rolled.keys():
            rolled[key] /= len(objects)

    return rolled

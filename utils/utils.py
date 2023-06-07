from dataclasses import asdict, dataclass

import yaml


@dataclass
class ConfigStruct:
    """struct to store config parameters"""

    arch: str = "resnet18"  # resnet18 or resnet50, backbone model architecture
    name: str = "random"  # run name
    chkpt_file: str = "none"  # checkpoint to resume from
    dataset: str = "cifar"  # dataset
    dataset_location: str = "./data/"  # dataset filepath
    fold_num: int = 0  # data fold num, for nfs

    optimizer: str = "adam"  # optimizer
    batch_size: int = 8  # int, batch size
    workers: int = 0  # int, dataloader worker threads

    lr: float = 1.0  # float, learning rate
    encoder_lr: float = 1.0  # float, encoder learning rate, for tuning
    lr_schedule: str = "cosine-anneal"  # learning rate schedule
    lr_gamma: float = 0.99  # learning rate scheduler parameter
    weight_decay: float = 1e-9  # float, optimizer weight decay
    epochs: int = 2  # int, num training epochs
    frozen_epochs: int = 20  # num frozen epochs, for tuning

    tau: float = 1.0  # float, NTXent parameter
    multiplier: int = 1  # int, NTXent parameter, set to 1 when not using simclr

    ce_loss_weight: float = 1  # float, supervised loss weight, for simreg
    ntxent_loss_weight: float = 1  # float, self-supervised loss weight, for simreg

    find_unused_parameters: bool = False  # should DDP find unused parameters
    multi_mag_majority_vote: bool = (
        False  # should eval use majority vote on all magnifications?
    )
    drop_classes: tuple = ()
    transforms: tuple = (
        "magnification",
        "horiz_flip",
        "vert_flip",
        "random_resized_crop",
        "normalize",
    )

    seed: int = -1  # random seed, -1 for random
    gpus: str | tuple = (0,)  # str or tuple, gpu(s) to use
    port: str = "1234"  # DDP port
    root: str = "runs"  # root of folder to save runs in
    mode: str = "frozen"  # model training mode, for simclr eval

    as_dict = asdict


def parse_config_file(filename: str):
    """load yaml config file"""
    with open(filename, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    configs = ConfigStruct()
    for key, val in config_dict.items():
        setattr(configs, key, val)

    return configs


def parse_config_file_and_overrule(filename: str, args):
    """if command line args are used, they will overrule the config file"""
    configs = parse_config_file(filename)

    for key, val in vars(args).items():
        if val:
            setattr(configs, key, val)

    return configs


# TODO: cleaner way of doing this?
def roll_objects(objects: list, method="mean"):
    """combine objects from all_gather into one dict"""
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

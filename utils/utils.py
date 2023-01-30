from dataclasses import dataclass, asdict
import yaml

import torch
from torch import Tensor


@dataclass
class ConfigStruct:
    arch: str = "resnet18"  # resnet18 or resnet50, backbone model architecture
    name: str = "random"  # run name
    chkpt_file: str = "./model.pth"  # checkpoint to resume from
    dataset: str = "cifar"  # cifar, dataset
    dataset_location: str = "./data/"  # dataset filepath

    optimizer: str = "adam"  # optimizer
    batch_size: int = 8  # int, batch size
    workers: int = 0  # int, dataloader worker threads

    lr: float = 1.0  # float, learning rate
    lr_schedule: str = "cosine-anneal"  # learning rate schedule
    lr_gamma: float = 0.99  # learning rate scheduler parameter
    epochs: int = 2  # int, num training epochs
    tau: float = 1.0  # float, NTXent parameter
    multiplier: int = 1  # int, NTXent parameter, set to 1 when not using simclr
    weight_decay: float = 1e-9  # float, optimizer weight decay

    find_unused_parameters: bool = False  # should DDP find unused parameters
    multi_mag_majority_vote: bool = (
        False  # should eval use majority vote on all magnifications?
    )

    seed: int = -1  # random seed, -1 for random
    gpus: str | tuple = (0,)  # str or tuple, gpu(s) to use
    port: str = "29500"  # DDP port
    root: str = "runs"  # root of folder to save runs in

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


def variance(model_pred: Tensor, branch_preds: Tensor) -> Tensor:
    """
    Computes variance metric for uncertainty quantification.
    Parameters
    ----------
    model_pred : `torch.Tensor`, voted or averaged prediction for ensemble model
    branch_preds : `torch.Tensor`, predictions for each branch or run, should be of shape `E * N`
    """

    var = 0.0
    for branch_pred in branch_preds:
        var += torch.sum(branch_pred != model_pred)

    return var / (branch_preds.shape[0] * branch_preds.shape[1])


def entropy(logits, base="two"):
    """compute entropy of logits"""
    if base == "two" or base == 2:
        # use nansum because 0 predictions give nan values (and log(0) = nan)
        return -torch.nansum(logits * torch.log2(logits), dim=1)
    elif base == "ten" or base == 10:
        return -torch.nansum(logits * torch.log10(logits), dim=1)
    elif base == "natural" or base == "e":
        return -torch.nansum(logits * torch.log(logits), dim=1)
    else:
        raise NotImplementedError(f"No log found for base {base}")

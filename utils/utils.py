from dataclasses import dataclass, asdict
import yaml


@dataclass
class ConfigStruct:
    # TODO: improve this so it doesn't break if args are out of order
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

    dict = asdict


def parse_config_file(filename: str):
    # load yaml config file
    with open(filename, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    return ConfigStruct(*config_dict.values())


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

"""
simclr model
"""
import os
import random
import argparse
import socket

import namegenerator
import mlflow

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist

import models
import utils


def worker(rank, world_size, configs):
    """
    single worker function
    """
    print(
        f"===> launched proc {rank}/{world_size}@{socket.gethostname()}",
        flush=True,
    )
    # setup the process groups
    utils.setup(rank, world_size, configs.port)
    # set device (for all_gather)
    torch.cuda.set_device(configs.gpus[rank])

    # prepare the dataloader
    train_dataloader, test_dataloader = utils.prepare_dataloaders(
        rank, world_size, configs
    )

    # set up model
    simclr = models.SimCLR(configs).to_ddp(rank)

    cudnn.benchmark = True

    if rank == 0:
        mlflow.set_tracking_uri("http://tularosa.sci.utah.edu:5000")
        mlflow.set_experiment("bartali")
        mlflow.start_run(run_name=configs.name)
        mlflow.log_params(configs.as_dict())

    # make structures for all_gather
    data = {
        "train_stats": None,
        "test_stats": None,
    }
    outputs = [None for _ in range(world_size)]

    for epoch in range(1, configs.epochs + 1):
        if rank == 0:
            print(f"epoch {epoch} of {configs.epochs}")

        data["train_stats"] = simclr.train_epoch(train_dataloader, verbose=False)
        data["test_stats"] = simclr.test_epoch(test_dataloader, verbose=False)

        dist.all_gather_object(outputs, data)

        if rank == 0:
            mlflow.log_metrics(utils.roll_objects(outputs), step=epoch)
            mlflow.log_metric(
                "learning_rate", simclr.scheduler.get_last_lr()[0], step=epoch
            )
            torch.save(simclr.get_ckpt(), f"{configs.root}/checkpoint-{epoch}.pth")

    if rank == 0:
        torch.save(simclr.get_ckpt(), f"{configs.root}/model.pth")

    print("done!")
    utils.cleanup()


if __name__ == "__main__":
    # parse args/config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="config file location"
    )
    args, _ = parser.parse_known_args()

    configs = utils.parse_config_file(args.config)

    if configs.seed != -1:
        random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        cudnn.deterministic = True

    if configs.name == "random":
        configs.name = "simclr-" + namegenerator.gen()
    else:
        configs.name = "simclr-" + configs.name

    print(f"Run name: {configs.name}")
    try:
        os.mkdir(f"{configs.root}/{configs.name}")
    except FileExistsError as error:
        pass
    configs.root = f"{configs.root}/{configs.name}"

    world_size = len(configs.gpus)

    mp.spawn(worker, args=(world_size, configs), nprocs=world_size, join=True)

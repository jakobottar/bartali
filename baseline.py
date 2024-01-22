"""
baseline resnet model
"""
import argparse
import os
import random
import shutil
import socket
import time

import mlflow
import namegenerator
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

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
    resnet = models.ResNet(configs).to_ddp(rank)

    cudnn.benchmark = True

    if rank == 0:
        mlflow.set_tracking_uri("http://tularosa.sci.utah.edu:5000")
        mlflow.set_experiment("bartali")
        mlflow.start_run(run_name=configs.name)
        mlflow.log_params(configs.as_dict())
        best_metric = -9999

    # make structures for all_gather
    data = {
        "train_stats": None,
        "test_stats": None,
    }
    outputs = [None for _ in range(world_size)]

    for epoch in range(1, configs.epochs + 1):
        if rank == 0:
            print(f"epoch {epoch} of {configs.epochs} ", end="")
            start_time = time.time()

        data["train_stats"] = resnet.train_epoch(train_dataloader, verbose=False)
        data["test_stats"] = resnet.test_epoch(test_dataloader, verbose=False)

        dist.all_gather_object(outputs, data)

        if rank == 0:
            metrics = utils.roll_objects(outputs)
            mlflow.log_metrics(metrics, step=epoch)
            mlflow.log_metric(
                "learning_rate", resnet.scheduler.get_last_lr()[0], step=epoch
            )
            if metrics["val_acc"] > best_metric:
                torch.save(resnet.get_ckpt(), f"{configs.root}/best.pth")
            print(f"{time.time() - start_time:.2f} sec")

        resnet.scheduler.step()

    if rank == 0:
        # torch.save(resnet.get_ckpt(), f"{configs.root}/last.pth")
        mlflow.pytorch.log_model(
            resnet.get_model(), "model", pip_requirements="requirements.txt"
        )
        mlflow.log_artifacts(configs.root)

    print("done!")
    utils.cleanup()


if __name__ == "__main__":
    # parse args/config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="config file location"
    )
    parser.add_argument("-e", "--epochs", type=int, default=None)
    parser.add_argument("--fold-num", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--chkpt-file", type=str, default=None)
    parser.add_argument("--drop-classes", type=str, default=None, nargs="+")
    args, _ = parser.parse_known_args()
    configs = utils.parse_config_file_and_overrule(args.config, args)

    if configs.seed != -1:
        random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        cudnn.deterministic = True

    if configs.name == "random":
        configs.name = "baseline-" + namegenerator.gen()
    else:
        configs.name = "baseline-" + configs.name

    print(f"Run name: {configs.name}")
    try:
        os.mkdir(f"runs/{configs.name}")
    except FileExistsError as error:
        pass
    configs.root = f"{configs.root}/{configs.name}"
    shutil.copy(args.config, os.path.join(configs.root, "config.yaml"))

    world_size = len(configs.gpus)

    mp.spawn(worker, args=(world_size, configs), nprocs=world_size, join=True)

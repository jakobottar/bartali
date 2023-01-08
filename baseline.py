"""
baseline resnet model
"""
import os
import argparse
import namegenerator
import mlflow

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import models
import utils


def main(rank, world_size, configs):
    print(f"model launched on gpu {configs.gpus[rank]}")
    # setup the process groups
    utils.setup(rank, world_size, configs.port)
    # set device (for all_gather)
    torch.cuda.set_device(configs.gpus[rank])
    # is head proc?
    is_head = rank == 0

    # prepare the dataloader
    train_dataloader, test_dataloader = utils.prepare_dataloaders(
        rank, world_size, configs
    )

    # set up model
    resnet = models.ResNet(configs).to_ddp(rank)
    resnet.set_up_optimizers()
    resnet.set_up_loss()

    if is_head:
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
        if is_head:
            print(f"epoch {epoch} of {configs.epochs}")

        data["train_stats"] = resnet.train_epoch(train_dataloader, verbose=False)
        data["test_stats"] = resnet.test_epoch(test_dataloader, verbose=False)

        dist.all_gather_object(outputs, data)

        if is_head:
            mlflow.log_metrics(utils.roll_objects(outputs), step=epoch)
            mlflow.log_metric(
                "learning_rate", resnet.scheduler.get_last_lr()[0], step=epoch
            )
            torch.save(resnet.get_ckpt(), f"runs/{configs.name}/checkpoint-{epoch}.pth")

        # resnet.scheduler.step()

    if is_head:
        torch.save(resnet.get_ckpt(), f"runs/{configs.name}/model.pth")

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

    if configs.name == "random":
        configs.name = "baseline-" + namegenerator.gen()
    else:
        configs.name = "baseline-" + configs.name

    print(f"Run name: {configs.name}")
    try:
        os.mkdir(f"runs/{configs.name}")
    except FileExistsError as error:
        pass

    world_size = len(configs.gpus)

    mp.spawn(main, args=(world_size, configs), nprocs=world_size, join=True)

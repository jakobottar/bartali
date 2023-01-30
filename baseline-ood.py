# pylint: disable=redefined-outer-name, C0103
"""
Baseline model, uses max softmax, as in Hendrycks and Gimpel 2018
"""

import os
import shutil
import argparse
import namegenerator
import matplotlib.pyplot as plt
import mlflow
from tqdm import tqdm
import sklearn.metrics as sk

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models, datasets, transforms

import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs", type=int, default=5, help="# of training epochs (default: 5)"
)
parser.add_argument("--gpu-id", type=int, default=0, help="gpu id (default: 0)")
parser.add_argument(
    "--name", type=str, default=None, help="model name (default: random)"
)
parser.add_argument(
    "--dataset", type=str, default="nfs", help="dataset name: nfs (default) or cifar"
)
FLAGS = parser.parse_args()

# Get cpu or gpu device for training.
DEVICE = f"cuda:{FLAGS.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

if FLAGS.name is None:
    NAME = f"base-ood-{namegenerator.gen()}"
else:
    NAME = FLAGS.name

print(f"Run name: {NAME}")
try:
    os.mkdir(f"runs/{NAME}")
except FileExistsError as error:
    pass


def train(model, train_loader, optimizer, loss_fn, epoch):
    """
    Train model
    """
    model.train()

    iter_loader = iter(train_loader)
    tbar_loader = tqdm(iter_loader, ncols=shutil.get_terminal_size().columns)

    correct, mean_entropy = 0, 0.0
    for batch_idx, (images, labels) in enumerate(tbar_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # get model predictions
        logits = model(images)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        logits = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        correct += preds.eq(labels.view_as(preds)).sum().item()
        mean_entropy += utils.entropy(logits).mean()

        if batch_idx % 10 == 0:
            itr = (epoch * len(iter_loader)) + batch_idx
            mlflow.log_metric("loss", loss.item(), step=itr)

        tbar_loader.set_description(f"train: {epoch}, l: {loss.item():.4f}")

    acc = correct / len(train_loader.dataset)
    mean_entropy /= len(train_loader)

    mlflow.log_metrics(
        {"train_acc": float(acc), "train_entropy": float(mean_entropy)},
        step=epoch,
    )


def test(model, test_loader, ood_loader, epoch):
    """
    Test model
    """
    model.eval()

    # load in-distribution data
    iter_test_loader = iter(test_loader)
    tbar_loader = tqdm(iter_test_loader, ncols=shutil.get_terminal_size().columns)

    entropy_right, entropy_wrong, conf_right, conf_wrong = [], [], [], []
    with torch.no_grad():
        for images, labels in tbar_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # put images through model
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            r = preds.eq(labels.view_as(preds))  # idx of correct preds

            entropy_right.append(utils.entropy(logits[r]))
            entropy_wrong.append(utils.entropy(logits[~r]))

            conf_right.append(torch.max(logits[r], dim=1).values)
            conf_wrong.append(torch.max(logits[~r], dim=1).values)

            tbar_loader.set_description(f"test d_in: {epoch}")

    entropy_right = torch.cat(entropy_right, dim=0)
    entropy_wrong = torch.cat(entropy_wrong, dim=0)
    conf_right = torch.cat(conf_right, dim=0)
    conf_wrong = torch.cat(conf_wrong, dim=0)

    labels = torch.cat(
        [
            torch.ones(conf_right.shape, dtype=int),
            torch.zeros(conf_wrong.shape, dtype=int),
        ]
    )
    values = torch.cat([conf_right, conf_wrong])

    auroc_in = sk.roc_auc_score(labels.cpu(), values.cpu())

    mlflow.log_metrics(
        {
            "val_acc": len(conf_right) / (len(conf_right) + len(conf_wrong)),
            "val_entropy_right": float(entropy_right.mean()),
            "val_entropy_wrong": float(entropy_wrong.mean()),
            "val_entropy_all": float((entropy_right.mean() + entropy_wrong.mean()) / 2),
            "val_conf_right": float(conf_right.mean()),
            "val_conf_wrong": float(conf_wrong.mean()),
            "val_conf_all": float((conf_right.mean() + conf_wrong.mean()) / 2),
            "val_auroc_in": auroc_in,
        },
        step=epoch,
    )

    # load out-of-distribution data
    iter_ood_loader = iter(ood_loader)
    tbar_loader = tqdm(iter_ood_loader, ncols=shutil.get_terminal_size().columns)

    entropy_ood, conf_ood = [], []
    with torch.no_grad():
        for images in tbar_loader:
            images = images.to(DEVICE)

            # put images through model
            logits = model(images)
            logits = F.softmax(logits, dim=1)

            entropy_ood.append(utils.entropy(logits))

            conf_ood.append(torch.max(logits, dim=1).values)

            tbar_loader.set_description(f"test d_out: {epoch}")

    entropy_ood = torch.cat(entropy_ood, dim=0)
    conf_ood = torch.cat(conf_ood, dim=0)

    labels = torch.cat(
        [
            torch.ones(conf_right.shape, dtype=int),
            torch.zeros(conf_ood.shape, dtype=int),
        ]
    )
    values = torch.cat([conf_right, conf_ood])

    auroc_out = sk.roc_auc_score(labels.cpu(), values.cpu())

    mlflow.log_metrics(
        {
            "val_entropy_ood": float(entropy_ood.mean()),
            "val_conf_ood": float(conf_ood.mean()),
            "val_auroc_out": auroc_out,
        },
        step=epoch,
    )

    torch.save(model, f"runs/{NAME}/model.pth")


def plot_from_dl(model, dataloader, filename, title):
    """
    plot examples of output from given dataloader
    """

    model.eval()

    class_key = [
        "SDU-U3O8",
        "AUCd-UO2",
        "AUC-UO3",
        "MDU-U3O8",
        "SDU-UO2",
        "AUC-U3O8",
        "AUCi-UO2",
        "UO4-UO2",
        "MDU-UO3",
        "ADU-UO2",
        "UO4-UO3",
        "ADU-U3O8",
        "SDU-UO3",
        "UO4-U3O8",
        "ADU-UO3",
        "MDU-UO2",
        "n/a",
    ]

    with torch.no_grad():
        batch = next(iter(dataloader))
        # handle normal dataloader case
        if isinstance(batch, list):
            images, labels = batch
        else:  # handle OOD dataloader case
            images = batch
            labels = [16 for _ in range(len(images))]

        images = images.to(DEVICE)

        # put images through model
        logits = model(images)
        logits = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        entropy = utils.entropy(logits)
        conf = torch.max(logits, dim=1).values

        figure = plt.figure(figsize=(8, 4))
        plt.suptitle(title)

        for i in range(3):

            figure.add_subplot(1, 3, i + 1)
            plt.title(
                f"real: {class_key[labels[i]]}\npred: {class_key[preds[i]]}\nent: {entropy[i]:.2f}, conf: {conf[i]:.2f}"
            )
            plt.axis("off")
            plt.imshow(images[i].cpu().numpy().transpose(1, 2, 0))

        mlflow.log_figure(figure, filename)
        plt.savefig(filename)


# =============================================================================================

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://tularosa.sci.utah.edu:5000")
    mlflow.set_experiment("bartali2")
    with mlflow.start_run(run_name=NAME):
        LEARNING_RATE = 0.001
        LR_GAMMA = 0.97
        BACKBONE = "densenet121"

        if FLAGS.dataset == "cifar":
            BATCH_SIZE = 256

            train_dataset = datasets.CIFAR10(
                "/scratch/jakobj/cifar",
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )
            test_dataset = datasets.CIFAR10(
                "/scratch/jakobj/cifar",
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
            ood_dataset = datasets.SVHN(
                "/scratch/jakobj/cifar",
                download=True,
                transform=transforms.ToTensor(),
            )

        if FLAGS.dataset == "nfs":
            BATCH_SIZE = 64

            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        256,
                        scale=(0.08, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    utils.Clamp(),
                ]
            )

            train_dataset = utils.MagImageDataset(
                "/scratch/jakobj/multimag/", train=True, transform=transform
            )
            test_dataset = utils.MagImageDataset(
                "/scratch/jakobj/multimag/", train=False, transform=transform
            )
            ood_dataset = utils.OODDataset(
                "/scratch/jakobj/multimag/", transform=transform
            )

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
        )
        ood_dataloader = DataLoader(
            ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
        )

        mlflow.log_params(
            {
                "inital_lr": LEARNING_RATE,
                "lr_gamma": LR_GAMMA,
                "batch_size": BATCH_SIZE,
                "training_epochs": FLAGS.epochs,
                "device": DEVICE,
                "conf_metric": "max_softmax",
                "dataset": FLAGS.dataset,
                "backbone": BACKBONE,
            }
        )

        X, y = next(iter(train_dataloader))
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")

        if BACKBONE == "resnet34":
            model = models.resnet34(weights="DEFAULT")
        elif BACKBONE == "resnet50":
            model = models.resnet50(weights="DEFAULT")
        elif BACKBONE == "resnet101":
            model = models.resnet101(weights="DEFAULT")
        elif BACKBONE == "densenet121":
            model = models.densenet121(weights="DEFAULT")
        else:
            raise NotImplementedError("No model found for backbone {BACKBONE}")

        model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
        loss_fn = nn.CrossEntropyLoss()

        print(f"\nTraining network for {FLAGS.epochs} epochs")
        for epoch in range(FLAGS.epochs):
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)
            train(model, train_dataloader, optimizer, loss_fn, epoch)
            if epoch % 5 == 0 or epoch == FLAGS.epochs - 1:
                test(model, test_dataloader, ood_dataloader, epoch)
            scheduler.step()

        print("Done!")
        mlflow.pytorch.log_model(
            model, f"runs/{NAME}/model.pth", pip_requirements="requirements.txt"
        )

        plot_from_dl(
            model,
            train_dataloader,
            f"runs/{NAME}/train_examples.png",
            "Training Examples",
        )
        plot_from_dl(
            model, test_dataloader, f"runs/{NAME}/test_examples.png", "Testing Examples"
        )
        plot_from_dl(
            model, ood_dataloader, f"runs/{NAME}/ood_examples.png", "OOD Examples"
        )

        mlflow.end_run()

"""
data utils
"""

import json
import os
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler

ROUTES = [
    "U3O8ADU",
    "U3O8AUC",
    "U3O8MDU",
    "U3O8SDU",
    "UO2ADU",
    "UO2AUCd",
    "UO2AUCi",
    "UO2SDU",
    "UO3ADU",
    "UO3AUC",
    "UO3MDU",
    "UO3SDU",
]


class MultiplyBatchSampler(BatchSampler):
    multiplier = 2

    def __iter__(self):
        for batch in super().__iter__():
            yield batch * self.multiplier


class MagImageDataset(Dataset):
    """Custom processing route dataset"""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        get_all_views: bool = False,
        ignore_views: bool = False,
        drop_classes=[],
        fold=0,
    ) -> None:
        super().__init__()
        match split:
            case "train_full":
                with open(
                    os.path.join(root, f"full_train_{fold}.json"), "r", encoding="utf-8"
                ) as f:
                    self.df = json.load(f)
            case "test_full" | "val_full":
                with open(
                    os.path.join(root, f"full_val_{fold}.json"), "r", encoding="utf-8"
                ) as f:
                    self.df = json.load(f)
            case "train_nova":
                with open(
                    os.path.join(root, f"nova_train_{fold}.json"), "r", encoding="utf-8"
                ) as f:
                    self.df = json.load(f)
            case "test_nova" | "val_nova":
                with open(
                    os.path.join(root, f"nova_val_{fold}.json"), "r", encoding="utf-8"
                ) as f:
                    self.df = json.load(f)
            case _:
                raise ValueError(
                    "Not a valid split, try 'train', 'test', 'val', or 'ood'"
                )

        # filter df
        self.df = [s for s in self.df if s["route"] not in drop_classes]

        # ignore magnification, concat dataframe
        self.ignore_views = ignore_views
        if ignore_views:
            raise NotImplementedError
            if get_all_mag:
                print("Warning: `get_all_mag` is overridden by `no_mag`!")

            temp_df = pd.DataFrame()
            for mag in MAGS:
                new_df = self.df[[mag, "label"]].rename({mag: "filename"}, axis=1)
                temp_df = pd.concat([temp_df, new_df], ignore_index=True)

            self.df = temp_df

        self.split = split
        self.transform = transform
        self.get_all_views = get_all_views
        self.root = os.path.join(root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df[idx]
        sample["route_int"] = int(ROUTES.index(sample["route"]))

        # ignores magnification
        if self.ignore_views:
            raise NotImplementedError
            image_path = os.path.join(self.root, sample["filename"])
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

        # get all views
        elif self.get_all_views:
            image = []
            for file in sample["files"]:
                image_path = os.path.join(self.root, file)
                image_mag = Image.open(image_path).convert("RGB")

                if self.transform:
                    image_mag = self.transform(image_mag)

                image.append(image_mag)

        # get a random view
        else:
            image_path = os.path.join(self.root, random.choice(sample["files"]))
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

        return image, sample["route_int"]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                256,
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )

    dataset = MagImageDataset(
        "/scratch_nvme/jakobj/multimag",
        split="train_nova",
        transform=transform,
        get_all_views=True,
        drop_classes=["UO3AUC", "U3O8MDU"],
    )

    sampler = RandomSampler(dataset)

    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        num_workers=2,
        batch_sampler=BatchSampler(sampler, batch_size=2, drop_last=False),
    )

    print(len(dataset))
    print(next(iter(dataloader)))

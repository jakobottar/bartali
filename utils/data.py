"""
data utils
"""

import os
import random
import pandas as pd
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import Dataset

from PIL import Image

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

MAGS = ["10000x", "25000x", "50000x", "100000x"]


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
        get_all_mag: bool = False,
        ood_classes=[],
    ) -> None:
        super().__init__()
        match split:
            case "train":
                self.df = pd.read_csv(os.path.join(root, "train_0.csv"))
                self.df = self.df[self.df.label.isin(ood_classes) == False]
            case "test" | "val":
                self.df = pd.read_csv(os.path.join(root, "val_0.csv"))
                self.df = self.df[self.df.label.isin(ood_classes) == False]
            case "ood":
                train_half = pd.read_csv(os.path.join(root, "train_0.csv"))
                val_half = pd.read_csv(os.path.join(root, "val_0.csv"))
                self.df = pd.concat([train_half, val_half])
                self.df = self.df[self.df.label.isin(ood_classes)]
            case _:
                raise ValueError(
                    "Not a valid split, try 'train', 'test', 'val', or 'ood'"
                )
        self.df = self.df.reset_index()

        self.split = split
        self.transform = transform
        self.all_mag = get_all_mag
        self.root = os.path.join(root)

    def __len__(self):
        return len(self.df) - 2

    def __getitem__(self, idx):
        row = self.df.loc[idx].to_dict()
        row["label_int"] = int(ROUTES.index(row["label"]))

        # get all magnifications
        if self.all_mag:
            image = []
            for mag in range(4):
                image_path = os.path.join(self.root, row[MAGS[mag]])
                image_mag = Image.open(image_path).convert("RGB")

                if self.transform:
                    image_mag = self.transform(image_mag)

                image.append(image_mag)

        # get a random magnification
        else:
            rand_mag = random.randint(0, 3)
            image_path = os.path.join(self.root, row[MAGS[rand_mag]])
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

        if self.split == "ood":
            return image

        return image, row["label_int"]


class OODDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform=None,
        get_all_mag: bool = False,
        random_noise: bool = False,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(f"{root}/ood.csv")
        self.transform = transform
        self.all_mag = get_all_mag
        self.random_noise = random_noise
        self.root = os.path.join(root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.random_noise:
            pass
        else:
            files = self.df.iloc[idx].dropna().values

            # get all magnifications
            if self.all_mag:
                image = []
                for filepath in files:
                    filepath = os.path.join(self.root, filepath)
                    image_mag = Image.open(filepath).convert("RGB")

                    if self.transform:
                        image_mag = self.transform(image_mag)

                    image.append(image_mag)

            # get a random magnification
            else:
                rand_mag = random.randint(0, len(files) - 1)
                image_path = os.path.join(self.root, files[rand_mag])
                image = Image.open(image_path).convert("RGB")

                if self.transform:
                    image = self.transform(image)

        return image


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
        "/scratch/jakobj/multimag",
        get_all_mag=False,
        ood_classes=["UO3AUC", "U3O8MDU"],
        transform=transform,
    )

    sampler = RandomSampler(dataset)

    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        num_workers=2,
        batch_sampler=BatchSampler(sampler, batch_size=64, drop_last=True),
    )

    count = 0
    for img, label in iter(dataloader):
        count += len(label)
        print(f"batch size: {len(label)}, total seen: {count}")

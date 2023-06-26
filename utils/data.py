"""
data utils
"""

import os
import random
import shutil

import numpy as np
import pandas as pd
import pyxis as px
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

VIEWS = ["10000x", "25000x", "50000x", "100000x"]


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
        no_mag: bool = False,
        ood_classes=[],
        fold=0,
    ) -> None:
        super().__init__()
        match split:
            case "train":
                self.df = pd.read_csv(os.path.join(root, f"train_{fold}.csv"))
                self.df = self.df[self.df.label.isin(ood_classes) == False]
            case "test" | "val":
                self.df = pd.read_csv(os.path.join(root, f"val_{fold}.csv"))
                self.df = self.df[self.df.label.isin(ood_classes) == False]
            case "ood":
                train_half = pd.read_csv(os.path.join(root, f"train_{fold}.csv"))
                val_half = pd.read_csv(os.path.join(root, f"val_{fold}.csv"))
                self.df = pd.concat([train_half, val_half])
                self.df = self.df[self.df.label.isin(ood_classes)]
            case _:
                raise ValueError(
                    "Not a valid split, try 'train', 'test', 'val', or 'ood'"
                )
        self.df = self.df.reset_index()

        # ignore magnification, concat dataframe
        self.no_mag = no_mag
        if no_mag:
            if get_all_mag:
                print("Warning: `get_all_mag` is overridden by `no_mag`!")

            temp_df = pd.DataFrame()
            for mag in VIEWS:
                new_df = self.df[[mag, "label"]].rename({mag: "filename"}, axis=1)
                temp_df = pd.concat([temp_df, new_df], ignore_index=True)

            self.df = temp_df

        self.split = split
        self.fold = fold
        self.transform = transform
        self.all_mag = get_all_mag
        self.ood_classes = ood_classes
        self.root = os.path.join(root)

        self.db = px.Reader(self.__make_lmdb(), lock=False)

    def __make_lmdb(self, force_regen=False):
        dirpath = os.path.join(
            self.root,
            f"temp_{self.split}_{self.fold}_{'no_mag' if self.no_mag else 'mag'}_{'_'.join(self.ood_classes)}",
        )
        if os.path.isdir(dirpath):
            if not force_regen:
                return dirpath
            shutil.rmtree(dirpath)
        with px.Writer(
            dirpath=dirpath,
            map_size_limit=11000,
        ) as db:
            print("making LMDB...")
            for _, row in self.df.iterrows():
                db_row = {"label": np.array([row["label"]], dtype=str)}

                images = []
                for view in VIEWS:
                    imagepath = os.path.join(self.root, row[view])
                    image = np.array(Image.open(imagepath).convert("RGB"))
                    # TODO: might need to reshape the image here
                    images.append(image)

                images = np.array([images])
                db_row["views"] = images

                # print(db_row["label"])
                db.put_samples(db_row)

            return db.dirpath

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.db.get_sample(idx)
        target = int(ROUTES.index(sample["label"]))

        # ignores magnification
        if self.no_mag:
            raise (NotImplementedError())
            # image_path = os.path.join(self.root, row["filename"])
            # image = Image.open(image_path).convert("RGB")

            # if self.transform:
            #     image = self.transform(image)

        # get all magnifications
        elif self.all_mag:
            image = []
            for mag in range(4):
                image_mag = Image.fromarray(sample["views"][mag])

                if self.transform:
                    image_mag = self.transform(image_mag)

                image.append(image_mag)

        # get a random magnification
        else:
            rand_mag = random.randint(0, 3)
            image = Image.fromarray(sample["views"][rand_mag])

            if self.transform:
                image = self.transform(image)

        if self.split == "ood":
            return image

        return image, target


# class OODDataset(Dataset):
#     def __init__(
#         self,
#         root: str,
#         transform=None,
#         get_all_mag: bool = False,
#         random_noise: bool = False,
#     ) -> None:
#         super().__init__()
#         self.df = pd.read_csv(f"{root}/ood.csv")
#         self.transform = transform
#         self.all_mag = get_all_mag
#         self.random_noise = random_noise
#         self.root = os.path.join(root)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         if self.random_noise:
#             pass
#         else:
#             files = self.df.iloc[idx].dropna().values

#             # get all magnifications
#             if self.all_mag:
#                 image = []
#                 for filepath in files:
#                     filepath = os.path.join(self.root, filepath)
#                     image_mag = Image.open(filepath).convert("RGB")

#                     if self.transform:
#                         image_mag = self.transform(image_mag)

#                     image.append(image_mag)

#             # get a random magnification
#             else:
#                 rand_mag = random.randint(0, len(files) - 1)
#                 image_path = os.path.join(self.root, files[rand_mag])
#                 image = Image.open(image_path).convert("RGB")

#                 if self.transform:
#                     image = self.transform(image)

#         return image


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
        split="ood",
        transform=transform,
        get_all_mag=False,
        ood_classes=["UO3AUC", "U3O8MDU"],
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

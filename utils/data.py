"""
data utils
"""

import os
import random
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset

# from torchvision import io
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


class MultiplyBatchSampler(BatchSampler):
    multiplier = 2

    def __iter__(self):
        for batch in super().__iter__():
            yield batch * self.multiplier


class MagImageDataset(Dataset):
    """Custom processing route dataset"""

    def __init__(self, root: str, train: bool = True, transform=None) -> None:
        super().__init__()
        self.images, self.labels = self.__parse_datafile(
            os.path.join(root, "train.txt" if train else "test.txt")
        )
        self.transform = transform
        self.root = os.path.join(root)

    # TODO: I'm not a huge fan of the data file format, can I make it better?
    def __parse_datafile(self, datafile) -> None:
        """
        Parse train/val text file into a dictionary input
        -> train/val text file template
            mag0/img0 mag1/img0 ... label
            mag0/img1 mag1/img1 ... label
            ...
            mag0/imgN mag1/imgN ... label
        """

        with open(datafile, "r") as file:
            img_files = {}
            labels = []

            curr_val = file.readline().strip("\n").split(" ")
            for vi, v in enumerate(curr_val[:-1]):
                img_files[f"mag{vi}"] = [v]

            labels.append(ROUTES.index(curr_val[-1]))

            for line in file:
                try:
                    curr_val = line.strip("\n").split(" ")
                except ValueError:  # Adhoc for test.
                    print("Incompatible text format in data file!")

                ## Obtain file name for each magnification input ##
                for vi, v in enumerate(curr_val[:-1]):
                    img_files[f"mag{vi}"].append(v)

                labels.append(ROUTES.index(curr_val[-1]))

            return img_files, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rand_mag = random.randint(0, 3)
        image_path = os.path.join(self.root, self.images[f"mag{rand_mag}"][idx])
        image = Image.open(image_path).convert("RGB")
        # image = io.read_image(image_path, mode=io.ImageReadMode.RGB).float() / 255
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    dataset = RouteImageDataset("/scratch/jakobj/multimag/")
    print(next(iter(dataset)))

"""
data utils
"""

import os
import pandas
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset
from torchvision import io


class MultiplyBatchSampler(BatchSampler):
    multiplier = 2

    def __iter__(self):
        for batch in super().__iter__():
            yield batch * self.multiplier


class RouteImageDataset(Dataset):
    """Custom processing route dataset"""

    def __init__(self, root: str, train: bool = True, transform=None) -> None:
        super().__init__()
        self.dataset = pandas.read_csv(os.path.join(root, "processed.csv"))

        test_split = int(len(self.dataset) * 0.25)
        self.dataset = self.dataset[test_split:] if train else self.dataset[:test_split]
        self.transform = transform
        self.root = os.path.join(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.dataset.iloc[idx]["filename"])
        image = io.read_image(image_path, mode=io.ImageReadMode.RGB).float() / 255
        label = int(self.dataset.iloc[idx]["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

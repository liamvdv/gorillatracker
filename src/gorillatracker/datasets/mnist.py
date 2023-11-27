from typing import Literal

from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST


class MNISTDataset(Dataset):
    def __init__(self, data_dir, partition: Literal["train", "val", "test"], transform=None):
        """
        Assumes directory structure:
            data_dir/
                train/
                    ...
                val/
                    ...
                test/
                    ...
        """
        self.partition = partition
        if partition in ("train", "val"):
            self.train, self.val = random_split(
                MNIST(data_dir, train=True, download=True, transform=transform), [0.8, 0.2]
            )
        elif partition == "test":
            self.test = MNIST(data_dir, train=False, download=True, trainsform=transform)
        else:
            raise ValueError(f"unknown partition '{partition}'")

    def __len__(self):
        return len(getattr(self, self.partition))

    def __getitem__(self, idx):
        return getattr(self, self.partition)[idx]

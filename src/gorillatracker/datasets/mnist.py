from typing import Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import gorillatracker.type_helper as gtypes


class MNISTDataset(Dataset[Tuple[gtypes.Id, torch.Tensor, gtypes.Label]]):
    def __init__(
        self, data_dir: str, partition: Literal["train", "val", "test"], transform: Optional[gtypes.Transform] = None
    ) -> None:
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
            self.test = MNIST(data_dir, train=False, download=True, transform=transform)
        else:
            raise ValueError(f"unknown partition '{partition}'")

    def __len__(self) -> int:
        return len(getattr(self, self.partition))

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, int]:
        tensor, label = getattr(self, self.partition)[idx]
        id = str(idx)
        return id, tensor, label

    def get_num_classes(self) -> int:
        return 10

    @classmethod
    def get_transforms(cls) -> gtypes.Transform:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

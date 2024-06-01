from pathlib import Path
from typing import List, Literal, Optional, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.transform_utils import SquarePad
from gorillatracker.type_helper import Id, Label
from gorillatracker.utils.labelencoder import LabelEncoder


def get_samples(dirpath: Path) -> List[Tuple[Path, str]]:
    """
    Assumed directory structure:
        dirpath/
            <label>_<...>.png

    """
    samples = []
    image_paths = dirpath.glob("*.jpg")
    for image_path in image_paths:
        if "_" in image_path.name:
            label = image_path.name.split("_")[0]
        else:
            label = image_path.name.split("-")[0]
        samples.append((image_path, label))
    return samples


def cast_label_to_int(labels: List[str]) -> List[int]:
    return LabelEncoder.encode_list(labels)


class BristolDataset(Dataset[Tuple[Id, Tensor, Label]]):
    def __init__(
        self, data_dir: str, partition: Literal["train", "val", "test"], transform: Optional[gtypes.Transform] = None
    ):
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
        dirpath = data_dir / Path(partition)
        samples = get_samples(dirpath)

        # new
        labels_string = [label for _, label in samples]
        labels_int = cast_label_to_int(labels_string)
        self.mapping = dict(zip(labels_int, labels_string))
        self.samples = list(zip([path for path, _ in samples], labels_int))

        self.transform = transform

        self.partition = partition

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Id, Tensor, Label]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return str(img_path), img, label

    @classmethod
    def get_transforms(cls) -> gtypes.Transform:
        return transforms.Compose(
            [
                SquarePad(),
                # Uniform input, you may choose higher/lower sizes.
                transforms.Resize(224),
                transforms.ToTensor(),
            ]
        )

    def get_num_classes(self) -> int:
        labels = [label for _, label in self.samples]
        return len(set(labels))

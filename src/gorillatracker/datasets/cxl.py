from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torchvision.transforms.v2 as transforms_v2
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
    image_paths = dirpath.glob("*.png")
    for image_path in image_paths:
        label = image_path.name.split("_")[0]
        samples.append((image_path, label))
    return samples


def cast_label_to_int(labels: List[str]) -> List[int]:
    return LabelEncoder.encode_list(labels)


class CXLDataset(Dataset[Tuple[Id, Tensor, Label]]):
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

    def __getitem__(self, idx: int) -> tuple[Id, Tensor, Label]:
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

    def get_class_distribution(self) -> Dict[int, int]:
        labels = [label for _, label in self.samples]
        class_distribution = {label: labels.count(label) for label in set(labels)}
        return class_distribution


if __name__ == "__main__":
    cxl = CXLDataset(
        "data/DO-NOT-USE-PING-BEN-IF-IN-DOUBT-joined_splits/cxl_face-openset=True_0",
        "train",
        CXLDataset.get_transforms(),
    )
    image = cxl[0][0]
    transformations = transforms.Compose(
        [
            transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms_v2.RandomHorizontalFlip(p=0.5),
            transforms.RandomErasing(p=1, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
        ]
    )
    image = transformations(image)
    pil_image = transforms.ToPILImage()(image)
    pil_image.save("test.png")

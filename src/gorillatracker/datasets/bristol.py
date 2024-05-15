# NOTE(liamvdv): missing ground_truth for bristol dataset

from pathlib import Path
from typing import List, Literal, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

import gorillatracker.type_helper as gtypes
from gorillatracker.type_helper import Id, Label


def get_samples(data_dir: Path) -> List[Tuple[Path, str]]:
    """
    Assumed directory structure:
        data_dir/
            <label>/
               <image>
    """
    samples = []
    for individual in data_dir.iterdir():
        if not individual.is_dir():
            continue
        label = individual.name
        image_paths = [image.absolute() for image in individual.iterdir() if image.is_file()]
        # NOTE(liamvdv): We sort for deterministic file order.
        image_paths.sort()
        samples.extend([(path, label) for path in image_paths])
    return samples


class BristolDataset(Dataset[Tuple[Id, Image.Image, Label]]):
    def __init__(
        self, data_dir: str, partition: Literal["train", "val", "test"], transform: Optional[gtypes.Transform] = None
    ):
        dirpath = data_dir / Path(partition)
        self.samples = get_samples(dirpath)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Id, Image.Image, Label]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return str(img_path), img, label

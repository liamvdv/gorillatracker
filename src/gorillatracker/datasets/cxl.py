from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.transform_utils import SquarePad

Label = Union[int, str]


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


class CXLDataset(Dataset[Tuple[Image.Image, Label]]):
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
        self.samples = get_samples(dirpath)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

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


if __name__ == "__main__":
    cxl = CXLDataset(
        "data/splits/use_for_baseline_cxl_face-openset_origin_robert", "train", CXLDataset.get_transforms()
    )
    image = cxl[0][0]
    image = transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11))(image)
    pil_image = transforms.ToPILImage()(image)
    pil_image.save("test.png")

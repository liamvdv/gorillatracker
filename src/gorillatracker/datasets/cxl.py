from pathlib import Path
from typing import List, Literal, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from gorillatracker.transform_utils import SquarePad

Label = Union[int, str]


def get_samples(dirpath: Path) -> List[Tuple[Path, Label]]:
    """
    Assumed directory structure:
        dirpath/
            <label>_<...>.png

    """
    samples = []
    # quickfix for now
    image_paths = [img_path for img_path in dirpath.glob("*.png")]
    image_paths = image_paths + [img_path for img_path in dirpath.glob("*.jpg")]
    for image_path in image_paths:
        label = image_path.name.split("_")[0]
        samples.append((image_path, label))
    return samples


class CXLDataset(Dataset):
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
        dirpath = data_dir / Path(partition)
        self.samples = get_samples(dirpath)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

    @classmethod
    def get_transforms(cls):
        return transforms.Compose(
            [
                SquarePad(),
                transforms.ToTensor(),
            ]
        )


if __name__ == "__main__":
    for img, label in CXLDataset(
        "data/splits/ground_truth-cxl-closedset--mintraincount-3-seed-42-train-70-val-15-test-15"
    ):
        print(img, label, img.filename)

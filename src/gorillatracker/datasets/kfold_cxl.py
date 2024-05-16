from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torchvision.transforms.v2 as transforms_v2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.transform_utils import SquarePad
from gorillatracker.type_helper import Id, Label


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
    le = LabelEncoder()
    le.fit(labels)
    return le.transform(labels)


class KFoldCXLDataset(Dataset[Tuple[Id, Image.Image, Label]]):
    def __init__(
        self,
        data_dir: str,
        partition: Literal["train", "val", "test"],
        val_i: int,
        k: int,
        transform: Optional[gtypes.Transform] = None,
    ):
        """
        Assumes directory structure:
            data_dir/
                fold-0/
                    ...
                fold-(k-1)/
                    ...
                test/
                    ...
        """
        samples = []
        if partition == "val":
            dirpath = data_dir / Path(f"fold-{val_i}")
            samples.extend(get_samples(dirpath))
        elif partition == "train":
            for i in range(k):
                if i != val_i:
                    dirpath = data_dir / Path(f"fold-{i}")
                    samples.extend(get_samples(dirpath))
        else:
            dirpath = data_dir / Path(partition)
            samples.extend(get_samples(dirpath))

        # new
        labels_string = [label for _, label in samples]
        labels_int = cast_label_to_int(labels_string)
        self.mapping = dict(zip(labels_int, labels_string))
        self.samples = list(zip([path for path, _ in samples], labels_int))

        self.transform = transform

        self.partition = partition

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Id, Image.Image, Label]:
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


if __name__ == "__main__":
    cxl = KFoldCXLDataset(
        "./data/splits/ground_truth-cxl-face_images-kfold-openset-seed-42-trainval-80-test-20-k-5",
        "train",
        0,
        5,
        KFoldCXLDataset.get_transforms(),
    )
    print(cxl.get_num_classes())
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

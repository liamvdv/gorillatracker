from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.transform_utils import SquarePad

Label = Union[int, str]


def get_samples_video(dirpath: Path) -> List[Tuple[Path, str]]:
    """
    Assumed directory structure:
        dirpath/
            <video_name>-<id>-<frame_idx>.jpg

    Label is <video_name>-<id>
    """
    samples = []
    image_paths = sorted(dirpath.glob("*.png"))
    for image_path in image_paths:
        label = (
            image_path.name.split("-")[0] + "-" + image_path.name.split("-")[1]
        )  # expects image_path as video-id-frame
        samples.append((image_path, label))
    return samples


def get_samples_cxl(dirpath: Path) -> List[Tuple[Path, str]]:
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


class SPACVideosDataset(Dataset[Tuple[Image.Image, Label]]):
    def __init__(
        self, data_dir: str, partition: Literal["train", "val", "test"], transform: Optional[gtypes.Transform] = None
    ):
        """
        Assumes directory structure:
            data_dir/
                train/
                    video_data
                    negatives.json
                val/
                    cxl_data
                test/
                    cxl_data
        """
        dirpath = data_dir / Path(partition)
        if partition == "train":
            self.samples = get_samples_video(dirpath)
        else:
            self.samples = get_samples_cxl(dirpath)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
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
                transforms.ToTensor(),
                # Uniform input, you may choose higher/lower sizes.
            ]
        )


if __name__ == "__main__":
    spac_videos = SPACVideosDataset(
        "data/derived_data/spac_gorillas_converted_labels_cropped_faces",
        "train",
        SPACVideosDataset.get_transforms(),
    )

    # print first 10 labels and save images
    for i in range(10):
        print(spac_videos[i][1])
        pil_image = transforms.ToPILImage()(spac_videos[i][0])
        pil_image.save(f"test{i}.png")

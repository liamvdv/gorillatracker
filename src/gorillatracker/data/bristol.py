from pathlib import Path

from gorillatracker.data.contrastive_sampler import ContrastiveClassSampler, ContrastiveImage, group_contrastive_images
from gorillatracker.data.nlet import NletDataset
from gorillatracker.type_helper import Label
from gorillatracker.utils.labelencoder import LabelEncoder


def get_groups(dirpath: Path) -> dict[Label, list[ContrastiveImage]]:
    """
    Assumed directory structure:
        dirpath/
            <label>_<...>.jpg
            or 
            <label>-<...>.jpg

    """
    samples = []
    image_paths = dirpath.glob("*.jpg")
    for image_path in image_paths:
        if "_" in image_path.name:
            label = image_path.name.split("_")[0]
        else:
            label = image_path.name.split("-")[0]
        samples.append(ContrastiveImage(str(image_path), image_path, LabelEncoder.encode(label)))
    return group_contrastive_images(samples)


class BristolDataset(NletDataset):
    @property
    def num_classes(self) -> int:
        return len(self.contrastive_sampler.class_labels)

    def create_contrastive_sampler(self, base_dir) -> ContrastiveClassSampler:
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
        dirpath = base_dir / Path(self.partition)
        return ContrastiveClassSampler(get_groups(dirpath))

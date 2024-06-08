from collections import defaultdict
from pathlib import Path

from gorillatracker.data.contrastive_sampler import ContrastiveClassSampler, ContrastiveImage, group_contrastive_images
from gorillatracker.data.nlet import KFoldNletDataset, NletDataset
from gorillatracker.type_helper import Label
from gorillatracker.utils.labelencoder import LabelEncoder


def group_images_by_label(dirpath: Path) -> defaultdict[Label, list[ContrastiveImage]]:
    """
    Assumed directory structure:
        dirpath/
            <label>_<...>.png

    """
    samples = []
    image_paths = dirpath.glob("*.png")
    for image_path in image_paths:
        label = image_path.name.split("_")[0]
        samples.append(ContrastiveImage(str(image_path), image_path, LabelEncoder.encode(label)))
    return group_contrastive_images(samples)


class CXLDataset(NletDataset):
    @property
    def num_classes(self) -> int:
        return len(self.contrastive_sampler.class_labels)

    @property
    def class_distribution(self) -> dict[Label, int]:
        return {label: len(samples) for label, samples in self.classes.items()}

    def create_contrastive_sampler(self, base_dir: Path) -> ContrastiveClassSampler:
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
        self.classes = group_images_by_label(dirpath)
        return ContrastiveClassSampler(self.classes)


class KFoldCXLDataset(KFoldNletDataset):
    @property
    def num_classes(self) -> int:
        return len(self.contrastive_sampler.class_labels)

    @property
    def class_distribution(self) -> dict[Label, int]:
        return {label: len(samples) for label, samples in self.groups.items()}

    def create_contrastive_sampler(self, base_dir: Path) -> ContrastiveClassSampler:
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
        if self.partition == "train":
            self.groups: defaultdict[Label, list[ContrastiveImage]] = defaultdict(list)
            for i in range(self.k):
                if i == self.val_i:
                    continue
                dirpath = base_dir / Path(f"fold-{i}")
                new_groups = group_images_by_label(dirpath)
                for label, samples in new_groups.items():
                    # NOTE(memben): NO deduplication here (feel free to add it)
                    self.groups[label].extend(samples)
        elif self.partition == "test":
            dirpath = base_dir / Path(self.partition)
            self.groups = group_images_by_label(dirpath)
        elif self.partition == "val":
            dirpath = base_dir / Path(f"fold-{self.val_i}")
            self.groups = group_images_by_label(dirpath)
        else:
            raise ValueError(f"Invalid partition: {self.partition}")
        return ContrastiveClassSampler(self.groups)

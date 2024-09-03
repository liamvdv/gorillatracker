import logging
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torch import Tensor
from wildlife_datasets import datasets, loader

import gorillatracker.type_helper as gtypes
from gorillatracker.data.contrastive_sampler import ContrastiveImage, ContrastiveSampler, FlatNlet
from gorillatracker.data.nlet import NletDataset
from gorillatracker.type_helper import Id, Label, Nlet
from gorillatracker.utils.labelencoder import LabelEncoder

logger = logging.getLogger(__name__)


def get_ds_dfs() -> pd.DataFrame:
    dfs = []

    for ds_cls in datasets.names_small:
        name = str(ds_cls).split(".")[-1][:-2]
        if name.endswith("v2"):
            name = name[:-2]

        if name in {
            "GreenSeaTurtles",
            "StripeSpotter",
            "PolarBearVidID",
            "AerialCattle2017",
            "FriesianCattle2015",
            "FriesianCattle2017",
        }:
            continue
        print(f"Loading {name}")
        d = loader.load_dataset(
            ds_cls,
            "/workspaces/gorillatracker/data/WildlifeReID-10k/data",
            "/workspaces/gorillatracker/data/WildlifeReID-10k/dataframes",
        )
        columns = ["identity", "path", "bbox"] if "bbox" in d.df.columns else ["identity", "path"]
        df = d.df.loc[:, columns]
        df.drop(df[df["identity"] == d.unknown_name].index, inplace=True)
        df["origin"] = name
        df["path"] = name + "/" + df["path"]
        df["label"] = LabelEncoder().encode_list(df["identity"].values.tolist())

        dfs.append(df)
        print(len(df))

    combined_df = pd.concat(dfs)
    combined_df.reset_index(drop=False, inplace=True)

    return combined_df


# base_path = "/workspaces/gorillatracker/data/WildlifeReID-10k/data"
# ds = get_ds_dfs()
# for row in ds.iterrows():
#     if not os.path.exists(f"{base_path}/{row[1]['path']}"):
#         print(row[1]["path"])

# print(ds["identity"].nunique())
# print(len(ds))
# print(ds["label"].nunique())


class MultiSpeciesContrastiveSampler(ContrastiveSampler):
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.ds = get_ds_dfs()

    def __getitem__(self, idx: int) -> ContrastiveImage:
        img = ContrastiveImage(
            id=str(idx),
            image_path=self.base_dir / self.ds.loc[idx, "path"],
            class_label=self.ds.loc[idx, "label"],
        )
        return img

    def __len__(self) -> int:
        return len(self.ds)

    @property
    def class_labels(self) -> list[gtypes.Label]:
        return self.ds["label"].unique().tolist()

    def positive(
        self, sample: ContrastiveImage
    ) -> ContrastiveImage:  # must map whatever __getitem__ returns to another sample
        positive_class = sample.class_label
        if len(self.ds[self.ds["label"] == positive_class]) == 1:
            # logger.warning(f"Only one sample in class {positive_class}. Returning same sample as positive.")
            return sample
        positive_indices = self.ds[self.ds["label"] == positive_class].index
        positive_indices = positive_indices[positive_indices != sample.id]
        positive_index = random.choice(positive_indices)
        sample_row = self.ds.loc[positive_index]
        sample = ContrastiveImage(
            id=str(positive_index),
            image_path=self.base_dir / sample_row["path"],
            class_label=sample_row["label"],
        )
        return sample

    # NOTE(memben): First samples a negative class to ensure a more balanced distribution of negatives,
    # independent of the number of samples per class
    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Different class is sampled uniformly at random and a random sample from that class is returned"""
        # filter for the same origin -> species
        sample_origin = self.ds.loc[int(sample.id), "origin"]
        same_ds = self.ds[self.ds["origin"] == sample_origin]
        same_ds = same_ds[same_ds["label"] != sample.class_label]
        negative_class = random.choice(same_ds["label"].unique())
        negative_indices = same_ds[same_ds["label"] == negative_class].index
        negative_index = random.choice(negative_indices)
        sample_row = same_ds.loc[negative_index]
        sample = ContrastiveImage(
            id=str(negative_index),
            image_path=self.base_dir / sample_row["path"],
            class_label=sample_row["label"],
        )
        return sample

    def negative_classes(self, img: ContrastiveImage) -> list[Label]:
        class_label = img.class_label
        return [label for label in self.contrastive_sampler.class_labels if label != class_label]  # type: ignore


class MultiSpeciesSupervisedDataset(NletDataset):
    """
    A dataset that assumes the following directory structure:
        base_dir/
            train/
                ...
            val/
                ...
            test/
                ...
    Each file is prefixed with the class label, e.g. "label1_1.jpg"
    """

    def _get_item(self, idx: Label) -> tuple[tuple[Id, ...], tuple[Tensor, ...], tuple[Label, ...]]:
        return super()._get_item(idx)

    @property
    def num_classes(self) -> int:
        return len(self.contrastive_sampler)

    @property
    def class_distribution(self) -> dict[Label, int]:
        return self.contrastive_sampler.ds["label"].value_counts().to_dict()  # type: ignore

    def create_contrastive_sampler(self, base_dir: Path) -> MultiSpeciesContrastiveSampler:
        """
        Assumes directory structure:
            base_dir/
                train/
                    ...
                val/
                    ...
                test/
                    ...
        """
        return MultiSpeciesContrastiveSampler(base_dir)

    def _stack_flat_nlet(self, flat_nlet: FlatNlet) -> Nlet:
        ids = tuple(str(img.image_path) for img in flat_nlet)
        labels = tuple(img.class_label for img in flat_nlet)

        values = tuple(self.transform(self._crop_if_necessary(img)) for img in flat_nlet)
        return ids, values, labels

    def _crop_if_necessary(self, img: ContrastiveImage) -> Image.Image:
        pilimg = Image.open(img.image_path)
        if "bbox" in self.contrastive_sampler.ds.columns and isinstance(  # type: ignore
            self.contrastive_sampler.ds.loc[int(img.id), "bbox"], list  # type: ignore
        ):
            bbox = self.contrastive_sampler.ds.loc[int(img.id), "bbox"]  # type: ignore
            x, y, w, h = bbox
            bbox = (x, y, x + w, y + h)
            pilimg = pilimg.crop(bbox)  # type: ignore

        pilimg = pilimg.convert("RGB")  # type: ignore

        if pilimg.width > 300 and pilimg.height > 300:
            ratio = pilimg.width / pilimg.height
            if ratio > 1:
                pilimg = pilimg.resize((300, int(300 / ratio)))  # type: ignore
            else:
                pilimg = pilimg.resize((int(300 * ratio), 300))  # type: ignore
        return pilimg

    def __len__(self) -> Label:
        if self.partition == "val":
            return min(100, len(self.contrastive_sampler.ds))  # type: ignore
        else:
            return len(self.contrastive_sampler.ds)  # type: ignore

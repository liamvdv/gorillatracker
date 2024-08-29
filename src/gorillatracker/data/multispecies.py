import logging
import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from torch import Tensor
from wildlife_datasets import datasets, loader

import gorillatracker.type_helper as gtypes
from gorillatracker.data.contrastive_sampler import ContrastiveImage, ContrastiveSampler, FlatNlet
from gorillatracker.data.nlet import NletDataset, group_images_by_label
from gorillatracker.type_helper import Id, Label, Nlet
from gorillatracker.utils.labelencoder import LabelEncoder

logger = logging.getLogger(__name__)


def get_ds_dfs(
    use_primates: bool = True, train_factor: float = 0.7, val_factor: float = 0.15, test_factor: float = 0.15
) -> pd.DataFrame:
    rng = random.Random(42)
    dfs = []

    dataset_names = datasets.names_primates if use_primates else datasets.names_all

    for ds_cls in dataset_names:
        name = str(ds_cls).split(".")[-1][:-2]
        if name.endswith("v2"):
            name = name[:-2]

        if name in {
            "AAUZebraFish",
            "GreenSeaTurtles",
            "Drosophila",
            "SMALST",
            "AerialCattle2017",  # DATA off (too big here -> prob video processing needed)
            "PolarBearVidID",  # DATA off
            "SealIDSegmented",  # no segmentation
            "BirdIndividualID",  # License: None
            "BirdIndividualIDSegmented",  # License: None
            "Giraffes",  # License: None
            "IPanda50",  # License: None
            "NyalaData",  # License: None
            # check for license problems
            # "HumpbackWhaleID" # not in paper, fine license
            # "HappyWhale" # not in paper, fine license
            # "LionData" # not in paper, not found online? (license?)
            # "MacaqueFaces" # not in paper, fine license -> CC BY 4.0
            "NOAARightWhale",  # not in paper, license raises questions
        }:
            continue
        d = loader.load_dataset(
            ds_cls,
            "/workspaces/gorillatracker/data/WildlifeReID-10k/data",
            "/workspaces/gorillatracker/data/WildlifeReID-10k/dataframes",
        )
        columns = ["identity", "path", "bbox"] if "bbox" in d.df.columns else ["identity", "path"]
        df = d.df.loc[:, columns]
        df.drop(df[df["identity"] == d.unknown_name].index, inplace=True)
        to_drop = df["identity"].value_counts()[df["identity"].value_counts() < 2].index
        df.drop(df[df["identity"].isin(to_drop)].index, inplace=True)
        df["origin"] = name
        df["identity"] = df["identity"].apply(lambda x: f"{name}_{x}")
        df["path"] = name + "/" + df["path"]
        df["label"] = LabelEncoder().encode_list(df["identity"].values.tolist())

        dfs.append(df)
    combined_df = pd.concat(dfs)
    combined_df.reset_index(drop=False, inplace=True)

    # perform train-val-test split
    train_individuals = []
    val_individuals = []
    test_individuals = []
    for name in combined_df["origin"].unique():
        df = combined_df[combined_df["origin"] == name]
        individuals = df["label"].unique().tolist()
        rng.shuffle(individuals)

        # split into train-val-test
        train_individuals += individuals[: int(train_factor * len(individuals))]
        val_individuals += individuals[
            int(train_factor * len(individuals)) : int((train_factor + val_factor) * len(individuals))
        ]
        test_individuals += individuals[int((train_factor + val_factor) * len(individuals)) :]

    combined_df["split"] = "train"
    combined_df.loc[combined_df["label"].isin(val_individuals), "split"] = "val"
    combined_df.loc[combined_df["label"].isin(test_individuals), "split"] = "test"

    return combined_df


class MultiSpeciesContrastiveSampler(ContrastiveSampler):
    def __init__(
        self, base_dir: Path, partition: Literal["train", "val", "test"] = "train", use_primates: bool = True
    ) -> None:
        self.base_dir = base_dir
        self.ds = get_ds_dfs(use_primates=use_primates)
        self.ds = self.ds[self.ds["split"] == partition]
        self.ds.reset_index(drop=False, inplace=True)

    def __getitem__(self, idx: int) -> ContrastiveImage:
        img = ContrastiveImage(
            id=str(idx),
            image_path=self.base_dir / self.ds.iloc[idx]["path"],
            class_label=self.ds.iloc[idx]["label"],
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
        sample_row = self.ds.iloc[positive_index]

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
        sample_origin = self.ds.iloc[int(sample.id)]["origin"]
        same_ds = self.ds[self.ds["origin"] == sample_origin]
        same_ds = same_ds[same_ds["label"] != sample.class_label]
        negative_class = random.choice(same_ds["label"].unique())
        negative_indices = same_ds[same_ds["label"] == negative_class].index
        negative_index = random.choice(negative_indices)
        sample_row = self.ds.iloc[negative_index]

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

    def __init__(
        self, use_gorillas: bool = False, use_lfw: bool = False, use_primates: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        self.use_primates = use_primates
        super().__init__(*args, **kwargs)
        if use_gorillas:
            path = (
                Path("/workspaces/gorillatracker/data/supervised/splits/cxl_faces_openset_seed_42_square/")
                / self.partition
            )

            classes = group_images_by_label(path)
            # convert into dataframe with: path, label, origin, split, identity
            labels = []
            identities = []
            paths = []
            for _, contrastive_imgs in classes.items():
                paths += [img.image_path for img in contrastive_imgs]
                labels += [img.class_label for img in contrastive_imgs]
                identities += [img.id for img in contrastive_imgs]

            gorilla_df = pd.DataFrame(
                {
                    "path": paths,
                    "label": labels,
                    "identity": identities,
                    "origin": "gorillas",
                    "split": self.partition,
                    "bbox": None,
                }
            )
            gorilla_df.reset_index(drop=False, inplace=True)

            self.contrastive_sampler.ds = pd.concat([self.contrastive_sampler.ds, gorilla_df])  # type: ignore
            self.contrastive_sampler.ds = self.contrastive_sampler.ds.reset_index(drop=True)  # type: ignore

        if use_lfw:
            lfw_people = fetch_lfw_people(resize=1.0, color=True)
            self.lfw_images = lfw_people.images
            self.lfw_targets = lfw_people.target
            self.lfw_target_names = lfw_people.target_names
            self.lfw_topil = transforms.ToPILImage()

            # perform train-val-test split
            individuals = np.unique(self.lfw_targets)

            # only keep individuals with >=3 images
            individuals = [ind for ind in individuals if np.sum(self.lfw_targets == ind) >= 3]

            rng = random.Random(42)
            rng.shuffle(individuals)

            # filter for 'partition' individuals
            if self.partition == "train":
                individuals = individuals[: int(0.7 * len(individuals))]
            elif self.partition == "val":
                individuals = individuals[int(0.7 * len(individuals)) : int(0.85 * len(individuals))]
            else:
                individuals = individuals[int(0.85 * len(individuals)) :]

            indices = [idx for idx, target in enumerate(self.lfw_targets) if target in individuals]
            self.lfw_images = self.lfw_images[indices]
            self.lfw_targets = self.lfw_targets[indices]
            self.lfw_target_names = self.lfw_target_names[self.lfw_targets]

            os.makedirs(f"/workspaces/gorillatracker/data/LabeledFacesInTheWild/{self.partition}", exist_ok=True)

            for idx, img in enumerate(self.lfw_images):
                pil_img = self.lfw_topil(img)
                (
                    pil_img.save(f"/workspaces/gorillatracker/data/LabeledFacesInTheWild/{self.partition}/{idx}.jpg")
                    if not Path(
                        f"/workspaces/gorillatracker/data/LabeledFacesInTheWild/{self.partition}/{idx}.jpg"
                    ).exists()
                    else None
                )

            lfw_df = pd.DataFrame(
                {
                    "path": [
                        f"/workspaces/gorillatracker/data/LabeledFacesInTheWild/{self.partition}/{idx}.jpg"
                        for idx in range(len(self.lfw_images))
                    ],
                    "label": self.lfw_targets,
                    "identity": self.lfw_target_names,
                    "origin": "lfw",
                    "split": self.partition,
                    "bbox": None,
                }
            )

            self.contrastive_sampler.ds = pd.concat([self.contrastive_sampler.ds, lfw_df])  # type: ignore

            # TODO: add dataset versions for separate metric eval.

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
        return MultiSpeciesContrastiveSampler(base_dir, partition=self.partition, use_primates=self.use_primates)

    def _stack_flat_nlet(self, flat_nlet: FlatNlet) -> Nlet:
        ids = tuple(str(img.image_path) for img in flat_nlet)
        labels = tuple(img.class_label for img in flat_nlet)

        values = tuple(self.transform(self._crop_if_necessary(img)) for img in flat_nlet)
        return ids, values, labels

    def _crop_if_necessary(self, img: ContrastiveImage) -> Image.Image:
        pilimg = Image.open(img.image_path)
        if "bbox" in self.contrastive_sampler.ds.columns and isinstance(  # type: ignore
            self.contrastive_sampler.ds.iloc[int(img.id)]["bbox"], list  # type: ignore
        ):
            bbox = self.contrastive_sampler.ds.iloc[int(img.id)]["bbox"]  # type: ignore
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
        return len(self.contrastive_sampler)


class GorillasPrimatesSupervisedDataset(MultiSpeciesSupervisedDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(use_gorillas=True, use_lfw=False, use_primates=True, *args, **kwargs)


class GorillasPrimatesLFWSupervisedDataset(MultiSpeciesSupervisedDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(use_gorillas=True, use_lfw=True, use_primates=True, *args, **kwargs)


class PrimatesLFWSupervisedDataset(MultiSpeciesSupervisedDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(use_gorillas=False, use_lfw=True, use_primates=True, *args, **kwargs)


class CombinedMultiSpeciesSupervisedDataset(MultiSpeciesSupervisedDataset):
    """Everything combined"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(use_gorillas=True, use_lfw=True, use_primates=False, *args, **kwargs)


class LFW_only(MultiSpeciesSupervisedDataset):  # TODO
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(use_gorillas=True, use_lfw=True, use_primates=False, *args, **kwargs)
        self.contrastive_sampler.ds = self.contrastive_sampler.ds[self.contrastive_sampler.ds["origin"] == "lfw"].copy()


class Macaques_only(MultiSpeciesSupervisedDataset):  # TODO
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(use_gorillas=False, use_lfw=False, use_primates=True, *args, **kwargs)

        self.contrastive_sampler.ds = self.contrastive_sampler.ds[
            self.contrastive_sampler.ds["origin"] == "MacaqueFaces"
        ].copy()


class Chimpanzee_only(MultiSpeciesSupervisedDataset):  # TODO
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(use_gorillas=False, use_lfw=False, use_primates=True, *args, **kwargs)

        self.contrastive_sampler.ds = self.contrastive_sampler.ds[
            self.contrastive_sampler.ds["origin"] == "CTai" | self.contrastive_sampler.ds["origin"] == "CZoo"
        ].copy()

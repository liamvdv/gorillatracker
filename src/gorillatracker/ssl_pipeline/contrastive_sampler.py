# NOTE(memben): let's worry about how we parse configs from the yaml file later

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import groupby
from typing import Any

from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import TrackingFrameFeature


@dataclass(frozen=True, order=True)
class ContrastiveImage:
    id: str
    image_path: str
    class_label: int

    @property
    def image(self) -> Image.Image:
        return Image.open(self.image_path)


class ContrastiveSampler(ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> ContrastiveImage:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Return a different positive sample from the same class."""
        pass

    @abstractmethod
    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Return a negative sample from a different class."""
        pass


class ContrastiveClassSampler(ContrastiveSampler):
    """ContrastiveSampler that samples from a set of classes. Negatives are drawn from a uniformly sampled negative class"""

    def __init__(self, classes: dict[Any, list[ContrastiveImage]]) -> None:
        self.classes = classes
        self.class_labels = list(classes.keys())
        self.samples = [sample for samples in classes.values() for sample in samples]
        self.sample_to_class = {sample: label for label, samples in classes.items() for sample in samples}

        assert all([len(samples) > 1 for samples in classes.values()]), "Classes must have at least two samples"
        assert len(self.samples) == len(set(self.samples)), "Samples must be unique"

    def __getitem__(self, idx: int) -> ContrastiveImage:
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)

    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        positive_class = self.sample_to_class[sample]
        positives = [s for s in self.classes[positive_class] if s != sample]
        return random.choice(positives)

    # NOTE(memben): First samples a negative class to ensure a more balanced distribution of negatives,
    # independent of the number of samples per class
    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Different class is sampled uniformly at random and a random sample from that class is returned"""
        positive_class = self.sample_to_class[sample]
        negative_classes = [c for c in self.class_labels if c != positive_class]
        negative_class = random.choice(negative_classes)
        negatives = self.classes[negative_class]
        return random.choice(negatives)


# TODO(memben): This is only for demonstration purposes. We will need to replace this with a more general solution
def get_random_ssl_sampler(base_path: str) -> ContrastiveClassSampler:
    WHATEVER_PWD = "DEV_PWD_139u02riowenfgiw4y589wthfn"
    PUBLIC_DB_URI = f"postgresql+psycopg2://postgres:{WHATEVER_PWD}@postgres:5432/postgres"
    engine = create_engine(PUBLIC_DB_URI)
    with Session(engine) as session:
        tracked_features = list(
            session.execute(
                select(TrackingFrameFeature)
                .where(
                    TrackingFrameFeature.cached,
                    TrackingFrameFeature.tracking_id.isnot(None),
                    TrackingFrameFeature.feature_type == "body",
                )
                .order_by(TrackingFrameFeature.tracking_id)
            )
            .scalars()
            .all()
        )
        contrastive_images = [
            ContrastiveImage(str(f.tracking_frame_feature_id), f.cache_path(base_path), f.tracking_id) for f in tracked_features  # type: ignore
        ]
        groups = groupby(contrastive_images, lambda x: x.class_label)
        classes: dict[Any, list[ContrastiveImage]] = {}
        for group in groups:
            class_label, sample_iter = group
            samples = list(sample_iter)
            if len(samples) > 1:
                classes[class_label] = samples
        return ContrastiveClassSampler(classes)


if __name__ == "__main__":
    version = "2024-04-09"
    sampler = get_random_ssl_sampler(f"/workspaces/gorillatracker/cropped_images/{version}")
    print(len(sampler))
    sample = sampler[0]
    print(sample)
    print(sampler.positive)

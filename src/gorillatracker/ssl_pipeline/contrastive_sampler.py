# NOTE(memben): let's worry about how we parse configs from the yaml file later

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import groupby

import torch
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature


@dataclass(frozen=True, order=True)
class ContrastiveImage:
    id: str
    image_path: str
    class_label: int

    @property
    def image(self) -> Image.Image:
        return Image.open(self.image_path)

    @property
    def image_tensor(self) -> torch.Tensor:
        return transforms.ToTensor()(self.image)


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


class RandomClassSampler(ContrastiveSampler):
    """ContrastiveSampler that samples from a set of classes. Negatives are drawn evenly from all other classes."""

    def __init__(self, classes: dict[gtypes.Label, list[ContrastiveImage]]) -> None:
        assert all([len(samples) > 1 for samples in classes.values()]), "Classes must have at least two samples"
        self.classes = classes
        self.class_labels = list(classes.keys())
        self.flat_samples = [sample for samples in classes.values() for sample in samples]
        assert len(self.flat_samples) == len(set(self.flat_samples)), "Samples must be unique"
        self.sample_to_class = {sample: label for label, samples in classes.items() for sample in samples}

    def __getitem__(self, idx: int) -> ContrastiveImage:
        return self.flat_samples[idx]

    def __len__(self) -> int:
        return len(self.flat_samples)

    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        positive_class = self.sample_to_class[sample]
        positive_samples = [sample for sample in self.classes[positive_class] if sample != sample]
        return random.choice(list(positive_samples))

    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        positive_class = self.sample_to_class[sample]
        negative_classes = [label for label in self.class_labels if label != positive_class]
        negative_class = random.choice(negative_classes)
        return random.choice(self.classes[negative_class])


def get_random_sampler() -> RandomClassSampler:
    PUBLIC_DB_URI = "postgresql+psycopg2://postgres:DEV_PWD_139u02riowenfgiw4y589wthfn@postgres:5432/postgres"
    engine = create_engine(PUBLIC_DB_URI)
    with Session(engine) as session:
        tracked_features = list(
            session.execute(
                select(TrackingFrameFeature)
                .where(
                    TrackingFrameFeature.cache_path.isnot(None),
                    TrackingFrameFeature.tracking_id.isnot(None),
                    TrackingFrameFeature.feature_type == "body",
                )
                .order_by(TrackingFrameFeature.tracking_id)
            )
            .scalars()
            .all()
        )
        contrastive_images = [
            ContrastiveImage(str(f.tracking_frame_feature_id), f.cache_path, f.tracking_id) for f in tracked_features  # type: ignore
        ]
        groups = groupby(contrastive_images, lambda x: x.class_label)
        classes = {label: list(samples) for label, samples in groups}
        return RandomClassSampler(classes)


if __name__ == "__main__":
    get_random_sampler()

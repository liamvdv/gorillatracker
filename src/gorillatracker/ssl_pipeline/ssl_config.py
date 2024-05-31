from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List

from sqlalchemy import Select, create_engine, select
from sqlalchemy.orm import Session

import gorillatracker.type_helper as gtypes
from gorillatracker.ssl_pipeline.contrastive_sampler import (
    ContrastiveClassSampler,
    ContrastiveImage,
    ContrastiveSampler,
)
from gorillatracker.ssl_pipeline.dataset import GorillaDatasetKISZ
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature, Video
from gorillatracker.ssl_pipeline.queries import (
    associated_filter,
    cached_filter,
    confidence_filter,
    feature_type_filter,
    min_count_filter,
    multiple_videos_filter,
)
from gorillatracker.ssl_pipeline.sampler import EquidistantSampler, RandomSampler, Sampler


@dataclass(kw_only=True)  # type: ignore
class SSLConfig:
    tff_selection: str
    # NOTE(v1nce1): This should be removed once we have a proper split object
    n_videos: int
    n_samples: int
    feature_types: list[str]
    min_confidence: float
    min_images_per_tracking: int
    split: object

    def get_contrastive_sampler(self, base_path: str) -> ContrastiveSampler:
        engine = create_engine(GorillaDatasetKISZ.DB_URI)

        with Session(engine) as session:
            video_ids = list(session.execute(select(Video.video_id)).scalars().all())
            query = self._build_query(video_ids[: self.n_videos])
            sampler = self._create_tff_sampler(query)
            tracked_features = self._sample_tracked_features(sampler, session)
            contrastive_images = self._create_contrastive_images(tracked_features, base_path)
            classes = self._group_contrastive_images(contrastive_images)
            return ContrastiveClassSampler(classes)

    def _create_tff_sampler(self, query: Select[tuple[TrackingFrameFeature]]) -> Sampler:
        if self.tff_selection == "random":
            return RandomSampler(query, self.n_samples)
        elif self.tff_selection == "equidistant":
            return EquidistantSampler(query, self.n_samples)
        else:
            raise ValueError(f"Unknown TFF selection method: {self.tff_selection}")

    def _build_query(self, video_ids: List[int]) -> Select[tuple[TrackingFrameFeature]]:
        query = multiple_videos_filter(video_ids)
        query = cached_filter(query)
        query = associated_filter(query)
        query = feature_type_filter(query, self.feature_types)
        query = confidence_filter(query, self.min_confidence)
        query = min_count_filter(query, self.min_images_per_tracking)
        return query

    def _sample_tracked_features(self, sampler: Sampler, session: Session) -> List[TrackingFrameFeature]:
        print("Sampling TrackingFrameFeatures...")
        return list(sampler.sample(session))

    def _create_contrastive_images(
        self, tracked_features: List[TrackingFrameFeature], base_path: str
    ) -> List[ContrastiveImage]:
        return [
            ContrastiveImage(str(f.tracking_frame_feature_id), f.cache_path(Path(base_path)), f.tracking_id)  # type: ignore
            for f in tracked_features
        ]

    def _group_contrastive_images(
        self, contrastive_images: List[ContrastiveImage]
    ) -> dict[gtypes.Label, List[ContrastiveImage]]:
        groups = groupby(contrastive_images, lambda x: x.class_label)
        classes: dict[gtypes.Label, List[ContrastiveImage]] = {}
        for group in groups:
            class_label, sample_iter = group
            samples = list(sample_iter)
            classes[class_label] = samples
        return classes


if __name__ == "__main__":
    ssl_config = SSLConfig(
        tff_selection="equidistant",
        n_videos=200,
        n_samples=15,
        feature_types=["body"],
        min_confidence=0.5,
        min_images_per_tracking=10,
        split=object(),
    )
    contrastive_sampler = ssl_config.get_contrastive_sampler("cropped-images/2024-04-18")
    print(len(contrastive_sampler))
    contrastive_image = contrastive_sampler[0]
    print(contrastive_image)
    print(contrastive_sampler.positive(contrastive_image))
    print(contrastive_sampler.negative(contrastive_image))

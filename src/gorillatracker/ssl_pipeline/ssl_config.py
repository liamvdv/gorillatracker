from dataclasses import dataclass
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, List, Sequence

from sqlalchemy import Select, create_engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

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
    video_filter,
)
from gorillatracker.ssl_pipeline.sampler import EquidistantSampler, RandomSampler, Sampler

# dataclass SSLConfig


@dataclass(kw_only=True)  # type: ignore
class SSLConfig:
    tff_selection: str
    n_videos: int
    n_samples: int
    feature_types: list[str]
    min_confidence: float
    split: object

    def get_contrastive_sampler(self, base_path: str) -> ContrastiveSampler:
        engine = create_engine(GorillaDatasetKISZ.DB_URI)
        query_builder = partial(
            self._build_query,
            feature_types=self.feature_types,
            min_confidence=self.min_confidence,
        )
        sampler = self._create_tff_sampler(query_builder)

        with Session(engine) as session:
            video_ids = session.execute(select(Video.video_id)).scalars().all()
            tracked_features = self._sample_tracked_features(video_ids[: self.n_videos], sampler, session)
            contrastive_images = self._create_contrastive_images(tracked_features, base_path)
            classes = self._group_contrastive_images(contrastive_images)
            return ContrastiveClassSampler(classes)

    def _create_tff_sampler(self, query_builder: Callable[[int], Select[tuple[TrackingFrameFeature]]]) -> Sampler:
        return (
            EquidistantSampler(query_builder, n_samples=self.n_samples)
            if self.tff_selection == "equidistant"
            else RandomSampler(query_builder, n_samples=self.n_samples)
        )

    def _build_query(
        self, video_id: int, feature_types: List[str], min_confidence: float
    ) -> Select[tuple[TrackingFrameFeature]]:
        query = video_filter(video_id)
        query = cached_filter(query)
        query = associated_filter(query)
        query = feature_type_filter(query, feature_types)
        query = confidence_filter(query, min_confidence)
        return query

    def _sample_tracked_features(
        self, video_ids: Sequence[int], sampler: Sampler, session: Session
    ) -> List[TrackingFrameFeature]:
        tracked_features = []
        for video_id in tqdm(video_ids, unit="video", desc="Selecting TFFs"):
            for tracked_feature in sampler.sample(video_id, session):
                tracked_features.append(tracked_feature)
        return tracked_features

    def _create_contrastive_images(
        self, tracked_features: List[TrackingFrameFeature], base_path: str
    ) -> List[ContrastiveImage]:
        return [
            ContrastiveImage(str(f.tracking_frame_feature_id), f.cache_path(Path(base_path)), f.tracking_id)  # type: ignore
            for f in tracked_features
        ]

    def _group_contrastive_images(
        self, contrastive_images: List[ContrastiveImage]
    ) -> dict[Any, List[ContrastiveImage]]:
        groups = groupby(contrastive_images, lambda x: x.class_label)
        classes: dict[Any, List[ContrastiveImage]] = {}
        for group in groups:
            class_label, sample_iter = group
            samples = list(sample_iter)
            if len(samples) > 1:
                classes[class_label] = samples
        return classes


if __name__ == "__main__":
    ssl_config = SSLConfig(
        tff_selection="random", n_videos=200, n_samples=15, feature_types=["body"], min_confidence=0.5, split=object()
    )
    contrastive_sampler = ssl_config.get_contrastive_sampler("video_data/cropped-images/2024-04-18")
    print(len(contrastive_sampler))
    contrastive_image = contrastive_sampler[0]
    print(contrastive_image)
    print(contrastive_sampler.positive(contrastive_image))
    print(contrastive_sampler.negative(contrastive_image))

import random
from itertools import groupby
from typing import Iterator, List

from sqlalchemy import Select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import TrackingFrameFeature


class Sampler:
    """Defines how to sample TrackingFrameFeature instances from the database."""

    def __init__(self, query: Select[tuple[TrackingFrameFeature]]) -> None:
        self.query = query

    def sample(self, session: Session) -> Iterator[TrackingFrameFeature]:
        """Sample a subset of TrackingFrameFeature instances from the database. Defined by query and sampling strategy."""
        return iter(session.execute(self.query).scalars().all())

    def group_by_tracking_id(self, frame_features: list[TrackingFrameFeature]) -> dict[int, list[TrackingFrameFeature]]:
        frame_features.sort(key=lambda x: x.tracking.tracking_id)
        return {
            tracking_id: list(features)
            for tracking_id, features in groupby(frame_features, key=lambda x: x.tracking.tracking_id)
        }


class RandomSampler(Sampler):
    """Randomly sample a subset of TrackingFrameFeature instances per tracking."""

    def __init__(self, query: Select[tuple[TrackingFrameFeature]], n_samples: int, seed: int = 42) -> None:
        super().__init__(query)
        self.seed = seed
        self.n_samples = n_samples

    def sample(self, session: Session) -> Iterator[TrackingFrameFeature]:
        tracking_frame_features = list(session.execute(self.query).scalars().all())
        tracking_id_grouped = self.group_by_tracking_id(tracking_frame_features)
        random.seed(self.seed)
        for features in tracking_id_grouped.values():
            num_samples = min(len(features), self.n_samples)
            yield from random.sample(features, num_samples)


class EquidistantSampler(Sampler):
    """Sample a subset of TrackingFrameFeature instances per tracking that are equidistant in time."""

    def __init__(self, query: Select[tuple[TrackingFrameFeature]], n_samples: int) -> None:
        super().__init__(query)
        self.n_samples = n_samples

    def sample(self, session: Session) -> Iterator[TrackingFrameFeature]:
        tracking_frame_features = list(session.execute(self.query).scalars().all())
        tracking_id_grouped = self.group_by_tracking_id(tracking_frame_features)
        for features in tracking_id_grouped.values():
            sampled_features = self.sample_equidistant(features, self.n_samples)
            yield from sampled_features

    def sample_equidistant(self, features: List[TrackingFrameFeature], n_samples: int) -> List[TrackingFrameFeature]:
        sorted_features = sorted(features, key=lambda x: x.frame_nr)
        num_features = len(features)
        if num_features <= n_samples:
            return features
        interval = (num_features - 1) // (n_samples - 1) if n_samples > 1 else 0
        indices = [i * interval for i in range(n_samples)]
        return [sorted_features[i] for i in indices]


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.sql import select

    from gorillatracker.ssl_pipeline.dataset import GorillaDatasetKISZ
    from gorillatracker.ssl_pipeline.models import Video
    from gorillatracker.ssl_pipeline.queries import (
        associated_filter,
        cached_filter,
        confidence_filter,
        feature_type_filter,
        min_count_filter,
        multiple_videos_filter,
    )

    engine = create_engine(GorillaDatasetKISZ.DB_URI)

    def build_query(
        video_ids: List[int], feature_types: List[str], min_confidence: float, min_images_per_tracking: int
    ) -> Select[tuple[TrackingFrameFeature]]:
        query = multiple_videos_filter(video_ids)
        query = cached_filter(query)
        query = associated_filter(query)
        query = feature_type_filter(query, feature_types)
        query = confidence_filter(query, min_confidence)
        query = min_count_filter(query, min_images_per_tracking)
        return query

    session_cls = sessionmaker(bind=engine)
    version = "2024-04-18"  # TODO(memben)

    with session_cls() as session:
        video_ids = list(session.execute(select(Video.video_id).where(Video.version == version)).scalars().all())
        query = build_query(
            video_ids=video_ids[:200], feature_types=["body"], min_confidence=0.5, min_images_per_tracking=10
        )
        sampler = Sampler(query)
        frame_features = sampler.sample(session)
        grouped = sampler.group_by_tracking_id(list(frame_features))

    for tracking_id, features in grouped.items():
        print(f"Tracking ID: {tracking_id}")
        for feature in features:
            print(f"Frame Nr: {feature.frame_nr}")
        print()

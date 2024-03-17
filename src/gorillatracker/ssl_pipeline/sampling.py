import random
from abc import ABC
from functools import reduce
from itertools import groupby
from pathlib import Path
from typing import Iterator

from sqlalchemy import Select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import TrackingFrameFeature
from gorillatracker.ssl_pipeline.queries import (
    video_tracking_frame_features_min_count_query,
    video_tracking_frame_features_query,
)


def group_by_tracking_id(frame_features: list[TrackingFrameFeature]) -> dict[int, list[TrackingFrameFeature]]:
    frame_features.sort(key=lambda x: x.tracking.tracking_id)
    return {
        tracking_id: list(features)
        for tracking_id, features in groupby(frame_features, key=lambda x: x.tracking.tracking_id)
    }


class SamplingFilter(ABC):
    """Defines a sampling filter for selecting a subset of TrackingFrameFeature instances to crop."""

    def filter(self, frame_features: Iterator[TrackingFrameFeature]) -> Iterator[TrackingFrameFeature]:
        """Returns a subset of TrackingFrameFeature instances to crop, prefiltered by the sampling filter."""
        return frame_features


class FeatureTypeFilter(SamplingFilter):
    """Filters TrackingFrameFeature instances by their type."""

    def __init__(self, feature_types: list[str]):
        self.feature_types = feature_types

    def filter(self, frame_features: Iterator[TrackingFrameFeature]) -> Iterator[TrackingFrameFeature]:
        return filter(lambda x: x.type in self.feature_types, frame_features)


class RandomSamplingFilter(SamplingFilter):
    """Randomly samples a subset of TrackingFrameFeature instances to crop."""

    def __init__(self, n_images_per_tracking: int):
        self.n_images_per_tracking = n_images_per_tracking

    def filter(self, frame_features: Iterator[TrackingFrameFeature]) -> Iterator[TrackingFrameFeature]:
        tracking_id_grouped = group_by_tracking_id(list(frame_features))
        for features in tracking_id_grouped.values():
            num_samples = min(len(features), self.n_images_per_tracking)
            yield from random.sample(features, num_samples)


class DatabaseFilter:
    """Defines a database filter for selecting a subset of TrackingFrameFeature instances to crop."""

    def filter_query(self, video: Path) -> Select[tuple[TrackingFrameFeature]]:
        """Filter query to select a subset of TrackingFrameFeature instances on the database."""
        return video_tracking_frame_features_query(video)


class MinCountFilter(DatabaseFilter):
    def __init__(self, min_feature_count: int):
        self.min_feature_count = min_feature_count

    def filter_query(self, video: Path) -> Select[tuple[TrackingFrameFeature]]:
        return video_tracking_frame_features_min_count_query(video, self.min_feature_count)


class Sampler:
    """Defines a sampler for selecting a subset of TrackingFrameFeature instances to crop."""

    def __init__(self, filter_query: DatabaseFilter, *filters: SamplingFilter):
        self.filter_query = filter_query
        self.filters = filters

    def sample(self, video: Path, session: Session) -> list[TrackingFrameFeature]:
        frames_features = list(session.execute(self.filter_query.filter_query(video)).scalars().all())
        return self.filter(iter(frames_features))

    def filter(self, frame_features: Iterator[TrackingFrameFeature]) -> list[TrackingFrameFeature]:
        return list(reduce(lambda acc, f: f.filter(acc), self.filters, frame_features))

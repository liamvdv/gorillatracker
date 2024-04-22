import random
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Callable, Iterator, List

from sqlalchemy import Select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import TrackingFrameFeature


class Sampler:
    """Defines how to sample TrackingFrameFeature instances from the database."""

    def __init__(self, query_builder: Callable[[int], Select[tuple[TrackingFrameFeature]]]) -> None:
        self.query_builder = query_builder

    def sample(self, video_id: int, session: Session) -> Iterator[TrackingFrameFeature]:
        """Sample a subset of TrackingFrameFeature instances from the database. Defined by query and sampling strategy."""

        query = self.query_builder(video_id)
        return iter(session.execute(query).scalars().all())

    def group_by_tracking_id(self, frame_features: list[TrackingFrameFeature]) -> dict[int, list[TrackingFrameFeature]]:
        frame_features.sort(key=lambda x: x.tracking.tracking_id)
        return {
            tracking_id: list(features)
            for tracking_id, features in groupby(frame_features, key=lambda x: x.tracking.tracking_id)
        }


class RandomSampler(Sampler):
    """Randomly sample a subset of TrackingFrameFeature instances per tracking."""

    def __init__(
        self, query_builder: Callable[[int], Select[tuple[TrackingFrameFeature]]], n_samples: int, seed: int = 42
    ) -> None:
        super().__init__(query_builder)
        self.seed = seed
        self.n_samples = n_samples

    def sample(self, video_id: int, session: Session) -> Iterator[TrackingFrameFeature]:
        query = self.query_builder(video_id)
        tracking_frame_features = list(session.execute(query).scalars().all())
        tracking_id_grouped = self.group_by_tracking_id(tracking_frame_features)
        random.seed(self.seed)
        for features in tracking_id_grouped.values():
            num_samples = min(len(features), self.n_samples)
            yield from random.sample(features, num_samples)


class EquidistantSampler(Sampler):
    """Sample a subset of TrackingFrameFeature instances per tracking that are equidistant in time."""

    def __init__(self, query_builder: Callable[[int], Select[tuple[TrackingFrameFeature]]], n_samples: int) -> None:
        super().__init__(query_builder)
        self.n_samples = n_samples

    def sample(self, video_id: int, session: Session) -> Iterator[TrackingFrameFeature]:
        query = self.query_builder(video_id)
        tracking_frame_features = list(session.execute(query).scalars().all())
        tracking_id_grouped = self.group_by_tracking_id(tracking_frame_features)
        for features in tracking_id_grouped.values():
            sampled_features = self.sample_equidistant(features, self.n_samples)
            yield from sampled_features

    def sample_equidistant(self, features: List[TrackingFrameFeature], n_samples: int) -> List[TrackingFrameFeature]:
        sorted_features = sorted(features, key=lambda x: x.frame_nr)
        interval = max(1, len(sorted_features) // n_samples)
        return sorted_features[::interval]


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.sql import select

    from gorillatracker.ssl_pipeline.models import Video
    from gorillatracker.ssl_pipeline.queries import associated_filter, video_filter

    engine = create_engine("sqlite:///test.db")

    def sampling_strategy(video_id: int, min_n_images_per_tracking: int) -> Select[tuple[TrackingFrameFeature]]:
        query = video_filter(video_id)
        query = associated_filter(query)
        return query

    session_cls = sessionmaker(bind=engine)
    version = "2024-04-09"  # TODO(memben)

    with session_cls() as session:
        videos = session.execute(select(Video)).scalars().all()
        video_paths = [Path(video.path) for video in videos]

    query = partial(sampling_strategy, min_n_images_per_tracking=10)
    sampler = EquidistantSampler(query, n_samples=10)

    frame_features = sampler.sample(videos[1].video_id, session)
    # group by tracking_frame_feature_id and print frame_nrs
    grouped = sampler.group_by_tracking_id(list(frame_features))
    for tracking_id, features in grouped.items():
        print(f"Tracking ID: {tracking_id}")
        for feature in features:
            print(f"Frame Nr: {feature.frame_nr}")
        print()

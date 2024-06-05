from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List, Literal, Optional

from sqlalchemy import Select, create_engine
from sqlalchemy.orm import Session

import gorillatracker.type_helper as gtypes
from gorillatracker.ssl_pipeline.contrastive_sampler import (
    CliqueGraphSampler,
    ContrastiveClassSampler,
    ContrastiveImage,
    ContrastiveSampler,
)
from gorillatracker.ssl_pipeline.data_structures import IndexedCliqueGraph, MultiLayerCliqueGraph
from gorillatracker.ssl_pipeline.dataset import GorillaDatasetKISZ
from gorillatracker.ssl_pipeline.dataset_splitter import SplitArgs
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature
from gorillatracker.ssl_pipeline.negative_mining_queries import find_overlapping_trackings, tracking_ids_from_videos
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
    tff_selection: Literal["random", "equidistant"]
    negative_mining: Literal["random", "overlapping"]
    n_samples: int
    feature_types: list[str]
    min_confidence: float
    min_images_per_tracking: int
    split_path: str

    def get_contrastive_sampler(self, partition: Literal["train", "val", "test"], base_path: str) -> ContrastiveSampler:
        engine = create_engine(GorillaDatasetKISZ.DB_URI)

        with Session(engine) as session:
            video_ids = self._get_video_ids(partition)
            query = self._build_query(video_ids)
            sampler = self._create_tff_sampler(query)
            tracked_features = self._sample_tracked_features(sampler, session)
            contrastive_images = self._create_contrastive_images(tracked_features, base_path)
            return self._create_contrastive_sampler(contrastive_images, video_ids, session)

    def _get_video_ids(self, partition: Literal["train", "val", "test"]) -> List[int]:
        split = SplitArgs.load_pickle(self.split_path)
        if partition == "train":
            return split.train_video_ids()
        elif partition == "val":
            return split.val_video_ids()
        elif partition == "test":
            return split.test_video_ids()
        else:
            raise ValueError(f"Unknown partition: {partition}")

    def _create_tff_sampler(self, query: Select[tuple[TrackingFrameFeature]]) -> Sampler:
        if self.tff_selection == "random":
            return RandomSampler(query, self.n_samples)
        elif self.tff_selection == "equidistant":
            return EquidistantSampler(query, self.n_samples)
        else:
            raise ValueError(f"Unknown TFF selection method: {self.tff_selection}")

    def _create_contrastive_sampler(
        self,
        contrastive_images: List[ContrastiveImage],
        video_ids: List[int],
        session: Session,
    ) -> ContrastiveSampler:
        if self.negative_mining == "random":
            classes = self._group_contrastive_images(contrastive_images)
            return ContrastiveClassSampler(classes)
        elif self.negative_mining == "overlapping":
            tracking_ids = session.execute(tracking_ids_from_videos(video_ids)).scalars().all()
            first_layer: IndexedCliqueGraph[int] = IndexedCliqueGraph(list(tracking_ids))
            overlapping_trackings = find_overlapping_trackings(session, video_ids)
            for left, right in overlapping_trackings:
                first_layer.partition(left, right)

            parent_edges: dict[ContrastiveImage, Optional[int]] = {img: img.class_label for img in contrastive_images}
            second_layer = MultiLayerCliqueGraph(
                vertices=contrastive_images, parent=first_layer, parent_edges=parent_edges
            )
            self._merge_same_class_vertices(second_layer)
            second_layer.prune_cliques_without_neighbors()
            return CliqueGraphSampler(second_layer)
        else:
            raise ValueError(f"Unknown negative mining method: {self.negative_mining}")

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

    def _merge_same_class_vertices(self, graph: MultiLayerCliqueGraph[ContrastiveImage]) -> None:
        # NOTE(V1nce1): Should be functionality of MultiLayerCliqueGraph and could be extended
        for _, childrens in graph.inverse_parent_edges.items():
            children_list = list(childrens)
            for i in range(len(children_list) - 1):
                graph.merge(children_list[i], children_list[i + 1])


if __name__ == "__main__":
    ssl_config = SSLConfig(
        tff_selection="equidistant",
        negative_mining="overlapping",
        n_samples=15,
        feature_types=["body"],
        min_confidence=0.5,
        min_images_per_tracking=10,
        split_path="/workspaces/gorillatracker/data/splits/SSL/SSL-Video-Split_2024-04-18_percentage-80-10-10_split.pkl",
    )
    contrastive_sampler = ssl_config.get_contrastive_sampler("train", "cropped-images/2024-04-18")
    print(len(contrastive_sampler))
    for i in range(10):
        contrastive_image = contrastive_sampler[i * 10]
        print(contrastive_image)
        print(contrastive_sampler.positive(contrastive_image))
        print(contrastive_sampler.negative(contrastive_image))

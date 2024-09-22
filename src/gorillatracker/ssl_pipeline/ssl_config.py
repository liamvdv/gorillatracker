from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import List, Literal, Optional, Sequence

from sqlalchemy import Select, create_engine
from sqlalchemy.orm import Session, defer, load_only
from tqdm import tqdm

from gorillatracker.data.contrastive_sampler import (
    CliqueGraphSampler,
    ContrastiveClassSampler,
    ContrastiveImage,
    ContrastiveSampler,
    group_contrastive_images,
)
from gorillatracker.ssl_pipeline.data_structures import IndexedCliqueGraph, MultiLayerCliqueGraph
from gorillatracker.ssl_pipeline.dataset import GorillaDatasetKISZ
from gorillatracker.ssl_pipeline.dataset_splitter import SplitArgs
from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature
from gorillatracker.ssl_pipeline.negative_mining_queries import (
    find_overlapping_trackings,
    find_social_group_negatives,
    trackings_from_videos,
)
from gorillatracker.ssl_pipeline.queries import (
    associated_filter,
    bbox_filter,
    cached_filter,
    confidence_filter,
    feature_type_filter,
    min_count_filter,
    multiple_videos_filter,
)
from gorillatracker.ssl_pipeline.sampler import (
    embedding_distant_sample,
    equidistant_sample,
    movement_sample,
    random_sample,
)


@dataclass(kw_only=True)
class SSLConfig:
    tff_selection: Literal["random", "equidistant", "embeddingdistant", "movement"]
    negative_mining: Literal["random", "overlapping", "social_groups"]
    n_samples: int
    feature_types: List[str]
    min_confidence: float
    min_images_per_tracking: int
    split_path: Path
    width_range: tuple[Optional[int], Optional[int]]
    height_range: tuple[Optional[int], Optional[int]]
    forced_train_image_count: Optional[int] = None
    movement_delta: Optional[float] = None

    def __post_init__(self) -> None:
        assert self.tff_selection != "movement" or self.movement_delta is not None, "Combination not allowed"
    
    def generate_cache_key(self):
        key_data = (self.tff_selection, self.negative_mining, self.n_samples, 
                    tuple(self.feature_types), self.min_confidence, 
                    self.min_images_per_tracking, str(self.split_path), 
                    self.width_range, self.height_range, 
                    self.forced_train_image_count, self.movement_delta)
        key_bytes = pickle.dumps(key_data)
        return hashlib.md5(key_bytes).hexdigest()

    def get_contrastive_sampler(
        self,
        base_path: Path,
        partition: Literal["train", "val", "test"],
    ) -> ContrastiveSampler:
        # Check if we have a cached sampler
        cache_key = self.generate_cache_key()
        cached_sampler = self._load_cache(cache_key) if partition == "train" else None
        if cached_sampler is not None:
            print(f"Loaded cached sampler for key: {cache_key}")
            return cached_sampler
        
        engine = create_engine("postgresql+psycopg2://postgres:HyfCW95WnwmXmnQpBmiw@10.149.20.40:5432/postgres", echo=False)

        with Session(engine) as session:
            video_ids = self._get_video_ids(partition)
            tracking_frame_features = self._sample_tracking_frame_features(video_ids, session, partition)
            contrastive_images = self._create_contrastive_images(tracking_frame_features, base_path)
            if self.forced_train_image_count is not None and partition == "train":
                if len(contrastive_images) < self.forced_train_image_count:
                    raise ValueError(
                        f"Not enough images for training, required: {self.forced_train_image_count}, got {len(contrastive_images)}"
                    )
                contrastive_images = contrastive_images[: self.forced_train_image_count]

            sampler = self._create_contrastive_sampler(contrastive_images, video_ids, session)
            self._save_cache(cache_key, sampler) if partition == "train" else None
            return sampler

    def _get_video_ids(self, partition: Literal["train", "val", "test"]) -> List[int]:
        split = SplitArgs.load_pickle(str(self.split_path))
        if partition == "train":
            return split.train_video_ids()
        elif partition == "val":
            return split.val_video_ids()
        elif partition == "test":
            return split.test_video_ids()
        else:
            raise ValueError(f"Unknown partition: {partition}")

    def _create_contrastive_sampler(
        self,
        contrastive_images: List[ContrastiveImage],
        video_ids: List[int],
        session: Session,
    ) -> ContrastiveSampler:
        if self.negative_mining == "random":
            classes = group_contrastive_images(contrastive_images)
            return ContrastiveClassSampler(classes)
        elif self.negative_mining == "overlapping":
            return self._create_two_layer_clique_sampler(contrastive_images, video_ids, session)
        elif self.negative_mining == "social_groups":
            return self._create_three_layer_clique_sampler(contrastive_images, video_ids, session)
        else:
            raise ValueError(f"Unknown negative mining method: {self.negative_mining}")

    def _build_query(self, video_ids: List[int]) -> Select[tuple[TrackingFrameFeature]]:
        query = multiple_videos_filter(video_ids)
        query = cached_filter(query)
        query = associated_filter(query)
        query = feature_type_filter(query, self.feature_types)
        query = confidence_filter(query, self.min_confidence)
        query = bbox_filter(query, self.width_range[0], self.width_range[1], self.height_range[0], self.height_range[1])
        query = min_count_filter(query, self.min_images_per_tracking)
        # NOTE(memben): Use with care, as it might lead to performance issues if using other columns
        query = query.options(
            load_only(
                TrackingFrameFeature.tracking_frame_feature_id,
                TrackingFrameFeature.tracking_id,
                TrackingFrameFeature.frame_nr,
                TrackingFrameFeature.bbox_x_center_n,
                TrackingFrameFeature.bbox_y_center_n,
            ),
            # NOTE(memben): prevent n + 1 queries
            defer("*", raiseload=True),
        )
        return query

    def _sample_tracking_frame_features(self, video_ids: list[int], session: Session, partition: str) -> list[TrackingFrameFeature]:
        BATCH_SIZE = 200
        num_batches = len(video_ids) // BATCH_SIZE
        tffs: list[TrackingFrameFeature] = []
        for i in tqdm(
            range(num_batches + 1), desc="Sampling TrackingFrameFeatures", total=num_batches + 1, unit="batch"
        ):
            batch_video_ids = video_ids[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            batch_tffs = session.execute(self._build_query(batch_video_ids)).scalars().all()
            tffs += batch_tffs
        
        if partition == "val":
            return list(equidistant_sample(tffs, self.n_samples))

        if self.tff_selection == "random":
            return list(random_sample(tffs, self.n_samples))
        elif self.tff_selection == "equidistant":
            return list(equidistant_sample(tffs, self.n_samples))
        elif self.tff_selection == "embeddingdistant":
            return list(embedding_distant_sample(tffs, self.n_samples))
        elif self.tff_selection == "movement":
            assert self.movement_delta is not None, "Movement delta must be set"
            samples = list(movement_sample(tffs, self.n_samples, self.movement_delta))
            print(f"Sampled {len(samples)} tracking frame features through movement")
            return samples
        else:
            raise ValueError(f"Unknown TFF selection method: {self.tff_selection}")

    def _create_contrastive_images(
        self, tracked_features: List[TrackingFrameFeature], base_path: Path
    ) -> List[ContrastiveImage]:
        return [
            ContrastiveImage(str(f.tracking_frame_feature_id), f.cache_path(base_path), f.tracking_id)  # type: ignore
            for f in tracked_features
        ]

    def _merge_same_class_vertices(self, graph: MultiLayerCliqueGraph[ContrastiveImage]) -> None:
        # NOTE(V1nce1): Should be functionality of MultiLayerCliqueGraph and could be extended
        for _, childrens in graph.inverse_parent_edges.items():
            children_list = list(childrens)
            for i in range(len(children_list) - 1):
                graph.merge(children_list[i], children_list[i + 1])

    def _create_two_layer_clique_sampler(
        self, contrastive_images: List[ContrastiveImage], video_ids: List[int], session: Session
    ) -> CliqueGraphSampler:
        trackings = session.execute(trackings_from_videos(video_ids)).scalars().all()
        first_layer: IndexedCliqueGraph[int] = IndexedCliqueGraph([tracking.tracking_id for tracking in trackings])
        overlapping_trackings = find_overlapping_trackings(session, video_ids)
        for left, right in overlapping_trackings:
            first_layer.partition(left, right)

        parent_edges: dict[ContrastiveImage, Optional[int]] = {img: img.class_label for img in contrastive_images}
        second_layer = MultiLayerCliqueGraph(vertices=contrastive_images, parent=first_layer, parent_edges=parent_edges)
        self._merge_same_class_vertices(second_layer)
        second_layer.prune_cliques_without_neighbors()
        return CliqueGraphSampler(second_layer)

    def _create_three_layer_clique_sampler(
        self, contrastive_images: List[ContrastiveImage], video_ids: List[int], session: Session
    ) -> CliqueGraphSampler:
        first_layer: IndexedCliqueGraph[int] = IndexedCliqueGraph(list(video_ids))
        video_negatives = find_social_group_negatives(session, video_ids)
        for left, right in video_negatives:
            first_layer.partition(left, right)
        trackings: Sequence[Tracking] = session.execute(trackings_from_videos(video_ids)).scalars().all()
        first_parent_edges: dict[int, Optional[int]] = {
            trackings.tracking_id: trackings.video_id for trackings in trackings
        }
        second_layer = MultiLayerCliqueGraph(
            vertices=list(first_parent_edges.keys()), parent=first_layer, parent_edges=first_parent_edges
        )
        second_parent_edges: dict[ContrastiveImage, Optional[int]] = {
            img: img.class_label for img in contrastive_images
        }
        third_layer = MultiLayerCliqueGraph(
            vertices=contrastive_images, parent=second_layer, parent_edges=second_parent_edges
        )
        self._merge_same_class_vertices(third_layer)
        third_layer.prune_cliques_without_neighbors()
        return CliqueGraphSampler(third_layer)
    
    def _load_cache(self, cache_key):
        cache_file = Path(f"cache_ssl_config/{cache_key}.pkl")
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, cache_key, obj):
        cache_file = Path(f"cache_ssl_config/{cache_key}.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump(obj, f)


if __name__ == "__main__":
    ssl_config = SSLConfig(
        tff_selection="movement",
        movement_delta=0.03,
        negative_mining="overlapping",
        n_samples=10,
        feature_types=["face_45"],
        min_confidence=0.5,
        min_images_per_tracking=3,
        width_range=(100, 10000),
        height_range=(100, 10000),
        split_path=Path(
            "/workspaces/gorillatracker/data/splits/SSL/sweep/30k-SSL-Video-Split-1_2024-04-18_percentage-95-5-0_split_20240812_2337.pkl"
        ),
    )
    import time

    before = time.time()
    contrastive_sampler = ssl_config.get_contrastive_sampler(Path("cropped-images-squared/2024-04-18"), "train")
    after = time.time()
    print(f"Time: {after - before}")
    print(len(contrastive_sampler))
    # for i in range(10):
    #     contrastive_image = contrastive_sampler[i * 10]
    #     print(contrastive_image)
    #     print(contrastive_sampler.positive(contrastive_image))
    #     print(contrastive_sampler.negative(contrastive_image))
import logging
import multiprocessing
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from gorillatracker.ssl_pipeline.data_structures import DirectedBipartiteGraph
from gorillatracker.ssl_pipeline.helpers import BoundingBox, groupby_frame
from gorillatracker.ssl_pipeline.models import TaskType, TrackingFrameFeature
from gorillatracker.ssl_pipeline.queries import get_next_task, load_features, load_tracked_features

log = logging.getLogger(__name__)


@dataclass
class Association:
    reference: TrackingFrameFeature
    unassociated: TrackingFrameFeature


class Correlator(Protocol):
    def __call__(
        self,
        tracked_features: list[TrackingFrameFeature],
        untracked_features: list[TrackingFrameFeature],
        threshold: float = 0.7,
    ) -> list[Association]: ...


def build_bipartite_graph(
    tracked_features: list[TrackingFrameFeature],
    untracked_features: list[TrackingFrameFeature],
    threshold: float = 0.7,
) -> DirectedBipartiteGraph[TrackingFrameFeature]:
    graph = DirectedBipartiteGraph(tracked_features, untracked_features)

    tracked_frames = groupby_frame(tracked_features)
    untracked_frames = groupby_frame(untracked_features)

    for frame_nr, tracked_frame_features in tracked_frames.items():
        untracked_frame_features = untracked_frames[frame_nr]
        for tf in tracked_frame_features:
            tf_bbox = BoundingBox.from_tracking_frame_feature(tf)
            for uf in untracked_frame_features:
                uf_bbox = BoundingBox.from_tracking_frame_feature(uf)
                if tf_bbox.intersection_over_smallest_area(uf_bbox) > threshold:
                    graph.add_edge(tf, uf)

    return graph


def one_to_one_correlator(
    tracked_features: list[TrackingFrameFeature],
    untracked_features: list[TrackingFrameFeature],
    threshold: float = 0.7,
) -> list[Association]:
    graph = build_bipartite_graph(tracked_features, untracked_features, threshold)
    return list(map(lambda a: Association(*a), graph.bijective_relationships()))


def correlate_and_update(
    tracked_features: list[TrackingFrameFeature],
    untracked_features: list[TrackingFrameFeature],
    correlator: Correlator,
    threshold: float = 0.7,
) -> None:
    for untracked_feature in untracked_features:
        untracked_feature.tracking_id = None

    associations = correlator(tracked_features, untracked_features, threshold)
    for association in associations:
        association.unassociated.tracking_id = association.reference.tracking_id


def correlate_worker(
    feature_type: str,
    correlator: Correlator,
    engine: Engine,
) -> None:
    # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    engine.dispose(close=False)
    session_cls = sessionmaker(bind=engine)
    with session_cls() as session:
        for task in get_next_task(session, TaskType.CORRELATE, task_subtype=feature_type):
            tracked = task.get_key_value("tracked_feature_type")
            untracked = task.get_key_value("untracked_feature_type")
            threshold = float(task.get_key_value("threshold"))
            tracked_features = load_tracked_features(session, task.video_id, [tracked])
            untracked_features = load_features(session, task.video_id, [untracked])
            correlate_and_update(list(tracked_features), list(untracked_features), correlator, threshold)


def multiprocess_correlate(
    feature_type: str,
    correlator: Correlator,
    engine: Engine,
    process_count: int,
) -> None:
    processes: list[multiprocessing.Process] = []
    for _ in range(process_count):
        process = multiprocessing.Process(target=correlate_worker, args=(feature_type, correlator, engine))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    log.info("Correlating completed")

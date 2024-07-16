import logging
import random
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import shared_memory
from typing import Iterator, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance
from tqdm import tqdm

from gorillatracker.ssl_pipeline.models import TrackingFrameFeature

log = logging.getLogger(__name__)


def group_by_tracking_id(frame_features: list[TrackingFrameFeature]) -> defaultdict[int, list[TrackingFrameFeature]]:
    grouped = defaultdict(list)
    for feature in frame_features:
        assert feature.tracking_id is not None
        grouped[feature.tracking_id].append(feature)
    return grouped


### RandomSampler ###


def random_sample(frame_features: list[TrackingFrameFeature], n_samples: int) -> Iterator[TrackingFrameFeature]:
    grouped = group_by_tracking_id(frame_features)
    for features in grouped.values():
        num_samples = min(len(features), n_samples)
        yield from random.sample(features, num_samples)


### EquidistantSampler ###


def tracking_equidistant_sample(features: list[TrackingFrameFeature], n_samples: int) -> list[TrackingFrameFeature]:
    assert all(
        t1.tracking_id == t2.tracking_id for t1, t2 in zip(features, features[1:])
    ), "Features must belong to the same tracking"
    sorted_features = sorted(features, key=lambda x: x.frame_nr)
    num_features = len(features)
    if num_features <= n_samples:
        return features
    interval = (num_features - 1) // (n_samples - 1) if n_samples > 1 else 0
    indices = [i * interval for i in range(n_samples)]
    return [sorted_features[i] for i in indices]


def equidistant_sample(frame_features: list[TrackingFrameFeature], n_samples: int) -> Iterator[TrackingFrameFeature]:
    grouped = group_by_tracking_id(frame_features)
    for features in grouped.values():
        yield from tracking_equidistant_sample(features, n_samples)


### EmbeddingDistantSampler ###


def max_min_dispatch(
    shm_name_ids: str,
    shm_name_embeddings: str,
    tff_ids: list[int],
    n_samples: int,
    shape_ids: tuple[int, ...],
    shape_embeddings: tuple[int, ...],
) -> list[int]:
    shm_ids = shared_memory.SharedMemory(name=shm_name_ids)
    shm_embeddings = shared_memory.SharedMemory(name=shm_name_embeddings)
    ids: NDArray[np.int32] = np.ndarray(shape_ids, dtype=np.int64, buffer=shm_ids.buf)
    embeddings: NDArray[np.float32] = np.ndarray(shape_embeddings, dtype=np.float32, buffer=shm_embeddings.buf)

    mask = np.isin(ids, tff_ids, assume_unique=True)
    filtered_ids = ids[mask]
    if len(filtered_ids) < n_samples:
        return list(filtered_ids)
    filtered_embeddings = embeddings[mask]
    ranking = max_min_dist_ranking(filtered_embeddings, n_samples)
    assert len(ranking) == n_samples
    return list(filtered_ids[ranking])


def max_min_dist_ranking(embeddings: NDArray[np.float32], n_samples: int) -> NDArray[np.int32]:
    dist_matrix = distance.cdist(embeddings, embeddings, "cosine")
    ranked_points = [0]
    remaining_points = set(range(1, embeddings.shape[0]))
    while len(ranked_points) < n_samples and remaining_points:
        max_min_dist = -np.inf
        best_point: Optional[int] = None
        for point in remaining_points:
            min_dist = min(dist_matrix[point, ranked] for ranked in ranked_points)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_point = point
        assert best_point is not None
        ranked_points.append(best_point)
        remaining_points.remove(best_point)
    return np.array(ranked_points)


def embedding_distant_sample(
    frame_features: list[TrackingFrameFeature], n_samples: int
) -> Iterator[TrackingFrameFeature]:
    # HACK(memben): hardcoded for now
    IDS = "/workspaces/gorillatracker/video_data/ssl_embeddings/ids.npy"
    EMBEDDINGS = "/workspaces/gorillatracker/video_data/ssl_embeddings/embeddings.npy"

    ids: NDArray[np.int32] = np.load(IDS, mmap_mode="r")
    embeddings: NDArray[np.float32] = np.load(EMBEDDINGS, mmap_mode="r")

    tracking_id_grouped = group_by_tracking_id(frame_features)

    shm_ids = shared_memory.SharedMemory(create=True, size=ids.nbytes)
    shm_embeddings = shared_memory.SharedMemory(create=True, size=embeddings.nbytes)
    shared_ids: NDArray[np.int32] = np.ndarray(ids.shape, dtype=ids.dtype, buffer=shm_ids.buf)
    shared_embeddings: NDArray[np.float32] = np.ndarray(
        embeddings.shape, dtype=embeddings.dtype, buffer=shm_embeddings.buf
    )
    np.copyto(shared_ids, ids)
    np.copyto(shared_embeddings, embeddings)

    with ProcessPoolExecutor(max_workers=200) as executor:
        futures: list[tuple[Future[list[int]], list[TrackingFrameFeature]]] = []
        for features in tqdm(tracking_id_grouped.values(), desc="Dispatching Max Min Distance Tasks"):
            future = executor.submit(
                max_min_dispatch,
                shm_ids.name,
                shm_embeddings.name,
                [feature.tracking_frame_feature_id for feature in features],
                n_samples,
                ids.shape,
                embeddings.shape,
            )
            futures.append((future, features))

        for future, features in tqdm(futures, desc="Processing Max Min Distance Tasks"):
            sampled_ids = future.result()
            yield from (feature for feature in features if feature.tracking_frame_feature_id in sampled_ids)

    shm_ids.close()
    shm_ids.unlink()
    shm_embeddings.close()
    shm_embeddings.unlink()

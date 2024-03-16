"""
This is a demo file to show how to use the SSL pipeline to track any animal in a video.

The pipeline consists of the following steps:
1. Create a dataset adapter for the dataset of interest. (dataset_adapter.py)
2. Use a tracking model to track the animal in the video. (video_tracker.py) 
    a. This should in be a YOLOv8 model (single_cls) trained on the body of the animal of interest.
    b. A use case specific metadata extractor is used to extract the metadata from the video (camera, start time).
    c. The tracking is done using the tracker settings.
3. Store the tracking results in a database. (models.py)
4. (Optional) Add additional features to the tracking results. (video_feature_mapper.py)
5. ...
"""

import logging
import random
from pathlib import Path

from gorillatracker.ssl_pipeline.dataset_adapter import GorillaDatasetAdapter, SSLDatasetAdapter
from gorillatracker.ssl_pipeline.video_feature_mapper import multiproces_feature_mapping
from gorillatracker.ssl_pipeline.video_tracker import multiprocess_video_tracker
from gorillatracker.ssl_pipeline.visualizer import multiprocess_visualize_video

log = logging.getLogger(__name__)


def visualize_pipeline(
    dataset_adapter: SSLDatasetAdapter,
    dest_dir: Path,
    n_videos: int = 30,
    max_worker_per_gpu: int = 8,
    gpus: list[int] = [0],
) -> None:
    """
    Visualize the tracking results of the pipeline.

    Args:
        dataset_adapter (SSLDatasetAdapter): The dataset adapter to use.
        dest (Path): The destination to save the visualizations.
        n_videos (int, optional): The number of videos to visualize. Defaults to 20.
        gpus (list[int], optional): The GPUs to use for tracking. Defaults to [0].

    Returns:
        None, the visualizations are saved to the destination and to the SSLDatasetAdapter.
    """

    random.seed(42)
    # NOTE: unprocessed_videos is not idempotent
    videos = sorted(dataset_adapter.unprocessed_videos())
    to_track = random.sample(videos, n_videos)

    multiprocess_video_tracker(
        dataset_adapter.body_model,
        dataset_adapter.yolo_kwargs,
        to_track,
        dataset_adapter.tracker_config,
        dataset_adapter.metadata_extractor,
        dataset_adapter.engine,
        max_worker_per_gpu=max_worker_per_gpu,
        gpus=gpus,
    )

    for yolo_model, yolo_kwargs, correlator, type in dataset_adapter.feature_models():
        multiproces_feature_mapping(
            yolo_model,
            yolo_kwargs,
            type,
            to_track,
            dataset_adapter.engine,
            correlator,
            max_worker_per_gpu=max_worker_per_gpu,
            gpus=gpus,
        )

    multiprocess_visualize_video(to_track, dataset_adapter.engine, dest_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset_adapter = GorillaDatasetAdapter(db_uri="sqlite:///test.db")
    dataset_adapter.setup_database()
    visualize_pipeline(
        dataset_adapter, Path("/workspaces/gorillatracker/video_output"), n_videos=100, max_worker_per_gpu=12, gpus=[0]
    )

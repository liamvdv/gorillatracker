"""
This is a demo file to show how to use the SSL pipeline to track any animal in a video.

The pipeline consists of the following steps:
1. Create a dataset adapter for the dataset of interest. (dataset.py)
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

from gorillatracker.ssl_pipeline.dataset import GorillaDataset, SSLDataset
from gorillatracker.ssl_pipeline.feature_mapper import correlate_videos
from gorillatracker.ssl_pipeline.video_preprocessor import preprocess_videos
from gorillatracker.ssl_pipeline.video_processor import multiprocess_predict_and_store, multiprocess_track_and_store
from gorillatracker.ssl_pipeline.visualizer import multiprocess_visualize_video

log = logging.getLogger(__name__)


def visualize_pipeline(
    dataset: SSLDataset,
    version: str,
    dest_dir: Path,
    n_videos: int = 30,
    sampled_fps: int = 10,
    max_worker_per_gpu: int = 8,
    gpus: list[int] = [0],
) -> None:
    """
    Visualize the tracking results of the pipeline.

    Args:
        dataset (SSLDataset): The dataset to use.
        dest (Path): The destination to save the visualizations.
        n_videos (int, optional): The number of videos to visualize. Defaults to 20.
        sampled_fps (int, optional): The FPS to sample the video at. Defaults to 10.
        max_worker_per_gpu (int, optional): The maximum number of workers per GPU. Defaults to 8.
        gpus (list[int], optional): The GPUs to use for tracking. Defaults to [0].

    Returns:
        None, the visualizations are saved to the destination and to the SSLDataset.
    """

    random.seed(42)  # For reproducibility
    videos = sorted(dataset.video_paths)
    to_track = random.sample(videos, n_videos)

    # preprocess_videos(to_track, version, sampled_fps, dataset.engine, dataset.metadata_extractor)

    # multiprocess_track_and_store(
    #     version,
    #     dataset.body_model,
    #     dataset.yolo_kwargs,
    #     to_track,
    #     dataset.tracker_config,
    #     dataset.engine,
    #     "body",  # NOTE(memben): Tracking will always be done on bodies
    #     max_worker_per_gpu=max_worker_per_gpu,
    #     gpus=gpus,
    # )

    # for yolo_model, yolo_kwargs, correlator, type in dataset.feature_models():
    #     multiprocess_predict_and_store(
    #         version,
    #         yolo_model,
    #         yolo_kwargs,
    #         to_track,
    #         dataset.engine,
    #         type,
    #         max_worker_per_gpu=max_worker_per_gpu,
    #         gpus=gpus,
    #     )

    for _, _, correlator, type in dataset.feature_models():
        correlate_videos(
            version,
            to_track,
            dataset.engine,
            correlator,
            type,
        )

    multiprocess_visualize_video(to_track, version, dataset.engine, dest_dir)


if __name__ == "__main__":
    import os

    # os.remove("test.db")
    logging.basicConfig(level=logging.INFO)
    dataset = GorillaDataset("sqlite:///test.db")
    # dataset.setup_database()
    # dataset.setup_cameras()
    visualize_pipeline(
        dataset,
        "2024-04-03",
        Path("/workspaces/gorillatracker/video_output"),
        n_videos=5,
        max_worker_per_gpu=12,
        gpus=[0],
    )

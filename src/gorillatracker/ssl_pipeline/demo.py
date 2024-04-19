"""
This is a demo file to show how to use the SSL pipeline to track any animal in a video.

The pipeline consists of the following steps:
1. Create a dataset adapter for the dataset of interest. (dataset.py)
2. Extract the metadata from the videos. (video_preprocessor.py)
3. Use a tracking model to track the animal in the video. (video_processor.py) 
    - This should in be a YOLOv8 model (single_cls) trained on the body of the animal of interest.
4. Store the tracking results in a database. (models.py)
5. (Optional) Add additional features to the tracking results and correlate them. (video_processor.py and feature_mapper.py)
6. (Optional) Visualize the tracking results. (visualizer.py)
7. ...
"""

import logging
import random
from pathlib import Path

from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.dataset import GorillaDataset, SSLDataset
from gorillatracker.ssl_pipeline.feature_mapper import multiprocess_correlate_videos
from gorillatracker.ssl_pipeline.helpers import remove_processed_videos
from gorillatracker.ssl_pipeline.queries import load_processed_videos
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
        version (str): The version of the pipeline.
        dest_dir (Path): The destination to save the visualizations.
        n_videos (int, optional): The number of videos to visualize. Defaults to 20.
        sampled_fps (int, optional): The FPS to sample the video at. Defaults to 10.
        max_worker_per_gpu (int, optional): The maximum number of workers per GPU. Defaults to 8.
        gpus (list[int], optional): The GPUs to use for tracking. Defaults to [0].

    Returns:
        None, the visualizations are saved to the destination and to the SSLDataset.
    """

    video_paths = sorted(dataset.video_paths)

    # NOTE(memben): For the production pipeline we should do this for every step
    # owever, in this context, we want the process to fail if not all videos are preprocessed for debugging.
    with Session(dataset.engine) as session:
        preprocessed_videos = list(load_processed_videos(session, version, []))
        video_paths = remove_processed_videos(video_paths, preprocessed_videos)

    random.seed(42)  # For reproducibility
    to_track = random.sample(video_paths, n_videos)

    preprocess_videos(to_track, version, sampled_fps, dataset.engine, dataset.metadata_extractor)

    multiprocess_track_and_store(
        version,
        dataset.body_model_path,
        dataset.yolo_kwargs,
        to_track,
        dataset.tracker_config,
        dataset.engine,
        "body",  # NOTE(memben): Tracking will always be done on bodies
        max_worker_per_gpu=max_worker_per_gpu,
        gpus=gpus,
    )

    for yolo_model, yolo_kwargs, _, type in dataset.feature_models():
        multiprocess_predict_and_store(
            version,
            yolo_model,
            yolo_kwargs,
            to_track,
            dataset.engine,
            type,
            max_worker_per_gpu=max_worker_per_gpu,
            gpus=gpus,
        )

    for _, _, correlator, type in dataset.feature_models():
        multiprocess_correlate_videos(
            version,
            to_track,
            dataset.engine,
            correlator,
            type,
        )

    multiprocess_visualize_video(to_track, version, dataset.engine, dest_dir)


if __name__ == "__main__":
    version = "2024-04-09"
    logging.basicConfig(level=logging.INFO)
    dataset = GorillaDataset("sqlite:///test.db")
    # NOTE(memben): for setup only once
    visualize_pipeline(
        dataset,
        version,
        Path("/workspaces/gorillatracker/video_output"),
        n_videos=8,
        max_worker_per_gpu=4,
        gpus=[0],
    )
    dataset.post_setup(version)

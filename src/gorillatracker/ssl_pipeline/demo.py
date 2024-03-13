"""
This is a demo file to show how to use the SSL pipeline to track any animal in a video.

The pipeline consists of the following steps:
1. Use a tracking model to track the animal in the video. (video_tracker.py) 
    a. This should in be a YOLOv8 model (single_cls) trained on the body of the animal of interest.
    b. A use case specific metadata extractor is used to extract the metadata from the video (camera, start time).
2. Store the tracking results in a database. (models.py)
3. (Optional) Add additional features to the tracking results. (video_feature_mapper.py)
4. 
"""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from gorillatracker.ssl_pipeline.dataset_adapter import GorillaDatasetAdapter, SSLDatasetAdapter
from gorillatracker.ssl_pipeline.models import Base
from gorillatracker.ssl_pipeline.video_tracker import multiprocess_video_tracker
from gorillatracker.ssl_pipeline.visualizer import visualize_video


def visualize_pipeline(videos: list[Path], dataset_adapter: SSLDatasetAdapter, engine: Engine, dest: Path) -> None:
    """
    Visualize the tracking results of the pipeline.

    Args:
        videos (list[Path]): The videos to visualize.
        dataset_adapter (SSLDatasetAdapter): The dataset adapter to use.
        engine (Engine): The database engine to use.
        dest (Path): The destination to save the visualizations to.

    Returns:
        None, the visualizations are saved to the destination.
    """

    multiprocess_video_tracker(
        dataset_adapter.get_body_model(),
        videos,
        dataset_adapter.get_tracker_config(),
        engine,
        dataset_adapter.get_metadata_extractor(),
    )

    for video in tqdm(videos, desc="Visualizing videos", unit="video"):
        visualize_video(video, engine, dest)


if __name__ == "__main__":
    engine = create_engine("sqlite:///test.db")
    Base.metadata.create_all(engine)
    dataset_adapter = GorillaDatasetAdapter()
    video = Path("video_data/R108_20230213_301.mp4")
    visualize_pipeline([video], dataset_adapter, engine, Path("output.mp4"))
    print("Visualizations saved to output.mp4")

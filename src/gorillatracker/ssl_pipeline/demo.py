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
4. 
"""

from pathlib import Path


from tqdm import tqdm

from gorillatracker.ssl_pipeline.dataset_adapter import GorillaDatasetAdapter, SSLDatasetAdapter
from gorillatracker.ssl_pipeline.video_tracker import multiprocess_video_tracker
from gorillatracker.ssl_pipeline.visualizer import visualize_video


def visualize_pipeline(dataset_adapter: SSLDatasetAdapter, dest: Path) -> None:
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
        dataset_adapter.body_model,
        dataset_adapter.videos[:30],
        # [Path("video_data/R108_20230213_301.mp4")],
        dataset_adapter.tracker_config,
        dataset_adapter.metadata_extractor,
        dataset_adapter.engine,
    )
    
    print("Visualizing videos")

    # for video in tqdm(dataset_adapter.videos, desc="Visualizing videos", unit="video"):
    #     visualize_video(video, dataset_adapter.engine, dest)


if __name__ == "__main__":
    dataset_adapter = GorillaDatasetAdapter(db_uri="sqlite:///test.db")
    # dataset_adapter.setup_database()
    visualize_pipeline(dataset_adapter, Path("/workspaces/gorillatracker/video_output"))

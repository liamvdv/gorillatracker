import logging

from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.dataset import GorillaDatasetKISZ, SSLDataset
from gorillatracker.ssl_pipeline.feature_mapper import multiprocess_correlate, one_to_one_correlator
from gorillatracker.ssl_pipeline.helpers import remove_processed_videos
from gorillatracker.ssl_pipeline.models import TaskType
from gorillatracker.ssl_pipeline.queries import load_preprocessed_videos, reset_dependent_tasks_status
from gorillatracker.ssl_pipeline.video_preprocessor import preprocess_videos
from gorillatracker.ssl_pipeline.video_processor import multiprocess_predict, multiprocess_track

log = logging.getLogger(__name__)


def run_pipeline(
    dataset: SSLDataset,
    version: str,
    target_output_fps: int = 10,
    max_worker_per_gpu: int = 8,
    gpu_ids: list[int] = [0],
) -> None:
    video_paths = sorted(dataset.video_paths)

    with Session(dataset.engine) as session:
        preprocessed_videos = list(load_preprocessed_videos(session, version))
        videos_to_track = remove_processed_videos(video_paths, preprocessed_videos)

    max_workers = len(gpu_ids) * max_worker_per_gpu

    preprocess_videos(
        videos_to_track,
        version,
        target_output_fps,
        dataset.engine,
        dataset.metadata_extractor,
        dataset.video_insert_hook,
    )

    body_model_path, yolo_body_kwargs = dataset.get_yolo_model_config(dataset.BODY)
    multiprocess_track(
        dataset.BODY,  # NOTE(memben): Tracking will always be done on bodies
        body_model_path,
        yolo_body_kwargs,
        dataset.tracker_config,
        dataset.engine,
        max_worker_per_gpu=max_worker_per_gpu,
        gpu_ids=gpu_ids,
    )

    for feature_type in dataset.features:
        yolo_model, yolo_kwargs = dataset.get_yolo_model_config(feature_type)
        multiprocess_predict(
            feature_type,
            yolo_model,
            yolo_kwargs,
            dataset.engine,
            max_worker_per_gpu=max_worker_per_gpu,
            gpu_ids=gpu_ids,
        )

        multiprocess_correlate(feature_type, one_to_one_correlator, dataset.engine, max_workers)


def redo_failed_correlation(
    dataset: SSLDataset,
    max_workers: int,
) -> None:
    """Correlation has no dependencies set on the previous steps,
    so it need be redone for failed previous steps."""

    with Session(dataset.engine) as session:
        reset_dependent_tasks_status(session, dependent=TaskType.CORRELATE, provider=TaskType.TRACK)
        reset_dependent_tasks_status(session, dependent=TaskType.CORRELATE, provider=TaskType.PREDICT)

    for feature_type in dataset.features:
        multiprocess_correlate(feature_type, one_to_one_correlator, dataset.engine, max_workers)


if __name__ == "__main__":
    version = "2024-04-18"
    logging.basicConfig(level=logging.INFO)
    dataset = GorillaDatasetKISZ()
    redo_failed_correlation(dataset, max_workers=90)
    # dataset.post_setup()

import json
import multiprocessing
import os
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, cast

import ultralytics
from ultralytics import YOLO

import gorillatracker.scripts.get_timestamp as gt

# setting a type here breaks the queue for some reason - i dont know why, but it makes me want to nuke mypy off the face of the earth
gpu_queue: multiprocessing.Queue = multiprocessing.Queue()  # type: ignore


class Config:
    def __init__(self) -> None:
        self.models: List[YOLO] = []
        self.post_process_functions: List[Callable[[List[ultralytics.engine.results.Results], str], None]] = []
        self.yolo_args: Dict[str, Union[bool, int, str]] = {}
        self.checkpoint_path: str = ""


config: Config = Config()


def save_result_to_json(
    results: List[ultralytics.engine.results.Results],
    video_path: str = "",
    json_folder: str = "",
    overwrite: bool = False,
    min_conf: float = 0.5,
) -> None:
    """
    Save the results to a JSON file.

    Args:
        results (List[ultralytics.engine.results.Results]): The results to save.
        video_path (str): The path to the video file. Used to get the frame count.
        json_folder (str): The folder to save the JSON file to.
        overwrite (bool): Whether to overwrite the JSON file if it already exists.
        min_conf (float): The minimum confidence for a box to be saved.

    Returns:
        None
    """

    assert video_path != "", "video_path must be specified"
    assert json_folder != "", "json_folder must be specified"

    file_name = Path(video_path).stem

    json_path = f"{json_folder}/{file_name}.json"
    if os.path.exists(json_path) and not overwrite:
        print(f"JSON file {json_path} already exists, skipping")
        return
    time_stamp = gt.get_time_stamp(video_path)
    frame_count = os.popen(
        f"ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 {video_path}"
    ).read()
    labeled_video_frames: List[List[Dict[str, float]]] = [[] for _ in range(int(frame_count))]

    for idx, result in enumerate(results):
        frame_index = 0
        for frame in result:
            boxes = frame.boxes.xywhn.tolist()
            confs = frame.boxes.conf.tolist()
            for box, conf in zip(boxes, confs):
                if conf < min_conf:
                    continue
                x, y, w, h = box
                box = {"class": idx, "center_x": x, "center_y": y, "w": w, "h": h, "conf": conf}
                labeled_video_frames[frame_index].append(box)
            frame_index += 1
    json.dump({"time_stamp": time_stamp, "labels": labeled_video_frames}, open(json_path, "w"), indent=4)


def predict_video(
    input_path: str,
    models: List[YOLO],
    gpu_id: int = 0,
) -> None:
    """
    Predicts labels for objects in a video using multiple YOLO models.
    Most of the parameters are passed through the global config variable.
    Config must be set before calling this function.

    Parameters:
    - input_path (str): The path to the input video file.
    - models (List[YOLO]): A list of YOLO models to use for prediction.
    - gpu_id (int): The ID of the GPU to use for prediction.
    Returns:
        None
    """

    global config

    # error checking

    assert config != {}, "config must be set before calling predict_video"
    assert len(models) > 0, "models must be a list of at least one model"
    assert os.path.exists(input_path), f"input_path {input_path} does not exist"

    # grabbing parameters from config

    post_process_functions: List[
        Callable[[List[ultralytics.engine.results.Results], str], None]
    ] = config.post_process_functions
    yolo_args = config.yolo_args
    checkpoint_path = config.checkpoint_path
    # file_name = os.path.basename(input_path)
    # file_name = input_path.split("/")[-1]
    # file_name_split = file_name.split(".")[:-1]
    # file_name = ".".join(file_name_split)

    # getting the generators for each model

    results = []
    for model in models:
        results.append(model.predict(input_path, stream=True, device=gpu_id, **yolo_args))

    # using the generators in the post-processing function

    if post_process_functions is not None:
        for post_process_function in post_process_functions:
            post_process_function(results, input_path)

    # updating the checkpoint

    if checkpoint_path != "":
        open(checkpoint_path, "w").write(input_path)


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton. # TODO UNTESTED
    """

    _instances: Dict[int, multiprocessing.Process] = {}

    def __call__(cls, *args, **kwargs) -> multiprocessing.Process:  # type: ignore
        pid = multiprocessing.current_process().pid
        if pid not in cls._instances:
            instance: multiprocessing.Process = super().__call__(*args, **kwargs)
            pid = cast(int, multiprocessing.current_process().pid)
            cls._instances[pid] = instance
        return cls._instances[pid]


class Singleton(metaclass=SingletonMeta):
    def __init__(self, models: List[YOLO]):
        self.value = models

    def get_models(self) -> List[YOLO]:
        return self.value


def worker_function(input_path: str) -> None:
    """
    Process the given input video file.

    Parameters:
    - input_path (str): The path of the input video file.
    Returns:
        None
    """

    global config, gpu_queue
    singleton = Singleton(config.models)
    gpu = gpu_queue.get()
    print(f"Processing {input_path} on GPU {gpu}")
    predict_video(input_path, singleton.get_models(), gpu)
    print(f"Finished processing {input_path} on GPU {gpu}")
    gpu_queue.put(gpu)


def predict_video_multiprocessing(
    post_process_functions: List[Callable[[List[ultralytics.engine.results.Results], str], None]],
    models: List[YOLO] = [
        YOLO("/workspaces/gorillatracker/src/gorillatracker/scripts/spac_tracking/weights/body.pt"),
        YOLO("/workspaces/gorillatracker/src/gorillatracker/scripts/spac_tracking/weights/face.pt"),
    ],
    yolo_args: Dict[str, Union[bool, str, int]] = {"verbose": False},
    pool_per_gpu: int = 4,
    gpu_ids: List[int] = [0],
    checkpoint_path: str = "",
    video_dir: Optional[str] = None,
    video_paths: Optional[List[str]] = None,
    **kwargs: Union[str, List[str], bool],
) -> None:
    """
    Perform video prediction using multiple YOLO models in parallel using multiprocessing.

    Parameters:
    - post_process_functions (List[Callable[[List[List[Dict]]], None]]): List of function to apply post-processing to the predicted results.
            The function will get the results passed as the first argument and the video path as the second argument.
            The function shouldn't return anything.
            If you want to pass additional arguments to the function, use functools.partial.
    - models (List[YOLO]): List of YOLO models to use for prediction.
    - yolo_args (Dict): Additional arguments to pass to the YOLO models.
    - pool_per_gpu (int): Number of processes to use for parallel prediction per GPU.
    - gpu_ids (List[int]): List of GPU IDs to use for prediction.
    - checkpoint_path (str): Path to a checkpoint file. If the file exists, the script will skip videos that have already been processed.
    - video_dir (str): Path to a directory containing videos to process. If specified, video_paths must be None.
    - video_paths (List[str]): List of paths to videos to process. If specified, video_dir must be None.
    - **kwargs: Additional keyword arguments.
    Returns:
    None
    """

    # setting global variables for other functions to use

    global config
    config.models = models
    config.post_process_functions = post_process_functions
    config.yolo_args = yolo_args
    config.checkpoint_path = checkpoint_path

    # creating a queue of gpu ids to be used by the worker function

    global gpu_queue
    for gpu_id in gpu_ids:
        for _ in range(pool_per_gpu):
            gpu_queue.put(gpu_id)

    # error checking

    assert len(models) > 0, "models must be a list of at least one model"
    assert len(gpu_ids) > 0, "gpu_ids must be a list of at least one GPU ID"
    assert len(post_process_functions) > 0, "post_process_functions must be a list of at least one function"
    assert video_paths is not None or video_dir is not None, "Either video_paths or video_dir must be specified"
    assert video_paths is None or video_dir is None, "Only one of video_paths or video_dir can be specified"

    video_paths_: List[str] = []
    # getting the video paths
    if video_dir is not None:
        video_paths = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
        video_paths_ = video_paths
    else:
        video_paths_ = cast(List[str], video_paths)

    # if a checkpoint file is specified, skip videos that have already been processed (also some error checking)

    print(f"Processing {len(video_paths_)} videos")
    if checkpoint_path != "":
        if os.path.exists(checkpoint_path):
            print("Checkpoint found, resuming from last processed video")
            last_processed_video = open(checkpoint_path, "r").read()
            print(last_processed_video)
            if last_processed_video not in video_paths_:
                print("Checkpoint file contains a video that doesn't exist, starting from the beginning")
                open(checkpoint_path, "w").close()
            last_processed_video_index = video_paths_.index(last_processed_video)
            print(f"Skipping {last_processed_video_index} videos")
            video_paths_ = video_paths_[last_processed_video_index + 1 :]
            print(f"Remaining videos: {len(video_paths_)}")
        else:
            open(checkpoint_path, "w").close()
            print("Checkpoint file not found, starting from the beginning")

    # start multiprocessing

    pool = multiprocessing.Pool(pool_per_gpu)
    pool.map(worker_function, video_paths_)
    pool.close()
    pool.join()


if __name__ == "__main__":
    # example usage, feel free to modify or just import the functions and use them in your own script
    video_dir = "/workspaces/gorillatracker/spac_gorillas_converted"
    video_paths = [os.path.join(video_dir, file_name) for file_name in os.listdir(video_dir)]
    # these are just some random splits of the video paths for debugging
    debug_vid_paths = video_paths[:200]
    debug_vid_paths_2 = video_paths[100:200]
    debug_vid_paths_3 = [f"{video_dir}/M002_20220328_015.mp4"]

    largest_200 = sorted(video_paths, key=lambda file: os.path.getsize(file), reverse=True)[:200]

    predict_video_multiprocessing(
        video_paths=debug_vid_paths_3,
        pool_per_gpu=1,
        yolo_args={"verbose": True},
        overwrite=True,
        post_process_functions=[
            partial(
                save_result_to_json,
                json_folder="/workspaces/gorillatracker/src/gorillatracker/scripts/spac_tracking/jsonTest",
                overwrite=True,
            )
        ],
        # checkpoint_path="./checkpoint.txt",
    )

from ultralytics import YOLO
from functools import partial
import os
import json
import multiprocessing
from typing import List, Dict, Callable, Optional

gpu_queue = multiprocessing.Queue()

config = {}


def save_result_to_json(
    results: List[List[Dict]], 
    file_name: str, 
    json_folder: str, 
    overwrite: bool = False,
    ):
    """
    Save the results to a JSON file.

    Args:
        results (List[List[Dict]]): The results to be saved. (a list of result generators from YOLO.predict)
        file_name (str): The name of the JSON file.
        json_folder (str): The folder path where the JSON file will be saved.
        overwrite (bool, optional): Whether to overwrite an existing JSON file with the same name. 
            Defaults to False.

    Returns:
        None
    """
    
    json_path = f"{json_folder}/{file_name}.json"
    if os.path.exists(json_path) and not overwrite:
        return
    
    label_frames = [[] for f in results[0]]
    for result_index in range(len(results)):
        result = results[result_index]
        frame_index = 0
        for frame in result:
            boxes = frame.boxes.xywhn.tolist()
            confs = frame.boxes.conf.tolist()
            for box, conf in zip(boxes, confs):
                x, y, w, h = box
                box = {
                    "class": result_index,
                    "center_x": x,
                    "center_y": y,
                    "w": w,
                    "h": h,
                    "conf": conf
                }
                label_frames[frame_index].append(box)
            frame_index += 1
    json.dump({"labels": label_frames}, open(json_path, "w"), indent=4)
    
    
    
    

def predict_video(
    input_path: str, 
    models: List[YOLO],
    gpu_id: int = 0,
    ):
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
    
    assert config != {}, "config must be set before calling predict_video"
    assert len(models) > 0, "models must be a list of at least one model"
    assert os.path.exists(input_path), f"input_path {input_path} does not exist"
        
    post_process_function = config["post_process_function"]
    yolo_args = config["yolo_args"]
    checkpoint_path = config["checkpoint_path"]
    file_name = input_path.split("/")[-1]
    file_name = file_name.split(".")[:-1]
    file_name = ".".join(file_name)
    
    
    results = []
    for model in models:
        results.append(model.predict(input_path, stream = True, device = gpu_id, **yolo_args))

    if post_process_function is not None:
        post_process_function(results = results, file_name = file_name)
    if checkpoint_path is not None:
        open(checkpoint_path, "w").write(input_path)





class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton. # TODO UNTESTED
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if multiprocessing.current_process().pid not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[multiprocessing.current_process().pid] = instance
        return cls._instances[multiprocessing.current_process().pid]

class Singleton(metaclass=SingletonMeta):
    def __init__(self, models: List[YOLO]):
        self.value = models

    def get_models(self): 
        return self.value


def worker_function(input_path: str):
    
    """
    Process the given input video file.

    Parameters:
    - input_path (str): The path of the input video file.
    Returns:
        None
    """
    
    global config
    singleton = Singleton(config["models"])
    gpu = gpu_queue.get()
    print(f"Processing {input_path} on GPU {gpu}")
    predict_video(input_path, singleton.get_models())
    print(f"Finished processing {input_path} on GPU {gpu}")
    gpu_queue.put(gpu)
    
    
    
    
    
def predict_video_multiprocessing(
    models: List[YOLO] = [
        YOLO("/workspaces/gorillatracker/models/body_s_Ben.pt"),
        YOLO("/workspaces/gorillatracker/src/gorillatracker/scripts/spac_tracking/weights/body.pt")
        ], 
    post_process_function: Callable[[List[List[Dict]], str], None] = None,
    yolo_args: Dict = {"verbose":False},
    pool_per_gpu: int = 4,
    gpu_ids: List[int] = [0],
    checkpoint_path: Optional[str] = None,
    **kwargs
    ):
    
    """
    Perform video prediction using multiple YOLO models in parallel using multiprocessing.
    
    Parameters:
    - models (List[YOLO]): List of YOLO models to use for prediction.
    - post_process_function (Callable[[List[List[Dict]]], None]): Function to apply post-processing to the predicted results. 
            The function will get the results passed as the first argument and the file name as the second argument.
            The function shouldn't return anything.
            If you want to pass additional arguments to the function, use functools.partial.
    - yolo_args (Dict): Additional arguments to pass to the YOLO models.
    - pool_per_gpu (int): Number of processes to use for parallel prediction per GPU.
    - **kwargs: Additional keyword arguments. Either "video_dir" or "video_paths" must be specified.
    Returns:
    None
    """
    
    global config
    config = {
        "models": models,
        "post_process_function": post_process_function,
        "yolo_args": yolo_args,
        "checkpoint_path": checkpoint_path
    }
    
    global gpu_queue
    for gpu_id in gpu_ids:
        for _ in range(pool_per_gpu):
            gpu_queue.put(gpu_id)
    
    assert "video_dir" in kwargs or "video_paths" in kwargs, "Either video_dir or video_paths must be specified"
    assert not ("video_dir" in kwargs and "video_paths" in kwargs), "Only one of video_dir or video_paths must be specified"
    
    video_paths = kwargs["video_paths"]
    
    if "video_dir" in kwargs:
        video_dir = kwargs["video_dir"]
        video_paths = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
        
    print(f"Processing {len(video_paths)} videos")
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            print("Checkpoint found, resuming from last processed video")
            last_processed_video = open(checkpoint_path, "r").read()
            print(last_processed_video)
            if last_processed_video not in video_paths:
                print("Checkpoint file contains a video that doesn't exist, starting from the beginning")
                open(checkpoint_path, "w").close()
            last_processed_video_index = video_paths.index(last_processed_video)
            print(f"Skipping {last_processed_video_index} videos")
            video_paths = video_paths[last_processed_video_index+1:]
            print(f"Remaining videos: {len(video_paths)}")
        else:
            open(checkpoint_path, "w").close()
            print("Checkpoint file not found, starting from the beginning")
    
    pool = multiprocessing.Pool(pool_per_gpu)
    pool.map(worker_function, video_paths)
    pool.close()
    pool.join()
    
    
    

if __name__ == "__main__":
    video_dir = "/workspaces/gorillatracker/spac_gorillas_converted"
    video_paths = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
    debug_vid_paths = video_paths[:200]
    debug_vid_paths_2 = video_paths[100:200]
    debug_vid_paths_3 = [f"{video_dir}/Trc099_20220705_115.mp4"]
    predict_video_multiprocessing(
        video_paths=debug_vid_paths, 
        pool_per_gpu=4, 
        yolo_args={"verbose":False}, 
        overwrite=True,
        post_process_function=partial(
            save_result_to_json, 
            json_folder="/workspaces/gorillatracker/src/gorillatracker/scripts/spac_tracking/jsonTest",
            overwrite=True,
            ),
        checkpoint_path="./checkpoint.txt"
        )
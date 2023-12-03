from ultralytics import YOLO
import os
import json
import multiprocessing
from typing import List, Dict, Callable, Optional


config = {}

def predict_video(
    input_path: str, 
    models: List[YOLO],
    ):
    """
    Predicts labels for objects in a video using multiple YOLO models.

    Parameters:
    - input_path (str): The path to the input video file.
    - models (List[YOLO]): A list of YOLO models to use for prediction.
    Returns:
        None
    """
    
    global config
    
    assert config != {}, "config must be set before calling predict_video"
    assert len(models) > 0, "models must be a list of at least one model"
    assert os.path.exists(input_path), f"input_path {input_path} does not exist"
        
    json_folder = config["json_folder"]
    save_json = config["save_json"]
    overwrite_json = config["overwrite_json"]
    post_process_function = config["post_process_function"]
    yolo_args = config["yolo_args"]
    file_name = input_path.split("/")[-1]
    file_name = file_name.split(".")[:-1]
    file_name = ".".join(file_name)
    json_path = f"{json_folder}/{file_name}.json"
    if os.path.exists(json_path) and not overwrite_json:
        return
    
    results = []
    for model in models:
        results.append(model.predict(input_path, stream = True, **yolo_args))
    
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

    if post_process_function is not None:
        post_process_function(label_frames)
    if save_json:
        json.dump({"labels": label_frames}, open(json_path, "w"), indent=4)

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
    print(f"Processing {input_path}")
    predict_video(input_path, singleton.get_models())
    
def predict_video_multiprocessing(
    models: List[YOLO] = [
        YOLO("/workspaces/gorillatracker/models/body_s_Ben.pt"),
        YOLO("/workspaces/gorillatracker/src/gorillatracker/scripts/spac_tracking/weights/body.pt")
        ], 
    json_folder: str = "/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels", 
    save_json: bool = True,
    overwrite_json: bool = False, 
    post_process_function: Callable[[List[List[Dict]]], None] = None,
    yolo_args: Dict = {"verbose":False},
    pool_size: int = 4,
    **kwargs
    ):
    
    """
    Perform video prediction using multiple YOLO models in parallel using multiprocessing.
    
    Parameters:
    - models (List[YOLO]): List of YOLO models to use for prediction.
    - json_folder (str): Path to the folder where the predicted JSON files will be saved.
    - save_json (bool): Flag indicating whether to save the predicted JSON files.
    - overwrite_json (bool): Flag indicating whether to overwrite existing JSON files.
    - post_process_function (Callable[[List[List[Dict]]], None]): Function to apply post-processing to the predicted results.
    - yolo_args (Dict): Additional arguments to pass to the YOLO models.
    - pool_size (int): Number of processes to use for parallel prediction.
    - **kwargs: Additional keyword arguments. Either "video_dir" or "video_paths" must be specified.
    Returns:
    None
    """
    
    global config
    config = {
        "models": models,
        "json_folder": json_folder,
        "save_json": save_json,
        "overwrite_json": overwrite_json,
        "post_process_function": post_process_function,
        "yolo_args": yolo_args
    }
    
    assert "video_dir" in kwargs or "video_paths" in kwargs, "Either video_dir or video_paths must be specified"
    assert not ("video_dir" in kwargs and "video_paths" in kwargs), "Only one of video_dir or video_paths must be specified"
    
    video_paths = kwargs["video_paths"]
    
    if "video_dir" in kwargs:
        video_dir = kwargs["video_dir"]
        video_paths = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
    
    pool = multiprocessing.Pool(pool_size)
    pool.map(worker_function, video_paths)
    pool.close()
    pool.join()

if __name__ == "__main__":
    video_dir = "/workspaces/gorillatracker/spac_gorillas_converted"
    video_paths = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
    debug_vid_paths = video_paths[:100]
    debug_vid_paths_2 = video_paths[100:200]
    debug_vid_paths_3 = [f"{video_dir}/Trc099_20220705_115.mp4"]
    predict_video_multiprocessing(video_paths=debug_vid_paths_3, pool_size=1, yolo_args={"verbose":True}, overwrite_json=True)
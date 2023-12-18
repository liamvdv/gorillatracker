import cv2
import os
import json
from gorillatracker.scripts.video_json_tracker import GorillaVideoTracker
from typing import Dict
import multiprocessing as mp


def crop_and_save_image(frame, x: float, y: float, w: float, h: float, output_path: str) -> None:
    """Crop the image at the given path using the given bounding box coordinates and save it to the given output path.

    Args:
        image_path: Path to the image to crop.
        x: Relative x coordinate of the center of the bounding box.
        y: Relative y coordinate of the center of the bounding box.
        w: Relative width of the bounding box.
        h: Relative height of the bounding box.
        output_path: Path to save the cropped image to.
    """

    # calculate the bounding box coordinates
    frame_height, frame_width, _ = frame.shape
    left = int((x - (w / 2)) * frame_width)
    right = int((x + (w / 2)) * frame_width)
    top = int((y - (h / 2)) * frame_height)
    bottom = int((y + (h / 2)) * frame_height)
    
    cropped_frame = frame[top:bottom, left:right]
    cv2.imwrite(output_path, cropped_frame)
 
def add_labels_to_json(json_input_path: str, video_name: str, json_output_path: str):
    """Add the labels from one video to the given JSON file.

    Args:
        json_input_path: Path to the JSON file to read.
        video_name: Name of the video.
        json_output_path: Path to the JSON file to write.
    """
    # read the JSON file
    with open(json_input_path, "r") as f:
        data = json.load(f)
        
    if not os.path.exists(json_output_path):
        with open(json_output_path, "w") as f:
            json.dump({}, f, indent=4)
            
    # add the ids to the JSON file
    with open(json_output_path, "r") as f:
        out_data = json.load(f)
    out_data[video_name] = data["tracked_IDs"]
    with open(json_output_path, "w") as f:
        json.dump(out_data, f, indent=4)
    
def get_data_from_video(video_path: str, json_path: str, output_dir: str) -> None:
    """crop images from the video in the given path and copy negative list to negatives.json

    Args:
        video_path: Path to the video.
        output_dir: Path to the directory to save the cropped images to.
    """ 
    
    images_per_individual = 15
    # create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    id_frames = get_frames_for_ids(json_path)
    # open the video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video = cv2.VideoCapture(video_path)

    for id, frames in id_frames.items():
        if(len(frames) < images_per_individual):
            continue
        step_size = len(frames)//images_per_individual
        frame_list = [frames[i] for i in range(0, images_per_individual * step_size, step_size)]
        for frame_idx, bbox in frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            frame = video.read()[1]
            crop_and_save_image(frame, bbox[0], bbox[1], bbox[2], bbox[3], os.path.join(output_dir, f"{video_name}-{id}-{frame_idx}.jpg"))
    video.release()
    
    add_labels_to_json(json_path, video_name, os.path.join(output_dir, "negatives.json"))
 
def create_dataset_from_videos(video_dir: str, json_dir: str, output_dir: str) -> None:
    """Create a dataset of cropped images from the videos in the given directory.
    args:
        video_dir: Path to the directory containing the videos.
        json_dir: Path to the directory containing the tracked JSON files.
        output_dir: Path to the directory to save the cropped images to.
    """
    video_list = []
    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_name)
        json_path = os.path.join(json_dir, f"{os.path.splitext(video_name)[0]}_tracked.json")
        if(not os.path.exists(json_path)):
            continue
        video_list.append((video_path, json_path, output_dir))
    # multiprocess the video processing
    pool_size = min(int(mp.cpu_count()//8), len(video_list))
    pool = mp.Pool(pool_size)
    pool.starmap(get_data_from_video, video_list)
    pool.close()

def get_frames_for_ids(json_path: str) -> dict[int, list[(int, (float, float, float, float))]]:
    """Get the frames for the given IDs.

    Args:
        json_path: Path to the JSON file containing the IDs.

    Returns:
        A list of lists of frames for each ID.
    """
    id_frames: Dict[int, list[(int, (float, float, float, float))]] = {}
    face_class: int = 0
    # read the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    for frame_idx, frame in enumerate(data["labels"]):
        for bbox in frame:
            if bbox["class"] != face_class:
                continue
            id = int(bbox["id"])
            if id not in id_frames:
                id_frames[id] = []
            id_frames[id].append((frame_idx,(bbox["center_x"], bbox["center_y"], bbox["w"], bbox["h"])))
                        
    return id_frames
    
#get_frames_for_ids("/workspaces/gorillatracker/tmp/R014_20220628_151_tracked.json")
#tracker = GorillaVideoTracker("/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels_backup", "/workspaces/gorillatracker/tmp/", "/workspaces/gorillatracker/videos")
#tracker.track_file("/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels_backup/R033_20220707_100.json")
#tracker.track_file("/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels_backup/R092_20220717_054.json")
#tracker.save_video("/workspaces/gorillatracker/videos/R033_20220707_100.mp4")
create_dataset_from_videos("/workspaces/gorillatracker/videos", "/workspaces/gorillatracker/tmp/", "/workspaces/gorillatracker/tmp/")   
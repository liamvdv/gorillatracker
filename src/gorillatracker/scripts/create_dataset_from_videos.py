import json
import os

import cv2
import cv2.typing as cvt

BBox = tuple[float, float, float, float]  # x, y, w, h
BBoxFrame = tuple[int, BBox]  # frame_idx, x, y, w, h
IdFrameDict = dict[int, list[BBoxFrame]]  # id -> list of frames
IdDict = dict[int, list[int]]  # id -> list of negatives
JsonDict = dict[str, list[str]]  # video_name-id -> list of negatives


def _get_json_data(json_path: str) -> JsonDict:
    """Return the data from the given JSON file and create it if it doesn't exist.

    Args:
        json_path: Path to the JSON file.

    Returns:
        The data from the JSON file.
    """
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            json.dump({}, f)
    with open(json_path, "r") as f:
        data = json.loads(f.read())
    return data


def _add_labels_to_json(id_negatives: IdDict, video_name: str, json_output_path: str) -> None:
    """Add the labels from one video to the given JSON file.

    Args:
        id_negatives: negatives for each ID.
        video_name: Name of the video.
        json_output_path: Path to the JSON file to write.
    """
    out_dict: JsonDict = {}
    out_data = _get_json_data(json_output_path)
    for id, negatives in id_negatives.items():
        out_dict[f"{video_name}-{id}"] = [f"{video_name}-{negative}" for negative in negatives]
    out_data.update(out_dict)
    with open(json_output_path, "w") as f:
        json.dump(out_data, f, indent=4)


def _get_negatives(id_frames: IdFrameDict, min_frames: int, json_path: str) -> tuple[IdFrameDict, IdDict]:
    """Return negatives for each ID and remove IDs with too few frames from the given dictionary.

    Args:
        id_frames: Dictionary of IDs to frames.
        min_frames: Minimum number of frames an ID must have to be kept.
        json_path: Path to the JSON file containing the IDs.

    Returns:
        The filtered dictionary.
        The negatives for each ID.
    """
    # filter out IDs with too few frames
    id_frames = {id: frames for id, frames in id_frames.items() if len(frames) >= min_frames}
    with open(json_path, "r") as f:
        data = json.load(f)
        ids = data["tracked_IDs"]
    # get the negatives from the json file for each ID
    id_negatives = {entry["id"]: entry["negatives"] for entry in ids if entry["id"] in id_frames}
    # filter out negatives that are not in id_frames and remove IDs with no negatives
    for id in id_negatives:
        id_negatives[id] = [negative for negative in id_negatives[id] if negative in id_frames]
        if len(id_negatives[id]) < 1:
            del id_frames[id]
    # filter negatives again after removing IDs
    id_negatives = {id: negatives for id, negatives in id_negatives.items() if id in id_frames}

    return id_frames, id_negatives


def _get_frames_for_ids(json_path: str) -> IdFrameDict:
    """Get the frames for the given IDs.

    Args:
        json_path: Path to the JSON file containing the IDs.

    Returns:
        Dictionary of IDs to frames.
    """
    id_frames: IdFrameDict = {}
    face_class: int = 1
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
            id_frames[id].append((frame_idx, (bbox["center_x"], bbox["center_y"], bbox["w"], bbox["h"])))

    return id_frames


def _crop_and_save_image(frame: cvt.MatLike, x: float, y: float, w: float, h: float, output_path: str) -> None:
    """Crop the image at the given path using the given bounding box coordinates and save it to the given output path.

    Args:
        frame: Image to crop.
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


def _get_data_from_video(video_path: str, json_path: str, output_dir: str) -> None:
    """crop images from the video in the given path and copy negative list to negatives.json

    Args:
        video_path: Path to the video.
        json_path: Path to the tracked json file.
        output_dir: Path to the directory to save the cropped images to.
    """

    images_per_individual = 15
    # create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    id_frames = _get_frames_for_ids(json_path)
    # open the video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video = cv2.VideoCapture(video_path)
    id_frames, id_negatives = _get_negatives(id_frames, images_per_individual, json_path)
    for id, frames in id_frames.items():
        step_size = len(frames) // images_per_individual
        frame_list = [frames[i] for i in range(0, images_per_individual * step_size, step_size)]
        for frame_idx, bbox in frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            frame = video.read()[1]  # read the frame. read() returns a tuple of (success, frame)
            _crop_and_save_image(
                frame,
                bbox[0],  # x
                bbox[1],  # y
                bbox[2],  # w
                bbox[3],  # h
                os.path.join(output_dir, f"{video_name}-{id}-{frame_idx}.png"),
            )
    video.release()

    _add_labels_to_json(id_negatives, video_name, os.path.join(output_dir, "negatives.json"))


def create_dataset_from_videos(video_dir: str, json_dir: str, output_dir: str) -> None:
    """Create a dataset of cropped images from the videos in the given directory.
    args:
        video_dir: Path to the directory containing the videos.
        json_dir: Path to the directory containing the tracked JSON files.
        output_dir: Path to the directory to save the cropped images to.
    """
    negative_json = os.path.join(output_dir, "negatives.json")
    video_skip_list = set([id.split("-")[0] for id in _get_json_data(negative_json).keys()])
    print(f"Skipping {len(video_skip_list)} videos.")
    video_list = [video for video in os.listdir(video_dir) if os.path.splitext(video)[0] not in video_skip_list]
    for idx, video in enumerate(video_list):
        print(f"processing videos: {idx + 1}/{len(video_list)}", end="\r")
        video_name = os.path.splitext(video)[0]
        video_path = os.path.join(video_dir, video)
        json_path = os.path.join(json_dir, f"{video_name}_tracked.json")
        if not os.path.exists(json_path):
            continue
        _get_data_from_video(video_path, json_path, output_dir)
    print("" * 80)  # clear line
    print("all videos processed")


# example usage
# if __name__ == "__main__":
#    create_dataset_from_videos(
#        "/workspaces/gorillatracker/videos",
#        "/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels_tracked",
#        "/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels_cropped_faces/train",
#    )

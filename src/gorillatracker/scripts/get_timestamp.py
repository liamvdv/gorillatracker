import json
import os

import cv2
import easyocr


def get_time_stamp(video_path: str) -> str:
    """
    Extracts the time stamp from the video file.
    Args:
        video_path (str): path to the video file
    Returns:
        str: time stamp as string in the format of HH:MMAM/PM e.g. 01:23AM
    """

    # open the video file
    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # crop area for the time stamp
    x_left, y_top, x_right, y_bottom = int(0.61 * width), int(0.9 * height), int(0.75 * width), int(1.0 * height)
    crop_area = (x_left, y_top, x_right, y_bottom)

    # crop and read the timestamp
    frame = video.read()[1]
    cropped_frame = frame[crop_area[1] : crop_area[3], crop_area[0] : crop_area[2]]
    cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    extracted_time_stamp_raw = reader.readtext(cropped_frame_rgb)  # returns a list of detected text elements

    # read from the extracted text, the text information is in the second elements of each tuple in the list
    time_stamp = "".join([text[1] for text in extracted_time_stamp_raw])
    time_stamp = time_stamp.replace(" ", "")  # remove spaces, cause they can be different between detections
    time_stamp = time_stamp.replace(":", "")  # remove the colon, cause it can be different between detections
    time_stamp = time_stamp.replace("O", "0")  # exchanges O with 0, cause it can be false detected
    time_stamp = time_stamp.replace("o", "0")  # exchanges o with 0, cause it can be false detected
    # time_stamp is in hh:mmAM/PM format e.g. 01:23AM
    try:
        h = int(time_stamp[:2])
        m = int(time_stamp[2:4])
        am = True if time_stamp[4:6].lower() == "am" else False
    except ValueError:
        raise ValueError(f"Could not extract time stamp from {video_path}")

    video.release()

    return f"{str(h).zfill(2)}:{str(m).zfill(2)} {'AM' if am else 'PM'}"


def write_timestamps_to_json(json_path: str, video_path: str) -> None:
    """
    Writes the time stamps of the given video file to the json file.
    Args:
        json_path (str): path to the json file
        video_path (str): path to the video files
    """
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            json.dump({}, f)

    with open(json_path, "r") as f:
        data = json.load(f)

    for idx, video in enumerate(os.listdir(video_path)):
        print(f"reading timestamp {idx}/{len(os.listdir(video_path))}", end="\r")
        video_name = os.path.splitext(video)[0]
        if os.path.splitext(video)[1] != ".mp4":
            continue
        try:
            time_stamp = get_time_stamp(os.path.join(video_path, video))
        except ValueError as e:
            print(e)
            continue
        data[video_name] = time_stamp
        if idx % 10 == 0:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


write_timestamps_to_json(
    "/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels_cropped_faces/timestamps.json",
    "/workspaces/gorillatracker/videos",
)

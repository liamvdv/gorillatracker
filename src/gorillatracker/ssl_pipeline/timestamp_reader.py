from pathlib import Path
import easyocr
from datetime import time
import cv2
from gorillatracker.ssl_pipeline.helpers import video_reader, BoundingBox


def read_timestamp(
    video_path: Path, bbox: BoundingBox = BoundingBox(0.674036, 0.977824, 0.0980, 0.0296, 1, 1920, 1080)
) -> time:
    """
    Extracts the time stamp from the video file.
    Args:
        video_path (Path): path to the video file
        bbox (BoundingBox): bounding box for the time stamp
    Returns:
        time: time stamp as time object
    """
    with video_reader(video_path) as video_feed:
        frame = next(video_feed).frame
    cropped_frame = crop_frame(frame, bbox)
    time_stamp = _extract_time_stamp(cropped_frame)
    return time_stamp

def _extract_time_stamp(cropped_frame: cv2.typing.MatLike) -> time:
    """Extracts the time stamp from the cropped frame."""
    allow_list = "0123456789A:PMapm"
    rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    extracted_time_stamp_raw = reader.readtext(rgb_frame, allowlist=allow_list)
    time_stamp = "".join([text[1] for text in extracted_time_stamp_raw])
    time_stamp = time_stamp.replace(":", "")
    try:
        h = int(time_stamp[:2])
        m = int(time_stamp[2:4])
        am = True if time_stamp[4:6].lower() == "am" else False
    except ValueError:
        raise ValueError(f"Could not extract time stamp from frame")
    if not am and h < 12:
        h += 12
    if am and h == 12:
        h = 0
    return time(h, m)

def crop_frame(frame: cv2.typing.MatLike, bbox: BoundingBox) -> cv2.typing.MatLike:
    """Crops a frame according to the bounding box."""
    cropped_frame = frame[
                    bbox.y_top_left : bbox.y_bottom_right,
                    bbox.x_top_left : bbox.x_bottom_right,
                ]
    return cropped_frame

if __name__ == "__main__":
    video_path = Path("/workspaces/gorillatracker/video_data/R185_20221026_054.mp4")
    time_stamp = read_timestamp(video_path)
    print(time_stamp)
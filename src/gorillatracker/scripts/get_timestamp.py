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
    reader = easyocr.Reader(["en"])
    extracted_time_stamp_raw = reader.readtext(cropped_frame_rgb)  # returns a list of detected text elements

    # read from the extracted text, the text information is in the second elements of each tuple in the list
    time_stamp = "".join([text[1] for text in extracted_time_stamp_raw])
    time_stamp = time_stamp.replace(" ", "")  # remove spaces, cause they can be different between detections
    # time_stamp is in hh:mmAM/PM format e.g. 01:23AM
    h = int(time_stamp[:2])
    m = int(time_stamp[3:5])
    am = True if time_stamp[5:7].lower() == "am" else False

    video.release()

    return f"{h}:{m} {'AM' if am else 'PM'}"


get_time_stamp("/workspaces/gorillaracker/data/videos/R465_20220506_058.mp4")

import cv2
import easyocr


def get_time_stamp(video_path: str) -> str:
    """
    Extracts the time stamp from the video file.
    Args:
        video_path (str): path to the video file
    Returns:
        str: time stamp in the format of HH:MM AM/PM
    """

    # open the video file
    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_area = (int(0.61 * width), int(0.9 * height), int(0.75 * width), int(height))

    # crop and read the timestamp
    frame = video.read()[1]
    cropped_frame = frame[crop_area[1] : crop_area[3], crop_area[0] : crop_area[2]]
    cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    reader = easyocr.Reader(["en"])
    extracted_text = reader.readtext(cropped_frame_rgb)

    # read from the extracted text
    extracted_text = "".join([text[1] for text in extracted_text])
    extracted_text = extracted_text.replace(" ", "")
    h = int(extracted_text[:2])
    m = int(extracted_text[3:5])
    am = True if extracted_text[5:7].lower() == "am" else False

    video.release()

    return f"{h}:{m} {'AM' if am else 'PM'}"

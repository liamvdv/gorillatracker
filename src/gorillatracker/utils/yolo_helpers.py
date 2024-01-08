# NOTE: https://github.com/ultralytics/ultralytics/issues/5800
# also did not work for the bbox export
from typing import List, Tuple

BOUNDING_BOX = Tuple[Tuple[int, int], Tuple[int, int]]


def convert_to_yolo_format(box: BOUNDING_BOX, img_width: int, img_height: int) -> List[float]:
    x_min, y_min = box[0]
    x_max, y_max = box[1]
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    assert 0 <= x_center <= 1
    assert 0 <= y_center <= 1
    assert 0 <= width <= 1
    assert 0 <= height <= 1
    return [x_center, y_center, width, height]


def convert_from_yolo_format(box: List[float], img_width: int, img_height: int) -> BOUNDING_BOX:
    x_center, y_center, width, height = box
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return ((x_min, y_min), (x_max, y_max))

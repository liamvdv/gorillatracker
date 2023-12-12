from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt


def get_cutout_bbox(
    full_image: npt.NDArray[np.uint8], cutout: npt.NDArray[np.uint8], threshold: float = 0.95
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Get the bounding box of a cutout in a full image.

    Args:
    full_image: cv2 image of the full image
    cutout: cv2 image of the cutout
    threshold: how similar the cutout must be to the full image to be considered a match

    Returns:
    (top_left, bottom_right) of the cutout in the full image
    """
    res = cv2.matchTemplate(full_image, cutout, cv2.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
    if maxVal < threshold:
        raise Exception("Cutout not found in full image")
    cutout_height, cutout_width, _ = cutout.shape
    top_left = maxLoc
    bottom_right = (top_left[0] + cutout_width, top_left[1] + cutout_height)
    return [top_left, bottom_right]
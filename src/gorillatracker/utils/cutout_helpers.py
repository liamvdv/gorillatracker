import cv2

import gorillatracker.type_helper as gtyping


def get_cutout_bbox(full_image: gtyping.Image, cutout: gtyping.Image, threshold: float = 0.95) -> gtyping.BoundingBox:
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
    assert maxVal > threshold, "Cutout not found in full image"
    cutout_height, cutout_width, _ = cutout.shape  # type: ignore
    top_left_x, top_left_y = maxLoc[:2]
    top_left = (top_left_x, top_left_y)
    bottom_right = (top_left[0] + cutout_width, top_left[1] + cutout_height)
    return (top_left, bottom_right)


def cutout_image(full_image: gtyping.Image, bbox: gtyping.BoundingBox, target_path: str) -> None:
    """
    Cut out a section of an image.

    Args:
    full_image: cv2 image of the full image
    bbox: ((x_top_left, y_top_left), (x_bottom_right, y_bottom_right))
    target_path: path to save cutout image to

    """
    top_left, bottom_right = bbox
    cutout = full_image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]  # type: ignore
    cv2.imwrite(target_path, cutout)

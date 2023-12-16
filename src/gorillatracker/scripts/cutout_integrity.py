import os
from typing import Set

import cv2
import numpy as np
import numpy.typing as npt


def _is_coutout_in_image(image: npt.NDArray[np.uint8], cutout: npt.NDArray[np.uint8]) -> bool:
    res = cv2.matchTemplate(image, cutout, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    return len(loc[0]) > 0


def assert_matching_cutouts(cutout_dir: str, image_dir: str) -> Set[str]:
    outlier_files = set()

    cutout_files = set(os.listdir(cutout_dir))
    image_files = set(os.listdir(image_dir))

    assert all([cutout_file.endswith(".png") for cutout_file in cutout_files])
    assert all([image_file.endswith(".png") for image_file in image_files])

    not_in_cutout_files = image_files - cutout_files
    if len(not_in_cutout_files) > 0:
        print(f"WARNING: {len(not_in_cutout_files)} image files not in cutout files ({not_in_cutout_files})")

    for cutout_file in cutout_files:
        if cutout_file not in image_files:
            print(f"WARNING: {cutout_file} not in image files")
            continue

        image_path = os.path.join(image_dir, cutout_file)
        cutout_path = os.path.join(cutout_dir, cutout_file)

        image = cv2.imread(image_path)
        cutout = cv2.imread(cutout_path)

        if cutout.shape >= image.shape:
            outlier_files.add(cutout_file)
            print(f"WARNING: {cutout_file} has larger shape than corresponding image")
            continue

        if not _is_coutout_in_image(image, cutout):
            outlier_files.add(cutout_file)
            print(f"WARNING: {cutout_file} not in corresponding image")

        # TODO(remove)
        if len(outlier_files) > 3:
            break
    return outlier_files


if __name__ == "__main__":
    image_dir = "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/body_images"
    cutout_dir = "/workspaces/gorillatracker/data/ground_truth/cxl/face_images"
    outliers = assert_matching_cutouts(cutout_dir, image_dir)
    print(len(outliers), "outliers found")

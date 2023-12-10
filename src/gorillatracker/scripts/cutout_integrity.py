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

    return outlier_files


if __name__ == "__main__":
    cutout_dir = "/workspaces/gorillatracker/data/ground_truth/rohan-cxl/face_images"
    image_dir = "/workspaces/gorillatracker/data/ground_truth/rohan-cxl/body_images"
    filtered_dir = "/workspaces/gorillatracker/data/derived_data/rohan-cxl/filtered_body_images"
    outliers = assert_matching_cutouts(cutout_dir, image_dir)
    print(len(outliers), "outliers found")
    filtered_images = set(os.listdir(image_dir)) - outliers
    os.makedirs(filtered_dir, exist_ok=True)
    for image in filtered_images:
        os.system(f"cp '{os.path.join(image_dir, image)}' '{os.path.join(filtered_dir, image)}'")
    # print("Asserting for filtered images:")
    assert_matching_cutouts(cutout_dir, filtered_dir)

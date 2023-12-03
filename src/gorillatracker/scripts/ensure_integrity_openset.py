"""Ensure that the given train, val and test sets are valid.

This means that the train set does not contain any images that by accident contain a subject that for testing/validation purposes should be considered as unknown (as it should be test/val proprietary).
The exact same applies to the val set.

This is an issue because the bristol dataset names images with the subject that is (probably) most visible in the image. 
The problem is that there are images containing multiple subjects. So the name of the image does not ensure that only this particular subject is visible in the image.
So we have to check the annotation files and move images that contain a subject that should be considered as unknown to the test/val set.

Another issue is that the bristol dataset contains images where subject encoded in the name is not part of the annotations for this image.
As it cannot be verified which label is correct, these images are removed from the dataset.
"""


import os
import shutil
import logging

from typing import Set, List

bristol_index_to_name = {0: "afia", 1: "ayana", 2: "jock", 3: "kala", 4: "kera", 5: "kukuena", 6: "touni"}
bristol_name_to_index = {value: key for key, value in bristol_index_to_name.items()}

logger = logging.getLogger(__name__)


def move_images_of_subjects(image_dir: str, bbox_dir: str, output_dir: str, subjects_indicies: List[int]) -> int: # TODO(rob2u): make more compact
    """Move all images from the image folder to the output folder that contain bounding boxes of the given subjects.

    Args:
        image_folder (str): Folder containing the images.
        bbox_folder (str): Folder containing the bounding box files.
        output_folder (str): Folder to move the images to.
        subjects_indicies (list): List of subject indicies to move.

    Returns:
        int: Number of images moved.
    """
    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    move_count = 0
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        bbox_path = os.path.join(bbox_dir, image_file.replace(".jpg", ".txt"))

        assert os.path.exists(bbox_path), f"Bounding box file '{bbox_path}' does not exist for image '{image_file}'"
        
        # Read bounding box coordinates from the text file
        bbox_data = []
        with open(bbox_path, "r") as bbox_file:
            bbox_data = bbox_file.read().strip().split()

        assert len(bbox_data) % 5 == 0, f"Bounding box file '{bbox_path}' has invalid format"

        for i in range(0, len(bbox_data), 5):
            index, x, y, w, h = map(
                float, bbox_data[i : (i + 5)]
            )  # x,y is the center of the bbox and w,h are the width and height

            if index in subjects_indicies:
                shutil.move(image_path, output_dir)
                logger.info("Moved image %s to %s", image_file, output_dir)
                move_count += 1

    return move_count


def filter_images_bristol(image_dir: str, bbox_dir: str) -> int: # TODO(rob2u): make more compact
    """Remove all images from the image folder that do not contain a bounding box of the actual subject."""

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    remove_count = 0
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        bbox_path = os.path.join(bbox_dir, image_file.replace(".jpg", ".txt"))

        assert os.path.exists(bbox_path), f"Bounding box file '{bbox_path}' does not exist for image '{image_file}'"
        
        # Read bounding box coordinates from the text file
        bbox_data = []

        with open(bbox_path, "r") as bbox_file:
            bbox_data = bbox_file.read().strip().split()

        actual_subject = image_file.split("-")[0]
        actual_subject_index = bristol_name_to_index[actual_subject]
        seen_actual_subject = actual_subject_index in map(int, bbox_data[::5])
        if not seen_actual_subject:
            logger.warn(
                "Warning: Actual subject %s not found in bounding box file %s for image %s",
                actual_subject,
                bbox_path,
                image_file,
            )
            logger.info("Removing image %s from folder %s", image_file, image_dir)
            os.remove(image_path)
            remove_count += 1

    return remove_count


def get_subjects_in_directory(test_dir: str) -> Set[str]:
    """Get all subjects in the given directory. Subjects are identified by the prefix of the image file name."""
    # Get list of image files
    image_files = [f for f in os.listdir(test_dir) if f.endswith(".jpg")]
    logger.info("Found %d images in folder %s", len(image_files), test_dir)
    test_subjects = set()
    for image_file in image_files:
        test_subjects.add(image_file.split("-")[0])
    return test_subjects


def ensure_integrity(train_set_dir: str, val_set_dir: str, test_set_dir: str, bbox_dir: str) -> None:
    """Ensure that the given train, val and test sets are valid.
    This means that the train set does not contain any images that by accident contain the subject of the val or test set.
    """
    test_subjects = get_subjects_in_directory(test_set_dir)
    val_subjects = get_subjects_in_directory(val_set_dir)
    train_subjects = get_subjects_in_directory(train_set_dir)

    test_proprietary_subjects_set = test_subjects - val_subjects - train_subjects
    test_proprietary_subjects = [bristol_name_to_index[s] for s in test_proprietary_subjects_set]

    val_proprietary_subjects_set = val_subjects - train_subjects  # dont substract the test subjects
    val_proprietary_subjects = [bristol_name_to_index[s] for s in val_proprietary_subjects_set]

    logger.info("Test proprietary subjects: %d", test_proprietary_subjects)
    logger.info("Val proprietary subjects: %d", val_proprietary_subjects)

    assert len(test_proprietary_subjects) != 0, f"Test proprietary subjects is empty: {test_proprietary_subjects}"
    assert len(val_proprietary_subjects) != 0, f"Val proprietary subjects is empty: {val_proprietary_subjects}"

    # ensure that every image has a bounding box for the actual subject
    remove_count = filter_images_bristol(train_set_dir, bbox_dir)
    remove_count += filter_images_bristol(val_set_dir, bbox_dir)
    remove_count += filter_images_bristol(test_set_dir, bbox_dir)
    logger.info("Removed %d images", remove_count)

    # filter out images in train and val that contain test_proprietary_subjects
    move_count_train_to_test = move_images_of_subjects(
        train_set_dir, bbox_dir, test_set_dir, test_proprietary_subjects
    )
    logger.info("Moved %d images from train to test", move_count_train_to_test)
    move_count_val_to_test = move_images_of_subjects(
        val_set_dir, bbox_dir, test_set_dir, test_proprietary_subjects
    )
    logger.info("Moved %d images from val to test", move_count_val_to_test)

    # filter out images in train and test that contain val_proprietary_subjects
    move_count_train_to_val = move_images_of_subjects(
        train_set_dir, bbox_dir, val_set_dir, val_proprietary_subjects
    )
    logger.info("Moved %d images from train to val", move_count_train_to_val)

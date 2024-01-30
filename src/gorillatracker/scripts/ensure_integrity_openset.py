"""Ensure that the given train, val and test sets are valid.

This means that the train set does not contain any images that by accident contain a subject that for testing/validation purposes should be considered as unknown (as it should be test/val proprietary).
The exact same applies to the val set.
"""

import logging
import os
import shutil
from typing import List, Literal, Set, Tuple

import regex as re

from gorillatracker.scripts.crop_dataset import read_bbox_data
from gorillatracker.scripts.dataset_splitter import generate_split

bristol_index_to_name = {0: "afia", 1: "ayana", 2: "jock", 3: "kala", 4: "kera", 5: "kukuena", 6: "touni"}
bristol_name_to_index = {value: key for key, value in bristol_index_to_name.items()}

logger = logging.getLogger(__name__)


def move_image(image_path: str, bbox_path: str, output_dir: str, subject_indices: List[int]) -> int:
    """Move the given image to the given output directory if it contains a bounding box of the actual subject.

    Args:
        image_path: Path to the image to move.
        bbox_path: Path to the bounding box file.
        output_dir: Directory to move the image to.

    Returns:
        1 if the image was moved, 0 otherwise.
    """
    bbox_lines = read_bbox_data(bbox_path)
    subjects_in_image = [int(bbox_line[0]) for bbox_line in bbox_lines]

    if any([subject_index in subjects_in_image for subject_index in subject_indices]) and not os.path.exists(
        os.path.join(output_dir, os.path.basename(image_path))
    ):
        shutil.move(image_path, output_dir)
        logger.info("Moved image %s to %s", image_path, output_dir)
        return 1
    else:
        return 0


def move_images_of_subjects(image_dir: str, bbox_dir: str, output_dir: str, subject_names: List[str]) -> int:
    """Move all images from the image folder to the output folder that contain bounding boxes of the given subjects.

    Args:
        image_folder: Folder containing the images.
        bbox_folder: Folder containing the bounding box files.
        output_folder: Folder to move the images to.
        subject_names: List of subject indicies to move.

    Returns:
        Number of images moved.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    subject_indices = [bristol_name_to_index[s] for s in subject_names]

    move_count = 0
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        bbox_path = os.path.join(bbox_dir, image_file.replace(".jpg", ".txt"))

        assert os.path.exists(bbox_path), f"Bounding box file '{bbox_path}' does not exist for image '{image_file}'"
        move_count += move_image(image_path, bbox_path, output_dir, subject_indices)
    return move_count


def filter_images_bristol(image_dir: str, bbox_dir: str) -> int:
    """Remove all images from the image folder that do not contain a bounding box of the actual subject."""
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    remove_count = 0
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        bbox_path = os.path.join(bbox_dir, image_file.replace(".jpg", ".txt"))

        assert os.path.exists(bbox_path), f"Bounding box file '{bbox_path}' does not exist for image '{image_file}'"

        bbox_subjects = [int(bbox[0]) for bbox in read_bbox_data(bbox_path)]
        actual_subject = re.split(r"[_\s-]", image_file, maxsplit=1)[0]
        actual_subject_index = bristol_name_to_index[actual_subject]
        if actual_subject_index not in bbox_subjects:
            logger.warning(
                "Actual subject %s not found in bounding box file %s for image %s, removing image",
                actual_subject,
                bbox_path,
                image_file,
            )
            os.remove(image_path)
            remove_count += 1
    return remove_count


def get_subjects_in_directory(test_dir: str, file_extension: Literal[".jpg", ".png"] = ".jpg") -> Set[str]:
    """Get all subjects in the given directory. Subjects are identified by the prefix of the image file name."""
    image_files = [f for f in os.listdir(test_dir) if f.endswith(file_extension)]
    logger.info("Found %d images in folder %s", len(image_files), test_dir)
    test_subjects = set()
    for image_file in image_files:
        image_file = image_file.lower()
        subject, _ = re.split(r"[_\s-]", image_file, maxsplit=1)
        test_subjects.add(subject)
    return test_subjects


def get_test_val_train_proprietary_subjects(
    split_base_dir: str, file_extension: Literal[".png", ".jpg"] = ".png"
) -> Tuple[List[str], List[str], List[str]]:
    """Get the test and val proprietary subjects from the given split."""
    train_set_dir = os.path.join(split_base_dir, "train")
    val_set_dir = os.path.join(split_base_dir, "val")
    test_set_dir = os.path.join(split_base_dir, "test")

    test_subjects = get_subjects_in_directory(test_set_dir, file_extension=file_extension)
    val_subjects = get_subjects_in_directory(val_set_dir, file_extension=file_extension)
    train_subjects = get_subjects_in_directory(train_set_dir, file_extension=file_extension)

    test_proprietary_subjects = list(test_subjects - val_subjects - train_subjects)
    val_proprietary_subjects = list(val_subjects - train_subjects - test_subjects)
    train_proprietary_subjects = list(train_subjects - val_subjects - test_subjects)

    logger.info("Split %s has test proprietary subjects: %d ", test_proprietary_subjects)
    logger.info("Split %s has val proprietary subjects: %d", val_proprietary_subjects)
    return test_proprietary_subjects, val_proprietary_subjects, train_proprietary_subjects


def ensure_integrity_openset(
    split_base_dir: str, bbox_dir: str = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images_face_bbox"
) -> None:
    """Ensure that the given train, val and test sets are valid.
    This means that the train set does not contain any images that by accident contain the subject of the val or test set.
    """
    test_proprietary_subjects, val_proprietary_subjects, _ = get_test_val_train_proprietary_subjects(
        split_base_dir, file_extension=".jpg"
    )

    assert len(test_proprietary_subjects) != 0, f"Test proprietary subjects is empty: {test_proprietary_subjects}"
    assert len(val_proprietary_subjects) != 0, f"Val proprietary subjects is empty: {val_proprietary_subjects}"

    # ensure that every image has a bounding box for the actual subject
    train_set_dir = os.path.join(split_base_dir, "train")
    val_set_dir = os.path.join(split_base_dir, "val")
    test_set_dir = os.path.join(split_base_dir, "test")

    remove_count = filter_images_bristol(train_set_dir, bbox_dir)
    remove_count += filter_images_bristol(val_set_dir, bbox_dir)
    remove_count += filter_images_bristol(test_set_dir, bbox_dir)
    logger.info("Removed %d images", remove_count)

    # filter out images in train and val that contain test_proprietary_subjects
    move_count_train_to_test = move_images_of_subjects(train_set_dir, bbox_dir, test_set_dir, test_proprietary_subjects)
    logger.info("Moved %d images from train to test", move_count_train_to_test)
    move_count_val_to_test = move_images_of_subjects(val_set_dir, bbox_dir, test_set_dir, test_proprietary_subjects)
    logger.info("Moved %d images from val to test", move_count_val_to_test)

    # filter out images in train and test that contain val_proprietary_subjects
    move_count_train_to_val = move_images_of_subjects(train_set_dir, bbox_dir, val_set_dir, val_proprietary_subjects)
    logger.info("Moved %d images from train to val", move_count_train_to_val)


if __name__ == "__main__":
    bristol_split_dir = str(
        generate_split(
            dataset="ground_truth/bristol/full_images",
            mode="openset",
            seed=69,
            reid_factor_test=0,
            reid_factor_val=0,
        )
    )
    bristol_split_dir = str(os.path.abspath(bristol_split_dir))
    ensure_integrity_openset(bristol_split_dir)

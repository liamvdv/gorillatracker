"""Builds the different splits for the bristol dataset and the cxl dataset. If specified also trains a yolo model on the bristol dataset."""

import json
import os
from typing import Any, Dict, Tuple

import ultralytics

from gorillatracker.scripts.crop_dataset import crop_images
from gorillatracker.scripts.dataset_splitter import generate_split
from gorillatracker.scripts.ensure_integrity_openset import (
    ensure_integrity_openset,
    get_test_val_train_proprietary_subjects,
)
from gorillatracker.scripts.train_yolo import detect_gorillafaces_cxl, modify_dataset_train_yolo


def save_dict_json(dict: Dict[Any, Any], file_path: str) -> None:
    """Saves the given dictionary to the given file path as json."""
    with open(file_path, "w") as file:
        json.dump(dict, file, indent=4, sort_keys=True)


def crop_images_save_metadata_cxl(
    model_name: str,
    bristol_split_dir: str,
    cxl_imgs_dir: str = "/workspaces/gorillatracker/data/ground_truth/cxl/full_images",
) -> Tuple[Dict[str, Any], str]:
    """Crops the cxl images according to the predicted bounding boxes and saves the metadata to metadata.json.
    The location of the cropped images and the annotations is derived from the model_name."""

    # get image where cropped imgs and annotations should be saved from model_name
    cxl_model_dir = os.path.join("/workspaces/gorillatracker/data/derived_data/cxl", model_name)

    cxl_imgs_crop_dir = os.path.join(cxl_model_dir, "face_crop")
    cxl_annotation_dir = os.path.join(cxl_model_dir, "face_bbox")

    os.makedirs(cxl_imgs_crop_dir, exist_ok=True)

    # crop cxl images according to predicted bounding boxes
    imgs_without_bbox, imgs_with_no_bbox_prediction, imgs_with_low_confidence = crop_images(
        cxl_imgs_dir, cxl_annotation_dir, cxl_imgs_crop_dir, is_bristol=False, file_extension=".png"
    )

    # save information for the cropped images to file metadata.json (in the cxl only and the joined split as well)
    meta_data = {
        "yolo-model": str(model_name),
        "bristol-split": str(bristol_split_dir),
        "imgs-without-bbox": imgs_without_bbox,
        "imgs-with-no-bbox-prediction": imgs_with_no_bbox_prediction,
        "imgs-with-low-confidence": imgs_with_low_confidence,
        "cxl-annotation-dir": str(cxl_annotation_dir),
    }
    save_dict_json(meta_data, os.path.join(cxl_imgs_crop_dir, "metadata.json"))
    save_dict_json(meta_data, os.path.join(cxl_annotation_dir, "metadata.json"))

    return meta_data, cxl_imgs_crop_dir


def generate_split_save_metadata_cxl(
    dataset: str, meta_data: Dict[str, Any], seed: int = 42, reid_factor_test: int = 0, reid_factor_val: int = 0
) -> str:
    """Generates a split for the cropped cxl dataset and saves the metadata to metadata.json.

    Args:
        meta_data: Metadata of the cropped cxl faces to save to metadata.json.
        seed: Seed for the random number generator. Defaults to 42.
        reid_factor_test: The reid factor for the test set. Defaults to 10.
        reid_factor_val: The reid factor for the val set. Defaults to 10.

    Returns:
        The path to the generated split.
    """
    cxl_cropped_split_path = generate_split(
        dataset=dataset,
        mode="openset",
        seed=seed,
        reid_factor_test=reid_factor_test,
        reid_factor_val=reid_factor_val,
    )

    # information on subjects in different split sets
    (
        test_proprietary_subjects,
        val_proprietary_subjects,
        train_proprietary_subjects,
    ) = get_test_val_train_proprietary_subjects(os.path.abspath(cxl_cropped_split_path))

    meta_data.update([("subjects_train_proprietary", train_proprietary_subjects)])
    meta_data.update([("subjects_val_proprietary", val_proprietary_subjects)])
    meta_data.update([("subjects_test_proprietary", test_proprietary_subjects)])

    save_dict_json(meta_data, os.path.join(cxl_cropped_split_path, "metadata.json"))

    return str(cxl_cropped_split_path)


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

    # NOTE: The path to the bristol dataset split has to be inside the gorilla_yml_path file. Set breakpoint here to check / change it.
    model, model_name = modify_dataset_train_yolo(
        bristol_split_dir,
        model_type="yolov8x",
        epochs=1,
        batch_size=16,
    )

    # If you don't want to train a yolo model
    model_name = "yolov8x-e30-b163"
    model_path = "/workspaces/gorillatracker/models/yolov8x-e30-b163/weights/best.pt"
    model = ultralytics.YOLO(model_path)

    bristol_split_dir = "/workspaces/gorillatracker/data/splits/ground_truth-bristol-full_images-openset-reid-val-0-test-0-mintraincount-3-seed-69-train-70-val-15-test-15"
    detect_gorillafaces_cxl(model, model_name)
    meta_data, face_crop_dataset = crop_images_save_metadata_cxl(model_name, bristol_split_dir)
    # convert absolute to relative paths
    face_crop_dataset = os.path.relpath(face_crop_dataset, "/workspaces/gorillatracker/data")
    generate_split_save_metadata_cxl(face_crop_dataset, meta_data, seed=69)

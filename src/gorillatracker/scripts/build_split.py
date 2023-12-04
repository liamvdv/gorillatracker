"""Builds the different splits for the bristol dataset and the cxl dataset. If specified also trains a yolo model on the bristol dataset."""

import json
import os
import shutil
from typing import Any, Dict

import ultralytics

from gorillatracker.scripts.crop_dataset import crop_images
from gorillatracker.scripts.dataset_splitter import generate_split
from gorillatracker.scripts.ensure_integrity_openset import ensure_integrity, get_subjects_in_directory
from gorillatracker.scripts.train_yolo import build_dataset_train_remove, detect_gorillafaces_cxl


def merge_every_single_set_of_splits(split1_dir: str, split2_dir: str, output_dir: str) -> None:
    """Merges two splits into a new split.

    Args:
        split1_dir (str): Directory of the first split.
        split2_dir (str): Directory of the second split.
        output_dir (str): Directory to save the merged split to.
    """

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Copy train, val and test folders from split1 to output_dir
    shutil.copytree(os.path.join(split1_dir, "train"), os.path.join(output_dir, "train"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(split1_dir, "val"), os.path.join(output_dir, "val"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(split1_dir, "test"), os.path.join(output_dir, "test"), dirs_exist_ok=True)
    # Copy train, val and test folders from split2 to output_dir
    shutil.copytree(os.path.join(split2_dir, "train"), os.path.join(output_dir, "train"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(split2_dir, "val"), os.path.join(output_dir, "val"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(split2_dir, "test"), os.path.join(output_dir, "test"), dirs_exist_ok=True)


def merge_split2_into_train_set_of_split1(split1_dir: str, split2_dir: str, output_dir: str) -> None:
    """Merges all sets of split2 into the train set of split1.

    Args:
        split1_dir (str): Directory of the first split.
        split2_dir (str): Directory of the second split.
        output_dir (str): Directory to save the merged split to.
    """
    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Copy train, val and test folders from split1 to output_dir
    shutil.copytree(os.path.join(split1_dir, "train"), os.path.join(output_dir, "train"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(split1_dir, "val"), os.path.join(output_dir, "val"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(split1_dir, "test"), os.path.join(output_dir, "test"), dirs_exist_ok=True)
    # Copy train, val and test folders from split2 to output_dir
    shutil.copytree(os.path.join(split2_dir, "train"), os.path.join(output_dir, "train"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(split2_dir, "val"), os.path.join(output_dir, "train"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(split2_dir, "test"), os.path.join(output_dir, "train"), dirs_exist_ok=True)


def save_dict_json(dict: Dict[Any, Any], file_path: str) -> None:
    """Saves the given dictionary to the given file path as json."""
    with open(file_path, "w") as file:
        json.dump(dict, file, indent=4, sort_keys=True)


if __name__ == "__main__":
    # Define all needed paths
    bristol_dir = "/workspaces/gorillatracker/data/ground_truth/bristol"
    relative_path_to_bristol = "ground_truth/bristol"
    bristol_annotation_dir = os.path.join(bristol_dir, "full_images_face_bbox")
    bristol_yolo_annotation_dir = os.path.join(bristol_dir, "full_images_face_bbox_class0")

    gorilla_yml_path = "/workspaces/gorillatracker/data/ground_truth/bristol/gorilla.yaml"

    cxl_dir = "/workspaces/gorillatracker/data/ground_truth/cxl"
    cxl_imgs_dir = os.path.join(cxl_dir, "full_images")
    cxl_annotation_dir = "/workspaces/gorillatracker/data/derived_data/cxl_annotations_yolov8x-e30-b163"
    crop_cxl_imgs_dir = "/workspaces/gorillatracker/data/derived_data/cxl_faces_cropped_yolov8x-e30-b163"

    model_path = "/workspaces/gorillatracker/models/yolov8x-e30-b163/weights/best.pt"

    # 1. split the bristol dataset into train, val and test
    bristol_split_dir = generate_split(
        dataset=os.path.join(relative_path_to_bristol, "full_images"),
        mode="openset-strict",
        seed=69,
        reid_factor_test=10,
        reid_factor_val=10,
    )

    bristol_split_train_dir = os.path.join(bristol_split_dir, "train")
    bristol_split_val_dir = os.path.join(bristol_split_dir, "val")
    bristol_split_test_dir = os.path.join(bristol_split_dir, "test")
    bristol_split_bbox_dir = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images_face_bbox"

    # 2. ensure integrity of the bristol split (only necessary for openset)
    ensure_integrity(bristol_split_train_dir, bristol_split_val_dir, bristol_split_test_dir, bristol_split_bbox_dir)

    # 3. train yolo model on the bristol dataset split (openset)
    model_name = "yolov8x"
    epochs = 2
    batch_size = 16
    model = build_dataset_train_remove(
        bristol_annotation_dir,
        bristol_yolo_annotation_dir,
        bristol_split_dir,
        model_name,
        epochs,
        batch_size,
        gorilla_yml_path,
        wandb_project="Detection-YOLOv8-Bristol-OpenSet",
    )

    # 4. predict on the cxl dataset -> save to directory xy -> set model_path
    model = ultralytics.YOLO(model_path)
    detect_gorillafaces_cxl(model, cxl_imgs_dir, output_dir=cxl_annotation_dir)

    # 5a crop cxl images  according to predicted bounding boxes
    imgs_without_bbox, imgs_with_no_bbox_prediction, imgs_with_low_confidence = crop_images(
        cxl_imgs_dir, cxl_annotation_dir, crop_cxl_imgs_dir, is_bristol=False, file_extension=".png"
    )

    # 5b save information for the cropped images to file metadata.json (in the cxl only and the joined split as well)
    meta_data = {
        "yolo-model": str(model_name),
        "yolo-path": str(model_path),
        "bristol-split": str(bristol_split_dir),
        "imgs-without-bbox": imgs_without_bbox,
        "imgs-with-no-bbox-prediction": imgs_with_no_bbox_prediction,
        "imgs-with-low-confidence": imgs_with_low_confidence,
        "cxl-annotation-dir": str(cxl_annotation_dir),
    }
    save_dict_json(meta_data, os.path.join(crop_cxl_imgs_dir, "metadata.json"))

    # 5c create split for cxl dataset
    cxl_cropped_split_path = generate_split(
        dataset="derived_data/cxl_faces_cropped_yolov8x-e30-b163",
        mode="openset-strict-half-known",
        seed=69,
        reid_factor_test=10,
        reid_factor_val=10,
    )

    # information on subjects in different split sets
    val_subjects = get_subjects_in_directory(
        os.path.join(cxl_cropped_split_path, "val"), file_extension=".png", name_delimiter="_"
    )
    train_subjects = get_subjects_in_directory(
        os.path.join(cxl_cropped_split_path, "train"), file_extension=".png", name_delimiter="_"
    )
    test_subjects = get_subjects_in_directory(
        os.path.join(cxl_cropped_split_path, "test"), file_extension=".png", name_delimiter="_"
    )
    
    test_proprietary_subjects = test_subjects - train_subjects - val_subjects
    val_proprietary_subjects = val_subjects - train_subjects - test_subjects
    train_proprietary_subjects = train_subjects - val_subjects - test_subjects
    
    meta_data.update([("subjects_train", list(train_subjects))])
    meta_data.update([("subjects_val", list(val_subjects))])
    meta_data.update([("subjects_test", list(test_subjects))])
    meta_data.update([("subjects_train_proprietary", list(train_proprietary_subjects))])
    meta_data.update([("subjects_val_proprietary", list(val_proprietary_subjects))])
    meta_data.update([("subjects_test_proprietary", list(test_proprietary_subjects))])

    save_dict_json(meta_data, os.path.join(cxl_cropped_split_path, "metadata.json"))

    # crop bristol images
    bristol_cropped_path = os.path.join(bristol_dir, "cropped_images_face")
    bristol_img_dir = os.path.join(bristol_dir, "full_images")
    bristol_annotation_dir = os.path.join(bristol_dir, "full_images_face_bbox")
    crop_images(
        bristol_img_dir, bristol_annotation_dir, bristol_cropped_path, is_bristol=True, file_extension=".jpg"
    )


    bristol_cropped_path_relative = os.path.join(
        relative_path_to_bristol, "cropped_images_face"
    )  # NOTE generate_split wants relative path and returns absolute path
    bristol_cropped_split_path = generate_split(
        dataset=bristol_cropped_path_relative, mode="openset-strict-half-known", seed=69, reid_factor_test=10, reid_factor_val=10
    )

    # 6. merge bristol and cxl dataset if wanted    
    joined_base_dir = "/workspaces/gorillatracker/data/joined_splits"

    merge_dir = os.path.join(joined_base_dir, "faces_cropped_bristol_cxl_cropped_yolov8x-e30-b163_seed=69")
    merge_every_single_set_of_splits(bristol_cropped_split_path, cxl_cropped_split_path, merge_dir)
    merge_split2_into_train_set_of_split1(
        cxl_cropped_split_path, bristol_cropped_split_path, merge_dir + "_bristol_trainonly"
    )

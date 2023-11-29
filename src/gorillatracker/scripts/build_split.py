import json
import os
import shutil
import sys
from collections import defaultdict

import ultralytics

from gorillatracker.scripts.crop_dataset import crop_images
from gorillatracker.scripts.dataset_splitter import generate_split
from gorillatracker.scripts.ensure_integrity_openset import ensure_integrity
from gorillatracker.scripts.train_yolo import (
    detect_gorillafaces_cxl,
    join_annotations_and_imgs,
    remove_annotations_from_dir,
    set_annotation_class_0,
    train_yolo,
)


def merge_every_single_set_of_splits(split1_dir, split2_dir, output_dir):
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


def merge_split2_into_train_set_of_split1(split1_dir, split2_dir, output_dir):
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


def save_dict_json(dict, file_path):
    with open(file_path, "w") as file:
        json.dump(dict, file)


if __name__ == "__main__":
    # 1. split the bristol dataset into train, val and test
    bristol_dir = "/workspaces/gorillatracker/data/ground_truth/bristol"
    relative_path_to_bristol = "ground_truth/bristol"
    bristol_split_dir = generate_split(
        dataset=os.path.join(relative_path_to_bristol, "full_images"),
        mode="openset",
        seed=42,
        reid_factor_test=10,
        reid_factor_val=10,
    )

    # 2. ensure integrity of the bristol split (only necessary for openset)
    # bristol_split_dir = "/workspaces/gorillatracker/data/splits/ground_truth-bristol-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-42-train-70-val-15-test-15"
    bristol_split_train_dir = os.path.join(bristol_split_dir, "train")
    bristol_split_val_dir = os.path.join(bristol_split_dir, "val")
    bristol_split_test_dir = os.path.join(bristol_split_dir, "test")
    bristol_split_bbox_dir = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images_face_bbox"

    ensure_integrity(bristol_split_train_dir, bristol_split_val_dir, bristol_split_test_dir, bristol_split_bbox_dir)

    # # 3. train yolo model on the bristol dataset split (openset)
    model_name = "yolov8x"
    epochs = 2
    batch_size = 16

    # 3a build dataset for yolo
    bristol_annotation_dir = os.path.join(bristol_dir, "full_images_face_bbox")
    bristol_yolo_annotation_dir = os.path.join(bristol_dir, "full_images_face_bbox_class0")
    set_annotation_class_0(bristol_annotation_dir, bristol_yolo_annotation_dir)

    # 3b then train yolo (note that you have to set the path to the different sets in the yml file)
    gorilla_yml_path = "/workspaces/gorillatracker/data/ground_truth/bristol/gorilla.yaml"

    # take a split of the bristol dataset
    # bristol_split_dir = "/workspaces/gorillatracker/data/splits/ground_truth-bristol-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-42-train-70-val-15-test-15"
    # YOLO needs the images and annotations in the same folder
    join_annotations_and_imgs(
        os.path.join(bristol_split_dir, "train"), bristol_yolo_annotation_dir, os.path.join(bristol_split_dir, "train")
    )
    join_annotations_and_imgs(
        os.path.join(bristol_split_dir, "val"), bristol_yolo_annotation_dir, os.path.join(bristol_split_dir, "val")
    )
    join_annotations_and_imgs(
        os.path.join(bristol_split_dir, "test"), bristol_yolo_annotation_dir, os.path.join(bristol_split_dir, "test")
    )

    # train_yolo(model_name, epochs, batch_size, gorilla_yml_path, wandb_project="Detection-YOLOv8-Bristol-OpenSet", wandb_run_name="yolov8x")

    # 3c remove annotations from the bristol split
    for split in ["train", "val", "test"]:
        remove_annotations_from_dir(os.path.join(bristol_split_dir, split))

    # 4. predict on the cxl dataset -> save to directory xy -> set model_path
    cxl_dir = "/workspaces/gorillatracker/data/ground_truth/cxl"
    cxl_imgs_dir = os.path.join(cxl_dir, "full_images")
    cxl_annotation_dir = "/workspaces/gorillatracker/data/derived_data/cxl_annotations_yolov8x-e30-b163"
    model_path = "/workspaces/gorillatracker/models/yolov8x-e30-b163/weights/best.pt"
    model = ultralytics.YOLO(model_path)
    detect_gorillafaces_cxl(model, cxl_imgs_dir, output_dir=cxl_annotation_dir)

    # 5. crop cxl images  according to predicted bounding boxes
    crop_cxl_imgs_dir = "/workspaces/gorillatracker/data/derived_data/cxl_faces_cropped_yolov8x-e30-b163"
    imgs_without_bbox = crop_images(
        cxl_imgs_dir, cxl_annotation_dir, crop_cxl_imgs_dir, is_bristol=False, file_extension=".png"
    )

    # 5b save information for the cropped images to file metadata.json (in the cxl only and the joined split as well)
    meta_data = {
        "yolo-model": str(model_name),
        "yolo-path": str(model_path),
        "bristol-split": str(bristol_split_dir),
        "imgs-without-bbox": imgs_without_bbox,
        "cxl-annotation-dir": str(cxl_annotation_dir),
    }
    save_dict_json(meta_data, os.path.join(crop_cxl_imgs_dir, "metadata.json"))

    # 5b create split for cxl dataset
    cxl_cropped_split_path = generate_split(
        dataset="derived_data/cxl_faces_cropped_yolov8x-e30-b163",
        mode="openset",
        seed=42,
        reid_factor_test=10,
        reid_factor_val=10,
    )
    save_dict_json(meta_data, os.path.join(cxl_cropped_split_path, "metadata.json"))

    bristol_cropped_path = os.path.join(
        relative_path_to_bristol, "cropped_images_face"
    )  # NOTE generate_split wants relative path and returns absolute path
    bristol_cropped_split_path = generate_split(
        dataset=bristol_cropped_path, mode="openset", seed=42, reid_factor_test=10, reid_factor_val=10
    )

    # 6. merge bristol and cxl dataset if wanted
    joined_base_dir = "/workspaces/gorillatracker/data/joined_splits"

    merge_dir = os.path.join(joined_base_dir, "faces_cropped_bristol_cxl_cropped_yolov8x-e30-b163")
    merge_every_single_set_of_splits(bristol_cropped_split_path, cxl_cropped_split_path, merge_dir)
    merge_split2_into_train_set_of_split1(
        cxl_cropped_split_path, bristol_cropped_split_path, merge_dir + "_bristol_trainonly"
    )

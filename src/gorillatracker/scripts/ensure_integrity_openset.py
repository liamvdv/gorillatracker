import os
import shutil

index_to_name = {0: "afia", 1: "ayana", 2: "jock", 3: "kala", 4: "kera", 5: "kukuena", 6: "touni"}
name_to_index = {value: key for key, value in index_to_name.items()}


def move_images_of_subjects(image_folder, bbox_folder, output_folder, subjects_indicies):
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
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    move_count = 0
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        bbox_path = os.path.join(bbox_folder, image_file.replace(".jpg", ".txt"))

        if os.path.exists(bbox_path):
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
                    shutil.move(image_path, output_folder)
                    print(f"Moved image '{image_file}' to '{output_folder}'")
                    move_count += 1
        else:
            raise Exception(f"Bounding box file '{bbox_path}' does not exist for image '{image_file}'")

    return move_count


def filter_images(image_folder, bbox_folder):
    """Remove all images from the image folder that do not contain a bounding box of the actual subject."""

    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    remove_count = 0
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        bbox_path = os.path.join(bbox_folder, image_file.replace(".jpg", ".txt"))

        if os.path.exists(bbox_path):
            # Read bounding box coordinates from the text file
            bbox_data = []

            with open(bbox_path, "r") as bbox_file:
                bbox_data = bbox_file.read().strip().split()

            actual_subject = image_file.split("-")[0]
            actual_subject_index = name_to_index[actual_subject]
            seen_actual_subject = False
            for i in range(0, len(bbox_data), 5):
                index, x, y, w, h = map(
                    float, bbox_data[i : (i + 5)]
                )  # x,y is the center of the bbox and w,h are the width and height
                if index == actual_subject_index:
                    seen_actual_subject = True
            if not seen_actual_subject:
                print(
                    f"Warning: Actual subject '{actual_subject}' not found in bounding box file '{bbox_path}' for image '{image_file}'"
                )
                print(f"Removing image '{image_file}' from folder '{image_folder}'")
                os.remove(image_path)
                remove_count += 1
        else:
            raise Exception(f"Bounding box file '{bbox_path}' does not exist for image '{image_file}'")

    return remove_count


def get_subjects_in_directory(test_folder):
    """Get all subjects in the given directory. Subjects are identified by the prefix of the image file name."""
    # Get list of image files
    image_files = [f for f in os.listdir(test_folder) if f.endswith(".jpg")]
    print(f"Found {len(image_files)} images in folder '{test_folder}'")
    test_subjects = set()
    for image_file in image_files:
        test_subjects.add(image_file.split("-")[0])
    return test_subjects


def ensure_integrity(train_set_path, val_set_path, test_set_path, bbox_folder):
    """Ensure that the given train, val and test sets are valid.
    This means that the train set does not contain any images that by accident contain the subject of the val or test set.
    """
    test_subjects = get_subjects_in_directory(test_set_path)
    val_subjects = get_subjects_in_directory(val_set_path)
    train_subjects = get_subjects_in_directory(train_set_path)

    test_proprietary_subjects = test_subjects - val_subjects - train_subjects
    test_proprietary_subjects = [name_to_index[s] for s in test_proprietary_subjects]

    val_proprietary_subjects = val_subjects - train_subjects  # dont substract the test subjects
    val_proprietary_subjects = [name_to_index[s] for s in val_proprietary_subjects]

    print(f"Test proprietary subjects: {test_proprietary_subjects}")
    print(f"Val proprietary subjects: {val_proprietary_subjects}")

    assert len(test_proprietary_subjects) != 0, f"Test proprietary subjects is empty: {test_proprietary_subjects}"
    assert len(val_proprietary_subjects) != 0, f"Val proprietary subjects is empty: {val_proprietary_subjects}"

    # ensure that every image has a bounding box for the actual subject
    remove_count = filter_images(train_set_path, bbox_folder)
    remove_count += filter_images(val_set_path, bbox_folder)
    remove_count += filter_images(test_set_path, bbox_folder)
    print(f"Removed {remove_count} images")

    # filter out images in train and val that contain test_proprietary_subjects
    move_count_train_to_test = move_images_of_subjects(
        train_set_path, bbox_folder, test_set_path, test_proprietary_subjects
    )
    print(f"Moved {move_count_train_to_test} images from train to test")
    move_count_val_to_test = move_images_of_subjects(
        val_set_path, bbox_folder, test_set_path, test_proprietary_subjects
    )
    print(f"Moved {move_count_val_to_test} images from val to test")

    # filter out images in train and test that contain val_proprietary_subjects
    move_count_train_to_val = move_images_of_subjects(
        train_set_path, bbox_folder, val_set_path, val_proprietary_subjects
    )
    print(f"Moved {move_count_train_to_val} images from train to val")


if __name__ == "__main__":
    # raise Exception("This script is not ready for use")

    # check if split is valid
    base_folder = "/workspaces/gorillatracker/data/splits/ground_truth-bristol-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-43-train-70-val-15-test-15"
    test_set_path = os.path.join(base_folder, "test")
    val_set_path = os.path.join(base_folder, "val")
    train_set_path = os.path.join(base_folder, "train")
    bbox_folder = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images_face_bbox"

    ensure_integrity(train_set_path, val_set_path, test_set_path, bbox_folder)

"""Scripts to crop the images in the bristol dataset using the bounding boxes provided by the dataset."""

import os

from PIL import Image

index_to_name = {0: "afia", 1: "ayana", 2: "jock", 3: "kala", 4: "kera", 5: "kukuena", 6: "touni"}


def crop_and_save_image(image_path, x, y, w, h, output_path):
    """Crop the image at the given path using the given bounding box coordinates and save it to the given output path.

    Args:
        image_path (str): Path to the image to crop.
        x (float): Relative x coordinate of the center of the bounding box.
        y (float): Relative y coordinate of the center of the bounding box.
        w (float): Relative width of the bounding box.
        h (float): Relative height of the bounding box.
        output_path (str): Path to save the cropped image to.
    """
    # Open the image
    img = Image.open(image_path)

    # Calculate pixel coordinates from relative coordinates
    img_width, img_height = img.size
    left = int((x - w / 2) * img_width)
    right = int((x + w / 2) * img_width)
    top = int((y - h / 2) * img_height)
    bottom = int((y + h / 2) * img_height)

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Save the cropped image to the output folder
    cropped_img.save(output_path)


def read_bbox_data(bbox_path):
    """Read the bounding box data from the given file.

    Args:
        bbox_path (str): Path to the bounding box file.

    Returns:
        list: List of bounding box data lines."""
    # check if the file exists
    if not os.path.exists(bbox_path):
        print("warning: no bounding box found for image with path " + bbox_path)
        return []

    bbox_data_lines = []
    with open(bbox_path, "r") as bbox_file:
        bbox_data_lines = bbox_file.read().strip().split("\n")

    bbox_data_lines = [list(map(float, bbox_data_line.strip().split(" "))) for bbox_data_line in bbox_data_lines]

    return bbox_data_lines


def crop_bristol(image_path, bbox_path, output_dir):
    """Crops a single image from the bristol dataset."""
    bbox_data_lines = read_bbox_data(bbox_path)

    for index, x, y, w, h in bbox_data_lines:
        name = index_to_name[index]
        file_name = name + "_" + os.path.basename(image_path)
        output_path = os.path.join(output_dir, file_name)
        crop_and_save_image(image_path, x, y, w, h, output_path)


def crop_cxl(image_path, bbox_path, output_dir):
    """Crops a single image from the cxl dataset.
    NOTE: There is only one bounding box per image. Therefore, only the bounding box with the highest confidence score is used.
    """
    bbox_data_lines = read_bbox_data(bbox_path)
    bbo_max_confidence_idx, bbox_max_confidence = max(
        enumerate(bbox_data_lines), key=lambda x: x[1][-1], default=(-1, -1)
    )  # get the shortest line in the file

    if bbo_max_confidence_idx != -1:  # when there is a bounding box
        index, x, y, w, h, _ = bbox_max_confidence
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        crop_and_save_image(image_path, x, y, w, h, output_path)


def crop_images(image_dir, bbox_dir, output_dir, file_extension=".jpg", is_bristol=True):
    """Crop all images in the given directory using the bounding boxes in the given directory and save them to the given output directory.

    Args:
        image_dir (str): Directory containing the images.
        bbox_dir (str): Directory containing the bounding box files.
        output_dir (str): Directory to save the cropped images to.
        file_extension (str, optional): File extension of the images. Defaults to ".jpg".
        is_bristol (bool, optional): Whether the images are from the bristol dataset. Defaults to True."""

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    imgs_without_bbox = []
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(file_extension)]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        bbox_path = os.path.join(bbox_dir, image_file.replace(file_extension, ".txt"))

        if not os.path.exists(bbox_path):
            print("warning: no bounding box found for image " + image_file)
            imgs_without_bbox.append(image_file)
            continue

        if is_bristol:
            crop_bristol(image_path, bbox_path, output_dir)
        else:
            crop_cxl(image_path, bbox_path, output_dir)

    return imgs_without_bbox


if __name__ == "__main__":
    full_images_folder = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images"
    bbox_folder = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images_face_bbox"
    output_folder = "/workspaces/gorillatracker/data/ground_truth/bristol/cropped_images_face"

    # crop_images(full_images_folder, bbox_folder, output_folder)
    test_bbox = "/workspaces/gorillatracker/data/joined_splits/combined/train/afia-1-img-0.txt"
    data = read_bbox_data(test_bbox)
    print(data)

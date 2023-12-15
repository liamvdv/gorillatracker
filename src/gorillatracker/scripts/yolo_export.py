# NOTE: https://github.com/ultralytics/ultralytics/issues/5800
# also did not work for the bbox export

import cv2

import gorillatracker.utils.cvat_import as cvat_import
import gorillatracker.utils.yolo_helpers as yolo_helpers


# NOTE(memben): for now we are only storing the body bbox
# TODO(memben): generalize the export to segments
def export_cvat_to_yolo(xml_file: str, target_dir: str, full_images_dir: str) -> None:
    segmented_images = cvat_import.cvat_import(xml_file, full_images_dir)

    unique_labels = sorted({label for img in segmented_images for label in img.segments})
    class_labels = {label: idx for idx, label in enumerate(unique_labels)}

    for segmented_image in segmented_images:
        img = cv2.imread(segmented_image.path)
        img_width, img_height, _ = img.shape

        for class_label, segment_list in segmented_image.segments.items():
            yolo_boxes = []
            for mask, box in segment_list:
                x, y, w, h = yolo_helpers.convert_to_yolo_format(box, img_height, img_width)
                yolo_boxes.append(f"{class_labels[class_label]} {x} {y} {w} {h}")

            with open(f"{target_dir}/{segmented_image.filename}.txt", "w") as file:
                file.write("\n".join(yolo_boxes))


if __name__ == "__main__":
    xml_file = "/workspaces/gorillatracker/data/ground_truth/cxl/full_images_body_instance_segmentation/cvat_export.xml"
    target_dir = "/workspaces/gorillatracker/data/ground_truth/cxl/full_images_body_bbox"
    full_images = "/workspaces/gorillatracker/data/ground_truth/cxl/full_images"
    export_cvat_to_yolo(xml_file, target_dir, full_images)

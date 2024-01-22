import os

import cv2

import gorillatracker.utils.cutout_helpers as cutout_helpers


def _remove_face(body_image_path: str, segmented_body_image_path: str, face_image_path: str, target_path: str) -> None:
    BACKGROUND_COLOR = 255  # white

    body_image = cv2.imread(body_image_path)
    segmented_body_image = cv2.imread(segmented_body_image_path)
    face_image = cv2.imread(face_image_path)
    bbox = cutout_helpers.get_cutout_bbox(body_image, face_image)
    segmented_body_image[bbox[0][1] : bbox[1][1], bbox[0][0] : bbox[1][0]] = BACKGROUND_COLOR
    cv2.imwrite(target_path, segmented_body_image)


def remove_faces(body_image_dir: str, segmented_body_image_dir: str, face_image_dir: str, target_dir: str) -> None:
    body_image_names = os.listdir(body_image_dir)
    os.makedirs(target_dir, exist_ok=True)
    for body_image_name in body_image_names:
        body_image_path = os.path.join(body_image_dir, body_image_name)
        segmented_body_image_path = os.path.join(segmented_body_image_dir, body_image_name)
        face_image_path = os.path.join(face_image_dir, body_image_name)
        target_path = os.path.join(target_dir, body_image_name)

        _remove_face(body_image_path, segmented_body_image_path, face_image_path, target_path)


# if __name__ == "__main__":
#     remove_faces(
#         "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/body_images",
#         "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/segmented_body_images",
#         "/workspaces/gorillatracker/data/ground_truth/cxl/face_images",
#         "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/segmented_body_images_faceless",
#     )

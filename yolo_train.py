import shutil
import time

from ultralytics import YOLO

WANDB_PROJECT = "Detection-YoloV8-CXLBody-ClosedSet"
WANDB_ENTITY = "gorillas"


def train(
    model_path: str,
    training_name: str,
    data: str,
    epochs: int = 100,
    batch_size: int = -1,
    patience: int = 40,
) -> None:
    model = YOLO(model_path)
    training_name = f"{training_name}"
    model.train(
        project=WANDB_PROJECT, name=training_name, data=data, epochs=epochs, batch=batch_size, patience=patience
    )
    shutil.move(WANDB_PROJECT, f"logs/{WANDB_PROJECT}-{training_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")


if __name__ == "__main__":
    # train("/workspaces/gorillatracker/models/yolov8n_gorillaface_pkm2bzis.pt", "new-face-data", data="/workspaces/gorillatracker/cfgs/yolo_detection_face.yaml", epochs=200, patience=50)
    train("/workspaces/gorillatracker/yolov8n.pt", "#198-body-fmc", "/workspaces/gorillatracker/cfgs/yolo_detection_body.yaml", epochs=200, patience=50)

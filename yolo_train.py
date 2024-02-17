import shutil
import time

from ultralytics import YOLO

WANDB_PROJECT = "Detection-YoloV8-CXLFace-ClosedSet"
WANDB_ENTITY = "gorillas"


def train(
    model: YOLO,
    training_name: str,
    data: str,
    epochs: int = 100,
    batch_size: int = -1,
    patience: int = 40,
) -> None:
    training_name = f"{training_name}"
    model.train(
        project=WANDB_PROJECT, name=training_name, data=data, epochs=epochs, batch=batch_size, patience=patience
    )
    shutil.move(WANDB_PROJECT, f"logs/{WANDB_PROJECT}-{training_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")
    
def tune(
    model: YOLO,
    training_name: str,
    data: str,
    iterations: int = 30,
) -> None:
    model.tune(
        project=WANDB_PROJECT, name=training_name, data=data, iterations=iterations, epochs=150, batch=-1
    )
    shutil.move(WANDB_PROJECT, f"logs/{WANDB_PROJECT}-{training_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")

if __name__ == "__main__":
    model = YOLO("/workspaces/gorillatracker/models/OLD_MODELS/yolov8n_gorillaface_pkm2bzis.pt")
    run = "#198-tune-face-n"
    data = "/workspaces/gorillatracker/cfgs/yolo_detection_face.yaml"
    tune(model, run, data, iterations=10)

    #   /("/workspaces/gorillatracker/models/yolov8n_gorillaface_pkm2bzis.pt", "new-face-data", data="/workspaces/gorillatracker/cfgs/yolo_detection_face.yaml", epochs=200, patience=50)
    # train("/workspaces/gorillatracker/yolov8n.pt", "#198-hyperbody-fmc", "/workspaces/gorillatracker/cfgs/yolo_detection_body.yaml", epochs=200, patience=50)

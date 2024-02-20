import shutil
import time

from ultralytics import YOLO

WANDB_PROJECT = "Detection-YoloV8-CXLBody-ClosedSet"
WANDB_ENTITY = "gorillas"


def train(
    model: YOLO,
    training_name: str,
    data: str,
    epochs: int = 100,
    batch_size: int = -1, # -1 autobatch
    patience: int = 40,
) -> None:
    training_name = f"{training_name}"
    model.train(
        project=WANDB_PROJECT, name=training_name, data=data, epochs=epochs, batch=batch_size, patience=patience, single_cls=True
    )
    shutil.move(WANDB_PROJECT, f"logs/{WANDB_PROJECT}-{training_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")


def tune(
    model: YOLO,
    training_name: str,
    data: str,
    iterations: int = 30,
) -> None:
    model.tune(project=WANDB_PROJECT, name=training_name, data=data, iterations=iterations, epochs=200, batch=-1, single_cls=True) # -1 autobatch
    shutil.move(WANDB_PROJECT, f"logs/{WANDB_PROJECT}-{training_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")


if __name__ == "__main__":
    # model = YOLO("path/to/your/yolov8n.pt")
    # run = "#your-run-name"
    # data = "path/to/your/yolo_detection.yaml"    
    model = YOLO("/workspaces/gorillatracker/yolov8n.pt")
    # run = "#198-tune-gorilla-silverback-yolov8n"
    run = "#198-train-gorilla-yolov8n"
    data = "/workspaces/gorillatracker/cfgs/yolo_detection_body_gs.yaml"
    # tune(model, run, data, iterations=5)
    train(model, run, data, epochs=200, batch_size=32, patience=40)

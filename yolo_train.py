import shutil
import time

from ultralytics import YOLO

WANDB_PROJECT = "Detection-YoloV8-CXLBody"
WANDB_ENTITY = "gorillas"


model_paths = {
    "yolov8n": "./models/yolov8n.pt",
    "yolov8m": "./models/yolov8m.pt",
    "yolov8x": "./models/yolov8x.pt",
}


def train(
    model_name: str,
    training_name: str,
    data: str = "./gorilla.yaml",
    epochs: int = 100,
    batch_size: int = -1,
    patience: int = 40,
) -> None:
    model = YOLO(model_paths[model_name])
    training_name = f"{training_name}-{model_name}"
    model.train(
        project=WANDB_PROJECT, name=training_name, data=data, epochs=epochs, batch=batch_size, patience=patience
    )
    shutil.move(WANDB_PROJECT, f"logs/{WANDB_PROJECT}-{training_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")


# if __name__ == "__main__":
#     train("yolov8x", "#36-max-power-test", epochs=200, patience=50)

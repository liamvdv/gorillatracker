import shutil
import time
from typing import Optional

import typer
from ultralytics import YOLO

app = typer.Typer()

WANDB_PROJECT = "Detection-YoloV8-CXLFace-ClosedSet"
WANDB_ENTITY = "gorillas"


@app.command()
def train(
    model_path: str,
    training_name: str,
    data: str,
    epochs: int = 100,
    batch_size: int = -1,  # -1 for autobatch
    patience: int = 40,
) -> None:
    model = YOLO(model_path)
    training_name = f"{training_name}"
    print(f"Training {training_name}")
    model.train(
        project=WANDB_PROJECT, name=training_name, data=data, epochs=epochs, batch=batch_size, patience=patience, single_cls=True
    )
    print(f"Training of {training_name} finished")
    shutil.move(WANDB_PROJECT, f"logs/{WANDB_PROJECT}-{training_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")


@app.command()
def tune(
    model_path: str,
    training_name: str,
    data: str,
    iterations: int = 30,
) -> None:
    model = YOLO(model_path)
    model.tune(project=WANDB_PROJECT, name=training_name, data=data, iterations=iterations, epochs=200, single_cls=True)
    shutil.move(WANDB_PROJECT, f"logs/{WANDB_PROJECT}-{training_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")


if __name__ == "__main__":
    app()